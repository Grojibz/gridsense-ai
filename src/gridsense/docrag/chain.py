"""The DocRAG chain: retrieve -> ground -> structured, cited answer, with tracing.

Returns a :class:`RagAnswer` of ``{answer, citations[], confidence}``. The model is asked
to answer *only* from the retrieved context and to reference sources by their context index;
citations are then rebuilt deterministically from the retrieved chunks, so a citation can
never point at a document that wasn't actually retrieved.

Guardrails (M2): retrieved chunks below a relevance threshold are dropped, and if nothing
relevant remains — or the model's self-reported confidence is too low — DocRAG refuses with
an explicit "I don't know" instead of guessing.

Every call is traced to Langfuse when keys are configured (prompt, retrieved chunks +
relevance scores, latency, cost, and the guardrail decision). Tracing is best-effort: any
observability error is swallowed so it can never break answering.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field

from gridsense.config import Settings, get_settings
from gridsense.observability import get_langfuse

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from langchain_postgres import PGVector
    from langfuse import Langfuse

NO_ANSWER = "I don't know — I couldn't find relevant information in the documents."

#: Sentinel so callers can pass ``langfuse_client=None`` to disable tracing explicitly,
#: while the default (unset) builds a client from settings.
_UNSET: Any = object()

T = TypeVar("T")

SYSTEM_PROMPT = (
    "You are a precise technical assistant for battery energy-storage engineering. "
    "Answer the user's question using ONLY the numbered context passages provided. "
    "Do not use prior knowledge. If the context does not contain enough information to "
    "answer, say you don't know and set confidence to 0.\n\n"
    "In the `sources` field, list the numbers of every passage you used — if you gave an "
    "answer, `sources` must not be empty. "
    "Set `confidence` between 0 and 1 reflecting how well the context supports your answer."
)


class Citation(BaseModel):
    """A reference to a retrieved chunk that supports the answer."""

    source: str = Field(description="Source identifier (e.g. file path) of the chunk.")
    snippet: str = Field(description="Short excerpt of the cited chunk.")


class RagAnswer(BaseModel):
    """Structured answer returned by DocRAG."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="0..1 grounding confidence.")


class _LLMAnswer(BaseModel):
    """Raw structured output requested from the LLM (citations rebuilt from this)."""

    answer: str = Field(description="The answer, grounded only in the context.")
    confidence: float = Field(description="0..1 confidence the context supports the answer.")
    sources: list[int] = Field(
        default_factory=list,
        description="1-based indices of the context passages actually used.",
    )


def _safe(fn: Callable[[], T]) -> T | None:
    """Run a best-effort tracing call, swallowing any error."""
    try:
        return fn()
    except Exception:
        return None


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks as a numbered, source-tagged context block."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(blocks)


def _build_citations(llm_sources: list[int], docs: list[Document]) -> list[Citation]:
    """Map the LLM's 1-based source indices back to real retrieved chunks."""
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for idx in llm_sources:
        if not 1 <= idx <= len(docs):
            continue  # ignore out-of-range indices the model may invent
        doc = docs[idx - 1]
        source = str(doc.metadata.get("source", "unknown"))
        snippet = " ".join(doc.page_content.split())[:240]
        key = (source, snippet)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(source=source, snippet=snippet))
    return citations


def _finalize_trace(
    trace: Any,
    client: Langfuse | None,
    result: RagAnswer,
    *,
    reason: str,
    scores: list[float],
) -> None:
    """Record the outcome on the trace and flush it (best-effort)."""
    if trace is None:
        return
    _safe(
        lambda: trace.update(
            output={
                "answer": result.answer,
                "confidence": result.confidence,
                "num_citations": len(result.citations),
            },
            metadata={"guardrail": reason, "relevance_scores": scores},
        )
    )
    if client is not None:
        _safe(client.flush)


def answer_question(
    question: str,
    *,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    k: int | None = None,
    settings: Settings | None = None,
    langfuse_client: Langfuse | None = _UNSET,
) -> RagAnswer:
    """Answer ``question`` from the document store, returning a cited, scored answer.

    ``vectorstore``, ``chat_model`` and ``langfuse_client`` can be injected (e.g. fakes in
    tests, or ``langfuse_client=None`` to disable tracing); when omitted they default to the
    real pgvector store, the configured provider, and a Langfuse client built from settings.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    settings = settings or get_settings()
    k = k if k is not None else settings.docrag_top_k
    if vectorstore is None:
        from gridsense.docrag.retriever import get_vectorstore

        vectorstore = get_vectorstore()
    if chat_model is None:
        from gridsense.providers import get_chat_model

        chat_model = get_chat_model()
    if langfuse_client is _UNSET:
        langfuse_client = get_langfuse(settings)

    trace = None
    if langfuse_client is not None:
        trace = _safe(
            lambda: langfuse_client.trace(name="docrag.ask", input={"q": question, "k": k})
        )

    # --- Retrieve + relevance gate ----------------------------------------
    pairs = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    scores = [round(float(score), 4) for _, score in pairs]
    docs = [doc for doc, score in pairs if score >= settings.docrag_min_relevance]

    if trace is not None:
        _safe(
            lambda: trace.span(
                name="retrieve",
                input={"question": question, "k": k},
                output={
                    "scores": scores,
                    "kept": [d.metadata.get("source") for d in docs],
                },
            ).end()
        )

    if not docs:
        result = RagAnswer(answer=NO_ANSWER, citations=[], confidence=0.0)
        _finalize_trace(trace, langfuse_client, result, reason="no_relevant_context", scores=scores)
        return result

    # --- Generate ----------------------------------------------------------
    context = format_context(docs)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context passages:\n\n{context}\n\nQuestion: {question}"),
    ]
    structured = chat_model.with_structured_output(_LLMAnswer)

    # Time the generation ourselves and record it as a Langfuse generation observation.
    # (We don't use Langfuse's LangChain callback: its v2 integration imports the legacy
    # `langchain.callbacks` module, which LangChain v1 removed.)
    started = datetime.now()
    llm_out: _LLMAnswer = structured.invoke(messages)
    ended = datetime.now()

    if trace is not None:
        model_label = getattr(chat_model, "model", None) or settings.llm_provider.value
        _safe(
            lambda: trace.generation(
                name="generate",
                model=model_label,
                input=[{"role": m.type, "content": m.content} for m in messages],
                output=llm_out.model_dump(),
                start_time=started,
                end_time=ended,
            )
        )

    confidence = max(0.0, min(1.0, llm_out.confidence))

    # --- Confidence gate ---------------------------------------------------
    if confidence < settings.docrag_min_confidence:
        result = RagAnswer(answer=NO_ANSWER, citations=[], confidence=confidence)
        _finalize_trace(trace, langfuse_client, result, reason="low_confidence", scores=scores)
        return result

    citations = _build_citations(llm_out.sources, docs)
    if not citations:
        # The model answered but didn't enumerate sources (common with smaller models);
        # fall back to citing the retrieved passages it was grounded on.
        citations = _build_citations(list(range(1, len(docs) + 1)), docs)

    result = RagAnswer(answer=llm_out.answer, citations=citations, confidence=confidence)
    _finalize_trace(trace, langfuse_client, result, reason="answered", scores=scores)
    return result

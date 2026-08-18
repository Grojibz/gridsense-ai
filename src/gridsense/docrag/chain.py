"""The DocRAG chain: guard -> retrieve -> ground -> structured, cited answer, with tracing.

Returns a :class:`RagAnswer`. The model is asked to answer *only* from the retrieved
context and to reference sources by their context index; citations are then rebuilt
deterministically from the retrieved chunks, so a citation can never point at a document
that wasn't actually retrieved.

Guardrails run on both sides of the model (:mod:`gridsense.docrag.guardrails`):

- **Upstream** — the query is length-capped, scanned for prompt injection, routed for
  intent, and PII-scrubbed *before* it reaches the embedder, the model, or the trace store.
- **Retrieval** — chunks below a relevance threshold are dropped; if nothing relevant
  remains, DocRAG refuses rather than answer from noise.
- **Downstream** — structured output that fails validation is retried once and then falls
  back deterministically instead of raising; the answer is checked sentence-by-sentence
  against the retrieved chunks; and confidence, retrieval strength and grounding are
  combined into an explicit uncertainty verdict.

A shaky answer is *disclosed*, not suppressed: ``uncertainty_level`` and
``uncertainty_reasons`` say why, so the caller never has to guess whether a fluent answer
was actually well supported.

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
from gridsense.docrag.guardrails import (
    UncertaintyLevel,
    assess_uncertainty,
    check_input,
    groundedness,
    invoke_structured,
)
from gridsense.observability import get_langfuse

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from langchain_postgres import PGVector
    from langfuse import Langfuse

NO_ANSWER = "I don't know — I couldn't find relevant information in the documents."
#: Distinct from NO_ANSWER: the documents may well hold the answer, but the model failed to
#: return a usable object. Saying so beats implying the corpus is empty on the subject.
UNUSABLE_OUTPUT = (
    "I couldn't produce a reliable answer just now — the model's response was malformed. "
    "Please try again."
)

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
    refused: bool = Field(
        default=False,
        description="True when a guardrail declined to answer. Check this rather than "
        "string-matching the answer text.",
    )
    refusal_reason: str | None = Field(
        default=None, description="Which guardrail refused (e.g. no_relevant_context)."
    )
    uncertainty_level: UncertaintyLevel = Field(
        default="low", description="How much to trust this answer: low | medium | high."
    )
    uncertainty_reasons: list[str] = Field(
        default_factory=list, description="Signals behind the uncertainty verdict."
    )
    uncertainty_note: str = Field(
        default="", description="User-facing caveat to render with the answer."
    )
    groundedness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of answer sentences lexically supported by the cited chunks.",
    )


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


def retrieve_context(
    question: str,
    *,
    vectorstore: PGVector,
    k: int,
    settings: Settings,
) -> tuple[list[tuple[Document, float]], list[float]]:
    """Retrieve the top-``k`` chunks and drop those below the relevance floor.

    Returns ``(kept, all_scores)`` — the surviving ``(document, score)`` pairs, and the
    relevance scores of *all* ``k`` candidates. Keeping the kept chunks paired with their
    own scores is deliberate: zipping a filtered document list against an unfiltered score
    list silently misattributes scores as soon as one chunk is dropped. The full score list
    is what tells you a query fell off the corpus entirely.
    """
    pairs = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    scores = [round(float(score), 4) for _, score in pairs]
    kept = [
        (doc, round(float(score), 4))
        for doc, score in pairs
        if score >= settings.docrag_min_relevance
    ]
    return kept, scores


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
            metadata={
                "guardrail": reason,
                "relevance_scores": scores,
                "refused": result.refused,
                "uncertainty_level": result.uncertainty_level,
                "uncertainty_reasons": result.uncertainty_reasons,
                "groundedness": result.groundedness,
            },
        )
    )
    if client is not None:
        _safe(client.flush)


def _refuse(reason: str, *, message: str = NO_ANSWER, confidence: float = 0.0) -> RagAnswer:
    """Build the refusal for ``reason``, always flagged high-uncertainty and self-describing."""
    return RagAnswer(
        answer=message,
        citations=[],
        confidence=confidence,
        refused=True,
        refusal_reason=reason,
        uncertainty_level="high",
        uncertainty_reasons=[reason],
    )


def answer_question(
    question: str,
    *,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    k: int | None = None,
    settings: Settings | None = None,
    langfuse_client: Langfuse | None = _UNSET,
    on_context: Callable[[list[tuple[Document, float]]], None] | None = None,
) -> RagAnswer:
    """Answer ``question`` from the document store, returning a cited, scored answer.

    ``vectorstore``, ``chat_model`` and ``langfuse_client`` can be injected (e.g. fakes in
    tests, or ``langfuse_client=None`` to disable tracing); when omitted they default to the
    real pgvector store, the configured provider, and a Langfuse client built from settings.

    ``on_context`` is called with the chunks that survived the relevance gate — exactly what
    the model is about to see. The eval harness uses it to score the answer against the real
    prompt context; re-running retrieval afterwards is not equivalent, because a chunk
    sitting on the relevance threshold can fall on either side of it between two calls.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    settings = settings or get_settings()
    k = k if k is not None else settings.docrag_top_k
    if langfuse_client is _UNSET:
        langfuse_client = get_langfuse(settings)

    # --- Upstream guardrails ------------------------------------------------
    # Runs before the trace is opened and before anything is constructed, so a rejected
    # query costs no embedding call, no LLM call, and never puts raw PII (or an injection
    # payload) into the trace store.
    verdict = check_input(question, settings)
    question = verdict.query

    trace = None
    if langfuse_client is not None:
        trace = _safe(
            lambda: langfuse_client.trace(
                name="docrag.ask",
                input={"q": question, "k": k},
                metadata={
                    "intent": verdict.intent,
                    "input_tokens": verdict.estimated_tokens,
                    "injection_score": verdict.injection_score,
                    "pii_labels": verdict.pii_labels,
                },
            )
        )

    if not verdict.allowed:
        result = _refuse(verdict.reason, message=verdict.message)
        _finalize_trace(trace, langfuse_client, result, reason=verdict.reason, scores=[])
        return result

    if vectorstore is None:
        from gridsense.docrag.retriever import get_vectorstore

        vectorstore = get_vectorstore()
    if chat_model is None:
        from gridsense.providers import get_chat_model

        chat_model = get_chat_model()

    # --- Retrieve + relevance gate ----------------------------------------
    kept, scores = retrieve_context(question, vectorstore=vectorstore, k=k, settings=settings)
    docs = [doc for doc, _ in kept]
    kept_scores = [score for _, score in kept]
    if on_context is not None:
        on_context(kept)

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
        result = _refuse("no_relevant_context")
        _finalize_trace(trace, langfuse_client, result, reason="no_relevant_context", scores=scores)
        return result

    # --- Generate ----------------------------------------------------------
    context = format_context(docs)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context passages:\n\n{context}\n\nQuestion: {question}"),
    ]

    # Time the generation ourselves and record it as a Langfuse generation observation.
    # (We don't use Langfuse's LangChain callback: its v2 integration imports the legacy
    # `langchain.callbacks` module, which LangChain v1 removed.)
    started = datetime.now()
    llm_out, attempts = invoke_structured(
        chat_model, _LLMAnswer, messages, retries=settings.docrag_output_retries
    )
    ended = datetime.now()

    if trace is not None:
        model_label = getattr(chat_model, "model", None) or settings.chat_provider.value
        _safe(
            lambda: trace.generation(
                name="generate",
                model=model_label,
                input=[{"role": m.type, "content": m.content} for m in messages],
                output=llm_out.model_dump() if llm_out is not None else None,
                metadata={"attempts": attempts, "schema_valid": llm_out is not None},
                start_time=started,
                end_time=ended,
            )
        )

    # --- Output schema gate ------------------------------------------------
    if llm_out is None:
        # Every attempt produced something unparseable. Fall back deterministically rather
        # than propagate the parser exception out of the API.
        result = _refuse("invalid_output", message=UNUSABLE_OUTPUT)
        _finalize_trace(trace, langfuse_client, result, reason="invalid_output", scores=scores)
        return result

    confidence = max(0.0, min(1.0, llm_out.confidence))

    # --- Confidence gate ---------------------------------------------------
    if confidence < settings.docrag_min_confidence:
        result = _refuse("low_confidence", confidence=confidence)
        _finalize_trace(trace, langfuse_client, result, reason="low_confidence", scores=scores)
        return result

    citations = _build_citations(llm_out.sources, docs)
    if not citations:
        # The model answered but didn't enumerate sources (common with smaller models);
        # fall back to citing the retrieved passages it was grounded on.
        citations = _build_citations(list(range(1, len(docs) + 1)), docs)

    # --- Faithfulness post-check + uncertainty disclosure -------------------
    grounding = groundedness(
        llm_out.answer,
        [d.page_content for d in docs],
        support_threshold=settings.docrag_min_groundedness,
    )
    uncertainty = assess_uncertainty(
        confidence=confidence,
        relevance_scores=kept_scores,
        grounding=grounding.ratio,
        settings=settings,
    )

    if trace is not None:
        _safe(
            lambda: trace.span(
                name="output_validate",
                input={"attempts": attempts},
                output={
                    "groundedness": grounding.ratio,
                    "unsupported": grounding.unsupported,
                    "uncertainty": uncertainty.level,
                    "reasons": uncertainty.reasons,
                },
            ).end()
        )

    result = RagAnswer(
        answer=llm_out.answer,
        citations=citations,
        confidence=confidence,
        uncertainty_level=uncertainty.level,
        uncertainty_reasons=uncertainty.reasons,
        uncertainty_note=uncertainty.note,
        groundedness=grounding.ratio,
    )
    _finalize_trace(trace, langfuse_client, result, reason="answered", scores=scores)
    return result

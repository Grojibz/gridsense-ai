"""The DocRAG chain: retrieve -> ground -> structured, cited answer.

Returns a :class:`RagAnswer` of ``{answer, citations[], confidence}``. The model is asked
to answer *only* from the retrieved context and to reference sources by their context index;
citations are then rebuilt deterministically from the retrieved chunks, so a citation can
never point at a document that wasn't actually retrieved. If nothing is retrieved we
short-circuit to an explicit "I don't know" (the guardrail is hardened further in M2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from langchain_postgres import PGVector

NO_ANSWER = "I don't know — I couldn't find relevant information in the documents."

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


def answer_question(
    question: str,
    *,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    k: int = 4,
) -> RagAnswer:
    """Answer ``question`` from the document store, returning a cited, scored answer.

    ``vectorstore`` and ``chat_model`` can be injected (e.g. fakes in tests); when omitted
    they default to the real pgvector store and the configured provider.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if vectorstore is None:
        from gridsense.docrag.retriever import get_vectorstore

        vectorstore = get_vectorstore()
    if chat_model is None:
        from gridsense.providers import get_chat_model

        chat_model = get_chat_model()

    docs = vectorstore.similarity_search(question, k=k)
    if not docs:
        return RagAnswer(answer=NO_ANSWER, citations=[], confidence=0.0)

    context = format_context(docs)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context passages:\n\n{context}\n\nQuestion: {question}"),
    ]

    structured = chat_model.with_structured_output(_LLMAnswer)
    result: _LLMAnswer = structured.invoke(messages)

    confidence = max(0.0, min(1.0, result.confidence))
    citations = _build_citations(result.sources, docs)
    if not citations and confidence > 0.0:
        # The model answered but didn't enumerate sources (common with smaller models);
        # fall back to citing the retrieved passages it was grounded on.
        citations = _build_citations(list(range(1, len(docs) + 1)), docs)
    return RagAnswer(answer=result.answer, citations=citations, confidence=confidence)

"""Unit tests for the DocRAG chain, using fakes so no DB/LLM/Langfuse is required.

All calls pass ``langfuse_client=None`` to keep tracing disabled and the tests offline.
"""

from __future__ import annotations

from langchain_core.documents import Document

from gridsense.config import Settings
from gridsense.docrag.chain import (
    NO_ANSWER,
    RagAnswer,
    _LLMAnswer,
    answer_question,
    format_context,
)

SETTINGS = Settings(docrag_min_relevance=0.5, docrag_min_confidence=0.3)


class FakeStructured:
    def __init__(self, output: _LLMAnswer) -> None:
        self._output = output

    def invoke(self, _messages: object, config: object = None) -> _LLMAnswer:
        return self._output


class FakeChatModel:
    """Stands in for a chat model whose structured output we control."""

    def __init__(self, output: _LLMAnswer) -> None:
        self._output = output

    def with_structured_output(self, _schema: object) -> FakeStructured:
        return FakeStructured(self._output)


class FakeVectorStore:
    def __init__(self, docs: list[Document], score: float = 0.9) -> None:
        self._docs = docs
        self._score = score

    def similarity_search_with_relevance_scores(
        self, _query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        return [(d, self._score) for d in self._docs[:k]]


def _docs() -> list[Document]:
    return [
        Document(page_content="SOH end of life is 80%.", metadata={"source": "a.md"}),
        Document(page_content="Thermal runaway is exothermic.", metadata={"source": "b.md"}),
    ]


def _ask(llm: FakeChatModel, store: FakeVectorStore) -> RagAnswer:
    return answer_question(
        "q", vectorstore=store, chat_model=llm, settings=SETTINGS, langfuse_client=None
    )


def test_answer_builds_citations_from_used_sources() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="End of life is 80% SOH.", confidence=0.9, sources=[1]))
    result = _ask(llm, FakeVectorStore(_docs()))

    assert isinstance(result, RagAnswer)
    assert result.answer == "End of life is 80% SOH."
    assert result.confidence == 0.9
    assert [c.source for c in result.citations] == ["a.md"]
    assert "80%" in result.citations[0].snippet


def test_out_of_range_source_indices_are_ignored() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="x", confidence=0.5, sources=[1, 99, 0]))
    result = _ask(llm, FakeVectorStore(_docs()))
    assert [c.source for c in result.citations] == ["a.md"]


def test_confidence_is_clamped_to_unit_interval() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="x", confidence=1.7, sources=[1]))
    result = _ask(llm, FakeVectorStore(_docs()))
    assert result.confidence == 1.0


def test_missing_sources_falls_back_to_retrieved_context() -> None:
    # Model answered confidently but didn't enumerate sources -> cite retrieved passages.
    llm = FakeChatModel(_LLMAnswer(answer="80%", confidence=1.0, sources=[]))
    result = _ask(llm, FakeVectorStore(_docs()))
    assert [c.source for c in result.citations] == ["a.md", "b.md"]


def test_duplicate_sources_are_deduplicated() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="x", confidence=0.6, sources=[1, 1]))
    result = _ask(llm, FakeVectorStore(_docs()))
    assert len(result.citations) == 1


def test_no_documents_short_circuits_to_dont_know() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="should not be used", confidence=1.0, sources=[1]))
    result = _ask(llm, FakeVectorStore([]))
    assert result.answer == NO_ANSWER
    assert result.confidence == 0.0
    assert result.citations == []


def test_low_relevance_chunks_are_gated_out() -> None:
    # All retrieved chunks score below the relevance threshold -> refuse.
    llm = FakeChatModel(_LLMAnswer(answer="should not be used", confidence=1.0, sources=[1]))
    result = _ask(llm, FakeVectorStore(_docs(), score=0.2))
    assert result.answer == NO_ANSWER
    assert result.citations == []
    assert result.confidence == 0.0


def test_low_confidence_answer_is_refused() -> None:
    # Context was relevant, but the model isn't confident -> refuse rather than guess.
    llm = FakeChatModel(_LLMAnswer(answer="maybe 70%?", confidence=0.1, sources=[1]))
    result = _ask(llm, FakeVectorStore(_docs()))
    assert result.answer == NO_ANSWER
    assert result.citations == []
    assert result.confidence == 0.1


def test_format_context_numbers_and_tags_sources() -> None:
    ctx = format_context(_docs())
    assert "[1] (source: a.md)" in ctx
    assert "[2] (source: b.md)" in ctx

"""Unit tests for the DocRAG chain, using fakes so no DB/LLM is required."""

from __future__ import annotations

from langchain_core.documents import Document

from gridsense.docrag.chain import (
    NO_ANSWER,
    RagAnswer,
    _LLMAnswer,
    answer_question,
    format_context,
)


class FakeStructured:
    def __init__(self, output: _LLMAnswer) -> None:
        self._output = output

    def invoke(self, _messages: object) -> _LLMAnswer:
        return self._output


class FakeChatModel:
    """Stands in for a chat model whose structured output we control."""

    def __init__(self, output: _LLMAnswer) -> None:
        self._output = output

    def with_structured_output(self, _schema: object) -> FakeStructured:
        return FakeStructured(self._output)


class FakeVectorStore:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def similarity_search(self, _query: str, k: int = 4) -> list[Document]:
        return self._docs[:k]


def _docs() -> list[Document]:
    return [
        Document(page_content="SOH end of life is 80%.", metadata={"source": "a.md"}),
        Document(page_content="Thermal runaway is exothermic.", metadata={"source": "b.md"}),
    ]


def test_answer_builds_citations_from_used_sources() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="End of life is 80% SOH.", confidence=0.9, sources=[1]))
    result = answer_question("eol?", vectorstore=FakeVectorStore(_docs()), chat_model=llm)

    assert isinstance(result, RagAnswer)
    assert result.answer == "End of life is 80% SOH."
    assert result.confidence == 0.9
    assert [c.source for c in result.citations] == ["a.md"]
    assert "80%" in result.citations[0].snippet


def test_out_of_range_source_indices_are_ignored() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="x", confidence=0.5, sources=[1, 99, 0]))
    result = answer_question("q", vectorstore=FakeVectorStore(_docs()), chat_model=llm)
    assert [c.source for c in result.citations] == ["a.md"]


def test_confidence_is_clamped_to_unit_interval() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="x", confidence=1.7, sources=[]))
    result = answer_question("q", vectorstore=FakeVectorStore(_docs()), chat_model=llm)
    assert result.confidence == 1.0


def test_missing_sources_falls_back_to_retrieved_context() -> None:
    # Model answered confidently but didn't enumerate sources -> cite retrieved passages.
    llm = FakeChatModel(_LLMAnswer(answer="80%", confidence=1.0, sources=[]))
    result = answer_question("q", vectorstore=FakeVectorStore(_docs()), chat_model=llm)
    assert [c.source for c in result.citations] == ["a.md", "b.md"]


def test_low_confidence_without_sources_yields_no_citations() -> None:
    # No claimed grounding (confidence 0) and no sources -> no fabricated citations.
    llm = FakeChatModel(_LLMAnswer(answer="unsure", confidence=0.0, sources=[]))
    result = answer_question("q", vectorstore=FakeVectorStore(_docs()), chat_model=llm)
    assert result.citations == []


def test_no_documents_short_circuits_to_dont_know() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="should not be used", confidence=1.0, sources=[1]))
    result = answer_question("q", vectorstore=FakeVectorStore([]), chat_model=llm)
    assert result.answer == NO_ANSWER
    assert result.confidence == 0.0
    assert result.citations == []


def test_duplicate_sources_are_deduplicated() -> None:
    llm = FakeChatModel(_LLMAnswer(answer="x", confidence=0.6, sources=[1, 1]))
    result = answer_question("q", vectorstore=FakeVectorStore(_docs()), chat_model=llm)
    assert len(result.citations) == 1


def test_format_context_numbers_and_tags_sources() -> None:
    ctx = format_context(_docs())
    assert "[1] (source: a.md)" in ctx
    assert "[2] (source: b.md)" in ctx

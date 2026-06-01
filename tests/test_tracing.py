"""Tests that the chain records the expected Langfuse spans via a fake client (offline)."""

from __future__ import annotations

from langchain_core.documents import Document

from gridsense.config import Settings
from gridsense.docrag.chain import _LLMAnswer, answer_question

SETTINGS = Settings(docrag_min_relevance=0.5, docrag_min_confidence=0.3)


class FakeStructured:
    def __init__(self, output: _LLMAnswer) -> None:
        self._output = output

    def invoke(self, _messages: object, config: object = None) -> _LLMAnswer:
        return self._output


class FakeChatModel:
    model = "fake-model"

    def __init__(self, output: _LLMAnswer) -> None:
        self._output = output

    def with_structured_output(self, _schema: object) -> FakeStructured:
        return FakeStructured(self._output)


class FakeVectorStore:
    def __init__(self, docs: list[Document], score: float = 0.9) -> None:
        self._docs = docs
        self._score = score

    def similarity_search_with_relevance_scores(self, _q: str, k: int = 4):
        return [(d, self._score) for d in self._docs[:k]]


class FakeSpan:
    def end(self, **_kw: object) -> None:
        pass


class FakeTrace:
    def __init__(self) -> None:
        self.generations: list[dict] = []
        self.updates: list[dict] = []
        self.spans: list[dict] = []

    def span(self, **kw: object) -> FakeSpan:
        self.spans.append(kw)
        return FakeSpan()

    def generation(self, **kw: object) -> None:
        self.generations.append(kw)

    def update(self, **kw: object) -> None:
        self.updates.append(kw)


class FakeLangfuse:
    def __init__(self) -> None:
        self.trace_obj = FakeTrace()
        self.trace_inputs: list[dict] = []
        self.flushed = 0

    def trace(self, **kw: object) -> FakeTrace:
        self.trace_inputs.append(kw)
        return self.trace_obj

    def flush(self) -> None:
        self.flushed += 1


def _docs() -> list[Document]:
    return [Document(page_content="SOH eol is 80%.", metadata={"source": "a.md"})]


def test_answered_call_records_retrieve_generation_update_and_flush() -> None:
    lf = FakeLangfuse()
    llm = FakeChatModel(_LLMAnswer(answer="80%", confidence=0.9, sources=[1]))

    answer_question(
        "q",
        vectorstore=FakeVectorStore(_docs()),
        chat_model=llm,
        settings=SETTINGS,
        langfuse_client=lf,
    )

    assert lf.trace_inputs and lf.trace_inputs[0]["input"] == {"q": "q", "k": 4}
    assert len(lf.trace_obj.spans) == 1  # retrieve span
    assert len(lf.trace_obj.generations) == 1
    assert lf.trace_obj.generations[0]["model"] == "fake-model"
    assert lf.trace_obj.updates[0]["metadata"]["guardrail"] == "answered"
    assert lf.flushed >= 1


def test_refusal_records_guardrail_reason_and_no_generation() -> None:
    lf = FakeLangfuse()
    llm = FakeChatModel(_LLMAnswer(answer="unused", confidence=1.0, sources=[1]))

    # All chunks below the relevance threshold -> refuse before generating.
    answer_question(
        "q",
        vectorstore=FakeVectorStore(_docs(), score=0.1),
        chat_model=llm,
        settings=SETTINGS,
        langfuse_client=lf,
    )

    assert lf.trace_obj.generations == []  # never generated
    assert lf.trace_obj.updates[0]["metadata"]["guardrail"] == "no_relevant_context"
    assert lf.flushed >= 1

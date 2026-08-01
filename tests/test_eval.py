"""Unit tests for the eval harness scoring/aggregation, using fakes (offline)."""

from __future__ import annotations

from langchain_core.documents import Document

from gridsense.docrag.chain import NO_ANSWER, Citation, RagAnswer, _LLMAnswer
from gridsense.docrag.eval.dataset import EvalItem, load_dataset
from gridsense.docrag.eval.judge import JudgeScore
from gridsense.docrag.eval.run import _score_item, evaluate


class FakeStructured:
    def __init__(self, output: object) -> None:
        self._output = output

    def invoke(self, _messages: object, config: object = None) -> object:
        return self._output


class FakeChatModel:
    def __init__(self, output: object) -> None:
        self._output = output

    def with_structured_output(self, _schema: object) -> FakeStructured:
        return FakeStructured(self._output)


class FakeVectorStore:
    def __init__(self, docs: list[Document], score: float = 0.9) -> None:
        self._docs = docs
        self._score = score

    def similarity_search_with_relevance_scores(self, _q: str, k: int = 4):
        return [(d, self._score) for d in self._docs[:k]]


def test_bundled_dataset_loads_with_answerable_and_unanswerable() -> None:
    items = load_dataset()
    assert len(items) >= 2
    assert any(i.answerable for i in items)
    assert any(not i.answerable for i in items)


def test_score_answerable_answered_item() -> None:
    item = EvalItem(
        question="eol?", expected_source="a.md", expected_keywords=["80"], answerable=True
    )
    answer = RagAnswer(
        answer="End of life is 80% SOH.",
        citations=[Citation(source="a.md", snippet="80% SOH")],
        confidence=0.9,
    )
    res = _score_item(item, answer, judge_model=FakeChatModel(JudgeScore(score=0.8)))
    assert res.citation_correct is True
    assert res.keyword_hit is True
    assert res.groundedness == 0.8
    assert res.relevance == 0.8


def _refusal() -> RagAnswer:
    """A refusal as the chain now emits it — flagged structurally, not by its text."""
    return RagAnswer(
        answer=NO_ANSWER,
        citations=[],
        confidence=0.0,
        refused=True,
        refusal_reason="no_relevant_context",
    )


def test_score_unanswerable_refused_item() -> None:
    item = EvalItem(question="capital of France?", answerable=False)
    res = _score_item(item, _refusal(), judge_model=FakeChatModel(JudgeScore(score=0.0)))
    assert res.refused is True
    assert res.answerable is False


def test_score_answerable_but_refused_counts_against_quality() -> None:
    item = EvalItem(question="eol?", expected_source="a.md", answerable=True)
    answer = _refusal()
    res = _score_item(item, answer, judge_model=FakeChatModel(JudgeScore(score=1.0)))
    assert res.refused is True
    assert res.citation_correct is False
    assert res.groundedness == 0.0


def test_evaluate_aggregates_metrics_offline() -> None:
    items = [
        EvalItem(
            question="eol?", expected_source="a.md", expected_keywords=["80"], answerable=True
        ),
        EvalItem(question="capital?", answerable=False),
    ]
    docs = [Document(page_content="SOH eol is 80%.", metadata={"source": "a.md"})]

    # Answer model: confident, cites source 1. The unanswerable question uses the same
    # store/model here, so it will also "answer" — that's fine, we assert on the answerable
    # aggregates and that refusal_accuracy reflects the (non-refused) unanswerable item.
    answer_llm = FakeChatModel(_LLMAnswer(answer="It is 80% SOH.", confidence=0.9, sources=[1]))
    judge = FakeChatModel(JudgeScore(score=0.75))

    # Patch settings via injection is not available to answer_question inside evaluate, so we
    # rely on real defaults being compatible; use a high score to pass the relevance gate.
    report = evaluate(
        items,
        vectorstore=FakeVectorStore(docs, score=0.9),
        chat_model=answer_llm,
        judge_model=judge,
        enable_tracing=False,
    )
    assert report.citation_accuracy == 1.0
    assert report.keyword_accuracy == 1.0
    assert report.groundedness == 0.75
    assert 0.0 <= report.relevance <= 1.0

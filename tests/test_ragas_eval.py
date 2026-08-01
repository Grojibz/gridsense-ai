"""Offline tests for the RAGAS eval plumbing: pipeline recording and aggregation.

RAGAS itself is not exercised here (it needs a judge LLM) — these cover the parts that
decide whether the gate is honest: what gets recorded, and how it is rolled up.
"""

from __future__ import annotations

from langchain_core.documents import Document

from gridsense.docrag.chain import NO_ANSWER, _LLMAnswer
from gridsense.docrag.eval.golden import ExpectedChunk, GoldenDataset, GoldenItem
from gridsense.docrag.eval.ragas_eval import CORE_METRICS, aggregate, run_pipeline, score_with_ragas


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
    """Returns per-query hits, so in-corpus and off-corpus queries behave differently."""

    def __init__(self, by_query: dict[str, list[tuple[Document, float]]]) -> None:
        self._by_query = by_query

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
        return self._by_query.get(query, [])[:k]


IN_CORPUS = Document(
    page_content="80% SOH is the conventional end-of-life threshold.",
    metadata={"source": "battery_soh.md"},
)


def _dataset() -> GoldenDataset:
    return GoldenDataset(
        corpus="data/docs",
        items=[
            GoldenItem(
                id="eol",
                category="factual",
                answerable=True,
                query="eol threshold?",
                expected_chunks=[ExpectedChunk(source="battery_soh.md", must_contain=["80% SOH"])],
                reference_answer="80% SOH is end of life.",
            ),
            GoldenItem(
                id="oos",
                category="out_of_scope",
                answerable=False,
                query="capital of France?",
                reference_answer="Out of scope.",
            ),
        ],
    )


def _run() -> list:
    return run_pipeline(
        _dataset(),
        vectorstore=FakeVectorStore({"eol threshold?": [(IN_CORPUS, 0.9)]}),
        chat_model=FakeChatModel(
            _LLMAnswer(answer="End of life is 80% SOH.", confidence=0.9, sources=[1])
        ),
    )


def test_pipeline_records_retrieved_chunks_and_recall() -> None:
    answered, refused = _run()

    assert answered.id == "eol"
    assert answered.refused is False
    assert [c.source for c in answered.contexts] == ["battery_soh.md"]
    assert answered.relevance_scores == [0.9]
    assert answered.retrieval_recall == 1.0

    # Nothing retrieved for the out-of-scope query, so the relevance gate refuses.
    assert refused.id == "oos"
    assert refused.answer == NO_ANSWER
    assert refused.refused is True
    assert refused.contexts == []


def test_recorded_context_is_the_one_the_chain_used_not_a_second_retrieval() -> None:
    """Regression: the eval used to retrieve a second time to record contexts. A chunk on
    the relevance threshold could fall the other way, so RAGAS scored the answer against
    context the model never saw — or recorded an empty context for an answered item."""

    class DriftingStore:
        """Returns the chunk on the first call and nothing after — the worst case."""

        def __init__(self) -> None:
            self.calls = 0

        def similarity_search_with_relevance_scores(self, _q: str, k: int = 4) -> list:
            self.calls += 1
            return [(IN_CORPUS, 0.9)] if self.calls == 1 else []

    store = DriftingStore()
    records = run_pipeline(
        _dataset(),
        vectorstore=store,
        chat_model=FakeChatModel(
            _LLMAnswer(answer="End of life is 80% SOH.", confidence=0.9, sources=[1])
        ),
    )
    answered = records[0]
    assert store.calls == 2  # one per dataset item, not two per item
    assert answered.refused is False
    assert [c.source for c in answered.contexts] == ["battery_soh.md"]


def test_aggregate_splits_answerable_from_refusal_metrics() -> None:
    records = _run()
    records[0].ragas = dict.fromkeys(CORE_METRICS, 0.9)
    report = aggregate(records)

    assert report.n_answerable == 1
    assert report.n_unanswerable == 1
    assert report.scores["faithfulness"] == 0.9
    assert report.scores["retrieval_recall"] == 1.0
    assert report.scores["refusal_accuracy"] == 1.0
    assert report.coverage["faithfulness"] == 1


def test_unscored_metric_aggregates_to_zero_with_zero_coverage() -> None:
    report = aggregate(_run())
    assert report.scores["faithfulness"] == 0.0
    assert report.coverage["faithfulness"] == 0


def test_answerable_but_refused_scores_zero_without_calling_the_judge() -> None:
    """An unjustified refusal is a failure, not a datapoint the judge gets to skip."""
    records = run_pipeline(
        _dataset(),
        vectorstore=FakeVectorStore({}),  # retrieves nothing for either query
        chat_model=FakeChatModel(_LLMAnswer(answer="unused", confidence=0.9, sources=[])),
    )
    score_with_ragas(records)  # no judged items -> ragas is never imported

    assert records[0].ragas == dict.fromkeys(CORE_METRICS, 0.0)
    assert aggregate(records).scores["faithfulness"] == 0.0


def test_refusal_caused_by_unparseable_output_does_not_count_as_correct() -> None:
    """Right answer, wrong reason: a model too broken to reply must not score as a
    well-behaved guardrail. This inflated refusal_accuracy from 0.78 to 1.00."""

    class BrokenModel:
        def with_structured_output(self, _schema: object) -> BrokenModel:
            return self

        def invoke(self, _messages: object, config: object = None) -> object:
            raise ValueError("Failed to parse")

    # Both queries retrieve context, so both reach the model and both fail to parse.
    records = run_pipeline(
        _dataset(),
        vectorstore=FakeVectorStore(
            {"eol threshold?": [(IN_CORPUS, 0.9)], "capital of France?": [(IN_CORPUS, 0.9)]}
        ),
        chat_model=BrokenModel(),
    )
    unanswerable = records[1]
    assert unanswerable.refused is True
    assert unanswerable.refusal_reason == "invalid_output"

    report = aggregate(records)
    assert report.scores["refusal_accuracy"] == 0.0  # not credited despite refusing
    assert report.scores["output_validity"] == 0.0


def test_guardrail_refusal_still_counts_and_keeps_output_validity_clean() -> None:
    report = aggregate(_run())
    assert report.scores["refusal_accuracy"] == 1.0
    assert report.scores["output_validity"] == 1.0


def test_report_renders_scores_with_coverage() -> None:
    records = _run()
    records[0].ragas = dict.fromkeys(CORE_METRICS, 0.9)
    rendered = aggregate(records).render()

    assert "faithfulness" in rendered
    assert "0.900" in rendered
    assert "[1/1 scored]" in rendered

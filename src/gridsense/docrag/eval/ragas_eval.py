"""RAGAS evaluation of the DocRAG pipeline over the golden dataset.

Runs the *real* chain (same retriever, same prompt, same guardrails as `/ask`) over every
golden query, then scores the answerable items with the four core RAGAS metrics:

===================== ==============================================================
``faithfulness``      are the answer's claims entailed by the retrieved chunks?
``answer_relevancy``  does the answer actually address the query?
``context_precision`` are the retrieved chunks relevant, and ranked relevance-first?
``context_recall``    do the retrieved chunks cover the reference answer?
===================== ==============================================================

Two deterministic metrics are reported alongside them, because LLM-judged scores alone
make a poor merge gate:

``retrieval_recall``   fraction of ground-truth chunks actually retrieved (no LLM involved).
``refusal_accuracy``   fraction of vague / out-of-scope / uncovered queries correctly refused.
``output_validity``    fraction of items where the model returned a parseable object at all.

``refusal_accuracy`` deliberately does not credit a refusal caused by unparseable model
output: that is the right answer for the wrong reason, and counting it would make a flakier
model look better behaved. Those failures land in ``output_validity`` instead, where they
belong.

Answerable items that the pipeline refused are scored 0.0 on all four RAGAS metrics rather
than being sent to the judge — an unjustified refusal is a failure, not a missing datapoint.

Per-item scores are pushed to Langfuse as custom scores on an eval trace, so a regression
is visible next to the production traces rather than only in CI logs.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from gridsense.config import Settings, get_settings
from gridsense.docrag.chain import answer_question
from gridsense.docrag.eval.golden import (
    GoldenDataset,
    RetrievedChunk,
    load_golden_dataset,
    retrieval_recall,
)
from gridsense.observability import get_langfuse

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_postgres import PGVector
    from langfuse import Langfuse

#: RAGAS's internal metric names -> the names we report and gate on.
RAGAS_METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}
CORE_METRICS = tuple(RAGAS_METRIC_NAMES.values())

#: Refusal reason meaning "the model's output could not be parsed" — an infrastructure
#: failure, not a guardrail decision, and scored as such.
INVALID_OUTPUT = "invalid_output"

#: Sized for a local Ollama judge, which serialises requests: few workers, long timeout.
#: RAGAS's own defaults (16 workers / 180 s) starve it and every job times out to NaN.
RAGAS_TIMEOUT_SECONDS = 900
RAGAS_MAX_WORKERS = 2

#: A hosted judge is the opposite problem — it answers in seconds and serves requests in
#: parallel, so the local settings turn a two-minute eval into a twenty-minute one. RAGAS
#: issues several internal prompts per metric per item, which multiplies the difference.
HOSTED_TIMEOUT_SECONDS = 180
HOSTED_MAX_WORKERS = 8


def judge_run_config(settings: Settings | None = None) -> tuple[int, int]:
    """Return ``(timeout, max_workers)`` suited to the configured judge.

    Local and hosted judges fail in opposite directions: too many workers starves Ollama
    into NaN scores, too few makes a hosted run needlessly slow. Both failures are silent —
    one looks like a weak judge, the other like a slow CI job.
    """
    from gridsense.config import ChatProvider, get_settings

    settings = settings or get_settings()
    if settings.chat_provider is ChatProvider.ollama:
        return RAGAS_TIMEOUT_SECONDS, RAGAS_MAX_WORKERS
    return HOSTED_TIMEOUT_SECONDS, HOSTED_MAX_WORKERS


class ItemRecord(BaseModel):
    """One golden item run through the pipeline, with its scores."""

    id: str
    category: str
    answerable: bool
    query: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    confidence: float
    reference_answer: str
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    relevance_scores: list[float] = Field(default_factory=list)
    retrieval_recall: float = 0.0
    ragas: dict[str, float] = Field(default_factory=dict)


class RagasReport(BaseModel):
    """Aggregate eval result — this is what the threshold gate reads."""

    n_items: int
    n_answerable: int
    n_unanswerable: int
    #: Aggregate scores, keyed by metric name. The gate compares these to the thresholds.
    scores: dict[str, float]
    #: How many items each RAGAS metric actually produced a number for. A metric with low
    #: coverage has an untrustworthy mean, so the gate checks this too.
    coverage: dict[str, int]
    items: list[ItemRecord]

    def render(self) -> str:
        width = max(len(k) for k in self.scores)
        lines = [
            "DocRAG RAGAS eval",
            "=================",
            f"items: {self.n_items}  (answerable {self.n_answerable}, "
            f"unanswerable {self.n_unanswerable})",
            "",
        ]
        for name, value in self.scores.items():
            covered = self.coverage.get(name)
            suffix = f"   [{covered}/{self.n_answerable} scored]" if covered is not None else ""
            lines.append(f"{name:<{width}}  {value:.3f}{suffix}")
        return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_pipeline(
    dataset: GoldenDataset,
    *,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    settings: Settings | None = None,
    k: int | None = None,
) -> list[ItemRecord]:
    """Run every golden query through the real DocRAG chain and record what came back."""
    settings = settings or get_settings()
    k = k if k is not None else settings.docrag_top_k
    if vectorstore is None:
        from gridsense.docrag.retriever import get_vectorstore

        vectorstore = get_vectorstore()

    records: list[ItemRecord] = []
    for item in dataset.items:
        # Capture the chunks the chain actually put in the prompt. Retrieving separately
        # would be a different sample: a chunk sitting on the relevance threshold can fall
        # on either side of it between two calls, and RAGAS would then score the answer
        # against context the model never saw.
        contexts: list[RetrievedChunk] = []

        def collect(kept: list[tuple[Document, float]], sink: list = contexts) -> None:
            sink.extend(
                RetrievedChunk(
                    source=str(doc.metadata.get("source", "unknown")),
                    text=doc.page_content,
                    score=score,
                )
                for doc, score in kept
            )

        answer = answer_question(
            item.query,
            vectorstore=vectorstore,
            chat_model=chat_model,
            k=k,
            settings=settings,
            langfuse_client=None,  # the eval publishes its own trace, with scores attached
            on_context=collect,
        )
        scores = [c.score for c in contexts]
        records.append(
            ItemRecord(
                id=item.id,
                category=item.category,
                answerable=item.answerable,
                query=item.query,
                answer=answer.answer,
                refused=answer.refused,
                refusal_reason=answer.refusal_reason,
                confidence=answer.confidence,
                reference_answer=item.reference_answer,
                contexts=contexts,
                relevance_scores=scores,
                retrieval_recall=retrieval_recall(item, contexts),
            )
        )
    return records


def score_with_ragas(
    records: list[ItemRecord],
    *,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
    timeout: int | None = None,
    max_workers: int | None = None,
) -> None:
    """Fill in ``record.ragas`` for the answerable records, in place.

    Refused-but-answerable records are scored 0.0 without calling the judge.

    ``timeout`` and ``max_workers`` matter more than they look: RAGAS fans out one job per
    metric per item and defaults to 16 workers on a 180 s timeout. Against a local Ollama
    model that saturates the queue and every job times out, which yields a report full of
    NaN — a *silent* zero. Left unset, both are derived from the configured judge by
    :func:`judge_run_config`, because local and hosted judges need opposite settings.
    """
    settings = settings or get_settings()
    default_timeout, default_workers = judge_run_config(settings)
    timeout = default_timeout if timeout is None else timeout
    max_workers = default_workers if max_workers is None else max_workers
    for record in records:
        if record.answerable and record.refused:
            record.ragas = dict.fromkeys(CORE_METRICS, 0.0)

    judged = [r for r in records if r.answerable and not r.refused]
    if not judged:
        return

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    if llm is None or embeddings is None:
        from gridsense.providers import get_chat_model, get_embeddings

        llm = llm or get_chat_model(settings)
        embeddings = embeddings or get_embeddings(settings)

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r.query,
                response=r.answer,
                retrieved_contexts=[c.text for c in r.contexts],
                reference=r.reference_answer,
            )
            for r in judged
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(timeout=timeout, max_workers=max_workers),
        show_progress=False,
    )

    for record, row in zip(judged, result.scores, strict=True):
        record.ragas = {
            RAGAS_METRIC_NAMES[key]: float(value)
            for key, value in row.items()
            if key in RAGAS_METRIC_NAMES
            and isinstance(value, int | float)
            # RAGAS emits NaN when a metric could not be computed (e.g. the judge returned
            # nothing parseable). Drop it rather than poison the mean; `coverage` reports it.
            and not math.isnan(float(value))
        }


def aggregate(records: list[ItemRecord]) -> RagasReport:
    """Roll per-item scores up into the report the threshold gate consumes."""
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]

    scores: dict[str, float] = {}
    coverage: dict[str, int] = {}
    for metric in CORE_METRICS:
        values = [r.ragas[metric] for r in answerable if metric in r.ragas]
        scores[metric] = _mean(values)
        coverage[metric] = len(values)

    scores["retrieval_recall"] = _mean([r.retrieval_recall for r in answerable])
    # A refusal only counts as correct when a guardrail *decided* to refuse. Refusing
    # because the model emitted garbage is the right answer for the wrong reason, and
    # letting it score would mean a flakier model reads as a better-behaved one.
    scores["refusal_accuracy"] = _mean(
        [float(r.refused and r.refusal_reason != INVALID_OUTPUT) for r in unanswerable]
    )
    scores["output_validity"] = _mean([float(r.refusal_reason != INVALID_OUTPUT) for r in records])

    return RagasReport(
        n_items=len(records),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        scores=scores,
        coverage=coverage,
        items=records,
    )


def push_to_langfuse(
    report: RagasReport,
    *,
    client: Langfuse | None = None,
    run_name: str = "docrag.eval",
) -> int:
    """Publish one Langfuse trace per golden item, carrying its scores. Returns trace count.

    No-ops (returning 0) when Langfuse is not configured, so the eval still runs offline.
    """
    client = client if client is not None else get_langfuse()
    if client is None:
        return 0

    published = 0
    for record in report.items:
        trace = client.trace(
            name=run_name,
            input={"query": record.query, "category": record.category},
            output={"answer": record.answer, "refused": record.refused},
            metadata={
                "golden_id": record.id,
                "answerable": record.answerable,
                "sources": [c.source for c in record.contexts],
                "relevance_scores": record.relevance_scores,
            },
            tags=["eval", "ragas", record.category],
        )
        for metric, value in record.ragas.items():
            trace.score(name=metric, value=value, comment=f"RAGAS / {record.id}")
        if record.answerable:
            trace.score(name="retrieval_recall", value=record.retrieval_recall)
        else:
            trace.score(name="refused_correctly", value=float(record.refused))
        published += 1

    client.flush()
    return published


def run_eval(
    dataset: GoldenDataset | None = None,
    *,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    judge_model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
    k: int | None = None,
    publish: bool = True,
    timeout: int | None = None,
    max_workers: int | None = None,
) -> RagasReport:
    """Run the full golden-set evaluation and return the aggregate report."""
    dataset = dataset if dataset is not None else load_golden_dataset()
    records = run_pipeline(
        dataset, vectorstore=vectorstore, chat_model=chat_model, settings=settings, k=k
    )
    score_with_ragas(
        records,
        llm=judge_model,
        embeddings=embeddings,
        settings=settings,
        timeout=timeout,
        max_workers=max_workers,
    )
    report = aggregate(records)
    if publish:
        # Observability must never fail the eval.
        with contextlib.suppress(Exception):
            push_to_langfuse(report)
    return report

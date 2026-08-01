"""Run the DocRAG eval harness: score answers on groundedness, relevance, and citations.

Run as::

    python -m gridsense.docrag.eval.run            # ingest sample docs, then evaluate
    python -m gridsense.docrag.eval.run --no-ingest

Metrics (aggregated over the labelled set in ``questions.json``):

- **refusal_accuracy** — fraction of unanswerable questions correctly refused.
- **citation_accuracy** — fraction of answered questions citing the expected source.
- **keyword_accuracy** — fraction of answered questions containing the expected keywords.
- **groundedness / relevance** — mean LLM-as-judge scores over answered questions.

Citation/keyword/refusal are deterministic; groundedness/relevance use an LLM judge.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from pydantic import BaseModel

from gridsense.docrag.chain import RagAnswer, answer_question
from gridsense.docrag.eval.dataset import EvalItem, load_dataset
from gridsense.docrag.eval.judge import judge_groundedness, judge_relevance

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_postgres import PGVector

SAMPLE_DOCS = "data/docs"


class ItemResult(BaseModel):
    question: str
    answerable: bool
    answer: str
    refused: bool
    citation_correct: bool | None = None
    keyword_hit: bool | None = None
    groundedness: float | None = None
    relevance: float | None = None


class EvalReport(BaseModel):
    results: list[ItemResult]
    refusal_accuracy: float
    citation_accuracy: float
    keyword_accuracy: float
    groundedness: float
    relevance: float

    def render(self) -> str:
        lines = [
            "DocRAG eval report",
            "==================",
            f"items:             {len(self.results)}",
            f"refusal_accuracy:  {self.refusal_accuracy:.2f}",
            f"citation_accuracy: {self.citation_accuracy:.2f}",
            f"keyword_accuracy:  {self.keyword_accuracy:.2f}",
            f"groundedness:      {self.groundedness:.2f}",
            f"relevance:         {self.relevance:.2f}",
        ]
        return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score_item(
    item: EvalItem,
    answer: RagAnswer,
    *,
    judge_model: BaseChatModel,
) -> ItemResult:
    refused = answer.refused

    if not item.answerable:
        return ItemResult(
            question=item.question, answerable=False, answer=answer.answer, refused=refused
        )

    if refused:
        # Answerable but the system refused — counts against answer-quality metrics.
        return ItemResult(
            question=item.question,
            answerable=True,
            answer=answer.answer,
            refused=True,
            citation_correct=False,
            keyword_hit=False,
            groundedness=0.0,
            relevance=0.0,
        )

    sources = {c.source for c in answer.citations}
    citation_correct = item.expected_source in sources
    lower = answer.answer.lower()
    keyword_hit = all(kw.lower() in lower for kw in item.expected_keywords)
    context = "\n\n".join(c.snippet for c in answer.citations)

    return ItemResult(
        question=item.question,
        answerable=True,
        answer=answer.answer,
        refused=False,
        citation_correct=citation_correct,
        keyword_hit=keyword_hit,
        groundedness=judge_groundedness(answer.answer, context, chat_model=judge_model),
        relevance=judge_relevance(item.question, answer.answer, chat_model=judge_model),
    )


def evaluate(
    items: list[EvalItem] | None = None,
    *,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    judge_model: BaseChatModel | None = None,
    enable_tracing: bool = True,
) -> EvalReport:
    """Evaluate DocRAG over ``items`` (defaults to the bundled dataset).

    Set ``enable_tracing=False`` to disable Langfuse tracing (used by offline unit tests).
    """
    items = items if items is not None else load_dataset()
    if judge_model is None:
        from gridsense.providers import get_chat_model

        judge_model = get_chat_model()

    ask_kwargs: dict = {} if enable_tracing else {"langfuse_client": None}
    results: list[ItemResult] = []
    for item in items:
        answer = answer_question(
            item.question, vectorstore=vectorstore, chat_model=chat_model, **ask_kwargs
        )
        results.append(_score_item(item, answer, judge_model=judge_model))

    answerable = [r for r in results if r.answerable]
    answered = [r for r in answerable if not r.refused]
    unanswerable = [r for r in results if not r.answerable]

    return EvalReport(
        results=results,
        refusal_accuracy=_mean([float(r.refused) for r in unanswerable]),
        citation_accuracy=_mean([float(bool(r.citation_correct)) for r in answerable]),
        keyword_accuracy=_mean([float(bool(r.keyword_hit)) for r in answerable]),
        groundedness=_mean([r.groundedness for r in answered if r.groundedness is not None]),
        relevance=_mean([r.relevance for r in answered if r.relevance is not None]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DocRAG eval harness.")
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip ingesting the sample docs (evaluate against the current store).",
    )
    args = parser.parse_args(argv)

    if not args.no_ingest:
        from gridsense.docrag.ingest import ingest_path

        n = ingest_path(SAMPLE_DOCS)
        print(f"Ingested sample docs -> {n} chunk(s).\n")

    report = evaluate()
    print(report.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())

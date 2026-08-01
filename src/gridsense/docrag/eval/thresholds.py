"""Merge-gate thresholds for the DocRAG eval, and the check that enforces them.

The numbers below are the contract a PR has to satisfy. They are deliberately split:

- ``faithfulness`` and ``context_precision`` are the hard gate the roadmap calls for — an
  ungrounded answer or a retriever pulling in noise is a correctness bug, not a nuance.
- ``retrieval_recall`` and ``refusal_accuracy`` are deterministic, so they are the metrics
  that stay meaningful when the judge model is small; they are gated tightly.
- ``answer_relevancy`` and ``context_recall`` are gated loosely because they are the
  noisiest LLM-judged scores.

``MIN_COVERAGE_RATIO`` guards the failure mode where a metric's mean looks fine only
because the judge silently failed on most items.
"""

from __future__ import annotations

from pydantic import BaseModel

#: Metric -> minimum acceptable aggregate score.
THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.60,
    "answer_relevancy": 0.60,
    # Measured at 0.957 (k=4) on a sound index; 0.90 leaves room for corpus growth without
    # letting a real regression through. Raising k to 6 scores 1.000 only because it
    # returns the entire 6-chunk corpus — that is not retrieval, so k stays at 4.
    "retrieval_recall": 0.90,
    "refusal_accuracy": 0.80,
    # A model that cannot return a parseable object most of the time is not shippable,
    # whatever the answer-quality scores say. Local 7B models sit well under this; a
    # hosted model should be at 1.0.
    "output_validity": 0.95,
}

#: A judged metric must produce a score for at least this share of answerable items,
#: otherwise its mean is not trustworthy enough to gate on.
MIN_COVERAGE_RATIO = 0.80


class ThresholdViolation(BaseModel):
    """One metric that failed its gate."""

    metric: str
    actual: float
    threshold: float
    kind: str  # "score" or "coverage"

    def render(self) -> str:
        if self.kind == "coverage":
            return (
                f"  {self.metric}: only {self.actual:.0%} of items scored "
                f"(need {self.threshold:.0%}) — the judge failed too often to trust the mean"
            )
        return f"  {self.metric}: {self.actual:.3f} < {self.threshold:.2f}"


def check_thresholds(
    scores: dict[str, float],
    coverage: dict[str, int] | None = None,
    n_answerable: int = 0,
    thresholds: dict[str, float] | None = None,
) -> list[ThresholdViolation]:
    """Return every threshold violation in ``scores`` (empty list means the gate passes).

    A metric present in ``thresholds`` but missing from ``scores`` counts as a violation
    with an actual of 0.0 — a metric that silently vanished must not pass the gate.
    """
    thresholds = thresholds if thresholds is not None else THRESHOLDS
    violations = [
        ThresholdViolation(
            metric=metric, actual=scores.get(metric, 0.0), threshold=minimum, kind="score"
        )
        for metric, minimum in thresholds.items()
        if scores.get(metric, 0.0) < minimum
    ]

    if coverage and n_answerable:
        violations += [
            ThresholdViolation(
                metric=metric,
                actual=scored / n_answerable,
                threshold=MIN_COVERAGE_RATIO,
                kind="coverage",
            )
            for metric, scored in coverage.items()
            if scored / n_answerable < MIN_COVERAGE_RATIO
        ]
    return violations

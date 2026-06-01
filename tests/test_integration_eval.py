"""End-to-end eval-harness run against the live stack (pgvector + Ollama).

Skipped unless RUN_INTEGRATION=1. Deterministic metrics (refusal/citation/keyword) are
asserted strictly; LLM-judge metrics are only checked to be in range, since a small local
judge model is noisy (they are meaningful with a strong model such as Azure OpenAI).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="integration test; set RUN_INTEGRATION=1 with the live stack running",
)


def test_eval_harness_meets_deterministic_thresholds() -> None:
    from gridsense.docrag.eval.run import SAMPLE_DOCS, evaluate
    from gridsense.docrag.ingest import ingest_path

    ingest_path(SAMPLE_DOCS)
    report = evaluate()

    assert report.refusal_accuracy == 1.0, "unanswerable question should be refused"
    assert report.citation_accuracy >= 0.75
    assert report.keyword_accuracy >= 0.75
    assert 0.0 <= report.groundedness <= 1.0
    assert 0.0 <= report.relevance <= 1.0

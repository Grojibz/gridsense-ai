"""End-to-end RAGAS run over the golden dataset against the live stack.

Skipped unless RUN_INTEGRATION=1. This is the same code path the CI merge gate runs, so a
green run here means the gate is reproducible locally. Deterministic metrics are asserted
against the real thresholds; the LLM-judged RAGAS scores are only checked to be in range,
because the offline default (llama3.2 3B) is a noisy judge.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="integration test; set RUN_INTEGRATION=1 with the live stack running",
)


def test_ragas_eval_runs_over_the_golden_dataset() -> None:
    from gridsense.docrag.eval.golden import load_golden_dataset
    from gridsense.docrag.eval.ragas_eval import CORE_METRICS, run_eval
    from gridsense.docrag.eval.thresholds import THRESHOLDS
    from gridsense.docrag.ingest import ingest_path

    dataset = load_golden_dataset()
    ingest_path(dataset.corpus)
    report = run_eval(dataset, publish=False)

    assert report.n_items == len(dataset.items)
    assert report.scores["retrieval_recall"] >= THRESHOLDS["retrieval_recall"]
    assert report.scores["refusal_accuracy"] >= THRESHOLDS["refusal_accuracy"]
    for metric in CORE_METRICS:
        assert 0.0 <= report.scores[metric] <= 1.0

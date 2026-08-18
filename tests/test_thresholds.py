"""The merge gate: a threshold check that lets a regression through is worse than none."""

from __future__ import annotations

from gridsense.docrag.eval.thresholds import THRESHOLDS, check_thresholds

PASSING = {metric: minimum + 0.05 for metric, minimum in THRESHOLDS.items()}
FULL_COVERAGE = {"faithfulness": 10, "context_precision": 10}


def test_scores_above_every_threshold_pass() -> None:
    assert check_thresholds(PASSING) == []


def test_score_below_threshold_is_reported() -> None:
    scores = PASSING | {"faithfulness": 0.79}
    violations = check_thresholds(scores)
    assert [v.metric for v in violations] == ["faithfulness"]
    assert violations[0].actual == 0.79
    assert violations[0].threshold == 0.80


def test_roadmap_gates_are_enforced_at_the_agreed_levels() -> None:
    assert THRESHOLDS["faithfulness"] == 0.80
    assert THRESHOLDS["context_precision"] == 0.70


def test_missing_metric_fails_rather_than_passes_silently() -> None:
    scores = {k: v for k, v in PASSING.items() if k != "context_precision"}
    violations = check_thresholds(scores)
    assert [v.metric for v in violations] == ["context_precision"]
    assert violations[0].actual == 0.0


def test_high_scores_on_too_few_items_fail_the_coverage_check() -> None:
    violations = check_thresholds(PASSING, coverage={"faithfulness": 3}, n_answerable=10)
    assert [(v.metric, v.kind) for v in violations] == [("faithfulness", "coverage")]


def test_full_coverage_adds_no_violation() -> None:
    assert check_thresholds(PASSING, coverage=FULL_COVERAGE, n_answerable=10) == []


def test_custom_thresholds_override_the_defaults() -> None:
    assert check_thresholds({"faithfulness": 0.5}, thresholds={"faithfulness": 0.4}) == []

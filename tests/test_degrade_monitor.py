"""Offline unit tests for drift computation (uses Evidently, no DB)."""

from __future__ import annotations

from gridsense.degrade.data import generate_synthetic
from gridsense.degrade.monitor import compute_drift
from gridsense.degrade.serve import RAW_INPUT_COLUMNS


def _raw(n: int, seed: int):
    return generate_synthetic(n, seed=seed)[RAW_INPUT_COLUMNS]


def test_no_drift_when_distributions_match() -> None:
    reference = _raw(800, seed=1)
    current = _raw(300, seed=2)  # same generator distribution
    result = compute_drift(reference, current, drift_share_threshold=0.5)
    assert result.number_of_columns == len(RAW_INPUT_COLUMNS)
    assert result.alert is False


def test_alert_when_many_features_drift() -> None:
    reference = _raw(800, seed=1)
    current = _raw(300, seed=2).copy()
    # Push several features well outside the training distribution.
    current["avg_temperature_c"] += 40
    current["cycle_count"] += 2500
    current["calendar_age_days"] += 1500
    result = compute_drift(reference, current, drift_share_threshold=0.5)
    assert result.dataset_drift is True
    assert result.share_of_drifted_columns > 0.5
    assert result.alert is True

"""Offline unit tests for metrics, model card, and the pure training routine (no DB/MLflow)."""

from __future__ import annotations

import numpy as np

from gridsense.degrade.evaluate import build_model_card, regression_metrics
from gridsense.degrade.train import train_model

# Raw features available directly from the generator (the SQL-derived ones need a DB).
RAW_FEATURES = ["cycle_count", "avg_temperature_c", "avg_dod", "avg_c_rate", "calendar_age_days"]


def test_regression_metrics_on_perfect_prediction() -> None:
    y = np.array([80.0, 90.0, 100.0])
    m = regression_metrics(y, y.copy())
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_regression_metrics_keys_and_types() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = regression_metrics(y_true, y_pred)
    assert set(m) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) for v in m.values())


def test_train_model_learns_synthetic_relationship() -> None:
    from gridsense.degrade.data import TARGET, generate_synthetic

    df = generate_synthetic(2000, seed=0)
    result = train_model(df[RAW_FEATURES], df[TARGET], n_estimators=60)

    assert result.metrics["r2"] > 0.6  # the model captures most of the variance
    assert len(result.x_test) == 400  # 20% test split
    assert result.model.predict(result.x_test).shape == (400,)
    assert result.n_samples == 2000


def test_build_model_card_contains_metrics_and_sections() -> None:
    card = build_model_card(
        model_name="degrade-soh",
        algorithm="RandomForestRegressor",
        params={"n_estimators": 200, "max_depth": None},
        metrics={"mae": 0.5, "rmse": 0.7, "r2": 0.95},
        n_samples=1000,
        n_train=800,
        n_test=200,
        features=RAW_FEATURES,
        target="soh",
    )
    assert "# Model Card — degrade-soh" in card
    assert "## Test metrics" in card
    assert "0.9500" in card  # r2 rendered
    assert "`cycle_count`" in card

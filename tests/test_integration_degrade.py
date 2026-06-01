"""End-to-end DegradeML test against the live stack (Postgres + MLflow).

Skipped unless RUN_INTEGRATION=1. Exercises SQL feature building, training, and MLflow
tracking + registry.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="integration test; set RUN_INTEGRATION=1 with the live stack running",
)


def test_sql_feature_build_derives_columns() -> None:
    from gridsense.degrade.data import (
        FEATURE_COLUMNS,
        build_features,
        count_samples,
        get_engine,
        seed_database,
    )

    engine = get_engine()
    seed_database(500, seed=1, engine=engine)
    assert count_samples(engine) == 500

    x, y = build_features(engine)
    assert list(x.columns) == FEATURE_COLUMNS
    assert len(x) == 500
    # SQL-derived feature matches its definition.
    expected_efc = x["cycle_count"] * x["avg_dod"]
    assert (x["equivalent_full_cycles"] - expected_efc).abs().max() < 1e-6
    assert y.between(70.0, 100.0).all()


def test_run_training_logs_and_registers_model() -> None:
    import mlflow

    from gridsense.degrade.train import REGISTERED_MODEL, run_training

    out = run_training(n_estimators=60)
    assert out["run_id"]
    assert out["metrics"]["r2"] > 0.6

    # The run is retrievable and a model version was registered.
    client = mlflow.MlflowClient()
    run = client.get_run(out["run_id"])
    assert run.data.metrics["r2"] == out["metrics"]["r2"]
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    assert len(versions) >= 1

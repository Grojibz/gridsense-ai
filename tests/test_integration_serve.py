"""End-to-end serve + monitor test against the live stack (MLflow + Postgres).

Skipped unless RUN_INTEGRATION=1. Trains/registers a model, predicts (logging to Postgres),
then runs a drift report over deliberately-shifted inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="integration test; set RUN_INTEGRATION=1 with the live stack running",
)


def _row_count(engine) -> int:
    from sqlalchemy import text

    from gridsense.degrade.serve import PREDICTIONS_TABLE, ensure_predictions_table

    ensure_predictions_table(engine)
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {PREDICTIONS_TABLE}")).scalar() or 0)


def test_predict_logs_to_postgres_and_drift_alerts(tmp_path: Path) -> None:
    import gridsense.degrade.serve as serve
    from gridsense.degrade.data import generate_synthetic, get_engine
    from gridsense.degrade.monitor import run_drift_report
    from gridsense.degrade.serve import RAW_INPUT_COLUMNS, predict_soh
    from gridsense.degrade.train import run_training

    run_training(n_estimators=50)
    serve._cached_model.cache_clear()  # pick up the freshly registered version

    engine = get_engine()
    before = _row_count(engine)

    # Send predictions drawn from a deliberately drifted distribution.
    sample = generate_synthetic(40, seed=99)
    for _, row in sample.iterrows():
        feats = {c: float(row[c]) for c in RAW_INPUT_COLUMNS}
        # Shift a majority of features well outside the training range to force an alert.
        feats["avg_temperature_c"] += 40
        feats["cycle_count"] += 2500
        feats["calendar_age_days"] += 1400
        feats["avg_c_rate"] += 3.0
        out = predict_soh(feats, engine=engine)
        assert 70.0 <= out["predicted_soh"] <= 100.0
        assert out["model_version"]

    assert _row_count(engine) - before == 40

    result = run_drift_report(engine=engine, html_path=tmp_path / "drift.html")
    assert result.number_of_columns == len(RAW_INPUT_COLUMNS)
    assert result.alert is True  # we deliberately drifted the inputs
    assert (tmp_path / "drift.html").exists()

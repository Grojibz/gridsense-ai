"""Data-drift monitoring for DegradeML using Evidently.

Compares the distribution of recent live ``/predict`` inputs (from ``degrade_predictions``)
against the training distribution (from ``degrade_samples``) over the raw input features, and
raises an alert when too large a share of features have drifted. Saves an HTML report.

Run as::

    python -m gridsense.degrade.monitor
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel

from gridsense.config import Settings
from gridsense.degrade.data import SAMPLES_TABLE
from gridsense.degrade.serve import PREDICTIONS_TABLE, RAW_INPUT_COLUMNS

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

DEFAULT_DRIFT_SHARE_THRESHOLD = 0.5
DEFAULT_REPORT_PATH = "reports/drift.html"


class DriftResult(BaseModel):
    """Summary of a drift comparison."""

    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    dataset_drift: bool
    #: True when the drifted share exceeds the alert threshold.
    alert: bool


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
    html_path: str | Path | None = None,
) -> DriftResult:
    """Run Evidently's data-drift preset over ``reference`` vs ``current`` (no DB)."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    if html_path is not None:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_path))

    result = report.as_dict()["metrics"][0]["result"]
    share = float(result["share_of_drifted_columns"])
    return DriftResult(
        number_of_columns=int(result["number_of_columns"]),
        number_of_drifted_columns=int(result["number_of_drifted_columns"]),
        share_of_drifted_columns=share,
        dataset_drift=bool(result["dataset_drift"]),
        alert=share > drift_share_threshold,
    )


def _read_columns(engine: Engine, table: str, limit: int | None = None) -> pd.DataFrame:
    cols = ", ".join(RAW_INPUT_COLUMNS)
    sql = f"SELECT {cols} FROM {table}"
    if limit is not None:
        sql += f" ORDER BY created_at DESC LIMIT {int(limit)}"
    return pd.read_sql(sql, engine)


def run_drift_report(
    *,
    engine: Engine | None = None,
    settings: Settings | None = None,
    current_limit: int = 500,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
    html_path: str | Path = DEFAULT_REPORT_PATH,
) -> DriftResult:
    """Load training (reference) and recent prediction (current) inputs, then compute drift."""
    if engine is None:
        from gridsense.degrade.data import get_engine

        engine = get_engine(settings)

    reference = _read_columns(engine, SAMPLES_TABLE)
    current = _read_columns(engine, PREDICTIONS_TABLE, limit=current_limit)
    if current.empty:
        raise RuntimeError(
            f"No rows in '{PREDICTIONS_TABLE}'. Make some /predict calls before monitoring."
        )
    return compute_drift(
        reference, current, drift_share_threshold=drift_share_threshold, html_path=html_path
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the DegradeML data-drift report.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_DRIFT_SHARE_THRESHOLD)
    parser.add_argument("--out", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    result = run_drift_report(drift_share_threshold=args.threshold, html_path=args.out)
    status = "ALERT" if result.alert else "ok"
    print(
        f"[{status}] drift: {result.number_of_drifted_columns}/{result.number_of_columns} "
        f"columns ({result.share_of_drifted_columns:.0%}); dataset_drift={result.dataset_drift}"
    )
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

"""Serve the registered DegradeML model behind predictions, logging each call to Postgres.

Loads the latest registered ``degrade-soh`` version from the MLflow registry (cached), turns
a request's raw battery features into the model's feature vector (mirroring training), and
records every prediction to ``degrade_predictions`` so drift can be monitored later (M4
:mod:`gridsense.degrade.monitor`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pandas as pd

from gridsense.config import Settings, get_settings
from gridsense.degrade.data import add_derived_features, get_engine
from gridsense.degrade.train import REGISTERED_MODEL

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

#: Raw features a caller provides (derived features are computed server-side).
RAW_INPUT_COLUMNS = [
    "cycle_count",
    "avg_temperature_c",
    "avg_dod",
    "avg_c_rate",
    "calendar_age_days",
]

PREDICTIONS_TABLE = "degrade_predictions"

_PREDICTIONS_DDL = f"""
CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
    id                SERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    cycle_count       DOUBLE PRECISION,
    avg_temperature_c DOUBLE PRECISION,
    avg_dod           DOUBLE PRECISION,
    avg_c_rate        DOUBLE PRECISION,
    calendar_age_days DOUBLE PRECISION,
    predicted_soh     DOUBLE PRECISION,
    model_version     TEXT
)
"""


@lru_cache(maxsize=1)
def _cached_model(tracking_uri: str) -> tuple[Any, str]:
    """Load (and cache) the latest registered model version for a tracking URI."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if not versions:
        raise RuntimeError(
            f"No registered model '{REGISTERED_MODEL}'. Run `python -m gridsense.degrade.train`."
        )
    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{latest.version}")
    return model, str(latest.version)


def get_model(settings: Settings | None = None) -> tuple[Any, str]:
    """Return the cached ``(model, version)`` for the configured MLflow server."""
    settings = settings or get_settings()
    return _cached_model(settings.mlflow_tracking_uri)


def ensure_predictions_table(engine: Engine) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(_PREDICTIONS_DDL))


def log_prediction(
    engine: Engine, features: dict[str, float], predicted_soh: float, model_version: str
) -> None:
    """Append one prediction (inputs + output + model version) to Postgres."""
    from sqlalchemy import text

    ensure_predictions_table(engine)
    stmt = text(
        f"INSERT INTO {PREDICTIONS_TABLE} "
        "(cycle_count, avg_temperature_c, avg_dod, avg_c_rate, calendar_age_days, "
        " predicted_soh, model_version) "
        "VALUES (:cycle_count, :avg_temperature_c, :avg_dod, :avg_c_rate, "
        " :calendar_age_days, :predicted_soh, :model_version)"
    )
    params = {col: float(features[col]) for col in RAW_INPUT_COLUMNS}
    params["predicted_soh"] = float(predicted_soh)
    params["model_version"] = model_version
    with engine.begin() as conn:
        conn.execute(stmt, params)


def predict_soh(
    features: dict[str, float],
    *,
    model: Any | None = None,
    model_version: str | None = None,
    settings: Settings | None = None,
    engine: Engine | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Predict SOH for one set of raw features, logging the call unless ``log=False``.

    ``model``/``model_version`` and ``engine`` can be injected (tests); otherwise the cached
    registered model and the app database are used.
    """
    settings = settings or get_settings()
    if model is None:
        model, model_version = get_model(settings)

    x = add_derived_features(pd.DataFrame([features]))
    predicted_soh = float(model.predict(x)[0])

    if log:
        engine = engine or get_engine(settings)
        log_prediction(engine, features, predicted_soh, model_version or "unknown")

    return {"predicted_soh": predicted_soh, "model_version": model_version or "unknown"}

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
from gridsense.db import PREDICTIONS_TABLE as _PREDICTIONS_TABLE
from gridsense.db import metadata, predictions
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

#: Re-exported so existing imports keep working; the definition lives in gridsense.db.
PREDICTIONS_TABLE = _PREDICTIONS_TABLE


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
    """Create the predictions table if it is missing.

    This used to hold its own hand-written ``CREATE TABLE IF NOT EXISTS``, which made it a
    second, silent source of truth for the schema: ``IF NOT EXISTS`` does nothing to a table
    that already exists with the old shape, so a column added in one place and not the other
    would diverge without any error until the next INSERT. It now builds from the same
    metadata Alembic migrates, so there is one definition and two ways to apply it.

    Kept for local development and the integration tests, where running a migration chain to
    get one table is friction with no payoff. **Run `make migrate` in a deployment** — this
    creates a missing table but will never alter an existing one, which is exactly the gap
    migrations exist to close.
    """
    metadata.create_all(engine, tables=[predictions])


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
    # Heterogeneous on purpose: every bound parameter is a float except the model version.
    # Inferred from the comprehension alone this reads as dict[str, float], which is what
    # made the next assignment a type error rather than the intended shape.
    params: dict[str, float | str] = {col: float(features[col]) for col in RAW_INPUT_COLUMNS}
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

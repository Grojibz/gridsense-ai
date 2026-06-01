"""Train the battery SOH model and log it to MLflow (tracking + registry + model card).

Run as::

    python -m gridsense.degrade.train          # seeds data if empty, trains, registers

The training step is split into a pure :func:`train_model` (no DB/MLflow — unit-tested) and
:func:`run_training` which wires in the SQL feature build, MLflow logging, and registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gridsense.config import Settings, get_settings
from gridsense.degrade.data import FEATURE_COLUMNS, TARGET, build_features
from gridsense.degrade.evaluate import build_model_card, regression_metrics

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

ALGORITHM = "RandomForestRegressor"
EXPERIMENT = "degrade-soh"
REGISTERED_MODEL = "degrade-soh"


class TrainResult:
    """Outcome of a training run."""

    def __init__(
        self,
        model: RandomForestRegressor,
        metrics: dict[str, float],
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_samples: int,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.x_test = x_test
        self.y_test = y_test
        self.n_samples = n_samples


def train_model(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """Fit a RandomForest SOH regressor and evaluate it on a held-out split (no I/O)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    metrics = regression_metrics(y_test.to_numpy(), model.predict(x_test))
    return TrainResult(model, metrics, x_test, y_test, n_samples=len(x))


def run_training(
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    settings: Settings | None = None,
    seed_if_empty: bool = True,
) -> dict[str, Any]:
    """Build features from Postgres, train, and log/register the model in MLflow."""
    import mlflow
    from mlflow.models import infer_signature

    from gridsense.degrade.data import count_samples, get_engine, seed_database

    settings = settings or get_settings()
    engine = get_engine(settings)

    if seed_if_empty and count_samples(engine) == 0:
        n = seed_database(engine=engine)
        print(f"Samples table was empty; seeded {n} synthetic samples.")

    x, y = build_features(engine)
    result = train_model(x, y, n_estimators=n_estimators, max_depth=max_depth)

    params = {
        "algorithm": ALGORITHM,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": 0.2,
        "n_features": len(FEATURE_COLUMNS),
    }
    card = build_model_card(
        model_name=REGISTERED_MODEL,
        algorithm=ALGORITHM,
        params=params,
        metrics=result.metrics,
        n_samples=result.n_samples,
        n_train=result.n_samples - len(result.x_test),
        n_test=len(result.x_test),
        features=FEATURE_COLUMNS,
        target=TARGET,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(result.metrics)
        mlflow.log_text(card, "model_card.md")
        signature = infer_signature(result.x_test, result.model.predict(result.x_test))
        mlflow.sklearn.log_model(
            sk_model=result.model,
            artifact_path="model",
            signature=signature,
            input_example=result.x_test.iloc[:2],
            registered_model_name=REGISTERED_MODEL,
        )
        run_id = run.info.run_id

    return {"run_id": run_id, "metrics": result.metrics, "registered_model": REGISTERED_MODEL}


def _reconfigure_stdout_utf8() -> None:
    """Force UTF-8 stdout/stderr so MLflow's emoji log lines don't crash on cp1252 consoles."""
    import contextlib
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    _reconfigure_stdout_utf8()
    parser = argparse.ArgumentParser(description="Train and register the DegradeML SOH model.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args(argv)

    out = run_training(n_estimators=args.n_estimators, max_depth=args.max_depth)
    print(f"Run {out['run_id']} | metrics: {out['metrics']}")
    print(f"Registered model: {out['registered_model']}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

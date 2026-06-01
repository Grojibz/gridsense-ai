"""Regression metrics and model-card generation for DegradeML."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE and R^2 for a regression model's predictions."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_model_card(
    *,
    model_name: str,
    algorithm: str,
    params: dict[str, object],
    metrics: dict[str, float],
    n_samples: int,
    n_train: int,
    n_test: int,
    features: list[str],
    target: str,
) -> str:
    """Render a Markdown model card describing the trained model."""
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    params_rows = "\n".join(f"| {k} | {v} |" for k, v in params.items())
    metrics_rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
    features_list = "\n".join(f"- `{f}`" for f in features)

    return f"""# Model Card — {model_name}

_Generated {generated}._

## Overview

- **Task:** regression — predict battery **State of Health (SOH, %)**.
- **Algorithm:** {algorithm}.
- **Target:** `{target}` (SOH percentage, 70–100).

## Data

Synthetic, physically-plausible battery operating data (see `gridsense.degrade.data`).

- Total samples: **{n_samples}** (train **{n_train}** / test **{n_test}**).

### Features
{features_list}

## Hyperparameters

| Parameter | Value |
|---|---|
{params_rows}

## Test metrics

| Metric | Value |
|---|---|
{metrics_rows}

## Intended use & limitations

For demonstration and MLOps tooling only. The training data is synthetic, so metrics do not
reflect real-world battery behaviour and the model must not be used for operational decisions.
"""

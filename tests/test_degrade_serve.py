"""Offline unit tests for serve-time feature derivation and the prediction service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridsense.degrade.data import FEATURE_COLUMNS, add_derived_features
from gridsense.degrade.serve import predict_soh


def test_add_derived_features_matches_formulas() -> None:
    df = pd.DataFrame(
        [
            {
                "cycle_count": 1000,
                "avg_temperature_c": 35.0,
                "avg_dod": 0.5,
                "avg_c_rate": 1.0,
                "calendar_age_days": 400.0,
            }
        ]
    )
    x = add_derived_features(df)
    assert list(x.columns) == FEATURE_COLUMNS
    assert x["equivalent_full_cycles"].iloc[0] == pytest.approx(500.0)
    assert x["temperature_stress"].iloc[0] == pytest.approx(2.0)  # 2^((35-25)/10)
    assert x["calendar_stress"].iloc[0] == pytest.approx(20.0)  # sqrt(400)


class RecordingModel:
    def __init__(self) -> None:
        self.last_columns: list[str] | None = None

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        self.last_columns = list(x.columns)
        return np.full(len(x), 85.0)


def test_predict_soh_feeds_full_feature_vector_and_returns_result() -> None:
    model = RecordingModel()
    features = {
        "cycle_count": 1200,
        "avg_temperature_c": 30.0,
        "avg_dod": 0.6,
        "avg_c_rate": 1.1,
        "calendar_age_days": 500.0,
    }
    out = predict_soh(features, model=model, model_version="7", log=False)

    assert out == {"predicted_soh": 85.0, "model_version": "7"}
    assert model.last_columns == FEATURE_COLUMNS  # derived features were computed

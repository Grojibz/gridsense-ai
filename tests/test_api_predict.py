"""Unit test for the /predict route with the prediction service monkeypatched."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gridsense.api import routes_predict
from gridsense.api.main import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def fake_predict(features: dict) -> dict:
        return {"predicted_soh": 88.5, "model_version": "3"}

    monkeypatch.setattr(routes_predict, "predict_soh", fake_predict)
    return TestClient(create_app())


def test_predict_returns_soh_and_version(client: TestClient) -> None:
    resp = client.post(
        "/predict",
        json={
            "cycle_count": 1000,
            "avg_temperature_c": 30,
            "avg_dod": 0.6,
            "avg_c_rate": 1.0,
            "calendar_age_days": 400,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"predicted_soh": 88.5, "model_version": "3"}


def test_predict_rejects_out_of_range_dod(client: TestClient) -> None:
    resp = client.post(
        "/predict",
        json={
            "cycle_count": 1000,
            "avg_temperature_c": 30,
            "avg_dod": 2.0,  # > 1
            "avg_c_rate": 1.0,
            "calendar_age_days": 400,
        },
    )
    assert resp.status_code == 422

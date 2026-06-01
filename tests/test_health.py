"""Smoke tests for the M0 scaffold: package imports and the health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from gridsense import __version__


def test_version_is_exposed() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_health_endpoint_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "gridsense"
    assert body["version"] == __version__

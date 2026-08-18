"""Readiness, and why it must not share an endpoint with liveness.

Both probes in `k8s/deployment.yaml` used to point at `/health`, which returns `ok` without
touching anything. A pod whose Postgres was unreachable therefore reported itself ready and
had traffic routed to it — every request failing, the deployment reporting healthy.

The reverse mistake is just as bad and is why these stay two endpoints: a liveness probe
that checks the database gets the container *killed* during a database blip, turning a
dependency outage into a restart loop that cannot fix it.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gridsense.api import routes_health


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.include_router(routes_health.router)
    return application


def test_ready_when_the_database_answers(app: FastAPI, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(routes_health, "_check_database", lambda: "ok")
    resp = TestClient(app).get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"ready": True, "checks": {"database": "ok"}}


def test_an_unreachable_database_is_503_not_200(app: FastAPI, monkeypatch: pytest.MonkeyPatch):
    """The failure this endpoint exists for: serving traffic it cannot answer."""

    def boom() -> str:
        raise OSError("could not connect to server")

    monkeypatch.setattr(routes_health, "_check_database", boom)
    resp = TestClient(app).get("/ready")
    assert resp.status_code == 503
    assert resp.json()["ready"] is False
    assert "could not connect" in resp.json()["checks"]["database"]


def test_liveness_stays_independent_of_the_database(monkeypatch: pytest.MonkeyPatch):
    """A restart cannot fix an unreachable Postgres, so it must not trigger one."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    from gridsense.config import get_settings

    get_settings.cache_clear()
    try:
        from gridsense.api.main import create_app

        client = TestClient(create_app())

        def boom() -> str:
            raise OSError("down")

        monkeypatch.setattr(routes_health, "_check_database", boom)
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
    finally:
        get_settings.cache_clear()


def test_the_provider_is_deliberately_not_probed(app: FastAPI, monkeypatch):
    """Probing it would bill on a timer and take every pod out during one provider blip."""
    monkeypatch.setattr(routes_health, "_check_database", lambda: "ok")
    assert set(TestClient(app).get("/ready").json()["checks"]) == {"database"}

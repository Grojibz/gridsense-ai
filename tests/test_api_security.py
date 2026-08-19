"""API-key authentication, and the startup check that stops it being forgotten.

The property that matters most here is not "a good key works". It is that the app **refuses
to start** when it is deployed without authentication. An auth layer that defaults to open
is the same gap it was written to close, because the default is what ships — so the test
that would catch a regression is the one asserting a `RuntimeError`, not the one asserting
a 200.

The second-order properties are about not leaking: a missing key and a wrong key must be
indistinguishable to the caller, and no key material may reach a log line.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gridsense.api import security
from gridsense.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- the startup contract ----------------------------------------------------


def test_deployed_without_keys_refuses_to_start():
    """The whole point. /ask and /agent spend provider credits on every request."""
    with pytest.raises(RuntimeError, match="API_KEYS"):
        security.verify_auth_configuration(_settings(environment="k8s"))


@pytest.mark.parametrize("environment", ["local", "test", "LOCAL"])
def test_local_and_test_may_run_open(environment):
    security.verify_auth_configuration(_settings(environment=environment))


def test_a_deployed_app_with_keys_starts():
    security.verify_auth_configuration(_settings(environment="prod", api_keys="k1"))


def test_the_escape_hatch_is_explicit_and_never_the_default():
    """A gateway in front is a real deployment; a silent default is not."""
    assert _settings().api_auth_disabled is False
    security.verify_auth_configuration(_settings(environment="prod", api_auth_disabled=True))


def test_an_unnamed_environment_is_treated_as_deployed():
    """A denylist would exempt every environment name nobody thought to add."""
    with pytest.raises(RuntimeError):
        security.verify_auth_configuration(_settings(environment="staging"))


# --- key parsing -------------------------------------------------------------


def test_multiple_keys_are_accepted_so_rotation_does_not_need_a_flag_day():
    assert security.parse_api_keys(" old , new ") == ["old", "new"]
    assert security.parse_api_keys("") == []
    assert security.parse_api_keys(None) == []


def test_auth_is_off_when_no_key_is_configured():
    assert security.auth_enabled(_settings()) is False
    assert security.auth_enabled(_settings(api_keys="k")) is True
    assert security.auth_enabled(_settings(api_keys="k", api_auth_disabled=True)) is False


# --- enforcement -------------------------------------------------------------


@pytest.fixture
def guarded_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A minimal app carrying only the dependency, so nothing else can explain a 401."""
    from fastapi import Depends

    monkeypatch.setattr(security, "get_settings", lambda: _settings(api_keys="good,also-good"))

    app = FastAPI()

    @app.get("/guarded", dependencies=[Depends(security.require_api_key)])
    def guarded() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


def test_a_configured_key_is_accepted(guarded_client: TestClient):
    resp = guarded_client.get("/guarded", headers={"X-API-Key": "good"})
    assert resp.status_code == 200


def test_every_configured_key_works_so_rotation_is_not_a_cutover(guarded_client: TestClient):
    assert guarded_client.get("/guarded", headers={"X-API-Key": "also-good"}).status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_missing_and_wrong_keys_are_indistinguishable(guarded_client: TestClient, headers):
    """Telling a prober which half is wrong halves their work."""
    resp = guarded_client.get("/guarded", headers=headers)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "A valid X-API-Key header is required."


def test_a_rejection_never_echoes_the_presented_key(guarded_client: TestClient):
    resp = guarded_client.get("/guarded", headers={"X-API-Key": "hunter2-secret"})
    assert "hunter2" not in resp.text


def test_probes_stay_open_so_a_liveness_check_cannot_401(monkeypatch: pytest.MonkeyPatch):
    """A pod removed from service because its probe lacks a key is an outage, not security."""
    monkeypatch.setenv("API_KEYS", "deploy-key")
    monkeypatch.setenv("ENVIRONMENT", "test")
    from gridsense.config import get_settings

    get_settings.cache_clear()
    try:
        from gridsense.api.main import create_app

        client = TestClient(create_app())
        assert client.get("/health").status_code == 200
        assert client.post("/ask", json={"question": "hi"}).status_code == 401
    finally:
        get_settings.cache_clear()


def test_the_chat_page_is_local_only_once_auth_is_on(monkeypatch: pytest.MonkeyPatch):
    """A browser cannot send X-API-Key, so the UI is unreachable in a deployment.

    Asserted rather than left implicit because it is a documented trade, not an accident:
    serving the page while guarding `/ask` would give a UI that loads and then 401s on every
    question, which looks broken rather than protected. If someone later opens `/` to make
    the page load, this test is where they have to argue for it.
    """
    monkeypatch.setenv("API_KEYS", "deploy-key")
    monkeypatch.setenv("ENVIRONMENT", "test")
    from gridsense.config import get_settings

    get_settings.cache_clear()
    try:
        from gridsense.api.main import create_app

        client = TestClient(create_app())
        assert client.get("/").status_code == 401
        assert client.get("/chat").status_code == 401
    finally:
        get_settings.cache_clear()

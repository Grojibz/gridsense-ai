"""What a caller sees when something breaks, and what the operator gets instead.

One property, split across two assertions that have to hold together: the traceback goes to
the log, and the response carries a request id and nothing else. Returning the traceback is
the default FastAPI behaviour this middleware exists to replace — it leaks file paths, local
variables, and on a `create_engine` failure a connection string with credentials in it.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from gridsense.api.middleware import REQUEST_ID_HEADER, install_middleware


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    install_middleware(app)
    router = APIRouter()

    @router.get("/boom")
    def boom() -> None:
        raise RuntimeError("connection to postgres://user:hunter2@db failed")

    @router.get("/fine")
    def fine() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(router)
    # raise_server_exceptions=False so the middleware handles it, as it would in production;
    # the default re-raises into the test, which is exactly the path under test here.
    return TestClient(app, raise_server_exceptions=False)


def test_an_unhandled_exception_never_leaks_internals_to_the_caller(client: TestClient):
    resp = client.get("/boom")
    assert resp.status_code == 500
    assert "hunter2" not in resp.text
    assert "RuntimeError" not in resp.text
    assert "Traceback" not in resp.text


def test_the_caller_gets_a_request_id_it_can_quote(client: TestClient):
    resp = client.get("/boom")
    assert resp.json()["request_id"] == resp.headers[REQUEST_ID_HEADER]


def test_the_traceback_reaches_the_log(client: TestClient, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.ERROR, logger="gridsense.api"):
        client.get("/boom")
    record = next(r for r in caplog.records if r.message == "unhandled exception")
    assert record.exc_info is not None
    assert record.status_code == 500
    assert record.endpoint == "/boom"


def test_every_response_carries_a_request_id(client: TestClient):
    assert client.get("/fine").headers[REQUEST_ID_HEADER]


def test_an_upstream_request_id_survives_instead_of_being_replaced(client: TestClient):
    """A proxy or gateway that already assigned one is the correlation everyone else uses."""
    resp = client.get("/fine", headers={REQUEST_ID_HEADER: "from-the-gateway"})
    assert resp.headers[REQUEST_ID_HEADER] == "from-the-gateway"


def test_the_access_log_records_the_outcome(client: TestClient, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="gridsense.api"):
        client.get("/fine")
    record = next(r for r in caplog.records if r.message == "request")
    assert record.status_code == 200
    assert record.method == "GET"
    assert record.duration_ms >= 0


def test_the_access_log_records_the_route_pattern_not_the_raw_path(
    caplog: pytest.LogCaptureFixture,
):
    """`/items/{id}` must group. Logging the raw path mints a dimension per id."""
    app = FastAPI()
    install_middleware(app)

    @app.get("/items/{item_id}")
    def item(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    with caplog.at_level(logging.INFO, logger="gridsense.api"):
        TestClient(app).get("/items/1234")

    record = next(r for r in caplog.records if r.message == "request")
    assert record.endpoint == "/items/{item_id}"


def test_a_route_that_never_matched_falls_back_to_the_path(caplog: pytest.LogCaptureFixture):
    app = FastAPI()
    install_middleware(app)
    with caplog.at_level(logging.INFO, logger="gridsense.api"):
        TestClient(app).get("/nope")

    record = next(r for r in caplog.records if r.message == "request")
    assert record.endpoint == "/nope"
    assert record.status_code == 404

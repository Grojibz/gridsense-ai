"""The chat UI is served at the app root."""

from __future__ import annotations

from fastapi.testclient import TestClient

from gridsense.api.main import create_app


def test_root_serves_chat_html() -> None:
    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "GridSense" in resp.text
    assert "/ask" in resp.text  # chat tab calls the ask endpoint
    assert "/predict" in resp.text  # prediction tab calls the predict endpoint

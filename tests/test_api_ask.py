"""Unit test for the /ask route with the chain monkeypatched (no DB/LLM)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gridsense.api import routes_ask
from gridsense.api.main import create_app
from gridsense.docrag.chain import Citation, RagAnswer


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def fake_answer(question: str, *, k: int = 4) -> RagAnswer:
        return RagAnswer(
            answer=f"answer to: {question} (k={k})",
            citations=[Citation(source="a.md", snippet="snippet")],
            confidence=0.8,
        )

    monkeypatch.setattr(routes_ask, "answer_question", fake_answer)
    return TestClient(create_app())


def test_ask_returns_structured_answer(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "What is SOH?", "k": 3})
    assert resp.status_code == 200

    body = resp.json()
    assert body["answer"] == "answer to: What is SOH? (k=3)"
    assert body["citations"] == [{"source": "a.md", "snippet": "snippet"}]
    assert body["confidence"] == 0.8


def test_ask_rejects_empty_question(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422

"""`/agent` route: the two things the HTTP layer owns."""

from __future__ import annotations

from gridsense.agent.loop import AgentAnswer, ToolCall
from gridsense.api import routes_agent


def test_agent_returns_the_trajectory_alongside_the_answer(client, monkeypatch):
    """`tool_calls` is what makes an agentic reply auditable — it must survive the response
    model, not just exist internally."""

    def fake_run_agent(question: str) -> AgentAnswer:
        return AgentAnswer(
            answer="Below the 80% threshold by year five.",
            tool_calls=[
                ToolCall(name="predict_soh", input={"cycle_count": 1500.0}),
                ToolCall(name="search_docs", input={"query": "end of life"}),
            ],
            usage={"input_tokens": 120},
        )

    monkeypatch.setattr(routes_agent, "run_agent", fake_run_agent)
    response = client.post("/agent", json={"question": "Will this pack last five years?"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert [c["name"] for c in body["tool_calls"]] == ["predict_soh", "search_docs"]
    assert body["usage"]["input_tokens"] == 120


def test_a_missing_api_key_is_reported_as_unavailable_not_as_a_server_error(client, monkeypatch):
    """No Anthropic key is a deployment fault. A 500 with a stack trace tells the caller
    nothing they can act on; a 503 naming the variable does."""

    def fake_run_agent(question: str) -> AgentAnswer:
        raise ValueError("The agent loop requires ANTHROPIC_API_KEY.")

    monkeypatch.setattr(routes_agent, "run_agent", fake_run_agent)
    response = client.post("/agent", json={"question": "anything"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_an_empty_question_is_rejected_before_any_token_is_spent(client):
    assert client.post("/agent", json={"question": ""}).status_code == 422

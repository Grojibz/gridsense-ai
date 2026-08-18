"""`/agent` route — Claude tool-use loop over both GridSense modules.

Distinct from `/ask`, which is a single-shot RAG call. This endpoint takes questions that
need a prediction *and* a documented threshold, and lets the model decide which tools to
reach for. The trade is the usual one: it answers questions `/ask` cannot, at several times
the latency and cost.

The response carries `tool_calls` because the trajectory is part of the answer here. It is
what makes an agentic reply auditable — and it is the same field the trajectory eval scores.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gridsense.agent.loop import AgentAnswer, run_agent

router = APIRouter(tags=["agent"])


class AgentRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="A question that may need both the documentation and the SOH model.",
    )


@router.post("/agent", response_model=AgentAnswer)
def agent(request: AgentRequest) -> AgentAnswer:
    """Answer a question by letting Claude choose and sequence GridSense's tools."""
    try:
        return run_agent(request.question)
    except ValueError as exc:
        # Raised when the deployment has no Anthropic key. That is a configuration fault,
        # not a bad request — say so rather than returning a 500 with a stack trace.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

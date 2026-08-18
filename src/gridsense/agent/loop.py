"""The Claude tool-use loop: the part of GridSense that neither module can do alone.

DocRAG answers from documents. DegradeML predicts a number. Neither can answer *"this pack
runs at 33 °C and 1500 cycles over 600 days — will it still be serviceable in five years,
and what does the standard say?"*, because that needs a prediction, a documented
end-of-life threshold, and a comparison between them. Deciding which of those to fetch, in
what order, is the agent's job.

This is the one place in the repo that talks to the native Anthropic SDK rather than to
LangChain. That is deliberate: the tool runner, prompt caching, the effort control and
`stop_reason: "refusal"` are the whole point here, and LangChain abstracts over exactly
those. LangChain stays where it earns its keep — the RAG chain, whose structured-output
contract it implements well.

The refusal path matters as much as the answer path. A safety decline arrives as a
successful HTTP 200 with an empty or partial ``content``, so code that reads
``content[0].text`` unconditionally raises on it. Refusals are surfaced structurally here,
the same way DocRAG surfaces a guardrail refusal — a `refused` flag and a reason, never a
string to pattern-match.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from anthropic import beta_tool
from pydantic import BaseModel, Field

from gridsense.agent import tools
from gridsense.agent.subagents import build_research_tool
from gridsense.config import Settings, get_settings

if TYPE_CHECKING:
    from anthropic import Anthropic
    from langchain_core.language_models import BaseChatModel
    from langchain_postgres import PGVector
    from sqlalchemy.engine import Engine

logger = logging.getLogger("gridsense.agent")

#: Ceiling on coordinator turns. An agent that has taken this many and still has not
#: answered is looping, and a bounded wrong answer beats an unbounded bill.
MAX_ITERATIONS = 12

#: Beta flag for the server-side refusal fallback.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Beta flag for `output_config.task_budget`.
TASK_BUDGET_BETA = "task-budgets-2026-03-13"

#: Usage fields that represent tokens someone is billed for. Cache reads are billed at a
#: discount rather than free, so they belong in a spend ceiling even though they are cheap.
_BILLABLE_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def billable_tokens(usage: dict[str, int]) -> int:
    """Total tokens spent so far. A proxy for cost, not a price — the rates differ."""
    return sum(usage.get(field, 0) for field in _BILLABLE_USAGE_FIELDS)


SYSTEM_PROMPT = """\
You are a battery energy-storage (BESS) engineering assistant. You have two very different
sources of truth and the whole job is knowing which one a question needs.

`ask_docs` and `search_docs` read the technical documentation — standards, datasheets, test
reports. `predict_soh` runs a regression model over a pack's operating conditions. A
question about what a pack will do needs the model; a question about what is permissible
needs the documentation; a question about whether a pack will remain serviceable needs
both, because a predicted state of health means nothing without the end-of-life threshold
to compare it against. Fetch both before judging, and say which number came from where.

Cite the documentation for every claim that came from it. When `ask_docs` returns
`refused: true`, report the refusal and its reason — do not answer from your own knowledge
of batteries instead, and do not soften a refusal into a hedge. When it returns an
`uncertainty_level` of medium or high, carry the caveat into your answer rather than
presenting the claim as settled.

State plainly when the documentation does not cover something. The corpus is small, and an
acknowledged gap is a usable answer where an invented figure is not — in technical
documentation a wrong number is worse than no number.

Answer in the language the question was asked in.\
"""


class ToolCall(BaseModel):
    """One tool invocation, recorded for the response and for trajectory evals."""

    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    """What the agent loop returns.

    `refused` and `refusal_reason` mirror ``RagAnswer``: a refusal is a structural fact the
    caller can branch on, not a phrasing to match against.
    """

    answer: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    stop_reason: str | None = None
    iterations: int = 0
    usage: dict[str, int] = Field(default_factory=dict)


def build_tools(
    *,
    settings: Settings,
    client: Anthropic | None = None,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    engine: Engine | None = None,
    soh_model: Any | None = None,
    log_predictions: bool = True,
) -> list[Any]:
    """Build the coordinator's tool set, closing over injected dependencies.

    The wrappers are closures rather than module-level functions for the same reason the
    MCP server wraps rather than registers: the shared tool layer carries injection
    parameters that exist for tests, and a schema is a contract with the model. Everything
    below delegates to :mod:`gridsense.agent.tools` — the logic lives in exactly one place,
    whichever surface it is reached through.
    """

    @beta_tool
    def search_docs(query: str, k: int = 4) -> str:
        """Retrieve raw passages from the BESS documentation, without answering.

        Call this when you need the source text itself — to quote a threshold verbatim, or
        to check whether the corpus covers a topic before building on it. Prefer `ask_docs`
        when you want a written, cited answer.

        Args:
            query: What to look for.
            k: How many candidate passages to consider.
        """
        return tools.as_tool_result(
            tools.search_docs(query, k, vectorstore=vectorstore, settings=settings)
        )

    @beta_tool
    def ask_docs(question: str) -> str:
        """Answer a question from the BESS documentation, with citations.

        Call this for anything about standards, state-of-health thresholds, operating
        limits, or test procedures. Check `refused` before using `answer`, and carry any
        `uncertainty_note` into your own reply.

        Args:
            question: The documentation question, in full.
        """
        return tools.as_tool_result(
            tools.ask_docs(
                question, vectorstore=vectorstore, chat_model=chat_model, settings=settings
            )
        )

    @beta_tool
    def predict_soh(
        cycle_count: float,
        avg_temperature_c: float,
        avg_dod: float,
        avg_c_rate: float,
        calendar_age_days: float,
    ) -> str:
        """Predict a pack's State of Health (%) from its operating conditions.

        Call this whenever the question turns on how a pack has aged or will age. It
        returns a number, not a verdict — pair it with the documentation's end-of-life
        threshold before concluding anything about serviceability.

        Args:
            cycle_count: Equivalent full charge/discharge cycles.
            avg_temperature_c: Average operating temperature in degrees Celsius.
            avg_dod: Average depth of discharge as a fraction between 0 and 1.
            avg_c_rate: Average C-rate.
            calendar_age_days: Calendar age in days.
        """
        return tools.as_tool_result(
            tools.predict_soh(
                cycle_count=cycle_count,
                avg_temperature_c=avg_temperature_c,
                avg_dod=avg_dod,
                avg_c_rate=avg_c_rate,
                calendar_age_days=calendar_age_days,
                model=soh_model,
                engine=engine,
                settings=settings,
                log=log_predictions,
            )
        )

    @beta_tool
    def drift_report(current_limit: int = 500) -> str:
        """Check whether recent predictions match the model's training distribution.

        Call this before trusting a prediction on unusual duty, or when asked whether the
        model is still valid. This is data drift only — no ground truth returns from
        production, so nothing here says whether predictions are becoming *wrong*.

        Args:
            current_limit: How many recent predictions to compare.
        """
        return tools.as_tool_result(
            tools.drift_report(current_limit=current_limit, engine=engine, settings=settings)
        )

    return [
        search_docs,
        ask_docs,
        predict_soh,
        drift_report,
        build_research_tool(settings=settings, client=client, vectorstore=vectorstore),
    ]


def _accumulate_usage(total: dict[str, int], usage: Any) -> None:
    """Sum the usage fields across every turn of the loop.

    Cache figures are summed alongside the token counts because they are the only evidence
    that the cached prefix is actually being read. A `cache_read_input_tokens` that stays
    at zero across turns means the prefix never cleared the minimum cacheable size, and no
    amount of `cache_control` will change that.
    """
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, field, None)
        if value:
            total[field] = total.get(field, 0) + int(value)


def run_agent(
    question: str,
    *,
    client: Anthropic | None = None,
    settings: Settings | None = None,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    engine: Engine | None = None,
    soh_model: Any | None = None,
    log_predictions: bool = True,
    agent_tools: list[Any] | None = None,
) -> AgentAnswer:
    """Answer ``question`` by letting Claude choose and sequence GridSense's tools."""
    settings = settings or get_settings()

    if client is None:
        if not settings.anthropic_api_key:
            raise ValueError(
                "The agent loop requires ANTHROPIC_API_KEY. It talks to Claude directly and "
                "has no Azure or Ollama equivalent — the tool runner is Anthropic-specific."
            )
        from anthropic import Anthropic

        client = Anthropic(
            api_key=settings.anthropic_api_key,
            # Per API call, not per loop: the loop's own ceiling is MAX_ITERATIONS turns
            # and `agent_max_total_tokens` of spend. This one stops a single hung turn
            # from holding the worker indefinitely.
            timeout=settings.request_timeout_seconds,
        )

    if agent_tools is None:
        agent_tools = build_tools(
            settings=settings,
            client=client,
            vectorstore=vectorstore,
            chat_model=chat_model,
            engine=engine,
            soh_model=soh_model,
            log_predictions=log_predictions,
        )

    request: dict[str, Any] = {
        "model": settings.anthropic_model,
        "max_tokens": settings.anthropic_max_tokens,
        # Thinking is on by default on Opus 5 and counts against max_tokens, so the ceiling
        # has to cover reasoning plus answer, not just the answer.
        "output_config": {"effort": settings.anthropic_effort},
        "system": SYSTEM_PROMPT,
        "max_iterations": MAX_ITERATIONS,
        "tools": agent_tools,
        "messages": [{"role": "user", "content": question}],
        # Caches the longest stable prefix — tool schemas plus the system prompt. Whether it
        # pays depends on that prefix clearing the model's minimum cacheable size; read
        # `usage.cache_read_input_tokens` on the second call rather than assuming it does.
        "cache_control": {"type": "ephemeral"},
    }
    betas: list[str] = []
    if settings.anthropic_refusal_fallback:
        # A safety decline otherwise just stops. `"default"` lets the API route by refusal
        # category instead of pinning a substitute model that would need migrating later.
        request["fallbacks"] = "default"
        betas.append(FALLBACK_BETA)

    if settings.anthropic_task_budget_tokens:
        # Advisory, and the model can see it: it paces itself and will say when the budget
        # was what constrained the answer. It is *not* the cost ceiling — that is the local
        # check below, which does not depend on the model cooperating.
        request["output_config"]["task_budget"] = {
            "type": "tokens",
            "total": settings.anthropic_task_budget_tokens,
        }
        betas.append(TASK_BUDGET_BETA)

    if betas:
        request["betas"] = betas

    runner = client.beta.messages.tool_runner(**request)

    calls: list[ToolCall] = []
    usage: dict[str, int] = {}
    final = None
    iterations = 0

    budget = settings.agent_max_total_tokens

    for message in runner:
        final = message
        iterations += 1
        _accumulate_usage(usage, getattr(message, "usage", None))
        calls += [
            ToolCall(name=block.name, input=dict(block.input or {}))
            for block in message.content
            if block.type == "tool_use"
        ]

        # The hard cost ceiling, checked here rather than trusted to the provider. It is
        # necessarily *after* the turn that crossed it — the spend is only known once the
        # response exists — so this bounds the total at roughly one turn's overshoot, which
        # is the best a client-side check can do. Stopping is reported as a refusal because
        # what remains is a fragment: an agent cut off mid-trajectory has usually gathered
        # facts without reaching the comparison they were for, and returning that as an
        # answer is how a half-finished analysis gets read as a conclusion.
        if budget and billable_tokens(usage) >= budget:
            logger.warning(
                "agent stopped: token budget exhausted",
                extra={
                    "iterations": iterations,
                    "billable_tokens": billable_tokens(usage),
                    "budget": budget,
                    "tools_called": [call.name for call in calls],
                },
            )
            return AgentAnswer(
                answer=(
                    "This question exhausted the per-request token budget before the agent "
                    "finished. The partial work is not reported as an answer because a "
                    "trajectory cut short usually stops before the step that draws the "
                    "conclusion. Narrow the question, or raise AGENT_MAX_TOTAL_TOKENS."
                ),
                refused=True,
                refusal_reason="budget_exceeded",
                stop_reason=getattr(final, "stop_reason", None),
                tool_calls=calls,
                iterations=iterations,
                usage=usage,
            )

    if final is None:
        return AgentAnswer(
            answer="The agent produced no response.",
            refused=True,
            refusal_reason="empty_response",
            tool_calls=calls,
            iterations=iterations,
            usage=usage,
        )

    # Check the stop reason before touching content: on a refusal it is empty or partial,
    # and indexing into it is how this path usually breaks.
    if final.stop_reason == "refusal":
        details = getattr(final, "stop_details", None)
        logger.warning(
            "agent refused by safety classifier",
            extra={"category": getattr(details, "category", None), "iterations": iterations},
        )
        return AgentAnswer(
            answer="This request was declined by a safety classifier.",
            refused=True,
            refusal_reason=getattr(details, "category", None) or "refusal",
            stop_reason=final.stop_reason,
            tool_calls=calls,
            iterations=iterations,
            usage=usage,
        )

    text = "\n".join(b.text for b in final.content if b.type == "text").strip()

    if final.stop_reason == "max_tokens":
        # Truncation is a failure the caller must be able to see. Returning the partial text
        # with no marker is how a half-answer gets read as a whole one.
        return AgentAnswer(
            answer=text,
            refused=True,
            refusal_reason="max_tokens",
            stop_reason=final.stop_reason,
            tool_calls=calls,
            iterations=iterations,
            usage=usage,
        )

    logger.info(
        "agent answered",
        extra={
            "iterations": iterations,
            "tools_called": [call.name for call in calls],
            "billable_tokens": billable_tokens(usage),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "stop_reason": final.stop_reason,
        },
    )
    return AgentAnswer(
        answer=text,
        tool_calls=calls,
        stop_reason=final.stop_reason,
        iterations=iterations,
        usage=usage,
    )

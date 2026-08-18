"""The agent loop, driven with a fake Anthropic client so the tests stay offline.

Three groups of properties, in rough order of how expensive they are to get wrong:

**The request shape.** `temperature` is a 400 on Opus 5, and a request that quietly omits
`effort` or `cache_control` still works — it just costs more and reasons less. Neither
failure announces itself, so both are asserted against the kwargs the runner actually
receives.

**The failure paths.** A safety refusal arrives as a *successful* response whose `content`
is empty, and a `max_tokens` stop returns real text that happens to be a fragment. Both are
answers a caller must not read as complete, so both are surfaced structurally — a `refused`
flag and a reason — rather than as a string to match on.

**The trajectory.** `tool_calls` is what the agent evals score, so it has to record every
call across every turn, in order.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gridsense.agent import loop as agent_loop
from gridsense.agent.loop import AgentAnswer, build_tools, run_agent
from gridsense.agent.subagents import RESEARCH_MODEL, build_research_tool
from gridsense.config import Settings

SETTINGS = Settings(_env_file=None, chat_provider="anthropic", anthropic_api_key="sk-test")


# --- fakes -------------------------------------------------------------------


def _text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def _tool_use(name: str, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=kwargs)


def _usage(**kwargs: int) -> SimpleNamespace:
    defaults = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    return SimpleNamespace(**{**defaults, **kwargs})


def _message(
    *content: SimpleNamespace,
    stop_reason: str = "end_turn",
    usage: SimpleNamespace | None = None,
    stop_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        usage=usage or _usage(),
        stop_details=stop_details,
    )


class FakeRunner:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self._messages = messages

    def __iter__(self):
        return iter(self._messages)


class FakeClient:
    """Records the request and replays a scripted conversation."""

    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self.messages = messages
        self.request: dict[str, Any] = {}
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

    def _tool_runner(self, **kwargs: Any) -> FakeRunner:
        self.request = kwargs
        return FakeRunner(self.messages)


def _run(messages: list[SimpleNamespace], **overrides: Any) -> tuple[AgentAnswer, FakeClient]:
    client = FakeClient(messages)
    answer = run_agent("q", client=client, settings=SETTINGS, **overrides)
    return answer, client


# --- the request shape -------------------------------------------------------


@pytest.mark.parametrize("removed", ["temperature", "top_p", "top_k"])
def test_request_never_carries_a_sampling_parameter(removed):
    """These are 400s on Opus 5, and the tool runner would forward them verbatim."""
    _, client = _run([_message(_text("done"))])
    assert removed not in client.request


def test_request_sets_effort_and_a_token_ceiling_that_covers_thinking():
    _, client = _run([_message(_text("done"))])
    assert client.request["output_config"] == {"effort": SETTINGS.anthropic_effort}
    assert client.request["max_tokens"] == SETTINGS.anthropic_max_tokens
    assert client.request["model"] == "claude-opus-5"


def test_request_enables_prompt_caching_and_bounds_the_loop():
    _, client = _run([_message(_text("done"))])
    assert client.request["cache_control"] == {"type": "ephemeral"}
    assert client.request["max_iterations"] == agent_loop.MAX_ITERATIONS


def test_refusal_fallback_is_opt_out_not_opt_in():
    """A declined request otherwise just stops; the fallback is what recovers it."""
    _, client = _run([_message(_text("done"))])
    assert client.request["fallbacks"] == "default"
    assert agent_loop.FALLBACK_BETA in client.request["betas"]

    off = Settings(_env_file=None, anthropic_api_key="sk-test", anthropic_refusal_fallback=False)
    client = FakeClient([_message(_text("done"))])
    run_agent("q", client=client, settings=off)
    assert "fallbacks" not in client.request
    assert "betas" not in client.request


def test_agent_without_an_api_key_says_why_it_has_no_fallback_provider():
    """Unlike the RAG chain, this path has no Azure or Ollama equivalent to degrade to."""
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        run_agent("q", settings=Settings(_env_file=None, anthropic_api_key=None))


# --- the trajectory ----------------------------------------------------------


def test_tool_calls_are_recorded_across_every_turn_in_order():
    """This list is what the trajectory eval scores, so a dropped turn is a wrong score."""
    answer, _ = _run(
        [
            _message(_tool_use("predict_soh", cycle_count=1500.0), stop_reason="tool_use"),
            _message(_tool_use("search_docs", query="end of life"), stop_reason="tool_use"),
            _message(_text("It falls below the 80% threshold.")),
        ]
    )
    assert [c.name for c in answer.tool_calls] == ["predict_soh", "search_docs"]
    assert answer.tool_calls[0].input == {"cycle_count": 1500.0}
    assert answer.iterations == 3
    assert answer.refused is False
    assert answer.answer == "It falls below the 80% threshold."


def test_usage_is_summed_across_turns_including_the_cache_fields():
    """`cache_read_input_tokens` is the only evidence the cached prefix is really read."""
    answer, _ = _run(
        [
            _message(
                _tool_use("search_docs", query="x"),
                stop_reason="tool_use",
                usage=_usage(input_tokens=100, output_tokens=20, cache_creation_input_tokens=1200),
            ),
            _message(_text("done"), usage=_usage(input_tokens=40, cache_read_input_tokens=1200)),
        ]
    )
    assert answer.usage["input_tokens"] == 140
    assert answer.usage["output_tokens"] == 20
    assert answer.usage["cache_creation_input_tokens"] == 1200
    assert answer.usage["cache_read_input_tokens"] == 1200


# --- the failure paths -------------------------------------------------------


def test_a_safety_refusal_is_surfaced_structurally_and_never_indexes_into_content():
    """The classifier returns HTTP 200 with empty content; `content[0]` would raise here."""
    answer, _ = _run(
        [
            _message(
                stop_reason="refusal",
                stop_details=SimpleNamespace(category="cyber", explanation="declined"),
            )
        ]
    )
    assert answer.refused is True
    assert answer.refusal_reason == "cyber"
    assert answer.stop_reason == "refusal"


def test_a_refusal_without_stop_details_still_reports_a_reason():
    """`stop_details` is informational and may be absent even on a refusal."""
    answer, _ = _run([_message(stop_reason="refusal")])
    assert answer.refused is True
    assert answer.refusal_reason == "refusal"


def test_truncation_is_flagged_rather_than_returned_as_a_whole_answer():
    """Thinking counts against max_tokens, so a fragment looks like a finished reply."""
    answer, _ = _run([_message(_text("The pack will remain above"), stop_reason="max_tokens")])
    assert answer.refused is True
    assert answer.refusal_reason == "max_tokens"
    assert answer.answer == "The pack will remain above"


def test_an_empty_conversation_is_a_refusal_not_an_empty_success():
    answer, _ = _run([])
    assert answer.refused is True
    assert answer.refusal_reason == "empty_response"


# --- the tool surface --------------------------------------------------------


def test_the_coordinator_exposes_both_modules_and_the_researcher():
    names = {t.name for t in build_tools(settings=SETTINGS)}
    assert names == {"search_docs", "ask_docs", "predict_soh", "drift_report", "delegate_research"}


def test_no_coordinator_tool_publishes_a_test_seam():
    seams = {"vectorstore", "chat_model", "engine", "settings", "soh_model", "log"}
    for tool in build_tools(settings=SETTINGS):
        leaked = set(tool.input_schema["properties"]) & seams
        assert not leaked, f"{tool.name} exposes {sorted(leaked)}"


def test_every_coordinator_tool_description_says_when_to_call_it():
    for tool in build_tools(settings=SETTINGS):
        assert "Call this" in (tool.description or ""), f"{tool.name} lacks a trigger condition"


# --- the subagent ------------------------------------------------------------


def test_research_is_delegated_to_the_cheap_model_not_the_coordinator_model():
    """The point of the subagent is that the reading is not billed at Opus rates."""
    client = FakeClient([_message(_text("End of life is 80% SOH (battery_soh.md)."))])
    tool = build_research_tool(settings=SETTINGS, client=client)

    finding = tool.call({"question": "What is the end-of-life SOH threshold?"})

    assert client.request["model"] == RESEARCH_MODEL
    assert finding == "End of life is 80% SOH (battery_soh.md)."


def test_the_researcher_is_given_only_search_and_a_bounded_budget():
    """A worker that can answer, predict or delegate is not a worker — and an unbounded
    one quietly becomes the expensive half of the run."""
    client = FakeClient([_message(_text("ok"))])
    build_research_tool(settings=SETTINGS, client=client).call({"question": "q"})

    assert [t.name for t in client.request["tools"]] == ["search_docs"]
    assert client.request["max_iterations"] > 0


def test_a_refused_subagent_reports_back_instead_of_propagating_the_refusal():
    """The coordinator must be able to carry on; a raised exception here would sink the
    whole answer over one failed sub-question."""
    client = FakeClient([_message(stop_reason="refusal")])
    finding = build_research_tool(settings=SETTINGS, client=client).call({"question": "q"})
    assert "declined" in finding.lower()


# --- the cost ceiling --------------------------------------------------------
#
# `MAX_ITERATIONS` bounds the number of turns, which is not the same as bounding the spend:
# twelve turns of a long-context tool result cost far more than twelve short ones. The token
# ceiling is the limit that is actually denominated in the thing being paid for, and it is
# enforced here rather than delegated to the provider — `task_budget` is advisory, and a
# limit the model may decline to respect is not a limit.


def test_the_loop_stops_when_the_token_budget_is_exhausted():
    settings = Settings(_env_file=None, anthropic_api_key="sk-test", agent_max_total_tokens=1000)
    client = FakeClient(
        [
            _message(
                _tool_use("search_docs", query="soh"),
                stop_reason="tool_use",
                usage=_usage(input_tokens=600, output_tokens=500),
            ),
            _message(_text("a complete answer nobody should see")),
        ]
    )
    answer = run_agent("q", client=client, settings=settings)

    assert answer.refused is True
    assert answer.refusal_reason == "budget_exceeded"
    assert answer.iterations == 1
    assert "nobody should see" not in answer.answer


def test_a_budget_stop_is_a_refusal_rather_than_a_partial_answer():
    """A trajectory cut short usually stops before the step that draws the conclusion."""
    settings = Settings(_env_file=None, anthropic_api_key="sk-test", agent_max_total_tokens=100)
    client = FakeClient(
        [
            _message(
                _tool_use("predict_soh", cycle_count=1500.0),
                stop_reason="tool_use",
                usage=_usage(input_tokens=200),
            )
        ]
    )
    answer = run_agent("q", client=client, settings=settings)

    assert answer.refused is True
    # The trajectory so far is still reported: it is what the caller needs to see why.
    assert [call.name for call in answer.tool_calls] == ["predict_soh"]
    assert answer.usage["input_tokens"] == 200


def test_a_run_inside_its_budget_is_untouched():
    settings = Settings(_env_file=None, anthropic_api_key="sk-test", agent_max_total_tokens=10_000)
    client = FakeClient([_message(_text("done"), usage=_usage(input_tokens=50, output_tokens=10))])
    answer = run_agent("q", client=client, settings=settings)

    assert answer.refused is False
    assert answer.answer == "done"


def test_a_zero_budget_disables_the_ceiling():
    settings = Settings(_env_file=None, anthropic_api_key="sk-test", agent_max_total_tokens=0)
    client = FakeClient([_message(_text("done"), usage=_usage(input_tokens=10**9))])
    assert run_agent("q", client=client, settings=settings).refused is False


def test_cache_reads_count_toward_the_ceiling_because_they_are_billed():
    """Discounted is not free, and a cached prefix is most of a long agentic prompt."""
    assert agent_loop.billable_tokens({"cache_read_input_tokens": 500, "input_tokens": 100}) == 600


# --- the advisory budget -----------------------------------------------------


def test_no_task_budget_is_sent_unless_one_was_configured():
    """A budget guessed without measuring degrades answers for no saving."""
    _, client = _run([_message(_text("done"))])
    assert "task_budget" not in client.request["output_config"]
    assert agent_loop.TASK_BUDGET_BETA not in client.request.get("betas", [])


def test_a_configured_task_budget_is_sent_with_its_beta_flag():
    settings = Settings(
        _env_file=None, anthropic_api_key="sk-test", anthropic_task_budget_tokens=64_000
    )
    client = FakeClient([_message(_text("done"))])
    run_agent("q", client=client, settings=settings)

    assert client.request["output_config"]["task_budget"] == {"type": "tokens", "total": 64_000}
    assert agent_loop.TASK_BUDGET_BETA in client.request["betas"]
    # The refusal fallback beta must survive alongside it, not be replaced by it.
    assert agent_loop.FALLBACK_BETA in client.request["betas"]

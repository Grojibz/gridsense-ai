"""What the agent puts on the wire, driven through the real Anthropic SDK.

Every other agent test replaces the client with a fake, which is the right trade for
asserting control flow — and it is exactly why they could not catch this. A fake client
never serialises a request, so a tool whose return value the API would reject looks
identical to one it would accept.

These tests keep the SDK's own machinery in the loop and replace only the transport, so the
runner builds real request bodies. That is the seam where the contract actually lives: the
Messages API defines `tool_result.content` as a string or a list of content blocks, and
`beta_tool`'s type says the same thing (its `FunctionT` is bound to a callable returning
`str`). Returning a dict put a bare JSON object there — accepted by every fake, and by
nothing else.

The cost of this style is that it breaks when the SDK's internals change, which is a fair
price for the one property no fake can express: that a real request is well-formed.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gridsense.agent.loop import build_tools, run_agent
from gridsense.config import Settings

SETTINGS = Settings(_env_file=None, anthropic_api_key="sk-test")


def _assistant_tool_use(name: str, **tool_input: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "tool_use", "id": "tu_1", "name": name, "input": tool_input}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _assistant_text(text: str) -> dict[str, Any]:
    return {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.fixture
def wire() -> tuple[Any, list[dict[str, Any]]]:
    """A real Anthropic client over a mock transport, plus the list of captured bodies."""
    from anthropic import Anthropic

    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        if len(captured) == 1:
            return httpx.Response(200, json=_assistant_tool_use("search_docs", query="soh", k=2))
        return httpx.Response(200, json=_assistant_text("done"))

    client = Anthropic(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, captured


def _tool_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for message in body["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            blocks += [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
    return blocks


def test_a_tool_result_goes_on_the_wire_as_a_string(wire, monkeypatch: pytest.MonkeyPatch):
    """The bug this file exists for: a dict here is not a shape the API accepts."""
    client, captured = wire
    monkeypatch.setattr(
        "gridsense.agent.tools.search_docs",
        lambda *a, **k: {"chunks": [{"source": "a.md", "text": "80% SOH"}], "n": 1},
    )

    run_agent("q", client=client, settings=SETTINGS)

    results = _tool_results(captured[1])
    assert len(results) == 1
    assert isinstance(results[0]["content"], str), (
        f"tool_result.content must be a string or a list of blocks, got "
        f"{type(results[0]['content']).__name__}"
    )


def test_the_serialised_result_still_carries_the_payload(wire, monkeypatch: pytest.MonkeyPatch):
    """Serialising must not flatten the content the model needs to reason over."""
    client, captured = wire
    monkeypatch.setattr(
        "gridsense.agent.tools.search_docs",
        lambda *a, **k: {"chunks": [{"source": "iec.md", "text": "end of life at 80%"}]},
    )

    run_agent("q", client=client, settings=SETTINGS)

    payload = json.loads(_tool_results(captured[1])[0]["content"])
    assert payload["chunks"][0]["source"] == "iec.md"
    assert "80%" in payload["chunks"][0]["text"]


def test_every_tool_declares_a_string_return(monkeypatch: pytest.MonkeyPatch):
    """A new tool added with a dict return would reintroduce the same bug silently."""
    import typing

    for tool in build_tools(settings=SETTINGS, client=None):
        # `from __future__ import annotations` leaves these as strings; resolve them.
        annotation = typing.get_type_hints(tool.func).get("return")
        assert annotation is str, f"{tool.name} returns {annotation!r}, must be str"


def test_a_non_serialisable_value_degrades_instead_of_raising_inside_the_loop():
    """An exception here would surface as an opaque tool error, not as its cause."""
    from gridsense.agent.tools import as_tool_result

    class Odd:
        def __repr__(self) -> str:
            return "<odd>"

    assert json.loads(as_tool_result({"x": Odd()})) == {"x": "<odd>"}


def test_non_ascii_survives_serialisation():
    """The corpus and the prompts are not ASCII-only; escaping them would be lossy noise."""
    from gridsense.agent.tools import as_tool_result

    assert "33 °C" in as_tool_result({"note": "33 °C"})

"""Trajectory scoring — the metric definitions, and the dataset that feeds them.

Scoring is pure and deterministic, so all of it is covered offline with a fake agent. The
dataset checks matter as much as the arithmetic: a label that says one thing and means
another produces a confident, wrong score, and nothing downstream would catch it.
"""

from __future__ import annotations

import pytest

from gridsense.agent.eval import (
    AGENT_THRESHOLDS,
    TrajectoryDataset,
    TrajectoryItem,
    aggregate,
    load_trajectories,
    run_trajectory_eval,
    score_item,
)
from gridsense.agent.loop import AgentAnswer, ToolCall, build_tools
from gridsense.config import Settings

CROSS = TrajectoryItem(
    id="cross",
    query="q",
    category="cross_module",
    expected_tools=["predict_soh", "ask_docs|search_docs"],
)
SCOPE = TrajectoryItem(
    id="scope",
    query="q",
    category="out_of_scope",
    expected_tools=[],
    forbidden_tools=["predict_soh", "drift_report"],
)


# --- trajectory_accuracy ------------------------------------------------------


def test_calling_one_of_two_required_tools_scores_a_half_not_a_pass():
    """A predicted SOH with no threshold to compare it against is half an answer."""
    record = score_item(CROSS, ["predict_soh"])
    assert record.accuracy == 0.5
    assert record.missed == ["ask_docs|search_docs"]


def test_either_alternative_satisfies_a_requirement():
    """`ask_docs` vs `search_docs` is a style choice, not a routing error."""
    assert score_item(CROSS, ["predict_soh", "search_docs"]).accuracy == 1.0
    assert score_item(CROSS, ["predict_soh", "ask_docs"]).accuracy == 1.0


def test_extra_tools_do_not_reduce_accuracy_but_forbidden_ones_break_precision():
    """The two metrics answer different questions; conflating them would hide one."""
    record = score_item(CROSS, ["predict_soh", "ask_docs", "drift_report"])
    assert record.accuracy == 1.0
    assert record.precise is True

    record = score_item(SCOPE, ["predict_soh"])
    assert record.accuracy == 1.0  # nothing was required
    assert record.precise is False
    assert record.violated == ["predict_soh"]


def test_an_item_expecting_no_tools_cannot_fail_on_recall():
    assert score_item(SCOPE, []).accuracy == 1.0
    assert score_item(SCOPE, []).precise is True


# --- the anti-cheat rule ------------------------------------------------------


def test_refusing_an_item_that_needed_tools_scores_zero_not_excused():
    """Same rule as the RAGAS gate: if a refusal were excused rather than scored, refusing
    more would be a way to raise the mean."""
    record = score_item(CROSS, ["predict_soh"], refused=True, refusal_reason="max_tokens")
    assert record.accuracy == 0.0
    assert record.missed == CROSS.expected_tools
    assert record.satisfied == []


def test_refusing_an_out_of_scope_item_is_not_penalised():
    """Declining something the corpus has nothing to say about is the right outcome."""
    record = score_item(SCOPE, [], refused=True, refusal_reason="out_of_scope")
    assert record.accuracy == 1.0
    assert record.precise is True


# --- aggregation --------------------------------------------------------------


def test_scores_are_means_over_items():
    records = [score_item(CROSS, ["predict_soh", "ask_docs"]), score_item(CROSS, ["predict_soh"])]
    report = aggregate(records)
    assert report.scores["trajectory_accuracy"] == 0.75
    assert report.scores["tool_precision"] == 1.0
    assert report.n_items == 2


def test_an_empty_run_scores_zero_rather_than_passing_vacuously():
    report = aggregate([])
    assert report.scores == dict.fromkeys(AGENT_THRESHOLDS, 0.0)
    assert report.violations()


def test_the_gate_reuses_the_docrag_threshold_logic():
    report = aggregate([score_item(CROSS, ["predict_soh"]) for _ in range(4)])
    failed = {v.metric for v in report.violations()}
    assert "trajectory_accuracy" in failed
    assert "tool_precision" not in failed


# --- end to end with a fake agent --------------------------------------------


def test_run_trajectory_eval_drives_the_agent_and_scores_every_item():
    dataset = TrajectoryDataset(items=[CROSS, SCOPE])

    def fake_run(query: str, **_kwargs):
        return AgentAnswer(
            answer="ok",
            tool_calls=[ToolCall(name="predict_soh"), ToolCall(name="ask_docs")],
        )

    report = run_trajectory_eval(dataset, run=fake_run)

    assert report.n_items == 2
    assert report.scores["trajectory_accuracy"] == 1.0
    # The fake calls predict_soh on the out-of-scope item, which is exactly what
    # tool_precision is there to catch.
    assert report.scores["tool_precision"] == 0.5


# --- the shipped dataset ------------------------------------------------------


def test_shipped_dataset_loads_and_covers_every_category():
    dataset = load_trajectories()
    categories = {item.category for item in dataset.items}
    assert {"documentation", "prediction", "cross_module", "out_of_scope"} <= categories
    assert len(dataset.items) >= 10


def test_every_referenced_tool_actually_exists_on_the_agent():
    """A typo in a tool name would silently score as 'never called' — a metric that only
    ever goes down, for a reason no one would find."""
    real = {t.name for t in build_tools(settings=Settings(_env_file=None))}
    for item in load_trajectories().items:
        for requirement in item.expected_tools:
            for option in requirement.split("|"):
                assert option in real, f"{item.id} expects unknown tool {option!r}"
        for tool in item.forbidden_tools:
            assert tool in real, f"{item.id} forbids unknown tool {tool!r}"


def test_no_item_both_expects_and_forbids_the_same_tool():
    for item in load_trajectories().items:
        expected = {opt for req in item.expected_tools for opt in req.split("|")}
        assert not (expected & set(item.forbidden_tools)), f"{item.id} contradicts itself"


@pytest.mark.parametrize("item_id", ["scope-code-generation", "scope-live-data"])
def test_out_of_scope_items_expect_no_tool_use(item_id):
    item = next(i for i in load_trajectories().items if i.id == item_id)
    assert item.expected_tools == []
    assert item.forbidden_tools, "an out-of-scope item with nothing forbidden scores nothing"


def test_cross_module_items_require_both_modules():
    """These are the whole reason the agent exists; if one drifts into a single-module item
    the eval stops testing the thing it was built for."""
    cross = [i for i in load_trajectories().items if i.category == "cross_module"]
    assert len(cross) >= 3
    for item in cross:
        assert "predict_soh" in item.expected_tools
        assert any("ask_docs" in req or "search_docs" in req for req in item.expected_tools)


def test_dataset_ids_are_unique():
    ids = [item.id for item in load_trajectories().items]
    assert len(ids) == len(set(ids))

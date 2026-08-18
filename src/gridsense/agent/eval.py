"""Trajectory evaluation for the agent loop: did it reach for the right tools?

The DocRAG gate scores an *answer*. That is the wrong unit for an agent, where the same
final sentence can come from sound reasoning or from a lucky guess that never consulted the
model. What distinguishes them is the trajectory, so that is what this scores.

Two metrics, both deterministic — no judge, no network:

``trajectory_accuracy``
    Of the tools an item needed, how many were actually called. A question about whether a
    pack stays serviceable needs *both* a prediction and the documented threshold; calling
    one of the two is half an answer, and the score says so.

``tool_precision``
    The share of items that called no forbidden tool. Precision matters as much as recall
    here, for the same reason it does in the refusal metric: an agent that runs the
    regression model on a question about thermal runaway is not being thorough, it is
    burning budget and inviting an irrelevant number into the answer.

Kept out of the main ``THRESHOLDS`` dict on purpose. A metric absent from a report counts as
0.0 there — which is the right behaviour for a metric that vanished, and the wrong one for
a metric the RAGAS run was never supposed to produce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gridsense.docrag.eval.thresholds import ThresholdViolation, check_thresholds

DEFAULT_TRAJECTORY_PATH = "eval/agent_trajectories.json"

#: Gate for the agent loop. Separate from the DocRAG thresholds — see the module docstring.
AGENT_THRESHOLDS: dict[str, float] = {
    # One missed tool out of five items is tolerable; two is a routing problem.
    "trajectory_accuracy": 0.80,
    # Tighter, because a forbidden call is unambiguous: nothing about the question invited
    # it. There is no noise to leave headroom for.
    "tool_precision": 0.85,
}

#: Separator for alternative tools in ``expected_tools``: any one of them satisfies the
#: requirement. "Read the documentation" is legitimately either ``ask_docs`` (answer me) or
#: ``search_docs`` (give me the passages), and pinning one would score style, not behaviour.
ALTERNATIVE = "|"


class TrajectoryItem(BaseModel):
    """One labelled question and the tool use it should — and should not — provoke."""

    id: str
    query: str
    category: str
    #: Each entry is one requirement; ``a|b`` is satisfied by either.
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    note: str = ""


class TrajectoryDataset(BaseModel):
    items: list[TrajectoryItem]


class TrajectoryRecord(BaseModel):
    """One item after it has been run through the agent."""

    id: str
    category: str
    called_tools: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    satisfied: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    violated: list[str] = Field(default_factory=list)
    accuracy: float = 0.0
    precise: bool = True


class TrajectoryReport(BaseModel):
    n_items: int
    scores: dict[str, float]
    items: list[TrajectoryRecord]

    def violations(self) -> list[ThresholdViolation]:
        """Reuses the DocRAG gate logic against the agent thresholds."""
        return check_thresholds(self.scores, thresholds=AGENT_THRESHOLDS)


def load_trajectories(path: str | Path | None = None) -> TrajectoryDataset:
    payload = Path(path or DEFAULT_TRAJECTORY_PATH).read_text(encoding="utf-8")
    return TrajectoryDataset.model_validate_json(payload)


def _satisfied(requirement: str, called: set[str]) -> bool:
    return any(option in called for option in requirement.split(ALTERNATIVE))


def score_item(
    item: TrajectoryItem,
    called_tools: list[str],
    *,
    refused: bool = False,
    refusal_reason: str | None = None,
) -> TrajectoryRecord:
    """Score one item's trajectory.

    A refusal on an item that *expected* tool use scores 0.0 rather than being excused.
    Same rule as the RAGAS gate applies to a refused answerable item, and for the same
    reason: without it, refusing more is a way to score better.
    """
    called = set(called_tools)
    satisfied = [req for req in item.expected_tools if _satisfied(req, called)]
    missed = [req for req in item.expected_tools if req not in satisfied]
    violated = [tool for tool in item.forbidden_tools if tool in called]

    if not item.expected_tools:
        # Nothing was required, so there is nothing to recall. Precision carries this item.
        accuracy = 1.0
    elif refused:
        accuracy = 0.0
        missed = list(item.expected_tools)
        satisfied = []
    else:
        accuracy = len(satisfied) / len(item.expected_tools)

    return TrajectoryRecord(
        id=item.id,
        category=item.category,
        called_tools=list(called_tools),
        refused=refused,
        refusal_reason=refusal_reason,
        satisfied=satisfied,
        missed=missed,
        violated=violated,
        accuracy=accuracy,
        precise=not violated,
    )


def aggregate(records: list[TrajectoryRecord]) -> TrajectoryReport:
    if not records:
        return TrajectoryReport(n_items=0, scores=dict.fromkeys(AGENT_THRESHOLDS, 0.0), items=[])
    return TrajectoryReport(
        n_items=len(records),
        scores={
            "trajectory_accuracy": sum(r.accuracy for r in records) / len(records),
            "tool_precision": sum(1 for r in records if r.precise) / len(records),
        },
        items=records,
    )


def run_trajectory_eval(
    dataset: TrajectoryDataset | None = None,
    *,
    run: Any = None,
    **run_kwargs: Any,
) -> TrajectoryReport:
    """Run every labelled question through the agent and score the trajectories.

    ``run`` is the callable under test — :func:`gridsense.agent.loop.run_agent` by default,
    a fake in tests. Keeping it injectable is what lets the scoring be covered offline while
    the real run needs an API key.
    """
    dataset = dataset if dataset is not None else load_trajectories()
    if run is None:
        from gridsense.agent.loop import run_agent as run

    records = []
    for item in dataset.items:
        answer = run(item.query, **run_kwargs)
        records.append(
            score_item(
                item,
                [call.name for call in answer.tool_calls],
                refused=answer.refused,
                refusal_reason=answer.refusal_reason,
            )
        )
    return aggregate(records)


def write_report(report: TrajectoryReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(), indent=2) + "\n", encoding="utf-8")

"""Run the agent trajectory eval and gate on it.

    python eval/run_agent_eval.py --out eval/agent_results.json

Scores whether the agent reached for the right tools, not whether its prose reads well.
Needs ANTHROPIC_API_KEY: the loop is the thing under test, and it talks to Claude directly.

Thresholds live in :mod:`gridsense.agent.eval` rather than in this script or the CI YAML,
so loosening one shows up in code review like any other change.
"""

from __future__ import annotations

import argparse
import sys

from gridsense.agent.eval import (
    AGENT_THRESHOLDS,
    load_trajectories,
    run_trajectory_eval,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score and gate the agent's tool trajectories.")
    parser.add_argument("--dataset", default=None, help="Path to agent_trajectories.json.")
    parser.add_argument("--out", default="eval/agent_results.json", help="Where to write scores.")
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Report the scores without failing on a threshold breach.",
    )
    args = parser.parse_args(argv)

    report = run_trajectory_eval(load_trajectories(args.dataset))
    write_report(report, args.out)

    for record in report.items:
        mark = "ok" if record.accuracy == 1.0 and record.precise else "FAIL"
        called = ", ".join(record.called_tools) or "-"
        print(f"[{mark:>4}] {record.id:<28} called: {called}")
        if record.missed:
            print(f"         missed: {', '.join(record.missed)}")
        if record.violated:
            print(f"         forbidden: {', '.join(record.violated)}")

    print()
    violations = report.violations()
    for metric, minimum in AGENT_THRESHOLDS.items():
        mark = "FAIL" if any(v.metric == metric for v in violations) else "ok"
        print(f"[{mark:>4}] {metric:<22} {report.scores.get(metric, 0.0):.3f}  (min {minimum:.2f})")

    if violations and not args.no_gate:
        print("\nAgent gate FAILED:")
        for violation in violations:
            print(violation.render())
        return 1

    print("\nAgent gate passed." if not violations else "\nThresholds breached (gate disabled).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

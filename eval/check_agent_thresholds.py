"""Turn an agent trajectory report into a merge gate.

    python eval/run_agent_eval.py --no-gate --out eval/agent_results.json
    python eval/check_agent_thresholds.py eval/agent_results.json

Split from the runner for the same reason the RAGAS gate is: scoring costs an API call and
the report is worth keeping even when the gate fails, so CI can upload the artefact between
the two steps. Needs no credentials — the scoring is already done.

Thresholds live in :mod:`gridsense.agent.eval`, not here and not in CI YAML, so loosening
one shows up in code review like any other change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gridsense.agent.eval import AGENT_THRESHOLDS, TrajectoryReport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate an agent trajectory report.")
    parser.add_argument(
        "results",
        nargs="?",
        default="eval/agent_results.json",
        help="Report written by eval/run_agent_eval.py --out.",
    )
    args = parser.parse_args(argv)

    report = TrajectoryReport.model_validate(
        json.loads(Path(args.results).read_text(encoding="utf-8"))
    )
    violations = report.violations()

    for metric, minimum in AGENT_THRESHOLDS.items():
        mark = "FAIL" if any(v.metric == metric for v in violations) else "ok"
        score = report.scores.get(metric, 0.0)
        print(f"[{mark:>4}] {metric:<22} {score:.3f}  (min {minimum:.2f})")

    if violations:
        print("\nAgent gate FAILED:")
        for violation in violations:
            print(violation.render())
        # The trajectories that caused it, so the log is enough to start debugging from.
        for record in report.items:
            if record.missed or record.violated:
                called = ", ".join(record.called_tools) or "nothing"
                print(f"\n  {record.id} (called: {called})")
                if record.missed:
                    print(f"    missed:    {', '.join(record.missed)}")
                if record.violated:
                    print(f"    forbidden: {', '.join(record.violated)}")
        return 1

    print("\nAgent gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

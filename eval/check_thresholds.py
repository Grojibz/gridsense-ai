"""Turn a RAGAS report into a merge gate: exit non-zero if any score is under threshold.

    python eval/run_ragas.py --out eval/results.json
    python eval/check_thresholds.py eval/results.json

Thresholds live in :mod:`gridsense.docrag.eval.thresholds` so they are reviewed in a PR
like any other change, rather than buried in CI YAML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gridsense.docrag.eval.thresholds import THRESHOLDS, check_thresholds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate a RAGAS report against the thresholds.")
    parser.add_argument(
        "results",
        nargs="?",
        default="eval/results.json",
        help="Report written by eval/run_ragas.py --out.",
    )
    args = parser.parse_args(argv)

    report = json.loads(Path(args.results).read_text(encoding="utf-8"))
    scores: dict[str, float] = report["scores"]
    violations = check_thresholds(
        scores,
        coverage=report.get("coverage"),
        n_answerable=report.get("n_answerable", 0),
    )

    for metric, minimum in THRESHOLDS.items():
        mark = "FAIL" if any(v.metric == metric and v.kind == "score" for v in violations) else "ok"
        print(f"[{mark:>4}] {metric:<18} {scores.get(metric, 0.0):.3f}  (min {minimum:.2f})")

    if violations:
        print("\nEval gate FAILED:")
        for violation in violations:
            print(violation.render())
        return 1

    print("\nEval gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

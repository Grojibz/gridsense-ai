"""Run the RAGAS evaluation over the golden dataset and write a machine-readable report.

    python eval/run_ragas.py                          # ingest the corpus, evaluate, print
    python eval/run_ragas.py --out eval/results.json  # also write the CI artefact
    python eval/run_ragas.py --no-ingest --no-publish # evaluate the current store, no Langfuse

Requires the eval extra (``pip install -e ".[docrag,eval]"``), a reachable pgvector store,
and a configured LLM provider. :mod:`eval.check_thresholds` turns the report into a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gridsense.docrag.eval.golden import load_golden_dataset
from gridsense.docrag.eval.ragas_eval import (
    RAGAS_MAX_WORKERS,
    RAGAS_TIMEOUT_SECONDS,
    run_eval,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAGAS over the DocRAG golden dataset.")
    parser.add_argument("--dataset", help="Path to the golden dataset JSON.")
    parser.add_argument("--out", help="Write the full report as JSON to this path.")
    parser.add_argument("-k", type=int, help="Chunks to retrieve per query (default: settings).")
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Evaluate the current vector store instead of re-ingesting the corpus first.",
    )
    parser.add_argument("--no-publish", action="store_true", help="Do not push scores to Langfuse.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=RAGAS_TIMEOUT_SECONDS,
        help="Per-job RAGAS timeout in seconds. Raise it for a slow local judge.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=RAGAS_MAX_WORKERS,
        help="Concurrent RAGAS jobs. Keep low for Ollama; raise for a hosted endpoint.",
    )
    args = parser.parse_args(argv)

    dataset = load_golden_dataset(args.dataset)

    if not args.no_ingest:
        from gridsense.docrag.ingest import ingest_path

        n = ingest_path(dataset.corpus)
        print(f"Ingested {dataset.corpus} -> {n} chunk(s).\n")

    report = run_eval(
        dataset,
        k=args.k,
        publish=not args.no_publish,
        timeout=args.timeout,
        max_workers=args.workers,
    )
    print(report.render())

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

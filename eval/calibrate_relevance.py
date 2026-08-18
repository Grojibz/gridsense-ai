"""Measure the relevance distribution of the configured embedding model.

    python eval/calibrate_relevance.py            # ingest, then measure
    python eval/calibrate_relevance.py --no-ingest

Retrieval only — no judge, no generation, no tokens spent on a chat model.

**Why this exists.** `docrag_min_relevance` (the refusal floor) and
`docrag_uncertain_relevance` (the weak-retrieval signal) are not universal constants. They
are percentile cuts through one embedding model's similarity distribution, and that
distribution is not comparable across models: the same pair of texts scores differently
under `nomic-embed-text`, `voyage-4-lite` and `text-embedding-3-small`. Carrying the
numbers across a provider switch does not preserve the behaviour they encode — it silently
redefines what counts as relevant enough to answer from, in whichever direction the new
model's scale happens to run.

The current values were set this way against `nomic-embed-text`: answerable queries scored
a median top-1 of 0.70, unanswerable ones 0.57, and 0.60 was chosen to sit in that gap.
This script reproduces that measurement so the same reasoning can be applied to whatever
model is configured now, rather than guessed at.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from gridsense.config import get_settings
from gridsense.docrag.eval.golden import load_golden_dataset


def _summary(scores: list[float]) -> str:
    if not scores:
        return "no items"
    return (
        f"n={len(scores):<3} min={min(scores):.3f}  p25={statistics.quantiles(scores, n=4)[0]:.3f}"
        f"  median={statistics.median(scores):.3f}  max={max(scores):.3f}"
        if len(scores) >= 4
        else f"n={len(scores):<3} min={min(scores):.3f}  median={statistics.median(scores):.3f}"
        f"  max={max(scores):.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure top-1 relevance on the golden set.")
    parser.add_argument("--dataset", help="Path to the golden dataset JSON.")
    parser.add_argument("--no-ingest", action="store_true", help="Use the current store as is.")
    parser.add_argument(
        "-k", type=int, default=None, help="Chunks to retrieve (default: settings)."
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    dataset = load_golden_dataset(args.dataset)
    k = args.k if args.k is not None else settings.docrag_top_k

    if not args.no_ingest:
        from gridsense.docrag.ingest import ingest_path

        print(f"Ingesting {dataset.corpus} with {settings.embedding_provider.value}...")
        print(f"  {ingest_path(dataset.corpus)} chunk(s)\n")

    from gridsense.docrag.retriever import get_vectorstore

    store = get_vectorstore(settings)

    answerable: list[float] = []
    unanswerable: list[float] = []
    for item in dataset.items:
        pairs = store.similarity_search_with_relevance_scores(item.query, k=k)
        top1 = round(float(pairs[0][1]), 4) if pairs else 0.0
        (answerable if item.answerable else unanswerable).append(top1)

    print(f"Embedding model: {settings.embedding_provider.value} / top-1 relevance, k={k}\n")
    print(f"  answerable    {_summary(answerable)}")
    print(f"  unanswerable  {_summary(unanswerable)}")

    print("\nCurrent thresholds:")
    print(f"  docrag_min_relevance        {settings.docrag_min_relevance:.2f}   (refuse below)")
    print(f"  docrag_uncertain_relevance  {settings.docrag_uncertain_relevance:.2f}   (flag below)")

    if not answerable or not unanswerable:
        print("\nNot enough items in one class to say anything useful.")
        return 0

    lo, hi = max(unanswerable), min(answerable)
    print("\nWhat the measurement says:")
    if lo < hi:
        print(f"  The two classes separate: unanswerable top out at {lo:.3f}, answerable")
        print(f"  bottom out at {hi:.3f}. Any floor inside ({lo:.3f}, {hi:.3f}) separates them")
        print(f"  cleanly on this set — {(lo + hi) / 2:.2f} is the middle of that gap.")
    else:
        print(f"  The classes overlap: some unanswerable queries score up to {lo:.3f} while")
        print(f"  some answerable ones fall to {hi:.3f}. No floor separates them perfectly,")
        print("  so the choice is a trade: a higher floor refuses more real questions, a")
        print("  lower one answers more from irrelevant context. Pick against")
        print("  refusal_accuracy on the full eval, not against this number alone.")

    print("\nThese are inputs to a decision, not the decision. Set the values in")
    print("src/gridsense/config.py, then confirm with `make eval-ragas && make eval-gate`:")
    print("retrieval_recall and refusal_accuracy are what actually grade the choice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

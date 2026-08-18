---
name: bess-eval-gate
description: Diagnose a failing GridSense eval gate — which of the seven metrics broke, whether the judge or the chain caused it, and what is and is not a legitimate fix. Use when eval/results.json shows a violation, when the eval CI job is red, or when someone proposes changing a threshold.
---

# Diagnosing the eval gate

The gate is the contract a change satisfies before it can merge. `make eval-gate` (or the
`eval_gate` tool) prints every metric against its threshold.

## First: judged or deterministic?

This is the split that matters. It tells you whether you are looking at a scoring problem
or a real regression.

| Deterministic — no LLM in the loop | Judged by an LLM |
|---|---|
| `retrieval_recall`, `refusal_accuracy`, `output_validity` | `faithfulness`, `context_precision`, `context_recall`, `answer_relevancy` |
| Trustworthy whatever the judge model is | Only trustworthy with a capable hosted judge |

A deterministic metric dropping is always a real regression in the pipeline. A judged
metric dropping might be the judge.

## Read coverage before you read the score

Each judged metric is gated on **coverage** as well as score: at least
`MIN_COVERAGE_RATIO` of answerable items must have produced a number. A mean drawn from
the third of items the judge happened to succeed on is not a mean worth blocking on.

So: if coverage is failing, fix the judge before interpreting the score. A small local
model scores roughly half the items and its scores mean little either way.

## Metric → likely cause

- **`retrieval_recall`** — retrieval, not generation. Suspect the index before the prompt.
  A recall that is flat across `k` values points at a corrupted vector rather than a
  ranking problem; re-ingest and check the post-write self-retrieval verification.
- **`refusal_accuracy`** — the pipeline is refusing answerable items, or answering
  unanswerable ones. Note a refusal for the reason `invalid_output` does *not* count as a
  correct refusal; those land in `output_validity` instead.
- **`output_validity`** — the model cannot hold a schema. A hosted model should sit at 1.0.
  Items that never reached the provider at all (`provider_error` — credits, auth, rate
  limit) are **excluded** from this metric rather than scored either way, and counted
  separately in the report. A non-zero provider-error count means part of the run measured
  nothing, whatever the scores say: fix the provider and re-run before reading them.
- **`faithfulness`** — the answer states things the retrieved context does not support.
  Check the deterministic groundedness post-check too; the two disagreeing is informative.
- **`context_precision` / `context_recall`** — retrieval is pulling noise, or missing what
  it needs. Both use `reference_answer` as ground truth.
- **`answer_relevancy`** — the answer does not address the question. The noisiest of the
  four; read it last.

## What is not a fix

Lowering a threshold. Thresholds live in `src/gridsense/docrag/eval/thresholds.py` rather
than in CI YAML precisely so that loosening one shows up in code review as a change someone
has to argue for. If a threshold genuinely no longer reflects the bar, that is a
conversation with a measurement attached, not a one-line edit.

Two other rules exist to stop the gate being gamed, and should not be worked around:

- an answerable item the pipeline **refused** scores 0.0 on all four judged metrics rather
  than being dropped — otherwise refusing more would raise the mean;
- a metric **missing** from `results.json` counts as 0.0 — otherwise it could vanish
  quietly and pass.

## Ground truth rots

`eval/golden_dataset.json` pins `must_contain` substrings that must still appear verbatim
in `data/docs`. `tests/test_golden_dataset.py` enforces that on every PR without an LLM or
a datastore. If that test fails, the corpus changed and the dataset needs updating — the
pipeline is not necessarily broken.

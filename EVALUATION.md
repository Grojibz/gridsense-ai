# Evaluation: what runs where, and what the numbers actually mean

Measured 2026-08-01 against the live local stack (pgvector + Ollama), 32-item golden set,
6 chunks over `data/docs`.

> Written in English to match the rest of the repo.

## Short answer

The eval reports seven metrics. **Three are computed deterministically and are trustworthy
locally. Four require an LLM acting as a judge, and the local models cannot serve that
role** — they need a hosted endpoint (the CI `ragas-gate` job is wired to Azure OpenAI).

The distinction is *not* local-vs-cloud. It is **whether the model can reliably emit
structured output**. RAGAS chains several internal structured-output prompts per metric; a
7B local model fails them. A strong enough local model would work in principle —
`qwen2.5:7b-instruct` is not it, and neither is `llama3.2`.

| Metric | Judged by | Runs locally? |
|---|---|---|
| `retrieval_recall` | string match vs. ground-truth chunks | ✅ yes, no judge involved |
| `refusal_accuracy` | guardrail decision on the refusal reason | ✅ yes, no judge involved |
| `output_validity` | did the model return a parseable object | ✅ yes, no judge involved |
| `faithfulness` | RAGAS (LLM judge) | ❌ needs a hosted model |
| `answer_relevancy` | RAGAS (LLM judge) | ❌ needs a hosted model |
| `context_precision` | RAGAS (LLM judge) | ❌ needs a hosted model |
| `context_recall` | RAGAS (LLM judge) | ❌ needs a hosted model |

Both deterministic metrics still need the *pipeline* to run (so embeddings, and a chat
model for generation). What they don't need is an LLM grading the output.

## What the local run produces

```
items: 32  (answerable 23, unanswerable 9)

faithfulness       0.000   [19/23 scored]
answer_relevancy   0.000   [19/23 scored]
context_precision  0.000   [19/23 scored]
context_recall     0.000   [19/23 scored]
retrieval_recall   0.957
refusal_accuracy   0.444
output_validity    0.344
```

**The four zeros are not scores.** They are the answerable items that the pipeline refused,
which are assigned 0.0 deterministically without calling the judge. Every item that
*was* sent to the judge came back `RagasOutputParserException` → NaN → dropped. Coverage
lands at 19/23 = 83% for the same reason the other numbers are bad: 19 of 23 answerable
items never produced an answer to judge.

Read together, the three deterministic metrics say something specific: **retrieval works,
and almost nothing else gets a chance to.** Only 4 of 23 answerable items were actually
answered — 16 failed output parsing, 3 found no relevant context.

`refusal_accuracy` at 0.444 is the honest version of what used to read 1.000, and it is
mostly a *symptom* of `output_validity`: 5 of the 9 unanswerable items were refused because
the model produced garbage, not because a guardrail decided anything. Only 1 of the 9 was
caught by the intent router.

The gate handles this correctly — it fails on **coverage**, not just on the score:

```
$ python eval/check_thresholds.py eval/results.json
...
  faithfulness: only 48% of items scored (need 80%) — the judge failed too often to trust the mean
GATE_EXIT=1
```

That guard exists precisely so a run like this cannot be mistaken for a passing one, and so
a metric can't look healthy just because the judge quietly failed on most items.

## Known problems in the numbers above

Three problems, in the order they were found. Two are now fixed; the third is open. None
were bugs in the eval harness — the harness is what surfaced all three, which is the
argument for having built it.

### 1. ~~`refusal_accuracy = 1.000` is inflated~~ — fixed

The metric counted *any* refusal as correct. On the run that first reported 1.000, 2 of the
9 unanswerable items were "refused" only because the model produced unparseable output and
hit the `invalid_output` fallback — right answer, wrong reason. A *flakier* model scored as
a better-behaved one.

`refusal_accuracy` now only credits a refusal when a guardrail actually decided to refuse.
Parse failures land in a new **`output_validity`** metric instead — the fraction of items
where the model returned a parseable object at all — which is where they belong.

The effect is large and in the honest direction: the same pipeline now reports **0.444**,
because on the latest run 5 of the 9 unanswerable items were refused by a parse failure
rather than by a decision.

Still worth knowing: only 1 of the 9 is caught by the intent router (`oos-write-python`);
the rest fall through to the retrieval relevance gate. That is the designed behaviour (the
router is precision-tuned and defers when unsure), but the router does less work than the
headline number suggests.

### 2. ~~Retrieval is the real ceiling~~ — corrected: it was one bad vector

**An earlier version of this document blamed the embedding model. That was wrong**, and the
mistake is worth keeping on the record because the false diagnosis was very convincing.

The symptom: for "What is State of Health?", the chunk that *defines* SOH and states the
80% end-of-life threshold came back **last of six**, at relevance 0.331. `retrieval_recall`
sat at 0.870 and was completely **flat from k=4 to k=10**, which read as "the embedding
model ranks this corpus badly, and no amount of `k` will save it".

What it actually was: **a single corrupted vector in the index.** Computing cosine
similarity by hand settled it in one run —

| | store | recomputed cosine |
|---|---|---|
| BMS chunk | 0.5514 | 0.5514 ✅ |
| Degradation mechanisms | 0.4521 | 0.4521 ✅ |
| Thermal management | 0.4405 | 0.4405 ✅ |
| **State of Health chunk** | **0.3311** | **0.7068 ❌** |

Four of six matched the store exactly, so the relevance conversion was fine and the
embedding model was fine. One chunk's *stored* vector simply wasn't the embedding of the
text sitting next to it. A single re-ingest fixed it:

```
before:  retrieval_recall 0.870   (misses: soh-eol-threshold, soh-definition, soh-vs-soc)
after:   retrieval_recall 0.957   (miss:   cycle-ageing-per-cycle-cost)
```

**The lesson is the failure mode, not the fix.** A bad vector is completely silent: no
error, no log, no failed request. The chunk just stops being retrievable, and every
downstream symptom points at the model instead. `ingest_path(..., verify=True)` (the
default) now checks that every chunk retrieves *itself* after a write, and raises
`IndexVerificationError` if it does not. That check would have caught this in seconds.

Note on `k`: recall reaches 1.000 at k=6, but the corpus only has 6 chunks — k=6 returns
everything, which is not retrieval. `docrag_top_k` stays at 4 and the gate is set at 0.90
against the measured 0.957.

### 3. The structured-output schema is unreliable on local models (open)

This is the dominant problem, and it masks everything else.

- `llama3.2` (the documented offline default) returns things like
  `{"answer": "...usch moss ace ...@@@@"}` with `confidence` missing entirely, and
  effectively never answers.
- `qwen2.5:7b-instruct` is materially better and still fails most of the time.

**The failure rate is not stable, and two runs is not enough to quote one.** Across two
runs on the same 32 items, `qwen2.5:7b-instruct` went from ~9 parse failures to 21 —
`output_validity` 0.72 → 0.344. Ollama had been under sustained load for hours by the
second run and the variable was not isolated, so treat both figures as "frequently, and
unpredictably" rather than as a measured rate.

Before the P1 downstream guardrail existed, this propagated an `OutputParserException`
straight out of `/ask` as a 500. It is now contained by a retry plus a deterministic
fallback — the reason this eval run completed at all instead of dying on the first
malformed completion. But *contained* is not *fixed*: the schema or the prompt needs work,
or `docrag_output_retries` needs raising, or the offline default needs to be a model that
can hold a schema.

## Relevance score distribution

Top-1 relevance with `nomic-embed-text`, used to calibrate `docrag_uncertain_relevance`:

| | min | p25 | median | max |
|---|---|---|---|---|
| answerable queries (23) | 0.542 | 0.651 | 0.701 | 0.853 |
| unanswerable queries (9) | 0.348 | — | 0.573 | 0.671 |

The bands overlap heavily. `docrag_uncertain_relevance = 0.60` sits in the gap; 0.65 would
have flagged a quarter of the good queries as weak.

**These numbers are provider-specific.** The relevance scale is not comparable across
embedding models — re-measure before trusting any relevance threshold after switching.

## How to run each mode

```bash
# Deterministic metrics only — works fully offline.
# The RAGAS metrics will report 0.000 at low coverage; ignore them.
make eval-ragas

# Slow local judge: fewer workers, longer timeout (these are already the defaults).
python eval/run_ragas.py --workers 2 --timeout 900 --out eval/results.json

# Real RAGAS numbers: point at a hosted model.
LLM_PROVIDER=azure AZURE_OPENAI_API_KEY=... python eval/run_ragas.py --out eval/results.json
python eval/check_thresholds.py eval/results.json
```

RAGAS's own defaults (16 workers / 180 s) starve a serialising Ollama and make every job
time out to NaN. The repo defaults to 2 workers / 900 s for that reason; raise the workers
for a hosted endpoint.

## In CI

`.github/workflows/eval.yml` splits along exactly this line:

- **`golden-dataset`** — always runs, on every PR. Validates the dataset schema, checks
  every `must_contain` string is still verbatim in `data/docs`, and exercises the gate
  logic. No LLM, no datastore, a few seconds.
- **`ragas-gate`** — the merge gate. Brings up pgvector, runs the full eval against Azure
  OpenAI, and fails the build on a threshold or coverage breach. Skips itself when the
  Azure secrets are absent (fork PRs), because a missing credential must not read as a
  passing gate.

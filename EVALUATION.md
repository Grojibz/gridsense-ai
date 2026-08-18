# Evaluation: what runs where, and what the numbers actually mean

Measured 2026-08-01 against the live local stack (pgvector + Ollama), 32-item golden set,
6 chunks over `data/docs`.

> Written in English to match the rest of the repo.

> **Status after the Claude switch (M8): not re-measured.**
> `CHAT_PROVIDER` now defaults to `anthropic`, which changes *both* halves of this document
> — the judge that scores the four RAGAS metrics, and the chain that produces the answers
> the three deterministic metrics grade. Every number below still comes from the
> 2026-08-01 local `llama3.2` run. They have deliberately not been edited to what a hosted
> model "should" score: the whole argument of this document is that measured numbers beat
> assumed ones.
>
> What to expect, and what to check when the run happens:
> - the four judged metrics should gain **coverage** first (a capable judge scores all 23
>   answerable items, not ~11) — read coverage before reading the score;
> - `output_validity` and `refusal_accuracy` are deterministic and depend on the **chain**,
>   not the judge, so they move because Claude is answering, not because Claude is grading;
> - if `output_validity` does not reach 1.0 on a hosted model, that is a bug to investigate,
>   not a threshold to loosen.
>
> ```bash
> make eval-ragas && make eval-gate     # then rewrite the tables below from results.json
> ```

## Short answer

The eval reports seven metrics. **Three are computed deterministically and are trustworthy
locally. Four go through an LLM judge, and a local model produces real numbers for them but
not on enough items to gate on** — hence the CI `ragas-gate` job pointing at Azure OpenAI.

The distinction is *not* local-vs-cloud, and it is not that RAGAS "cannot run locally" —
an earlier version of this document said exactly that, on measurements taken from a machine
whose GPU was computing incorrectly (see problem #3). On sound hardware `llama3.2` scores
all four metrics. What it does not do is score them *consistently*: RAGAS chains several
internal structured-output prompts per metric, and the judge still fails on roughly half
the items, which leaves the mean untrustworthy even though it is no longer empty.

| Metric | Judged by | Runs locally? |
|---|---|---|
| `retrieval_recall` | string match vs. ground-truth chunks | ✅ yes, no judge involved |
| `refusal_accuracy` | guardrail decision on the refusal reason | ✅ yes, no judge involved |
| `output_validity` | did the model return a parseable object | ✅ yes, no judge involved |
| `faithfulness` | RAGAS (LLM judge) | ⚠️ scores, 48% coverage |
| `answer_relevancy` | RAGAS (LLM judge) | ⚠️ scores, 48% coverage |
| `context_precision` | RAGAS (LLM judge) | ⚠️ scores, 43% coverage |
| `context_recall` | RAGAS (LLM judge) | ⚠️ scores, 96% coverage |

Both deterministic metrics still need the *pipeline* to run (so embeddings, and a chat
model for generation). What they don't need is an LLM grading the output.

## What the local run produces

`llama3.2`, CPU-only inference, sound index:

```
items: 32  (answerable 23, unanswerable 9)

faithfulness       0.689   [11/23 scored]
answer_relevancy   0.589   [11/23 scored]
context_precision  0.900   [10/23 scored]
context_recall     0.752   [22/23 scored]
retrieval_recall   0.957
refusal_accuracy   0.778
output_validity    1.000
```

**The three deterministic metrics are the trustworthy ones, and they are healthy.**
`output_validity` at 1.000 means every one of the 32 items produced a parseable object —
the schema-following problem was hardware, not the model. `refusal_accuracy` at 0.778 is
7 of 9, and is now measuring guardrail decisions rather than crashes. Only 1 of the 9 is
caught by the intent router; the rest fall through to the retrieval relevance gate, which
is the designed behaviour but means the router does less work than it appears to.

**The four RAGAS scores are real but under-covered.** `context_recall` is credible at 96%
coverage; the other three sit at 43–48%, meaning the judge failed on more than half the
items and the mean is drawn from a biased remainder. The gate fails on coverage for exactly
that reason, and would fail on `faithfulness` (0.689 < 0.80) and `answer_relevancy`
(0.589 < 0.60) besides — but those two numbers should not be quoted as the system's real
quality until they are measured against a judge that answers reliably.

The gate reads both, and fails on **coverage** as well as on score:

```
$ python eval/check_thresholds.py eval/results.json
[FAIL] faithfulness       0.689  (min 0.80)
[  ok] context_precision  0.900  (min 0.70)
[  ok] context_recall     0.752  (min 0.60)
[FAIL] answer_relevancy   0.589  (min 0.60)
[  ok] retrieval_recall   0.957  (min 0.90)
[FAIL] refusal_accuracy   0.778  (min 0.80)
[  ok] output_validity    1.000  (min 0.95)
  faithfulness: only 48% of items scored (need 80%) — the judge failed too often to trust the mean
GATE_EXIT=1
```

The coverage guard is what stops `context_precision = 0.900` from reading as good news: it
is an average over 10 of 23 items, and the 13 the judge dropped are not a random sample.

## Known problems in the numbers above

Five problems, in the order they were found. Four are fixed; the third is open.

The first three were surfaced *by* the eval harness, which is the argument for having built
it. The last two were faults in the harness's own plumbing, and are the more uncomfortable
ones: a gate that had never executed, and then a gate that reported success without
evaluating anything. A measurement you never take and a measurement that always passes fail
in the same direction.

### 1. ~~`refusal_accuracy = 1.000` is inflated~~ — fixed

The metric counted *any* refusal as correct. On the run that first reported 1.000, 2 of the
9 unanswerable items were "refused" only because the model produced unparseable output and
hit the `invalid_output` fallback — right answer, wrong reason. A *flakier* model scored as
a better-behaved one.

`refusal_accuracy` now only credits a refusal when a guardrail actually decided to refuse.
Parse failures land in a new **`output_validity`** metric instead — the fraction of items
where the model returned a parseable object at all — which is where they belong.

The fix earns its keep on broken hardware: while the GPU was corrupting completions, the
old metric read 1.000 and the corrected one read 0.444, because 5 of the 9 refusals were
crashes. On sound hardware the two converge — 0.778, all 7 of them real decisions.

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

### 3. ~~The structured-output schema is unreliable on local models~~ — it was the GPU

**This entry was wrong twice before it was right**, and the wrong versions are kept because
the false trail is the useful part.

The symptom: the model returned fluent-looking garbage —
`{"answer": "...usch moss ace 잡indsight interrupted@ Gone@@@@@@"}` — often enough that
`output_validity` fell to 0.344, with only 4 of 23 answerable items answered. Every
hypothesis fit the evidence and every one was wrong:

1. *"llama3.2 is too weak to hold a schema."* But qwen2.5:7b failed too.
2. *"Ollama defaults `num_ctx` to 2048 and the RAG prompt overflows it."* True about the
   default, and worth fixing — but a 31-token prompt failed just as hard.
3. *"qwen2.5:7b doesn't fit in VRAM and partial offload corrupts it."* It genuinely didn't
   fit (2.4 GB resident of 5.4 GB) — but llama3.2, fully resident, failed identically.

The actual cause, isolated by forcing CPU inference on the same model, same prompt, same
server:

| | output for `"Say OK"` |
|---|---|
| GPU | `-то fluoresiller expectationTickgowrud@@@` |
| CPU (`num_gpu: 0`) | `OK. How can I assist you today?` |

**The GPU was computing incorrectly.** On the 8-item probe: 0/8 on GPU, **8/8 on CPU**.
llama3.2 holds the schema perfectly; nothing was wrong with `_LLMAnswer`, the prompt, or
the model.

This also closes problem #2 above: `nomic-embed-text` ran on the same GPU, so the corrupted
index vectors and the garbage completions were **one root cause, not two**.

**The transferable lesson: a faulty GPU does not raise. It returns confident nonsense.**
Every layer above it — the model, the schema, the retriever, the eval — reports a plausible
domain-level failure instead, and each one is a convincing place to spend an afternoon.
When several independent components degrade at once, suspect the layer underneath them all
before theorising about any of them. Forcing CPU inference is a two-minute test that would
have ended this immediately.

`OLLAMA_NUM_GPU=0` is the escape hatch; `OLLAMA_NUM_CTX` (default 8192 here) fixes the
unrelated-but-real 2048 truncation.

The instability that made this so hard to pin down is itself the tell. Across runs on the
same 32 items, `output_validity` moved 0.72 → 0.344 → **1.000**, with no change to the
schema, the prompt, or the model — only to whether the GPU was in the loop. A software
defect does not drift like that.

The P1 downstream guardrail is what made any of this survivable. Before it, a malformed
completion propagated an `OutputParserException` straight out of `/ask` as a 500, and it
killed the first eval run outright. With the retry and deterministic fallback in place, the
pipeline degraded into clean refusals for hours on hardware that was returning noise, and
kept producing usable telemetry the whole time. That was written as a guardrail against a
weak model; it turned out to be a guardrail against a failing GPU.

### 4. ~~The eval workflow had never run~~ — fixed

The first pull request on this repository turned three checks red at once:
`lint-and-test (3.11)`, `lint-and-test (3.12)` and `golden-dataset`. All three passed
locally. Two independent faults, and the interesting part is why neither had ever shown up.

**Why nothing had caught them.** `eval.yml` triggers on `pull_request` or a push to `main`.
Work had been happening on a feature branch, with no PR open, so **the workflow had never
executed once since it was written.** A gate that has never run is not a gate; it is a file
that looks like one. The CI badge in the README was green throughout, because `ci.yml` runs
on the same triggers and had last succeeded on `main`, before any of this code existed.

**Fault one — the obvious one.** `ci.yml` installed `.[docrag,degrade,dev]`. `main.py` had
gained an import of `routes_agent`, which imports the Anthropic SDK, so the whole suite
failed at collection on a missing `anthropic`. The sibling workflow had been updated for
the new extra and this one had not. Ordinary oversight, caught the moment CI ran.

**Fault two — the one worth keeping.** `golden-dataset` failed on `ModuleNotFoundError: No
module named 'pandas'` — in a job that scores a JSON dataset and some threshold arithmetic,
and touches no dataframe anywhere. The chain:

```
conftest.py  ->  gridsense.api.main  ->  routes_predict  ->  degrade.serve  ->  import pandas
```

`conftest.py` imported `create_app` at module level. pytest loads `conftest.py` for the
whole directory before collecting *any* test, so importing the application there pulled the
entire dependency graph — FastAPI, the ML stack, MLflow — into every job, regardless of
what that job actually ran. The deterministic job needed scikit-learn installed to collect
a test that scores a dictionary.

This was latent from the day the job was written, and it would have stayed latent: it is
invisible in any environment where everything happens to be installed.

**Why the local test run could not have found it.** Both faults were masked by exactly the
thing that makes local testing convenient — a development environment with every extra
installed. `pytest` passed locally on all 256 tests while two CI jobs were structurally
incapable of collecting a single one. Reading the workflow files did not settle it either:
the first two hypotheses from tracing imports by hand were both incomplete. What settled it
was building **one clean virtualenv per job, with that job's exact extras**, and running
that job's exact command:

```
.[docrag,agent,dev]           golden-dataset   ModuleNotFoundError: pandas
.[docrag,degrade,agent,dev]   lint-and-test    ModuleNotFoundError: anthropic
```

Two minutes of setup turned a guess into a diagnosis.

**The fix.** Fault one: add the `agent` extra to `ci.yml`. Fault two: move both imports
*inside* the `client` fixture, so the application graph is imported by the tests that use
it and by nothing else. Adding `degrade` to the job would also have made it pass, at the
price of installing scikit-learn, MLflow and Evidently to run 103 tests that never touch
them — a symptom fix that makes every future job slower.

**The general lesson.** A CI job's install list is part of its contract, and the only
environment that tests that contract is a clean one. If a job has never run, treat it as
untested code, because that is what it is.

### 5. A skipped gate reported as a passing one — fixed

Immediately after the above, the two credentialed gates went **green** without evaluating
anything. The credential check was a *step*-level `if:`, so the job started, skipped every
step inside it, failed nothing, and reported success.

That is the precise reading these gates exist to prevent: this document already argues that
"a missing credential must never read as a passing gate", and the implementation was doing
exactly that. The earlier `Skipped` status had been incidental — the jobs were skipped
because `golden-dataset`, which they depend on, had failed.

The check now lives in a `preflight` job whose output gates the others at *job* level, so an
absent credential renders as **Skipped**, not green. One honest caveat remains: GitHub
treats a skipped required check as satisfied, so a green PR is still not proof the gates
ran. The check state has to be read, not just the tick.

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
`eval/calibrate_relevance.py` (or `make calibrate`) reproduces exactly this measurement for
whichever model is configured; it is retrieval only, so it costs no chat tokens.

> **Outstanding for the Voyage switch.** `EMBEDDING_PROVIDER=voyage` is wired but the table
> above is still `nomic-embed-text`. The thresholds have **not** been recalibrated, and
> carrying them across unchanged is the mistake this section warns about. Run
> `make calibrate` against Voyage, set the values in `config.py`, then confirm with the
> full gate — `retrieval_recall` and `refusal_accuracy` are what actually grade the choice.

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

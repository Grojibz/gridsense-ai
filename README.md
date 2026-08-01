# GridSense AI

> An open-source lab bringing **generative AI** and **classical ML** to energy-storage
> engineering. Two production-minded modules in one repo:
> a **RAG assistant over technical documentation** and an **MLOps pipeline for
> battery-degradation prediction**.

[![CI](https://github.com/Grojibz/gridsense-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Grojibz/gridsense-ai/actions/workflows/ci.yml)
[![Eval](https://github.com/Grojibz/gridsense-ai/actions/workflows/eval.yml/badge.svg)](https://github.com/Grojibz/gridsense-ai/actions/workflows/eval.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Why this project

Engineering teams working on battery energy-storage systems (BESS) juggle two very
different AI needs:

1. **Knowledge retrieval** — answers buried in standards, datasheets, and test
   reports, where a wrong citation is worse than no answer.
2. **Predictive modelling** — small, well-understood models (cell degradation,
   state-of-health) that must be *reproducible, monitored, and deployable*, not just
   accurate in a notebook.

`GridSense AI` demonstrates a clean, end-to-end approach to **both** — combining
classical ML with generative AI in a single, well-architected codebase.

This is a portfolio/lab project: the goal is to show **production engineering
discipline** (clean architecture, tests, CI/CD, observability, deployment), not just
a model that works once.

---

## What this demonstrates

| Area | How |
|---|---|
| **GenAI orchestration** | RAG pipeline built with **LangChain** (retrievers, chains, structured output) |
| **Vector search** | **pgvector** on Postgres; pluggable to Pinecone |
| **Cloud AI** | **Azure OpenAI** for LLM + embeddings, with a local **Ollama** fallback |
| **LLM observability** | Tracing, latency and cost dashboards via **Langfuse** |
| **GenAI optimisation** | Prompt templates, retrieval tuning, and an **eval harness** (LLM-as-judge + groundedness) |
| **RAG evaluation** | Curated **golden dataset** + **RAGAS** metrics, enforced as a **merge gate** in CI |
| **Classical ML** | Battery-degradation regressor (scikit-learn / PyTorch) |
| **MLOps** | Experiment tracking & model registry (**MLflow**), reproducible pipelines |
| **Monitoring** | Data/concept **drift detection** (Evidently) with alerting |
| **Software engineering** | Type hints, design patterns, **pytest**, **ruff**, pre-commit, Clean Code |
| **Delivery** | **Docker** + docker-compose, **GitHub Actions CI/CD**, optional **Kubernetes** manifests |
| **APIs** | **FastAPI** services with Pydantic schemas and OpenAPI docs |
| **SQL** | Postgres for documents, embeddings, run metadata, and prediction logs |

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │                FastAPI                    │
                        │   /ask  (RAG)        /predict  (ML)        │
                        └───────┬───────────────────────┬───────────┘
                                │                       │
              ┌─────────────────▼───────┐     ┌─────────▼──────────────┐
              │   Module A — DocRAG      │     │  Module B — DegradeML   │
              │                          │     │                         │
              │  LangChain orchestration │     │  Training pipeline      │
              │  ├─ ingestion / chunking │     │  ├─ feature build       │
              │  ├─ pgvector retriever   │     │  ├─ train + eval        │
              │  ├─ Azure OpenAI LLM     │     │  ├─ MLflow registry     │
              │  └─ cited answers        │     │  └─ Evidently drift     │
              └────────┬─────────────────┘     └──────────┬──────────────┘
                       │                                  │
        ┌──────────────▼───────────┐      ┌───────────────▼─────────────┐
        │  Postgres + pgvector      │      │  MLflow tracking + registry │
        └───────────────────────────┘      └─────────────────────────────┘
                       │
              ┌────────▼─────────┐
              │     Langfuse     │  (tracing, cost, latency)
              └──────────────────┘
```

Everything runs locally via `docker-compose` (Postgres+pgvector, MLflow, Langfuse).
Cloud LLM calls use Azure OpenAI; set `LLM_PROVIDER=ollama` to run fully offline.

---

## Repository structure

```
gridsense-ai/
├── README.md
├── pyproject.toml              # deps, ruff, pytest config
├── Dockerfile                  # API image (serves /ask + /predict)
├── docker-compose.yml          # postgres+pgvector, mlflow, langfuse, api
├── Makefile                    # common dev commands (make help)
├── .github/workflows/ci.yml    # lint + test + build on every PR
├── .github/workflows/eval.yml  # golden-dataset checks + RAGAS merge gate on every PR
├── .pre-commit-config.yaml
├── eval/
│   ├── golden_dataset.json     # curated Q&A + retrieval ground truth
│   ├── run_ragas.py            # run RAGAS over the golden set -> results.json
│   └── check_thresholds.py     # exit non-zero if a score is under its gate
├── src/
│   └── gridsense/
│       ├── config.py           # pydantic-settings, env-driven
│       ├── api/                # FastAPI app + routers
│       ├── docrag/             # Module A
│       │   ├── ingest.py       # loaders, chunking, embedding
│       │   ├── retriever.py    # pgvector retriever
│       │   ├── chain.py        # LangChain RAG chain, cited output
│       │   ├── guardrails/     # upstream input checks + downstream output validation
│       │   └── eval/           # eval harness: LLM-as-judge + RAGAS + gate thresholds
│       └── degrade/            # Module B
│           ├── data.py         # feature engineering (SQL-backed)
│           ├── train.py        # training + MLflow logging
│           ├── evaluate.py     # metrics, model card
│           ├── monitor.py      # Evidently drift report
│           └── serve.py        # load registered model, predict
├── k8s/                        # optional: deployment + service manifests
├── data/                       # sample docs + synthetic battery dataset
├── notebooks/                  # exploration only — not the source of truth
└── tests/                      # pytest, both modules
```

---

## Module A — DocRAG (generative AI)

A retrieval-augmented assistant over technical documents that **always cites its
sources** and refuses to answer when retrieval confidence is low.

- Ingestion: PDF/markdown loaders → semantic chunking → Azure OpenAI embeddings → pgvector,
  with a post-write check that every chunk still retrieves itself (a bad vector is otherwise
  silent — see [`EVALUATION.md`](EVALUATION.md)).
- Retrieval: top-k similarity + optional reranking; metadata filters (doc type, section).
- Generation: LangChain chain returning a structured answer `{answer, citations[], confidence}`.
- Guardrails: upstream input checks, retrieval gate, output validation, uncertainty disclosure.
- Observability: every call traced in Langfuse (prompt, retrieved chunks, latency, cost).

### Guardrails

`docrag/guardrails/` wraps the model on both sides. Everything is a pure function over
plain data, so the whole path is tested without a model, a datastore or a network.

**Upstream** — runs before a single token is spent, so a rejected query costs nothing:

| Check | Behaviour |
|---|---|
| Length cap | Hard token limit; over it the query is **rejected, never truncated** |
| Prompt injection | Weighted patterns over the known families (instruction override, role reassignment, system-prompt exfiltration, delimiter smuggling, encoded payloads). The payload is redacted before it reaches the trace store |
| Intent router | Hard-rejects clearly out-of-scope requests (code generation, translation, live data). Precision-tuned: anything ambiguous falls through to retrieval |
| PII scrubber | Email / phone / IP / IBAN / card redacted to `[LABEL]` before logging **and** before the LLM call. Optional spaCy NER layer for names and orgs (`DOCRAG_PII_NER_ENABLED`) |

The injection detector is pattern-based and deliberately not called a classifier — the
score is the sum of matched pattern weights. Its test suite asserts the false-positive
side too: a guardrail that blocks real BESS questions is worse than no guardrail.

**Downstream** — runs on the model's output:

- **Schema validation with a bounded retry.** Malformed structured output is retried once
  with a repair instruction, then falls back deterministically. Before this existed, one
  bad completion from the local model propagated an `OutputParserException` straight out
  of `/ask`.
- **Faithfulness post-check.** Sentence-level lexical support against the retrieved
  chunks, no LLM. A number stated in the answer but absent from the context marks its
  sentence unsupported outright — an invented figure is the failure that actually hurts in
  technical documentation.

**Uncertainty disclosure.** Confidence, retrieval strength and grounding are combined into
`uncertainty_level` (`low` / `medium` / `high`) plus the `uncertainty_reasons` behind it.
A shaky answer is **disclosed, not suppressed**: it still reaches the user, labelled, with
the caveat rendered above it in the chat UI. Refusals are flagged structurally
(`refused` + `refusal_reason`) rather than by string-matching the answer text.

The disclosure lives in its own field rather than being prefixed onto the answer, so it
never contaminates the text RAGAS scores for faithfulness.

**Eval harness** (`docrag/eval/`): a small labelled question set scored automatically
on *groundedness*, *answer relevance*, and *citation correctness* — run in CI so
regressions in prompt/retrieval are caught.

### RAG evaluation & the merge gate

A RAG system without a regression gate is a demo. `eval/` holds the contract a change has
to satisfy before it can merge.

**Golden dataset** (`eval/golden_dataset.json`) — 32 curated items over the real corpus:
23 answerable (factual lookups and cross-document synthesis) plus 9 deliberate edge cases
split across `vague` (no antecedent), `out_of_scope` (nothing to do with BESS), and
`insufficient_context` (in-domain but genuinely absent from the docs). Each item carries
two kinds of ground truth:

- `expected_chunks` — the source file plus *verbatim substrings* a correctly retrieved
  chunk must contain. This yields `retrieval_recall`, a deterministic score with no LLM in
  the loop, so a retrieval regression is caught even when the generator papers over it
  with a plausible answer. `tests/test_golden_dataset.py` asserts every substring is still
  present in `data/docs`, so the ground truth can't silently rot.
- `reference_answer` — the ground truth RAGAS uses for `context_precision` / `context_recall`.

**Metrics.** `eval/run_ragas.py` replays every query through the *real* chain (same
retriever, prompt and guardrails as `/ask`) and scores it:

| Metric | Source | Gate |
|---|---|---|
| `faithfulness` | RAGAS | ≥ 0.80 |
| `context_precision` | RAGAS | ≥ 0.70 |
| `context_recall` | RAGAS | ≥ 0.60 |
| `answer_relevancy` | RAGAS | ≥ 0.60 |
| `retrieval_recall` | deterministic (ground-truth chunks) | ≥ 0.85 |
| `refusal_accuracy` | deterministic (edge cases refused) | ≥ 0.80 |

Two deliberate design choices: an answerable item the pipeline *refused* scores 0.0 on all
four RAGAS metrics rather than being dropped (an unjustified refusal is a failure, not a
missing datapoint); and each metric's **coverage** is gated too, so a mean can't look
healthy just because the judge silently failed on most items.

```bash
make install-eval      # pip install -e ".[docrag,eval]"
make eval-ragas        # run + write eval/results.json, push scores to Langfuse
make eval-gate         # exit non-zero if any score is under threshold
```

Per-item scores are pushed to Langfuse as custom scores on a tagged eval trace, so a
faithfulness drop shows up next to the production traces, not only in CI logs.

**CI.** `.github/workflows/eval.yml` runs on every PR. The `golden-dataset` job always
runs (schema, ground-truth-vs-corpus, gate logic — no LLM, a few seconds). The
`ragas-gate` job spins up pgvector, runs the full eval against Azure OpenAI, uploads
`results.json` as an artifact, and fails the build on a threshold breach. Mark it as a
required status check in branch protection to actually block the merge; it skips itself
on fork PRs, where the secrets aren't available.

> Thresholds live in `src/gridsense/docrag/eval/thresholds.py`, not in the CI YAML, so
> loosening the gate shows up in code review like any other change.

**The four RAGAS metrics need a hosted model.** They use an LLM as judge, and the local
models cannot serve that role — RAGAS chains several internal structured-output prompts per
metric and a 7B model fails them, so all four come back NaN and the gate fails on coverage
rather than reporting a bogus mean. Only `retrieval_recall` and `refusal_accuracy` are
trustworthy offline; both are computed deterministically, with no judge involved.

**[`EVALUATION.md`](EVALUATION.md) has the measured numbers, what runs where, and the three
problems those numbers exposed** — an inflated `refusal_accuracy` (fixed), a chunk that had
silently stopped being retrievable (fixed, and now guarded at ingest), and an unreliable
structured-output schema on local models (open).

## Module B — DegradeML (classical ML + MLOps)

A small, honest predictive model for battery cell **state-of-health / degradation**,
treated as a production asset.

- Reproducible training pipeline; all runs logged to MLflow (params, metrics, artifacts).
- Model promoted to the MLflow registry with a generated **model card**.
- `serve.py` loads the registered model behind `/predict`.
- `monitor.py` produces an **Evidently** drift report comparing live inputs to the
  training distribution, with a simple alert threshold.
- Prediction requests/outcomes logged to Postgres for later analysis (SQL).

> Dataset: synthetic-but-physically-plausible battery degradation data is generated in
> `data/` so the repo is fully runnable with no proprietary data.

---

## Getting started

The default setup runs **fully offline** using a local [Ollama](https://ollama.com) for the
LLM + embeddings; set `LLM_PROVIDER=azure` (and the `AZURE_OPENAI_*` keys) to use Azure
OpenAI instead.

```bash
# 0. Prereqs: Docker, and Ollama with the models pulled (for the offline default)
ollama pull llama3.2 && ollama pull nomic-embed-text

# 1. Configure
cp .env.example .env        # defaults to LLM_PROVIDER=ollama; Langfuse keys auto-provisioned

# 2. Bring up the whole stack: postgres+pgvector, mlflow, langfuse, and the API
docker compose up -d --build     # or: make up

# 3. Module A — ingest sample docs, then ask
docker compose exec api python -m gridsense.docrag.ingest data/docs
curl localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What SOH threshold is end of life for grid batteries?"}'

# 4. Module B — train + register, then predict
docker compose exec api python -m gridsense.degrade.train
curl localhost:8000/predict -H 'content-type: application/json' \
  -d '{"cycle_count":1500,"avg_temperature_c":33,"avg_dod":0.7,"avg_c_rate":1.0,"calendar_age_days":600}'

# 5. Drift report over recent /predict inputs vs the training distribution
docker compose exec api python -m gridsense.degrade.monitor
```

**Local dev** (without the API container): `make install` (or `pip install -e ".[docrag,degrade,dev]"`),
then run the modules with your own interpreter and `uvicorn gridsense.api.main:app --reload`.
`make help` lists the common tasks (`lint`, `test`, `ingest`, `train`, `eval`, `monitor`).

Service URLs: API → `:8000` (**chat UI at `/`**, OpenAPI at `/docs`), MLflow → `:5000`,
Langfuse → `:3000`. The chat page is a dependency-free DocRAG assistant that calls `/ask`.

### Testing

```bash
make test               # offline unit tests (fakes — no infra needed; this is what CI runs)
make test-integration   # gated end-to-end tests against the live stack (RUN_INTEGRATION=1)
```

### Kubernetes (optional)

`k8s/` holds Namespace / ConfigMap / Secret / Deployment / Service manifests for the API
(datastores + Ollama are referenced as external endpoints). Build and push the image, then
`kubectl apply -f k8s/`.

---

## Roadmap / milestones

This is built milestone by milestone so progress is visible in the commit history.

- [x] **M0 — Scaffold.** Repo, `pyproject`, ruff+pytest+pre-commit, CI green, docker-compose up.
- [x] **M1 — DocRAG MVP.** Ingest → pgvector → LangChain chain → cited answer via FastAPI.
- [x] **M2 — DocRAG hardening.** Langfuse tracing, confidence/guardrails, eval harness.
- [x] **M3 — DegradeML training.** Feature build (SQL), training, MLflow tracking + registry, model card.
- [x] **M4 — Serve + monitor.** `/predict` from registry, Evidently drift report, prediction logging.
- [x] **M5 — Deploy + polish.** Dockerised API service, k8s manifests, Makefile, docs.
- [x] **M6 — Eval gate.** Golden dataset with retrieval ground truth, RAGAS scoring,
      Langfuse custom scores, threshold gate wired into CI on every PR.
- [x] **M7 — Guardrails.** Upstream input checks (injection, PII, token cap, intent),
      downstream output validation with retry + deterministic fallback, uncertainty disclosure.

**Definition of done (per module):** runs from a clean clone via documented commands,
covered by tests, traced/tracked (Langfuse / MLflow), and green in CI.

> **Notes on this build.** The offline default uses a local **llama3.2 (3B)**, so DocRAG
> answer/judge quality is indicative rather than production-grade — switch `LLM_PROVIDER`
> to Azure OpenAI for stronger results (the provider abstraction is built in). **CI** runs
> the offline unit tests; the integration/eval tests are gated (`RUN_INTEGRATION=1`) as they
> need a live Postgres/Ollama/MLflow stack.

---

## Tech stack

`Python 3.11` · `LangChain` · `pgvector` / `Postgres` · `Azure OpenAI` (+ `Ollama` fallback) ·
`Langfuse` · `RAGAS` · `MLflow` · `Evidently` · `FastAPI` · `Pydantic` · `Docker` / `docker-compose` ·
`Kubernetes` (optional) · `GitHub Actions` · `pytest` · `ruff` · `pre-commit`

## License

MIT — see `LICENSE`.

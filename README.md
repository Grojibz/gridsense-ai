# GridSense AI

> An open-source lab bringing **generative AI** and **classical ML** to energy-storage
> engineering. Two production-minded modules in one repo:
> a **RAG assistant over technical documentation** and an **MLOps pipeline for
> battery-degradation prediction**.

![status](https://img.shields.io/badge/status-WIP-orange)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-blueviolet)

*(Replace badges/links once the repo is live. Rename the project if you prefer —
`GridSense` is a placeholder.)*

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
├── .pre-commit-config.yaml
├── src/
│   └── gridsense/
│       ├── config.py           # pydantic-settings, env-driven
│       ├── api/                # FastAPI app + routers
│       ├── docrag/             # Module A
│       │   ├── ingest.py       # loaders, chunking, embedding
│       │   ├── retriever.py    # pgvector retriever
│       │   ├── chain.py        # LangChain RAG chain, cited output
│       │   └── eval/           # eval harness (LLM-as-judge, groundedness)
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

- Ingestion: PDF/markdown loaders → semantic chunking → Azure OpenAI embeddings → pgvector.
- Retrieval: top-k similarity + optional reranking; metadata filters (doc type, section).
- Generation: LangChain chain returning a structured answer `{answer, citations[], confidence}`.
- Guardrails: groundedness check; "I don't know" path when no relevant chunk.
- Observability: every call traced in Langfuse (prompt, retrieved chunks, latency, cost).

**Eval harness** (`docrag/eval/`): a small labelled question set scored automatically
on *groundedness*, *answer relevance*, and *citation correctness* — run in CI so
regressions in prompt/retrieval are caught.

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
`Langfuse` · `MLflow` · `Evidently` · `FastAPI` · `Pydantic` · `Docker` / `docker-compose` ·
`Kubernetes` (optional) · `GitHub Actions` · `pytest` · `ruff` · `pre-commit`

## License

MIT — see `LICENSE`.

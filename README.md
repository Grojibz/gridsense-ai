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
├── docker-compose.yml          # postgres+pgvector, mlflow, langfuse
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

```bash
# 1. Spin up infra (postgres+pgvector, mlflow, langfuse)
docker compose up -d

# 2. Install
pip install -e ".[dev]"
cp .env.example .env        # set AZURE_OPENAI_* or LLM_PROVIDER=ollama

# 3. Module A — ingest sample docs and ask a question
python -m gridsense.docrag.ingest data/docs
uvicorn gridsense.api.main:app --reload
# POST /ask {"question": "..."}

# 4. Module B — train, register, serve
python -m gridsense.degrade.train
python -m gridsense.degrade.monitor      # drift report
# POST /predict {...}
```

---

## Roadmap / milestones

This is built milestone by milestone so progress is visible in the commit history.

- [x] **M0 — Scaffold.** Repo, `pyproject`, ruff+pytest+pre-commit, CI green, docker-compose up.
- [x] **M1 — DocRAG MVP.** Ingest → pgvector → LangChain chain → cited answer via FastAPI.
- [x] **M2 — DocRAG hardening.** Langfuse tracing, confidence/guardrails, eval harness.
- [x] **M3 — DegradeML training.** Feature build (SQL), training, MLflow tracking + registry, model card.
- [x] **M4 — Serve + monitor.** `/predict` from registry, Evidently drift report, prediction logging.
- [ ] **M5 — Deploy + polish.** Dockerised services, k8s manifests, README/diagrams, demo GIFs.

**Definition of done (per module):** runs from a clean clone via documented commands,
covered by tests, traced/tracked (Langfuse / MLflow), and green in CI.

---

## Tech stack

`Python 3.11` · `LangChain` · `pgvector` / `Postgres` · `Azure OpenAI` (+ `Ollama` fallback) ·
`Langfuse` · `MLflow` · `Evidently` · `FastAPI` · `Pydantic` · `Docker` / `docker-compose` ·
`Kubernetes` (optional) · `GitHub Actions` · `pytest` · `ruff` · `pre-commit`

## License

MIT — see `LICENSE`.

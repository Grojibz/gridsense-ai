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
| **Claude models** | **Claude Opus 5** as the default chat provider — `langchain-anthropic` for the RAG chain, the **native Anthropic SDK** for the agentic surface |
| **Agent development** | Tool-use loop spanning both modules (`/agent`), with a **Claude Haiku 4.5** research subagent the coordinator delegates reading to |
| **MCP** | An **MCP server** exposing GridSense as five tools to any MCP client, sharing one tool definition with the agent loop |
| **Agent Skills** | Three versioned skills in `.claude/skills/` — eval-gate triage, refusal triage, BESS domain |
| **Agent evaluation** | Trajectory metrics (`trajectory_accuracy`, `tool_precision`) gated in CI beside the RAG metrics |
| **GenAI orchestration** | RAG pipeline built with **LangChain** (retrievers, chains, structured output) |
| **Vector search** | **pgvector** on Postgres; pluggable to Pinecone |
| **Cloud AI** | **Claude** for chat, **Voyage** for embeddings (Anthropic's own recommendation, since they ship none), **Azure OpenAI** and local **Ollama** as alternatives |
| **LLM observability** | Tracing, latency and cost dashboards via **Langfuse** |
| **GenAI optimisation** | Prompt templates, retrieval tuning, and an **eval harness** (LLM-as-judge + groundedness) |
| **RAG evaluation** | Curated **golden dataset** + **RAGAS** metrics, enforced as a **merge gate** in CI |
| **Classical ML** | Battery-degradation regressor (scikit-learn) |
| **MLOps** | Experiment tracking & model registry (**MLflow**), reproducible pipelines |
| **Monitoring** | Data/concept **drift detection** (Evidently) with alerting |
| **Software engineering** | Type hints, design patterns, **pytest**, **ruff**, pre-commit, Clean Code |
| **Delivery** | **Docker** + docker-compose, **GitHub Actions CI/CD**, optional **Kubernetes** manifests |
| **APIs** | **FastAPI** services with Pydantic schemas and OpenAPI docs |
| **SQL** | Postgres for documents, embeddings, run metadata, and prediction logs |

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │             FastAPI             │
                        │  /ask     /agent     /predict   │
                        └────┬─────────┬───────────┬──────┘
                             │         │           │
                             │    ┌────▼─────────────────────┐        MCP client
                             │    │   Module C — Agent       │            │
                             │    │                          │   stdio    │
                             │    │  Claude Opus 5           ├────────────┘
                             │    │  ├─ shared tool layer    │
                             │    │  ├─ Haiku subagent       │
                             │    │  └─ MCP server           │
                             │    └────┬──────────────┬──────┘
                             │         │              │
                   ┌─────────▼─────────▼──┐    ┌──────▼──────────────────┐
                   │  Module A — DocRAG   │    │  Module B — DegradeML   │
                   │                      │    │                         │
                   │  ingestion/chunking  │    │  feature build (SQL)    │
                   │  pgvector retriever  │    │  train + eval           │
                   │  Claude/Azure/Ollama │    │  MLflow registry        │
                   │  guardrails + cites  │    │  Evidently drift        │
                   └──────────┬───────────┘    └────────────┬────────────┘
                              │                             │
                   ┌──────────▼───────────┐    ┌────────────▼────────────┐
                   │ Postgres + pgvector  │    │ MLflow tracking + reg.  │
                   └──────────┬───────────┘    └─────────────────────────┘
                              │
                   ┌──────────▼───────────┐
                   │       Langfuse       │  tracing, cost, latency
                   └──────────────────────┘
```

Everything runs locally via `docker-compose` (Postgres+pgvector, MLflow, Langfuse).

Chat and embeddings are configured **separately** — `CHAT_PROVIDER` (`anthropic` | `azure` |
`ollama`) and `EMBEDDING_PROVIDER` (`voyage` | `azure` | `ollama`). That is not symmetry for
its own sake: Anthropic serves no embeddings API, so the default arrangement — Claude
generating, something else embedding — cannot be expressed with a single provider field. The
legacy `LLM_PROVIDER` still sets both at once, except for `anthropic`, which is rejected
rather than guessing an embedding provider on your behalf.

**Voyage is the hosted embedding default.** It is what Anthropic points at for embeddings,
and `voyage-4-lite` ships 200M free tokens — this corpus will not exhaust them. Ollama stays
the fully-offline path.

> A Voyage account with no payment method is limited to **3 requests per minute**, which an
> eval run exceeds within seconds. The free tokens apply regardless; the card raises the
> rate limit, it does not unlock the free tier.

> ⚠️ **The relevance scale is not comparable across embedding models.**
> `docrag_min_relevance` and `docrag_uncertain_relevance` are percentile cuts through one
> model's similarity distribution. Carrying them across a provider switch silently redefines
> what counts as relevant enough to answer from. Run `make calibrate` (retrieval only, no
> chat tokens) after any change to `EMBEDDING_PROVIDER`, then confirm with the full gate.

---

## Repository structure

```
gridsense-ai/
├── README.md
├── pyproject.toml              # deps, ruff, pytest config
├── Dockerfile                  # API image (serves /ask + /agent + /predict), non-root
├── docker-compose.yml          # postgres+pgvector, mlflow, langfuse, api
├── Makefile                    # common dev commands (make help)
├── alembic.ini                 # migrations config (URL comes from DATABASE_URL)
├── migrations/                 # Alembic chain; ships inside the API image
├── .github/workflows/ci.yml    # lint + mypy + test + build-and-start on every PR
├── .github/workflows/eval.yml  # golden-dataset checks + RAGAS merge gate on every PR
├── .pre-commit-config.yaml
├── .mcp.json                   # MCP server registration for Claude Code / Desktop
├── .claude/skills/             # agent skills, versioned with the repo
├── eval/
│   ├── calibrate_relevance.py  # measure an embedding model's relevance distribution
│   ├── golden_dataset.json     # curated Q&A + retrieval ground truth
│   ├── run_ragas.py            # run RAGAS over the golden set -> results.json
│   ├── check_thresholds.py     # exit non-zero if a score is under its gate
│   ├── agent_trajectories.json # labelled tool-use expectations
│   ├── run_agent_eval.py       # score the agent's trajectories
│   └── check_agent_thresholds.py  # the trajectory merge gate
├── src/
│   └── gridsense/
│       ├── config.py           # pydantic-settings, env-driven
│       ├── db.py               # the tables this app owns — one definition, two consumers
│       ├── logconfig.py        # JSON logging + the request-id ContextVar
│       ├── providers.py        # chat/embedding factories, per-provider timeouts
│       ├── api/                # FastAPI app + routers
│       │   ├── security.py     # API-key auth; refuses to start unauthenticated
│       │   ├── ratelimit.py    # per-caller token buckets, tighter on /agent
│       │   ├── middleware.py   # request id, access log, exception containment
│       │   └── routes_health.py  # /ready — readiness, distinct from /health
│       ├── docrag/             # Module A
│       │   ├── ingest.py       # loaders, chunking, embedding
│       │   ├── retriever.py    # pgvector retriever
│       │   ├── chain.py        # LangChain RAG chain, cited output
│       │   ├── guardrails/     # upstream input checks + downstream output validation
│       │   └── eval/           # eval harness: LLM-as-judge + RAGAS + gate thresholds
│       ├── agent/              # Module C
│       │   ├── tools.py        # the shared tool layer (one definition, two surfaces)
│       │   ├── loop.py         # Claude tool-use loop (native Anthropic SDK)
│       │   ├── subagents.py    # Haiku research worker
│       │   └── eval.py         # trajectory metrics + thresholds
│       ├── mcp/                # MCP server — thin wrappers over agent/tools.py
│       └── degrade/            # Module B
│           ├── data.py         # feature engineering (SQL-backed)
│           ├── train.py        # training + MLflow logging
│           ├── evaluate.py     # metrics, model card
│           ├── monitor.py      # Evidently drift report
│           └── serve.py        # load registered model, predict
├── k8s/                        # optional: deployment + service + migration-job manifests
├── data/                       # sample docs + synthetic battery dataset
└── tests/                      # pytest, both modules
```

---

## Module A — DocRAG (generative AI)

A retrieval-augmented assistant over technical documents that **always cites its
sources** and refuses to answer when retrieval confidence is low.

- Ingestion: PDF/markdown loaders → semantic chunking → embeddings (Voyage, Azure or Ollama) → pgvector,
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
`ragas-gate` job spins up pgvector, replays the golden set with Claude judging and Voyage
embedding, uploads `results.json` as an artifact, and fails the build on a threshold breach. Mark it as a
required status check in branch protection to actually block the merge; it skips itself
on fork PRs, where the secrets aren't available.

> Thresholds live in `src/gridsense/docrag/eval/thresholds.py`, not in the CI YAML, so
> loosening the gate shows up in code review like any other change.

**The four RAGAS metrics need a hosted model to gate on.** A local judge does score them,
but only on about half the items — RAGAS chains several internal structured-output prompts
per metric — and a mean drawn from the half that happened to succeed is not a mean worth
blocking a merge on. The three deterministic metrics (`retrieval_recall`,
`refusal_accuracy`, `output_validity`) need no judge and are trustworthy offline.

**[`EVALUATION.md`](EVALUATION.md) has the measured numbers and the three problems they
exposed** — an inflated `refusal_accuracy`, a chunk that had silently stopped being
retrievable, and a run of apparent "model weakness" that turned out to be a GPU computing
incorrectly. All three are fixed; the last one is worth reading for how convincingly a
hardware fault disguises itself as a software one.

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

## Module C — Agent (Claude tool use, MCP, skills)

Modules A and B each answer half of a real question. *"This pack runs at 33 °C and 1500
cycles over 600 days — will it still be serviceable in five years, and what does the
standard say?"* needs a prediction **and** a documented end-of-life threshold, and a
comparison between them. Deciding to fetch both is what the agent adds.

### One tool layer, two surfaces

`agent/tools.py` holds five plain functions — `search_docs`, `ask_docs`, `predict_soh`,
`drift_report`, `eval_gate` — each a thin adapter over an existing entry point. Two
surfaces expose them, and neither owns the logic:

| Surface | How | Where |
|---|---|---|
| **MCP server** | `MCPServer` over stdio | `gridsense/mcp/server.py`, `make mcp` |
| **Agent loop** | `client.beta.messages.tool_runner` | `gridsense/agent/loop.py`, `POST /agent` |

Because `ask_docs` goes through `answer_question()`, the guardrails apply identically on
both: an MCP client gets the same injection scan, PII scrub, relevance gate and output
validation as `/ask`. There is no second, weaker path into the model — which is the point
of the indirection, and is asserted in `tests/test_mcp_server.py`.

Both surfaces *wrap* rather than register the shared functions, because those carry
keyword-only injection parameters used by tests. A tool schema is a contract with the
model, and test seams do not belong in it.

### The subagent

`delegate_research` runs its own tool loop on **Claude Haiku 4.5** with `search_docs` as its
only tool, and returns a written finding. The coordinator reads one paragraph instead of
twenty passages, and the bulk of the reading is billed at roughly a fifth of Opus rates.

> The Messages API has no subagent primitive — this is a tool whose implementation happens
> to be another tool loop. **Managed Agents** does have one
> (`multiagent: {"type": "coordinator", ...}`), with real per-thread isolation and its own
> event stream, but it requires an Anthropic-hosted sandbox that this repo deliberately does
> not need. The difference is worth naming rather than blurring.

### Claude specifics that are easy to get wrong

- **`temperature` is removed on Claude Opus 5** — sending it is a 400, not a warning. It is
  dropped for the Anthropic backend and still passed for Azure/Ollama. Determinism is
  steered with `effort` and the prompt instead. `tests/test_providers.py` asserts it against
  the *built client*, not the kwargs dict: `langchain-anthropic` silently relocates
  parameters it recognises out of `model_kwargs`, so a parameter passed the wrong way is
  accepted, warned about once, and then never sent.
- **Thinking is on by default and counts against `max_tokens`.** A ceiling sized around the
  answer alone truncates the response mid-sentence, so `/agent` treats a `max_tokens` stop as
  a refusal rather than returning a fragment that reads like a finished reply.
- **A safety refusal is an HTTP 200** with empty or partial `content`. The loop checks
  `stop_reason` *before* touching `content`, and surfaces `refused` + `refusal_reason`
  structurally — the same contract `RagAnswer` uses. `fallbacks: "default"` is enabled so a
  decline is re-run on a fallback model rather than simply stopping; set
  `ANTHROPIC_REFUSAL_FALLBACK=false` for an account without the beta.
- **Prompt caching has a floor.** The minimum cacheable prefix on Opus 5 is 512 tokens. The
  agent's system prompt plus tool schemas measures ~1150 tokens (tiktoken estimate), so it
  should cache; the DocRAG system prompt alone (~120 tokens) would not. Confirm with
  `usage.cache_read_input_tokens` on a second call rather than assuming — the response body
  reports it.

### Agent Skills

`.claude/skills/` ships three skills, versioned with the repo and discovered when it is
mounted in a session: `bess-eval-gate` (why the gate failed and what is not a fix),
`docrag-triage` (which guardrail fired and what it implies), `bess-domain` (vocabulary, and
the train/serve skew to watch).

> Skills loaded from a repository are **executed as instructions**. Anyone who can commit to
> a mounted repository can edit them, and the platform loads them at session start with no
> review step. That is a trust boundary, and it is worth stating before mounting a repo with
> external contributors.

### Trajectory evaluation

Scoring the answer is the wrong unit for an agent: the same final sentence can come from
sound tool use or from a guess that consulted nothing. `eval/agent_trajectories.json` labels
12 questions with the tools each one should — and must not — provoke, across documentation,
prediction, cross-module, monitoring, out-of-scope and adversarial categories.

| Metric | Meaning | Gate |
|---|---|---|
| `trajectory_accuracy` | Share of required tools actually called | ≥ 0.80 |
| `tool_precision` | Share of items that called no forbidden tool | ≥ 0.85 |

Both deterministic — no judge, no network. The anti-cheat rules carry over from the RAGAS
gate: an item that *needed* tools and was refused scores 0.0 rather than being excused, and
a metric missing from the report counts as 0.0. Thresholds live in
`src/gridsense/agent/eval.py`, kept out of the RAG `THRESHOLDS` dict on purpose — a metric
absent from a report counts as zero there, which would fail the RAG gate on metrics its run
was never meant to produce.

```bash
make install-agent     # pip install -e ".[docrag,degrade,agent]"
make mcp               # serve the MCP tools over stdio
make eval-agent        # score trajectories -> eval/agent_results.json
make eval-agent-gate   # exit non-zero on a threshold breach
```

`.github/workflows/eval.yml` runs `agent-gate` on every PR beside `ragas-gate`, scoring and
gating in separate steps so the report is still uploaded when the gate fails.

---

## Getting started

The default setup runs **fully offline** using a local [Ollama](https://ollama.com) for both
chat and embeddings.

Because chat and embeddings are separate settings, they mix — and the mix is usually what you
want:

| Goal | Configuration |
|---|---|
| Everything offline, no key | the defaults in `.env.example` (Ollama for both) |
| **Local development against Claude** | `CHAT_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`, embeddings left on Ollama |
| CI, or no local Ollama | `CHAT_PROVIDER=anthropic` + `EMBEDDING_PROVIDER=voyage` |

The middle row is the one to reach for while developing. Keeping embeddings on Ollama means
the relevance thresholds stay the ones they were calibrated against, costs nothing, and is
not rate-limited — the only thing you gain by moving embeddings to a hosted provider is not
needing Ollama at all, which is a CI problem rather than a local one.

Azure OpenAI works for either half if you already have it.

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

# 6. Module C — the agent, which needs both modules to answer at all.
#    Requires ANTHROPIC_API_KEY and CHAT_PROVIDER=anthropic. This is the one endpoint with
#    no offline path: the tool runner is Anthropic-specific, and the code says so with an
#    explicit error rather than degrading quietly. /ask and the MCP server both run offline.
curl localhost:8000/agent -H 'content-type: application/json'   -d '{"question": "This pack runs at 33C, 1500 cycles over 600 days, DoD 0.7 at 1.0C. Will it still be serviceable in five years, and what does the documentation say the limit is?"}'
```

Step 3 ingests through the API container, which mounts `./data` read-only — the corpus is
not baked into the image. Ingestion verifies itself: if any chunk fails to retrieve its own
text it raises rather than leave a quietly unusable index.

> **If answers come back as fluent nonsense, set `OLLAMA_NUM_GPU=0` and re-ingest.**
> A GPU that computes incorrectly does not raise — Ollama returns confident garbage and
> writes corrupted embeddings, and every layer above reports a plausible domain-level
> failure instead. Verify with `ollama run llama3.2 "Say OK"`. See
> [`EVALUATION.md`](EVALUATION.md) for how long that took to find.

**Local dev** (without the API container): `make install` (or
`pip install -e ".[docrag,degrade,agent,dev]"`), then run the modules with your own
interpreter and `uvicorn gridsense.api.main:app --reload`. `make help` lists the common
tasks (`lint`, `test`, `ingest`, `train`, `monitor`, `eval-ragas`, `mcp`, `eval-agent`).

**MCP.** `.mcp.json` at the repo root registers the server, so a Claude Code or Claude
Desktop session in this directory picks up the five tools with no extra setup. `make mcp`
runs it directly over stdio.

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

The Deployment runs unprivileged (`runAsNonRoot`, `readOnlyRootFilesystem`, all capabilities
dropped) against the image's own `USER 10001`. Set `API_KEYS` in the Secret before applying:
the app refuses to start without it outside `local`/`test`.

> One thing the manifests deliberately do *not* fix: `image: gridsense-api:latest` with
> `imagePullPolicy: IfNotPresent`. Two nodes can serve different builds under that name and
> there is nothing to roll back to. Pin a digest for a real deployment.

## Operating the service

Everything in this section exists because the endpoints spend money on someone else's
behalf. `/ask` is a model call per request and `/agent` is a bounded loop of them plus a
sub-agent, so an open deployment is not a missing nicety — it is an unmetered bill payable
by whoever owns the provider key.

### Authentication fails closed, at startup

`API_KEYS` is comma-separated so a key can be rotated without a flag-day cutover. Outside
`ENVIRONMENT=local` or `test`, an app with no keys **refuses to start**:

```
RuntimeError: ENVIRONMENT='k8s' but no API_KEYS is set. The API would start with /ask and
/agent open to anyone, and both spend provider credits per request.
```

Failing at startup rather than at the first request puts the mistake in front of whoever is
deploying, instead of in a bill weeks later. A configuration flag defaulting to "no auth"
would have reproduced exactly the gap it was meant to close, because the default is what
ships — so the opt-out (`API_AUTH_DISABLED=true`, for a gateway that authenticates in front
of this service) is explicit and logged loudly at every startup.

`/health` and `/ready` stay open: a probe cannot present a key, and a readiness check that
401s takes the pod out of service for a reason unrelated to whether it can serve.

### Three limits, and which one is real

| Limit | Setting | Enforced by |
|---|---|---|
| Requests per caller | `RATE_LIMIT_PER_MINUTE` / `AGENT_RATE_LIMIT_PER_MINUTE` | This app, per process |
| Tokens per `/agent` request | `AGENT_MAX_TOTAL_TOKENS` | This app — the loop stops |
| Tokens the model aims for | `ANTHROPIC_TASK_BUDGET_TOKENS` | **Nothing. It is advisory** |

The third row is the one to read twice. `output_config.task_budget` is a suggestion the
model can see and paces itself against, not a cap — so it is left unset by default, and the
ceiling that actually stops the spend is the local one. A `/agent` request that crosses it
returns `refused: true, refusal_reason: "budget_exceeded"` rather than the partial work: a
trajectory cut short has usually gathered facts without reaching the comparison they were
for, and returning that as an answer is how a half-finished analysis reads as a conclusion.

The two rate limits differ because the endpoints do not cost the same. A single number
applied to `/ask` and `/agent` alike would have to be loose enough for the first, which
makes it useless for the second.

> **⚠️ Rate limiting is per process.** Two uvicorn workers, or the two replicas in
> `k8s/deployment.yaml`, each keep their own buckets — the effective limit is N times the
> configured one. This is a guardrail against a runaway client and an accidental loop, not a
> defence against a distributed attacker. `X-Forwarded-For` is deliberately not trusted for
> caller identity either, since any client can set it. A limit that must hold across
> replicas needs shared state or an ingress that enforces it, and both belong in front of
> the app rather than inside it.

### Liveness and readiness are different questions

`/health` reports that the process is serving and touches nothing external. `/ready` checks
Postgres. They must not share an endpoint in either direction:

- readiness on `/health` means a pod whose database is unreachable reports itself ready and
  gets traffic routed to it — every request failing, the deployment reporting healthy;
- liveness on `/ready` means a database blip **kills the container**, turning a dependency
  outage into a restart loop that cannot fix it.

The provider is deliberately not probed. Probing it would bill on a timer and take every pod
out of service during one provider blip; a provider outage is surfaced per request, as
`provider_error`.

### Logs

One JSON object per line on stdout, with a `request_id` that a `ContextVar` carries into log
calls deep in the RAG chain — which know nothing about HTTP — without threading it through
every signature. The id is read from `X-Request-ID` when a proxy already assigned one, and
echoed on every response.

```json
{"ts": "...", "level": "WARNING", "logger": "gridsense.agent", "request_id": "8746432ade6d",
 "message": "agent stopped: token budget exhausted", "billable_tokens": 201430,
 "tools_called": ["search_docs", "predict_soh"]}
```

The formatter redacts on the field *name* — anything containing `api_key`, `secret`,
`token`, `password` or `authorization` becomes `***` — so the guarantee holds for call sites
nobody has written yet. `*_tokens` is excluded from that rule, because usage counts are among
the most useful numbers in the log and the word would otherwise swallow them.

Unhandled exceptions are the visible change: the traceback goes to the log and the caller
gets a request id and nothing else. The FastAPI default returns the traceback, which leaks
file paths, local variables, and on a connection failure a URL with credentials in it.

### Timeouts

`REQUEST_TIMEOUT_SECONDS` reaches every backend — by three different parameter names, since
no two of these clients agree on one. Without it a hung provider holds a worker until the
process restarts, and enough of them take the service down without a single error being
raised. The default is generous (120s) because a local Ollama on CPU is genuinely slow;
tighten it for a hosted provider.

### Schema migrations

```bash
make migrate                      # alembic upgrade head
make migrate-new m="add a column" # autogenerate from src/gridsense/db.py
make migrate-status               # current revision vs head
```

`src/gridsense/db.py` is the single definition of the tables this app owns, with two
consumers: Alembic migrates it, and `create_all` builds it for local development and the
integration tests. `tests/test_migrations.py` asserts the two agree — a column added to the
metadata without a migration fails there rather than at the next INSERT.

Before this, `degrade_predictions` was created by a `CREATE TABLE IF NOT EXISTS` run on
every prediction. That is fine exactly until the first column change: `IF NOT EXISTS` does
nothing to a table that already exists with the old shape, so the DDL and the database
diverge with no error at all.

> **⚠️ MLflow shares this database.** Its backend store is the same Postgres, and it
> migrates its own schema with Alembic under the default `alembic_version`. GridSense
> therefore uses `gridsense_alembic_version`, and restricts autogenerate to the tables it
> declares. Without the first, the initial `alembic upgrade` reads MLflow's revision as its
> own; without the second, `--autogenerate` proposes a migration that drops the experiment
> tracking store. Both are guarded and tested.

In Kubernetes, migrations run as a Job (`k8s/migrate-job.yaml`) rather than an
initContainer: with two replicas an initContainer races two migrations against each other on
every rollout. Apply it and wait for completion *before* rolling out the Deployment. The
chain ships inside the image, so migrations and the code that needs them come from one
artefact rather than from a developer's checkout.

### Type checking and the coverage floor

```bash
make typecheck   # mypy
make test-cov    # pytest with the coverage floor enforced
```

`mypy` runs over `src/` in CI, as a step separate from ruff so a type error and a style error
are distinguishable in the checks list. It is deliberately not `strict`: the value here is
catching contradictions inside code that *is* annotated — nearly all of it — not forcing
annotations onto the ML scripts, where `Any` is often the honest type for a model object.

`providers.py` has a narrow, documented exemption for two error codes. It is 140 lines of
third-party constructor calls, and every error it raises is an artefact of how those
libraries are typed (pydantic aliases, `SecretStr` coercion, `**kwargs` into heterogeneous
models) rather than a defect. What covers that module instead is `tests/test_providers.py`,
which asserts against the **built client object** rather than the kwargs dict — because a
parameter passed the wrong way is accepted, warned about once, and then never sent.

The coverage floor (`fail_under = 80` in `pyproject.toml`) is set from the measured total,
not from an aspiration: a gate above what the suite reaches fails on the first PR and gets
deleted, and one far below never fires. It is there to catch a *drop*, so it moves up when
real coverage does and never down to accommodate a regression. It lives in `pyproject.toml`
rather than in a CI flag for the same reason the eval thresholds live in Python: lowering it
should show up in a code review as a change someone has to argue for.

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
- [x] **M8 — Claude + agentic surface.** Anthropic as the default chat provider (with chat
      and embedding providers split, because Anthropic serves no embeddings API), an MCP
      server, a Claude tool-use loop with a Haiku research subagent, agent skills, and a
      trajectory eval gated in CI.

- [x] **M9 — Service hardening.** API-key auth that fails closed at startup, per-caller
      rate limits with a tighter budget on `/agent`, a hard per-request token ceiling,
      structured JSON logs with a request id, provider timeouts, `/ready` split from
      `/health`, an unprivileged container, and a CI step that starts the image instead of
      only building it.

- [x] **M10 — Schema and type discipline.** Alembic migrations with the metadata as a single
      source of truth (and a version table kept clear of MLflow's, which shares the
      database), mypy over `src/` in CI, an enforced coverage floor, and a migration Job for
      Kubernetes.

**Definition of done (per module):** runs from a clean clone via documented commands,
covered by tests, traced/tracked (Langfuse / MLflow), and green in CI.

> **Notes on this build.** `CHAT_PROVIDER` defaults to `anthropic`; `docker-compose.yml`
> overrides it to `ollama` so the stack still comes up fully offline with no key. On that
> path a local **llama3.2 (3B)** answers, so DocRAG answer quality is indicative rather than
> production-grade. The **agent loop and MCP server have no offline equivalent** — the tool
> runner is Anthropic-specific and needs `ANTHROPIC_API_KEY`. **CI** runs the offline unit
> tests; the integration/eval tests are gated (`RUN_INTEGRATION=1`) as they need a live
> Postgres/Ollama/MLflow stack.

> **Not yet measured.** `EVALUATION.md` records a local llama3.2 run in which the gate
> fails. The Claude-judged run has not been recorded here — those numbers should be measured
> and written down, not assumed to have improved.

> **Still open after M10.** Rate limiting is per process, so it does not hold across
> replicas — a limit that must is an ingress concern, not an application one. The `/agent`
> endpoint and the Claude path remain unverified against a live API, and the eval gates have
> never run against Claude, because the account funding them has no credits. Neither is a
> code defect, and neither should be reported as measured.

---

## Tech stack

`Python 3.11` · `Claude Opus 5` / `Haiku 4.5` · `Anthropic SDK` · `MCP` · `LangChain` ·
`pgvector` / `Postgres` · `Azure OpenAI` · `Ollama` ·
`Langfuse` · `RAGAS` · `MLflow` · `Evidently` · `FastAPI` · `Pydantic` · `Docker` / `docker-compose` ·
`Kubernetes` (optional) · `GitHub Actions` · `pytest` · `ruff` · `mypy` · `Alembic` · `pre-commit`

## License

MIT — see `LICENSE`.

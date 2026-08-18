# Common GridSense developer commands. Override the interpreter with e.g. `make PY=.venv/bin/python`.
PY ?= python

.PHONY: help install install-eval install-agent up down logs lint fmt test test-integration \
        ingest ask-serve train eval eval-ragas eval-gate eval-agent eval-agent-gate \n        monitor build mcp

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with all extras (editable)
	$(PY) -m pip install -e ".[docrag,degrade,agent,dev]"

install-eval:  ## Add the RAGAS eval stack (heavier; not needed for the unit tests)
	$(PY) -m pip install -e ".[docrag,eval]"

install-agent:  ## Add just the agentic surface (Anthropic SDK + MCP)
	$(PY) -m pip install -e ".[docrag,degrade,agent]"

up:  ## Start infra + api (postgres+pgvector, mlflow, langfuse, api)
	docker compose up -d --build

down:  ## Stop all services
	docker compose down

logs:  ## Tail the api container logs
	docker compose logs -f api

lint:  ## Ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:  ## Apply ruff formatting + autofixes
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:  ## Run offline unit tests
	$(PY) -m pytest

test-integration:  ## Run gated integration tests (needs the live stack)
	RUN_INTEGRATION=1 $(PY) -m pytest

build:  ## Build the api Docker image
	docker build -t gridsense-api:latest .

# --- DocRAG (Module A) ---
ingest:  ## Ingest sample docs into pgvector
	$(PY) -m gridsense.docrag.ingest data/docs

eval-ragas:  ## Run RAGAS over the golden dataset -> eval/results.json
	$(PY) eval/run_ragas.py --out eval/results.json

eval-gate:  ## Fail if any RAGAS score is under its merge threshold
	$(PY) eval/check_thresholds.py eval/results.json

# --- Agent (Module C) ---
eval-agent:  ## Score the agent's tool trajectories -> eval/agent_results.json
	$(PY) eval/run_agent_eval.py --no-gate --out eval/agent_results.json

eval-agent-gate:  ## Fail if a trajectory score is under its merge threshold
	$(PY) eval/check_agent_thresholds.py eval/agent_results.json

mcp:  ## Serve GridSense as an MCP server over stdio
	$(PY) -m gridsense.mcp.server

# --- DegradeML (Module B) ---
train:  ## Train + register the SOH model in MLflow
	$(PY) -m gridsense.degrade.train

monitor:  ## Run the data-drift report
	$(PY) -m gridsense.degrade.monitor

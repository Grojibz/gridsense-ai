# Common GridSense developer commands. Override the interpreter with e.g. `make PY=.venv/bin/python`.
PY ?= python

.PHONY: help install up down logs lint fmt test test-integration \
        ingest ask-serve train eval monitor build

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with all extras (editable)
	$(PY) -m pip install -e ".[docrag,degrade,dev]"

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

eval:  ## Run the DocRAG eval harness
	$(PY) -m gridsense.docrag.eval.run

# --- DegradeML (Module B) ---
train:  ## Train + register the SOH model in MLflow
	$(PY) -m gridsense.degrade.train

monitor:  ## Run the data-drift report
	$(PY) -m gridsense.degrade.monitor

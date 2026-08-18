# Application image for the GridSense FastAPI service (/ask, /agent, /predict).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy metadata + sources, then install the package with the extras the API actually
# imports. `agent` is not optional here despite the name: `api/main.py` imports
# `routes_agent` unconditionally, which imports `anthropic` at module scope, so leaving it
# out means the whole service fails to start — not just /agent. It worked without it only
# because `langchain-anthropic` happens to pull `anthropic` in transitively, at whatever
# version it pins. Depending on that is depending on someone else's dependency graph
# staying put.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[docrag,degrade,agent]"

# Run unprivileged. A root process inside a container shares the host's user namespace by
# default, so a remote-code-execution bug on the API starts from root rather than from
# nobody. `--system` keeps it out of the login-capable range.
RUN useradd --system --create-home --uid 10001 gridsense \
    && chown -R gridsense:gridsense /app
USER 10001

EXPOSE 8000

# Liveness only — "is the process serving?". Readiness (can it reach Postgres?) is a
# separate question at /ready, and deliberately not wired to a healthcheck that would
# restart the container for a dependency outage a restart cannot fix.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "gridsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

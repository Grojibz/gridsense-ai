# Application image for the GridSense FastAPI service (serves /ask and /predict).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy metadata + sources, then install the package with both module extras so the
# image can serve DocRAG (/ask) and DegradeML (/predict).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[docrag,degrade]"

EXPOSE 8000

# Liveness probe hits the same endpoint as docker-compose / k8s.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "gridsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

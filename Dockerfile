# Application image for the GridSense FastAPI service.
# Module-specific extras (docrag/degrade) are layered in as those milestones land.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy metadata + sources, then install the package (base deps only for now).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

EXPOSE 8000

# Liveness probe hits the same endpoint as docker-compose / k8s.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "gridsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

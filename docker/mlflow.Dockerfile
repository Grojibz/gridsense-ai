# MLflow server image with a Postgres driver.
# The upstream image ships without psycopg2, which the Postgres backend-store
# URI (postgresql+psycopg2://...) requires — so we layer it in here.
FROM ghcr.io/mlflow/mlflow:v2.12.1

RUN pip install --no-cache-dir psycopg2-binary>=2.9

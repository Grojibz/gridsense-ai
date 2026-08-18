"""The migration chain, and the check that stops it drifting from the code's schema.

Two definitions of a table are one more than a codebase can keep honest. `gridsense.db`
holds the metadata; Alembic migrates it and `create_all` builds it for local development.
Nothing in Python notices when those diverge — the failure surfaces as an INSERT against a
column that one half believes exists.

So the load-bearing test here is `test_the_migration_chain_matches_the_declared_schema`. The
rest verify the chain is runnable and reversible.

**What SQLite proves and what it does not.** These run against SQLite so they need no
datastore and can gate every PR. That catches the drift that actually happens — a column
added to the metadata and not to a migration — and does not catch dialect-specific
divergence (`TIMESTAMPTZ`, `SERIAL`, server defaults), because SQLAlchemy translates those
per dialect. The Postgres-specific behaviour is exercised by the gated integration tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from gridsense.db import PREDICTIONS_TABLE, metadata

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def alembic_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Alembic pointed at a throwaway SQLite file."""
    from alembic.config import Config

    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    # env.py reads the URL from settings, which is the property that keeps a connection
    # string out of the tracked ini file.
    monkeypatch.setenv("DATABASE_URL", url)

    from gridsense.config import get_settings

    get_settings.cache_clear()

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    yield config, url
    get_settings.cache_clear()


def _columns(url: str, table: str) -> dict[str, str]:
    engine = create_engine(url)
    try:
        return {c["name"]: str(c["type"]) for c in inspect(engine).get_columns(table)}
    finally:
        engine.dispose()


def test_the_chain_runs_from_empty(alembic_config):
    from alembic import command

    config, url = alembic_config
    command.upgrade(config, "head")

    engine = create_engine(url)
    try:
        assert PREDICTIONS_TABLE in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_the_migration_chain_matches_the_declared_schema(alembic_config):
    """The one that matters: a column added to gridsense.db and not to a migration."""
    from alembic import command

    config, url = alembic_config
    command.upgrade(config, "head")

    migrated = set(_columns(url, PREDICTIONS_TABLE))
    declared = {column.name for column in metadata.tables[PREDICTIONS_TABLE].columns}
    assert migrated == declared, (
        f"The migration chain and gridsense.db disagree. "
        f"Only in migrations: {migrated - declared}. Only in metadata: {declared - migrated}. "
        f"Add a migration with `make migrate-new m='...'`."
    )


def test_the_migration_is_reversible(alembic_config):
    """A downgrade nobody has ever run is not a rollback plan."""
    from alembic import command

    config, url = alembic_config
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(url)
    try:
        assert PREDICTIONS_TABLE not in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_upgrading_over_an_existing_table_is_safe(alembic_config):
    """The table predates Alembic in every environment that has served /predict."""
    from alembic import command

    config, url = alembic_config

    engine = create_engine(url)
    try:
        metadata.create_all(engine)  # simulate the pre-Alembic runtime DDL
    finally:
        engine.dispose()

    command.upgrade(config, "head")  # must not raise "table already exists"
    assert set(_columns(url, PREDICTIONS_TABLE)) == {
        column.name for column in metadata.tables[PREDICTIONS_TABLE].columns
    }


def test_there_is_exactly_one_head():
    """Two heads mean two people migrated in parallel and the chain no longer linearises."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    assert len(ScriptDirectory.from_config(config).get_heads()) == 1


def test_runtime_create_and_migrations_build_the_same_table(tmp_path: Path):
    """`ensure_predictions_table` is the second consumer of the metadata, not a second DDL."""
    from gridsense.degrade.serve import ensure_predictions_table

    url = f"sqlite:///{tmp_path / 'runtime.db'}"
    engine = create_engine(url)
    try:
        ensure_predictions_table(engine)
        assert set(_columns(url, PREDICTIONS_TABLE)) == {
            column.name for column in metadata.tables[PREDICTIONS_TABLE].columns
        }
    finally:
        engine.dispose()


# --- sharing a database with MLflow ------------------------------------------
#
# `DATABASE_URL` and MLflow's `--backend-store-uri` point at the same Postgres database in
# docker-compose. MLflow migrates its own schema with Alembic, under the default
# `alembic_version`. Both guards below exist because of that overlap, and both were found by
# looking at the live database rather than by reasoning about the code.


def test_the_version_table_is_not_the_default():
    """Sharing `alembic_version` means each project reads the other's revision as its own."""
    from gridsense.db import ALEMBIC_VERSION_TABLE

    assert ALEMBIC_VERSION_TABLE == "gridsense_alembic_version"
    assert ALEMBIC_VERSION_TABLE != "alembic_version"


def test_the_chain_records_itself_in_its_own_version_table(alembic_config):
    """The rename is only real if the migration actually writes there."""
    from alembic import command

    config, url = alembic_config
    command.upgrade(config, "head")

    from gridsense.db import ALEMBIC_VERSION_TABLE

    engine = create_engine(url)
    try:
        tables = inspect(engine).get_table_names()
    finally:
        engine.dispose()

    assert ALEMBIC_VERSION_TABLE in tables
    assert "alembic_version" not in tables


def test_autogenerate_ignores_tables_this_app_does_not_declare():
    """Otherwise `--autogenerate` proposes dropping the experiment tracking store."""
    from gridsense.db import is_owned_table

    assert is_owned_table(PREDICTIONS_TABLE) is True
    for foreign in ("experiments", "runs", "langchain_pg_embedding", "alembic_version"):
        assert is_owned_table(foreign) is False, foreign

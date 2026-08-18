"""Adopt degrade_predictions into the migration chain.

Revision ID: 0001_predictions
Revises:
Create Date: 2026-08-18

The table already exists in every environment that has ever served `/predict`: it was
created by a `CREATE TABLE IF NOT EXISTS` run on each prediction. So this migration has to
be safe on a database that already has it *and* on an empty one, which is why it inspects
before creating rather than assuming either.

That check is not defensive noise — it is what makes adopting an existing table into Alembic
possible at all. Stamping instead (`alembic stamp head`) would claim the schema matches this
revision without verifying it, which is the failure mode this whole change exists to remove.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_predictions"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "degrade_predictions"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE in inspector.get_table_names():
        # Already created by the pre-Alembic runtime DDL. Adopt it as-is: the column set is
        # asserted against gridsense.db.metadata by tests/test_migrations.py, so a database
        # that diverged from it fails there rather than silently here.
        return

    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("cycle_count", sa.Float(), nullable=True),
        sa.Column("avg_temperature_c", sa.Float(), nullable=True),
        sa.Column("avg_dod", sa.Float(), nullable=True),
        sa.Column("avg_c_rate", sa.Float(), nullable=True),
        sa.Column("calendar_age_days", sa.Float(), nullable=True),
        sa.Column("predicted_soh", sa.Float(), nullable=True),
        sa.Column("model_version", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_degrade_predictions"),
    )


def downgrade() -> None:
    op.drop_table(TABLE)

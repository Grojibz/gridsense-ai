"""Synthetic battery dataset + SQL-backed feature engineering for DegradeML.

The repo ships no proprietary data, so we generate a synthetic-but-physically-plausible
dataset of battery operating conditions and resulting State of Health (SOH), persist it to
Postgres, and build model features with a SQL query. The degradation model mirrors the
physics described in ``data/docs/battery_soh.md``: cycle ageing scaled by depth-of-discharge,
C-rate and an Arrhenius-like temperature stress, plus calendar ageing growing with the square
root of age.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gridsense.config import Settings, get_settings

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

SAMPLES_TABLE = "degrade_samples"
TARGET = "soh"

#: Raw measurement columns produced by the generator and stored in Postgres.
RAW_COLUMNS = [
    "cell_id",
    "cycle_count",
    "avg_temperature_c",
    "avg_dod",
    "avg_c_rate",
    "calendar_age_days",
]

#: Feature columns the model trains on (raw + SQL-derived). Order matters for inference.
FEATURE_COLUMNS = [
    "cycle_count",
    "avg_temperature_c",
    "avg_dod",
    "avg_c_rate",
    "calendar_age_days",
    "equivalent_full_cycles",
    "temperature_stress",
    "calendar_stress",
]

# SQL feature build: derive stress features in the database, then read them out.
FEATURE_SQL = f"""
SELECT
    cycle_count,
    avg_temperature_c,
    avg_dod,
    avg_c_rate,
    calendar_age_days,
    cycle_count * avg_dod                              AS equivalent_full_cycles,
    POWER(2.0, (avg_temperature_c - 25.0) / 10.0)      AS temperature_stress,
    SQRT(calendar_age_days)                            AS calendar_stress,
    {TARGET}
FROM {SAMPLES_TABLE}
"""


def generate_synthetic(n: int = 4000, *, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` synthetic battery samples with a physically-plausible SOH target."""
    rng = np.random.default_rng(seed)

    cycle_count = rng.integers(0, 3000, size=n)
    avg_temperature_c = rng.uniform(15.0, 45.0, size=n)
    avg_dod = rng.uniform(0.2, 1.0, size=n)
    avg_c_rate = rng.uniform(0.2, 2.0, size=n)
    calendar_age_days = rng.uniform(0.0, 1500.0, size=n)

    temperature_stress = 2.0 ** ((avg_temperature_c - 25.0) / 10.0)
    equivalent_full_cycles = cycle_count * avg_dod

    cycle_fade = 0.0015 * equivalent_full_cycles * np.sqrt(avg_c_rate) * temperature_stress
    calendar_fade = 0.05 * np.sqrt(calendar_age_days) * temperature_stress
    noise = rng.normal(0.0, 0.5, size=n)

    soh = 100.0 - cycle_fade - calendar_fade + noise
    soh = np.clip(soh, 70.0, 100.0)

    return pd.DataFrame(
        {
            "cell_id": np.arange(n),
            "cycle_count": cycle_count,
            "avg_temperature_c": avg_temperature_c.round(2),
            "avg_dod": avg_dod.round(4),
            "avg_c_rate": avg_c_rate.round(4),
            "calendar_age_days": calendar_age_days.round(1),
            TARGET: soh.round(3),
        }
    )


def get_engine(settings: Settings | None = None) -> Engine:
    """Create a SQLAlchemy engine for the app's Postgres database."""
    from sqlalchemy import create_engine

    settings = settings or get_settings()
    return create_engine(settings.database_url)


def seed_database(
    n: int = 4000,
    *,
    seed: int = 42,
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> int:
    """Generate synthetic samples and (re)write them to the samples table. Returns row count."""
    engine = engine or get_engine(settings)
    df = generate_synthetic(n, seed=seed)
    df.to_sql(SAMPLES_TABLE, engine, if_exists="replace", index=False)
    return len(df)


def count_samples(engine: Engine) -> int:
    """Return the number of rows in the samples table (0 if it doesn't exist)."""
    from sqlalchemy import text

    with engine.connect() as conn:
        try:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {SAMPLES_TABLE}"))
        except Exception:
            return 0
        return int(result.scalar() or 0)


def build_features(
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run the SQL feature query and return ``(X, y)`` for modelling."""
    from sqlalchemy import text

    engine = engine or get_engine(settings)
    df = pd.read_sql(text(FEATURE_SQL), engine)
    return df[FEATURE_COLUMNS], df[TARGET]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Seed the synthetic battery dataset in Postgres.")
    parser.add_argument("--n", type=int, default=4000, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args(argv)

    n = seed_database(args.n, seed=args.seed)
    print(f"Seeded {n} samples into '{SAMPLES_TABLE}'.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

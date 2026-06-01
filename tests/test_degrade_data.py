"""Offline unit tests for synthetic data generation (no DB)."""

from __future__ import annotations

from gridsense.degrade.data import TARGET, generate_synthetic


def test_generate_synthetic_shape_and_ranges() -> None:
    df = generate_synthetic(500, seed=1)
    assert len(df) == 500
    for col in ["cell_id", "cycle_count", "avg_temperature_c", "avg_dod", "avg_c_rate", TARGET]:
        assert col in df.columns
    assert df[TARGET].between(70.0, 100.0).all()
    assert df["avg_dod"].between(0.2, 1.0).all()
    assert df["avg_c_rate"].between(0.2, 2.0).all()


def test_generation_is_deterministic_for_a_seed() -> None:
    a = generate_synthetic(200, seed=7)
    b = generate_synthetic(200, seed=7)
    assert a.equals(b)


def test_heavier_cycling_degrades_soh_on_average() -> None:
    df = generate_synthetic(4000, seed=2)
    low_use = df[df["cycle_count"] < 500][TARGET].mean()
    heavy_use = df[df["cycle_count"] > 2500][TARGET].mean()
    assert heavy_use < low_use


def test_higher_temperature_degrades_soh_on_average() -> None:
    df = generate_synthetic(4000, seed=3)
    cool = df[df["avg_temperature_c"] < 20][TARGET].mean()
    hot = df[df["avg_temperature_c"] > 40][TARGET].mean()
    assert hot < cool

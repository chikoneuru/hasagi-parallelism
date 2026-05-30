"""Tests for the ElectricityMaps CSV loader + published-stats parametric trace.

Pins the loader to the actual Data Portal column conventions and verifies the
parametric trace matches its documented annual statistics within tight bounds.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest

from hasagi.energy.carbon_trace import (
    _GRID_ZONES,
    CarbonTrace,
    load_csv_trace,
    load_electricitymaps_csv,
    published_grid_trace,
)

FIXTURE = Path(__file__).parent / "fixtures" / "electricitymaps_de_7day_sample.csv"

# --- load_electricitymaps_csv ---


def test_loader_reads_the_de_fixture() -> None:
    trace = load_electricitymaps_csv(FIXTURE)
    assert isinstance(trace, CarbonTrace)
    # 7 days hourly + endpoint = 169 samples.
    assert len(trace.timestamps) == 169
    assert len(trace.intensities) == 169


def test_loader_intensities_match_de_published_stats() -> None:
    trace = load_electricitymaps_csv(FIXTURE)
    mean = statistics.mean(trace.intensities)
    # DE annual mean ≈ 360 gCO2/kWh per Data Portal; ±60 g tolerance for a 7-day slice.
    assert abs(mean - _GRID_ZONES["DE"]["mean_g"]) < 60
    assert min(trace.intensities) >= 5.0
    assert max(trace.intensities) < 800


def test_loader_handles_iso_with_z(tmp_path: Path) -> None:
    csv_path = tmp_path / "iso_z.csv"
    csv_path.write_text(
        "Datetime (UTC),Carbon Intensity gCO₂eq/kWh (LCA)\n"
        "2024-07-01T00:00:00Z,300\n"
        "2024-07-01T01:00:00Z,320\n"
    )
    trace = load_electricitymaps_csv(csv_path)
    assert len(trace.intensities) == 2
    assert trace.intensities == [300.0, 320.0]


def test_loader_handles_ascii_column_alternative(tmp_path: Path) -> None:
    """Plain-ASCII export variants must also load (no ₂ subscript glyph)."""
    csv_path = tmp_path / "ascii.csv"
    csv_path.write_text(
        "Datetime (UTC),Carbon Intensity gCO2eq/kWh (LCA)\n"
        "2024-07-01 00:00:00,300\n"
        "2024-07-01 01:00:00,320\n"
    )
    trace = load_electricitymaps_csv(csv_path)
    assert trace.intensities == [300.0, 320.0]


def test_loader_handles_aggregated_public_dump(tmp_path: Path) -> None:
    """The public aggregated dumps use ``carbon_intensity_avg`` instead of the LCA column."""
    csv_path = tmp_path / "agg.csv"
    csv_path.write_text(
        "datetime_utc,carbon_intensity_avg\n"
        "2024-07-01T00:00:00,150\n"
        "2024-07-01T01:00:00,200\n"
    )
    trace = load_electricitymaps_csv(csv_path)
    assert trace.intensities == [150.0, 200.0]


def test_loader_skips_blank_rows(tmp_path: Path) -> None:
    """ElectricityMaps occasionally ships blank rows for missing data; loader must skip."""
    csv_path = tmp_path / "blanks.csv"
    csv_path.write_text(
        "Datetime (UTC),Carbon Intensity gCO₂eq/kWh (LCA)\n"
        "2024-07-01T00:00:00Z,300\n"
        "2024-07-01T01:00:00Z,\n"
        "2024-07-01T02:00:00Z,320\n"
    )
    trace = load_electricitymaps_csv(csv_path)
    assert trace.intensities == [300.0, 320.0]


def test_loader_rejects_unknown_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "wrong.csv"
    csv_path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="unrecognised CSV schema"):
        load_electricitymaps_csv(csv_path)


def test_loader_rejects_empty(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    with pytest.raises(ValueError):
        load_electricitymaps_csv(csv_path)


def test_loader_is_drop_in_for_existing_plain_csv(tmp_path: Path) -> None:
    """The ``intensity_g_per_kwh`` plain schema (existing ``load_csv_trace``) is one of
    the recognised fallbacks — both loaders should yield the same trace."""
    csv_path = tmp_path / "plain.csv"
    csv_path.write_text(
        "timestamp,intensity_g_per_kwh\n"
        "2024-07-01T00:00:00,300\n"
        "2024-07-01T01:00:00,320\n"
    )
    via_em = load_electricitymaps_csv(csv_path)
    via_plain = load_csv_trace(csv_path)
    assert via_em.intensities == via_plain.intensities
    assert via_em.timestamps == via_plain.timestamps


# --- published_grid_trace ---


def test_zones_table_is_non_empty_and_documented() -> None:
    assert set(_GRID_ZONES) == {
        "DE", "US-CA", "FR", "PL", "VN", "JP",
        "GB", "SG", "KR", "BR",
        "NO", "ZA", "AU", "IN",
        "CN", "AE",
    }
    for params in _GRID_ZONES.values():
        assert params["mean_g"] > 0
        assert params["daily_swing"] >= 0
        assert params["weekly_swing"] >= 0
        assert params["noise_sd"] >= 0


def test_published_de_trace_matches_published_mean() -> None:
    trace = published_grid_trace("DE", days=30, sample_minutes=60, seed=0)
    mean = statistics.mean(trace.intensities)
    # 30-day mean should be within ±30 g of the published annual mean.
    assert abs(mean - _GRID_ZONES["DE"]["mean_g"]) < 30


def test_published_traces_span_published_order_of_magnitude() -> None:
    """Anchor-zone ordering matches the published Data Portal annual dashboards.

    Mid-zone means (SG/VN/KR/JP) cluster within ~60 g of each other and may
    swap under noise, so we only assert ordering on the anchor zones whose
    published means are separated by clear gaps.
    """
    means = {
        z: statistics.mean(published_grid_trace(z, days=14, sample_minutes=60, seed=1).intensities)
        for z in _GRID_ZONES
    }
    assert means["NO"] < means["FR"] < means["BR"] < means["GB"] < means["US-CA"] < means["DE"]
    assert means["DE"] < means["PL"] < means["ZA"]
    # ZA/NO ratio ≥ 25× (published statistics give ~34×).
    assert means["ZA"] / means["NO"] >= 25


def test_each_published_trace_mean_within_tolerance() -> None:
    """Every zone's empirical 14-day mean must land within ±25% of its declared
    annual mean — looser tolerance for high-volatility zones like GB."""
    for zone, params in _GRID_ZONES.items():
        trace = published_grid_trace(zone, days=14, sample_minutes=60, seed=1)
        emp = statistics.mean(trace.intensities)
        declared = params["mean_g"]
        rel_err = abs(emp - declared) / declared
        assert rel_err < 0.25, (
            f"{zone}: empirical mean {emp:.0f} g deviates from declared "
            f"{declared:.0f} g by {rel_err:.1%}"
        )


def test_published_trace_is_deterministic_for_seed() -> None:
    a = published_grid_trace("DE", days=3, sample_minutes=60, seed=7).intensities
    b = published_grid_trace("DE", days=3, sample_minutes=60, seed=7).intensities
    assert a == b


def test_published_trace_seed_changes_realisation() -> None:
    a = published_grid_trace("DE", days=3, sample_minutes=60, seed=7).intensities
    b = published_grid_trace("DE", days=3, sample_minutes=60, seed=8).intensities
    assert a != b


def test_published_trace_never_emits_negative_intensity() -> None:
    """Noise is large enough on FR (mean 65, noise sd 6) to dip near zero; the clamp
    must keep values ≥ 5 gCO2/kWh."""
    trace = published_grid_trace("FR", days=14, sample_minutes=60, seed=999)
    assert min(trace.intensities) >= 5.0


def test_published_trace_diurnal_swing_visible() -> None:
    """For a solar-heavy zone (US-CA), the diurnal swing should be visible in the
    hour-of-day means over a multi-day window."""
    trace = published_grid_trace("US-CA", days=14, sample_minutes=60, seed=3)
    by_hour: dict[int, list[float]] = {}
    for ts, val in zip(trace.timestamps, trace.intensities, strict=True):
        by_hour.setdefault(ts.hour, []).append(val)
    hourly_mean = {h: statistics.mean(by_hour[h]) for h in by_hour}
    # The solar trough should be substantially lower than the late-evening peak.
    assert min(hourly_mean.values()) < max(hourly_mean.values()) * 0.85


def test_published_trace_rejects_unknown_zone() -> None:
    with pytest.raises(ValueError, match="unknown zone"):
        published_grid_trace("XX", days=1)


# --- end-to-end: load the published-trace-generated fixture and recover stats ---


def test_fixture_roundtrips_through_loader() -> None:
    """The shipped DE fixture was generated by ``published_grid_trace('DE', ..., seed=42)``;
    re-loading it should yield the same intensities."""
    via_loader = load_electricitymaps_csv(FIXTURE).intensities
    regenerated = published_grid_trace("DE", days=7, sample_minutes=60, seed=42).intensities
    # CSV serialisation rounds to 2 decimals, so the comparison is to that tolerance.
    assert len(via_loader) == len(regenerated)
    for v, g in zip(via_loader, regenerated, strict=True):
        assert abs(v - g) < 0.01


def test_fixture_csv_is_real_em_schema() -> None:
    """Sanity: the fixture file's header matches the ElectricityMaps Data Portal export."""
    with FIXTURE.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert "Datetime (UTC)" in header
    assert any("Carbon Intensity" in c and "(LCA)" in c for c in header)

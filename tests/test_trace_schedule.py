"""Tests for the real-trace → schedule adapter."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from hasagi.energy.carbon_trace import CarbonTrace
from hasagi.energy.trace_schedule import (
    GRID_ZONE_IDS,
    diurnal_offsets,
    find_zone_csv,
    load_zone_traces,
    quantile_threshold,
    rotate,
    trace_to_hourly,
    zone_stats,
)

_EM_CSV = (
    "Datetime (UTC),Carbon Intensity gCO₂eq/kWh (LCA)\n"
    "2024-07-01 00:00:00,400.0\n"
    "2024-07-01 01:00:00,410.0\n"
    "2024-07-01 02:00:00,395.0\n"
)


def _hourly_trace(values: list[float]) -> CarbonTrace:
    t0 = datetime(2024, 7, 1, 0, 0, 0)
    ts = [t0 + timedelta(hours=h) for h in range(len(values))]
    return CarbonTrace(timestamps=ts, intensities=values)


def test_trace_to_hourly_resamples() -> None:
    tr = _hourly_trace([100.0, 200.0, 300.0, 400.0, 500.0])
    assert trace_to_hourly(tr) == [100.0, 200.0, 300.0, 400.0, 500.0]


def test_rotate() -> None:
    assert rotate([1, 2, 3, 4], 0) == [1, 2, 3, 4]
    assert rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert rotate([1, 2, 3, 4], 5) == [2, 3, 4, 1]   # wraps (mod len)
    assert rotate([], 3) == []


def test_quantile_threshold() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    assert quantile_threshold(xs, 0.0) == 1.0
    assert quantile_threshold(xs, 1.0) == 4.0
    assert quantile_threshold(xs, 0.5) == pytest.approx(2.5)
    assert quantile_threshold([7.0], 0.5) == 7.0


def test_quantile_threshold_validates() -> None:
    with pytest.raises(ValueError):
        quantile_threshold([], 0.5)
    with pytest.raises(ValueError):
        quantile_threshold([1.0, 2.0], 1.5)


def test_diurnal_offsets() -> None:
    # 336-hour (14-day) trace, one sample/day, each needs 24 h → offsets 0..312
    offs = diurnal_offsets(336, stride_hours=24, span_hours=24)
    assert offs[0] == 0
    assert offs[-1] == 312
    assert all(o % 24 == 0 for o in offs)
    assert len(offs) == 14


def test_diurnal_offsets_small_trace() -> None:
    assert diurnal_offsets(0) == []
    # span larger than trace → single offset at 0
    assert diurnal_offsets(10, stride_hours=24, span_hours=48) == [0]


def test_zone_stats() -> None:
    s = zone_stats([10.0, 20.0, 30.0])
    assert s["mean"] == pytest.approx(20.0)
    assert s["median"] == 20.0
    assert s["min"] == 10.0
    assert s["max"] == 30.0
    assert s["swing"] == 20.0


def test_find_zone_csv(tmp_path) -> None:
    (tmp_path / "de_2024-07-01_2024-07-15_hourly.csv").write_text(_EM_CSV, encoding="utf-8")
    (tmp_path / "us-ca_2024.csv").write_text(_EM_CSV, encoding="utf-8")
    assert find_zone_csv(tmp_path, "DE").name.startswith("de_")
    assert find_zone_csv(tmp_path, "US-CA").name.startswith("us-ca_")
    assert find_zone_csv(tmp_path, "FR") is None
    assert find_zone_csv(tmp_path / "missing", "DE") is None


def test_load_zone_traces_real_and_synthetic(tmp_path) -> None:
    (tmp_path / "de_x.csv").write_text(_EM_CSV, encoding="utf-8")
    traces = load_zone_traces(tmp_path, zones=["DE", "FR", "NO"], synthetic_days=2)
    assert traces["DE"].source == "real-csv"
    assert traces["DE"].csv_path is not None
    assert traces["DE"].trace.intensities[0] == 400.0     # from the CSV
    assert traces["FR"].source == "synthetic-parametric"
    assert traces["FR"].csv_path is None
    assert traces["NO"].source == "synthetic-parametric"


def test_grid_zone_ids_is_the_agreed_16() -> None:
    assert len(GRID_ZONE_IDS) == 16
    assert "DE" in GRID_ZONE_IDS and "NO" in GRID_ZONE_IDS
    # default load covers all 16 zones (synthetic where no CSV present)
    traces = load_zone_traces("/nonexistent-dir", synthetic_days=1)
    assert len(traces) == 16
    assert all(t.source == "synthetic-parametric" for t in traces.values())

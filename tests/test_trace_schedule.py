"""Tests for the real-trace → schedule adapter."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from hasagi.energy.carbon_trace import CarbonTrace
from hasagi.energy.trace_schedule import (
    diurnal_offsets,
    quantile_threshold,
    rotate,
    trace_to_hourly,
    zone_stats,
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

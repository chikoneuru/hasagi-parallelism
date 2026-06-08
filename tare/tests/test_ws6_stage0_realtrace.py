"""Tests for the WS6 Stage-0 wall-clock real-trace replay (pure analysis, no GPU)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from experiments.exp_carbon_repartition_breakeven import Layout
from experiments.exp_ws6_stage0_realtrace import (
    _n_needed_to_exclude_zero,
    _quantile,
    wallclock_replay,
    zone_deltas,
)
from tare.energy.carbon_trace import CarbonTrace

_FAST = Layout("fast", energy_per_iter_j=1.0, throughput_iter_s=1.0)
_ECO = Layout("eco", energy_per_iter_j=0.80, throughput_iter_s=0.6)
_IPW = 3600


def _trace(intensities: list[float]) -> CarbonTrace:
    t0 = datetime(2024, 7, 1)
    ts = [t0 + timedelta(hours=i) for i in range(len(intensities))]
    return CarbonTrace(timestamps=ts, intensities=intensities)


def test_quantile_and_n_needed() -> None:
    assert _quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0
    assert _n_needed_to_exclude_zero(0.0, 1.0) == float("inf")   # zero mean never excludes
    assert _n_needed_to_exclude_zero(-1.0, 1.0) == float("inf")  # negative mean
    small = _n_needed_to_exclude_zero(1.0, 1.0)
    big = _n_needed_to_exclude_zero(1.0, 4.0)                      # more noise -> need more zones
    assert big > small > 0


def test_flat_trace_no_switches_and_zero_delta() -> None:
    # A flat grid has no dirty windows (intensity strictly > q-threshold never holds),
    # so throttle never throttles and repartition never switches -> identical to fast.
    trace = _trace([400.0] * 48)
    thr_v = _quantile(trace.intensities, 0.6)
    common = dict(start_s=0.0, n_windows=6, iters_per_window=_IPW, threshold=thr_v,
                  throttle_energy_frac=0.85, throttle_tput_frac=0.7)
    fastr = wallclock_replay(trace, _FAST, _ECO, "static_fast", **common)
    thr = wallclock_replay(trace, _FAST, _ECO, "throttle", **common)
    rep = wallclock_replay(trace, _FAST, _ECO, "repartition", **common,
                           migration_energy_j=510.0, migration_time_s=5.1)
    assert rep["switches"] == 0
    assert abs(thr["carbon_g"] - fastr["carbon_g"]) < 1e-9
    assert abs(rep["carbon_g"] - fastr["carbon_g"]) < 1e-9


def test_static_eco_uses_less_energy_but_more_makespan() -> None:
    trace = _trace([300.0 + 200.0 * (i % 24 >= 12) for i in range(72)])
    common = dict(start_s=0.0, n_windows=12, iters_per_window=_IPW,
                  threshold=_quantile(trace.intensities, 0.6),
                  throttle_energy_frac=0.85, throttle_tput_frac=0.7)
    fastr = wallclock_replay(trace, _FAST, _ECO, "static_fast", **common)
    eco = wallclock_replay(trace, _FAST, _ECO, "static_eco", **common)
    assert eco["energy_j"] < fastr["energy_j"]          # 0.80x energy/iter
    assert eco["makespan_h"] > fastr["makespan_h"]      # at 0.60x throughput


def test_repartition_switches_on_a_swing_trace() -> None:
    # A smooth diurnal swing produces dirty windows (q-threshold lands strictly below the
    # peak) -> repartition switches at least once.
    trace = _trace([400.0 + 200.0 * math.sin(2.0 * math.pi * i / 24.0) for i in range(96)])
    common = dict(start_s=0.0, n_windows=24, iters_per_window=_IPW,
                  threshold=_quantile(trace.intensities, 0.6),
                  throttle_energy_frac=0.85, throttle_tput_frac=0.7)
    rep = wallclock_replay(trace, _FAST, _ECO, "repartition", **common,
                           migration_energy_j=510.0, migration_time_s=5.1)
    assert rep["switches"] >= 1


def test_zone_deltas_shape() -> None:
    trace = _trace([200.0 + 400.0 * (i % 24 >= 12) for i in range(120)])
    zd = zone_deltas(trace, _FAST, _ECO, n_windows=12, iters_per_window=_IPW,
                     threshold_q=0.6, throttle_energy_frac=0.85, throttle_tput_frac=0.7,
                     migration_energy_j=510.0, migration_time_s=5.1, offset_stride_h=24)
    assert zd["n_offsets"] >= 1
    assert len(zd["deltas"]) == zd["n_offsets"]
    assert 0 <= zd["rep_win_offsets"] <= zd["n_offsets"]

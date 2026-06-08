"""Tests for the large-model carbon-ledger analysis (pure analysis, no GPU).

Locks the two properties the headline rests on: carbon-aware pausing yields a
positive reduction vs always-on on a diurnal trace, and the state-size resume tax
erodes (never increases) that reduction as model state grows."""
from __future__ import annotations

from datetime import datetime, timedelta

from experiments.exp_largemodel_carbon_ledger import zone_reductions
from tare.energy.carbon_trace import CarbonTrace

_RELOAD_KW = dict(cold_start_s=4.7, resume_power_per_gpu_w=100.0,
                  write_bw_gbps=3.9, reload_bw_gbps=7.7, warmup_s=0.761)


def _diurnal_trace(days: int = 14) -> CarbonTrace:
    # clean nights (200), dirty days (700): a clear within-zone diurnal swing
    t0 = datetime(2024, 7, 1)
    ints, ts = [], []
    for h in range(days * 24):
        ints.append(200.0 if (h % 24) < 12 else 700.0)
        ts.append(t0 + timedelta(hours=h))
    return CarbonTrace(timestamps=ts, intensities=ints)


def _reduction(state_gb: float) -> float:
    return zone_reductions(_diurnal_trace(), state_gb=state_gb, h_job=24.0, threshold_q=0.5,
                           n_full=8, e_fngh_kwh=2.4, reload_kwargs=_RELOAD_KW,
                           offset_stride_h=24)["mean_reduction_pct"]


def test_pause_beats_always_on_on_a_diurnal_trace():
    assert _reduction(1.0) > 0.0


def test_state_tax_erodes_the_reduction_monotonically():
    small = _reduction(1.0)
    big = _reduction(160.0)
    # larger state pays a larger per-resume reload tax, so the reduction can only shrink
    assert big <= small
    # and the erosion is bounded (the tax is amortised over a multi-day run, not catastrophic)
    assert small - big < small  # big stays positive-ish, not wiped out


def test_zero_state_is_the_ceiling():
    # ~stateless resume (tiny state) gives the largest reduction in the sweep
    assert _reduction(0.001) >= _reduction(80.0)

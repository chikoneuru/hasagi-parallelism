"""Tests for the carbon-throttle vs pause vs always-on Pareto simulator."""
from __future__ import annotations

import pytest

from tare.energy.throttle_pareto import (
    CapPoint,
    PowerCapProfile,
    pareto,
    simulate_masked_policy,
    simulate_policy,
)

# Synthetic profile: energy-optimal at 200 W (4.5 J/it), throughput-max at 300 W.
_PROFILE = PowerCapProfile(
    gpu_name="synthetic",
    points={
        100.0: CapPoint(100.0, 10.0, 100.0, 10.0 / 3_600_000.0),
        200.0: CapPoint(200.0, 40.0, 180.0, 4.5 / 3_600_000.0),
        300.0: CapPoint(300.0, 50.0, 270.0, 5.4 / 3_600_000.0),
    },
)

_SCHED = [200.0, 800.0]   # clean, dirty
_KW = dict(total_iters=1000, window_s=10.0, schedule_g=_SCHED, threshold_g=500.0)


def test_profile_optima() -> None:
    assert _PROFILE.energy_optimal_cap == 200.0
    assert _PROFILE.max_throughput_cap == 300.0
    assert _PROFILE.point(195.0).cap_w == 200.0   # nearest


def test_always_on_pays_dirty_intensity() -> None:
    r = simulate_policy(_PROFILE, name="always-on", clean_cap_w=300.0, dirty_cap_w=300.0, **_KW)
    # 500 iters clean@200 + 500 iters dirty@800, each 7.5e-4 kWh
    assert r.total_carbon_g == pytest.approx(7.5e-4 * 200 + 7.5e-4 * 800)   # 0.75 g
    assert r.makespan_s == pytest.approx(20.0)
    assert r.iters == 1000


def test_pause_defers_dirty_work_zero_idle() -> None:
    r = simulate_policy(
        _PROFILE, name="pause", clean_cap_w=300.0, dirty_cap_w=None,
        idle_power_w=0.0, **_KW,
    )
    # all work done in clean windows at 200 → 2×0.15 g; +10 s makespan for the pause
    assert r.total_carbon_g == pytest.approx(0.30)
    assert r.makespan_s == pytest.approx(30.0)
    assert r.idle_carbon_g == 0.0


def test_pause_dedicated_idle_adds_carbon() -> None:
    r = simulate_policy(
        _PROFILE, name="pause", clean_cap_w=300.0, dirty_cap_w=None,
        idle_power_w=50.0, **_KW,
    )
    idle_g = 50.0 * 10.0 / 3_600_000.0 * 800.0   # one dirty window idle @ 800
    assert r.idle_carbon_g == pytest.approx(idle_g)
    assert r.total_carbon_g == pytest.approx(0.30 + idle_g)


def test_pause_charges_resume_on_wakeup() -> None:
    r = simulate_policy(
        _PROFILE, name="pause", clean_cap_w=300.0, dirty_cap_w=None,
        resume_energy_kwh=1e-4, **_KW,
    )
    # one wake-up (the clean window after the dirty pause), billed at clean 200
    assert r.resume_carbon_g == pytest.approx(1e-4 * 200.0)


def test_throttle_between_alwayson_and_pause() -> None:
    r = simulate_policy(
        _PROFILE, name="throttle", clean_cap_w=300.0, dirty_cap_w=200.0, **_KW,
    )
    # clean 500@200 (0.15) + dirty 400@200cap@800 (0.4) + clean 100@200 (0.03)
    assert r.total_carbon_g == pytest.approx(0.58)
    assert r.makespan_s == pytest.approx(22.0)


def test_pareto_orders_carbon_and_makespan() -> None:
    res = pareto(_PROFILE, total_iters=1000, window_s=10.0, schedule_g=_SCHED, threshold_g=500.0)
    assert set(res) == {"always_on", "always_on_eco", "pause", "throttle"}
    # pause: least carbon, most makespan; always-on: most carbon, least makespan
    assert res["pause"].total_carbon_g < res["throttle"].total_carbon_g < res["always_on"].total_carbon_g
    assert res["pause"].makespan_s > res["always_on"].makespan_s
    # all complete the same work
    for r in res.values():
        assert r.iters >= 1000


def test_eco_baseline_is_carbon_blind_efficiency() -> None:
    """always-on@eco runs the energy-optimal cap throughout: lower carbon than
    always-on@full (the U-curve efficiency win) but a longer makespan, and it is
    carbon-BLIND (same cap regardless of intensity)."""
    res = pareto(_PROFILE, total_iters=1000, window_s=10.0, schedule_g=_SCHED, threshold_g=500.0)
    eco = res["always_on_eco"]
    full = res["always_on"]
    # eco (200 W, 4.5 J/it) is leaner than full (300 W, 5.4 J/it) → less carbon
    assert eco.total_carbon_g < full.total_carbon_g
    # ... but slower (40 vs 50 it/s) → longer makespan
    assert eco.makespan_s > full.makespan_s
    # eco never idles (it is always-on, just at a leaner cap)
    assert eco.idle_carbon_g == 0.0


# --- masked policy: an external decision rule drives the measured substrate ---

def test_masked_all_active_equals_always_on() -> None:
    """A mask that is active in every window is exactly always-on@full."""
    full = simulate_policy(_PROFILE, name="always-on", clean_cap_w=300.0, dirty_cap_w=300.0, **_KW)
    masked = simulate_masked_policy(
        _PROFILE, name="masked-on", active_mask=[1, 1], full_cap_w=300.0,
        total_iters=1000, window_s=10.0, schedule_g=_SCHED,
    )
    assert masked.total_carbon_g == pytest.approx(full.total_carbon_g)
    assert masked.makespan_s == pytest.approx(full.makespan_s)
    assert masked.iters == full.iters


def test_masked_pause_matches_threshold_pause() -> None:
    """Masking off the dirty window (pause) reproduces the threshold pause policy
    when the mask flags exactly the windows above the threshold."""
    thr = simulate_policy(
        _PROFILE, name="pause", clean_cap_w=300.0, dirty_cap_w=None,
        idle_power_w=50.0, resume_energy_kwh=1e-4, **_KW,
    )
    # schedule is [clean=200, dirty=800]; threshold 500 → dirty window is index 1.
    masked = simulate_masked_policy(
        _PROFILE, name="pause", active_mask=[1, 0], full_cap_w=300.0, off_cap_w=None,
        idle_power_w=50.0, resume_energy_kwh=1e-4,
        total_iters=1000, window_s=10.0, schedule_g=_SCHED,
    )
    assert masked.total_carbon_g == pytest.approx(thr.total_carbon_g)
    assert masked.idle_carbon_g == pytest.approx(thr.idle_carbon_g)
    assert masked.resume_carbon_g == pytest.approx(thr.resume_carbon_g)
    assert masked.makespan_s == pytest.approx(thr.makespan_s)


def test_masked_throttle_matches_threshold_throttle() -> None:
    """Masking off the dirty window with an off_cap (throttle) reproduces the
    threshold throttle policy."""
    thr = simulate_policy(
        _PROFILE, name="throttle", clean_cap_w=300.0, dirty_cap_w=200.0, **_KW,
    )
    masked = simulate_masked_policy(
        _PROFILE, name="throttle", active_mask=[1, 0], full_cap_w=300.0, off_cap_w=200.0,
        total_iters=1000, window_s=10.0, schedule_g=_SCHED,
    )
    assert masked.total_carbon_g == pytest.approx(thr.total_carbon_g)
    assert masked.makespan_s == pytest.approx(thr.makespan_s)


def test_masked_pause_vs_throttle_same_mask_isolates_mechanism() -> None:
    """Same dirty-window mask, two responses: pause defers (more makespan, no
    active dirty energy) while throttle keeps training at the lean cap. This is
    the head-to-head axis — the mask (decision rule) is held fixed."""
    mask = [1, 0]
    kw = dict(total_iters=1000, window_s=10.0, schedule_g=_SCHED, active_mask=mask, full_cap_w=300.0)
    pause = simulate_masked_policy(_PROFILE, name="pause", off_cap_w=None, idle_power_w=0.0, **kw)
    throttle = simulate_masked_policy(_PROFILE, name="throttle", off_cap_w=200.0, **kw)
    # pause never burns active energy in the dirty window; throttle does (at 200 W).
    assert pause.makespan_s > throttle.makespan_s
    # both complete the work
    assert pause.iters >= 1000 and throttle.iters >= 1000


def test_masked_empty_mask_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        simulate_masked_policy(
            _PROFILE, name="x", active_mask=[], full_cap_w=300.0,
            total_iters=10, window_s=10.0, schedule_g=_SCHED,
        )

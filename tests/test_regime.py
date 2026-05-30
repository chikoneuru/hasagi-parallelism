"""Tests for the three-regime carbon analysis and the break-even solver."""
from __future__ import annotations

import math

import pytest

from hasagi.energy.pod_ledger import PHASE_ACTIVE, PHASE_IDLE, PodEnergyLedger
from hasagi.energy.regime import (
    GpuRegime,
    break_even_window_s,
    regime_breakdown,
    regime_carbon,
)


def _report_with_idle():
    """One active interval (0.01 kWh @ 500) + one 60 s idle interval @ 800.

    Marginal idle energy is 0 (the paused job draws nothing of its own), which
    is exactly the shared-GPU case.
    """
    led = PodEnergyLedger(energy_kwh_fn=lambda: 0.0, clock=lambda: 0.0)
    led.mark(PHASE_ACTIVE, 500.0, t_s=0.0, cumulative_kwh=0.0)
    led.mark(PHASE_IDLE, 800.0, t_s=10.0, cumulative_kwh=0.01)
    return led.report(t_s=70.0, cumulative_kwh=0.01)


def test_reallocated_charges_zero_idle() -> None:
    rep = _report_with_idle()
    rc = regime_carbon(rep, GpuRegime.REALLOCATED)
    assert rc.idle_carbon_g == 0.0
    assert rc.measured_carbon_g == pytest.approx(5.0)   # 0.01 kWh × 500
    assert rc.total_carbon_g == pytest.approx(5.0)


def test_dedicated_charges_idle_floor() -> None:
    rep = _report_with_idle()
    rc = regime_carbon(rep, GpuRegime.DEDICATED, dedicated_idle_w=50.0)
    # idle: 50 W × 60 s / 3.6e6 = 8.333e-4 kWh × 800 = 0.6667 g
    assert rc.idle_carbon_g == pytest.approx(50.0 * 60 / 3_600_000.0 * 800.0)
    assert rc.total_carbon_g == pytest.approx(5.0 + 50.0 * 60 / 3_600_000.0 * 800.0)


def test_powered_down_charges_spin_only() -> None:
    rep = _report_with_idle()
    rc = regime_carbon(rep, GpuRegime.POWERED_DOWN, spin_down_kwh=0.001)
    assert rc.idle_carbon_g == pytest.approx(0.001 * 800.0)   # 0.8 g
    assert rc.total_carbon_g == pytest.approx(5.0 + 0.8)


def test_regime_breakdown_has_all_three() -> None:
    rep = _report_with_idle()
    bd = regime_breakdown(rep, dedicated_idle_w=50.0)
    assert set(bd) == {"dedicated", "reallocated", "powered_down"}
    assert bd["dedicated"].total_carbon_g > bd["reallocated"].total_carbon_g


def test_break_even_finite_when_idle_free() -> None:
    # reallocated/powered-down: idle 0 → finite break-even
    t = break_even_window_s(
        active_power_w=200.0, intensity_dirty=800.0, intensity_clean=200.0,
        resume_energy_kwh=0.0001, idle_power_w=0.0,
    )
    # T* = 0.0001 × 200 × 3.6e6 / (200 × 600) = 72000 / 120000 = 0.6 s
    assert t == pytest.approx(0.6)


def test_break_even_infinite_when_dedicated_idle_too_costly() -> None:
    # dedicated idle so high it outweighs the carbon arbitrage → never wins
    t = break_even_window_s(
        active_power_w=200.0, intensity_dirty=800.0, intensity_clean=200.0,
        resume_energy_kwh=0.0001, idle_power_w=200.0,
    )
    assert t == math.inf


def test_break_even_infinite_when_no_intensity_gap() -> None:
    t = break_even_window_s(
        active_power_w=200.0, intensity_dirty=300.0, intensity_clean=300.0,
        resume_energy_kwh=0.0001, idle_power_w=0.0,
    )
    assert t == math.inf

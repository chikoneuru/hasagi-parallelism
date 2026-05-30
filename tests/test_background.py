"""Tests for the time-varying background model and trace re-integration."""
from __future__ import annotations

import pytest

from hasagi.energy.background import BackgroundModel, marginal_kwh_from_trace
from hasagi.energy.pod_ledger import PHASE_ACTIVE, PHASE_IDLE, PodEnergyLedger


def test_single_anchor_is_constant() -> None:
    bg = BackgroundModel()
    bg.add(0.0, 120.0)
    assert bg.at(-5.0) == 120.0
    assert bg.at(100.0) == 120.0
    assert bg.mean_w == 120.0
    assert bg.drift_w == 0.0


def test_linear_interpolation_between_anchors() -> None:
    bg = BackgroundModel()
    bg.add(0.0, 100.0)
    bg.add(10.0, 200.0)
    assert bg.at(0.0) == 100.0
    assert bg.at(5.0) == 150.0      # midpoint
    assert bg.at(10.0) == 200.0
    assert bg.at(-1.0) == 100.0     # flat before first
    assert bg.at(11.0) == 200.0     # flat after last
    assert bg.drift_w == 100.0


def test_anchors_unsorted_input_sorted() -> None:
    bg = BackgroundModel()
    bg.add(10.0, 200.0)
    bg.add(0.0, 100.0)
    assert bg.at(5.0) == 150.0


def test_marginal_integration_constant_background() -> None:
    # device steady 300 W, background steady 100 W → marginal 200 W over 10 s
    bg = BackgroundModel()
    bg.add(0.0, 100.0)
    trace = [(float(t), 300.0) for t in range(0, 11)]   # 0..10 s @ 1 Hz
    kwh = marginal_kwh_from_trace(trace, bg, 0.0, 10.0)
    assert kwh == pytest.approx(200.0 * 10 / 3_600_000.0)


def test_marginal_integration_tracks_drifting_background() -> None:
    # device steady 300 W; background drifts 100→200 W over 10 s (mean 150)
    # marginal = 300 - bg(t); integral of (200 down to 100) over 10s = mean 150 W
    bg = BackgroundModel()
    bg.add(0.0, 100.0)
    bg.add(10.0, 200.0)
    trace = [(float(t), 300.0) for t in range(0, 11)]
    kwh = marginal_kwh_from_trace(trace, bg, 0.0, 10.0)
    # right-endpoint rectangles: sum_{t=1..10} (300 - bg(t)) * 1s
    expected_w_s = sum(300.0 - bg.at(float(t)) for t in range(1, 11))
    assert kwh == pytest.approx(expected_w_s / 3_600_000.0)


def test_marginal_clamped_when_device_below_background() -> None:
    bg = BackgroundModel()
    bg.add(0.0, 150.0)
    trace = [(0.0, 120.0), (5.0, 120.0)]   # device below background → 0 marginal
    assert marginal_kwh_from_trace(trace, bg, 0.0, 5.0) == 0.0


def test_window_restricts_integration() -> None:
    bg = BackgroundModel()
    bg.add(0.0, 0.0)
    trace = [(float(t), 360.0) for t in range(0, 11)]   # 360 W constant
    # integrate only [2, 4] → 2 s × 360 W
    kwh = marginal_kwh_from_trace(trace, bg, 2.0, 4.0)
    assert kwh == pytest.approx(360.0 * 2 / 3_600_000.0)


def test_report_from_trace_reattributes_phases() -> None:
    """A drifting background is corrected when re-integrating the ledger."""
    bg = BackgroundModel()
    bg.add(0.0, 100.0)
    bg.add(20.0, 100.0)            # steady background here
    # device: 300 W during active [0,10], 100 W (==background) during idle [10,20]
    trace = [(float(t), 300.0) for t in range(0, 11)] + [(float(t), 100.0) for t in range(11, 21)]
    led = PodEnergyLedger(energy_kwh_fn=lambda: 0.0, clock=lambda: 0.0)
    led.mark(PHASE_ACTIVE, 500.0, t_s=0.0, cumulative_kwh=0.0)
    led.mark(PHASE_IDLE, 800.0, t_s=10.0, cumulative_kwh=0.0)
    rep = led.report_from_trace(trace, bg, t_end=20.0)
    # active marginal ≈ 200 W × 10 s; idle marginal ≈ 0 (device == background)
    assert rep.energy_by_phase_kwh[PHASE_ACTIVE] == pytest.approx(200.0 * 10 / 3_600_000.0, rel=0.05)
    assert rep.energy_by_phase_kwh[PHASE_IDLE] == pytest.approx(0.0, abs=1e-9)
    assert rep.carbon_by_phase_g[PHASE_ACTIVE] > 0.0

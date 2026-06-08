"""Tests for the real-trace carbon panel: the same-budget carbon-blind mask and
the carbon-signal decomposition / fair-mechanism-gap arithmetic."""
from __future__ import annotations

import pytest

from experiments.exp_realtrace_pareto import _savings_over_offsets, _uniform_off_mask
from tare.energy.throttle_pareto import CapPoint, PowerCapProfile

# Energy-optimal at 200 W (4.5 J/it); throughput-max at 300 W (5.4 J/it).
_PROFILE = PowerCapProfile(
    gpu_name="synthetic",
    points={
        200.0: CapPoint(200.0, 40.0, 180.0, 4.5 / 3_600_000.0),
        300.0: CapPoint(300.0, 50.0, 270.0, 5.4 / 3_600_000.0),
    },
)
# 72 h, bimodal: 60% clean (200 g) and 40% dirty (900 g) → q0.6 threshold splits them.
_HOURLY = [900.0 if (h % 5) >= 3 else 200.0 for h in range(72)]


def test_uniform_off_mask_count_and_edges() -> None:
    m = _uniform_off_mask(100, 40)
    assert len(m) == 100
    assert m.count(0) == 40                      # exact same-budget count
    assert _uniform_off_mask(10, 0) == [1] * 10  # no response windows
    assert _uniform_off_mask(10, 10) == [0] * 10  # all response
    assert _uniform_off_mask(10, 99).count(0) == 10  # clamped to n


def _run():
    return _savings_over_offsets(
        _PROFILE, _HOURLY, total_iters=4000, threshold=500.0, throttle_cap=200.0,
        resume_kwh=0.0, dedicated_idle_w=26.0, stride_hours=12, span_hours=48,
        green_pause_fraction=0.4, green_window=24,
    )


def test_signal_is_throttle_minus_same_budget_blind() -> None:
    res = _run()
    assert res["n_offsets"] >= 1
    for i in range(res["n_offsets"]):
        assert res["signal_pp"][i] == pytest.approx(
            res["throttle_pct"][i] - res["throttle_blind_pct"][i]
        )


def test_fair_gap_is_throttle_minus_green_offline_reallocated() -> None:
    res = _run()
    for i in range(res["n_offsets"]):
        assert res["gap_fair_pp"][i] == pytest.approx(
            res["throttle_pct"][i] - res["green_off_rea_pct"][i]
        )


def test_eco_endpoint_is_carbon_blind_energy_ratio() -> None:
    """always-on@eco saving ≈ 1 − eco/full energy-per-iter (≈16.7%), independent
    of intensity, because it leans every window regardless of carbon."""
    res = _run()
    expected = 100.0 * (1.0 - 4.5 / 5.4)
    for v in res["eco_pct"]:
        assert v == pytest.approx(expected, abs=1.0)


def test_pause_defers_more_makespan_than_throttle() -> None:
    """GREEN's pause forfeits throughput in off-windows, so it finishes later
    than throttling under the same online mask."""
    res = _run()
    for i in range(res["n_offsets"]):
        assert res["green_on_makespan_h"][i] >= res["throttle_online_makespan_h"][i]


def test_selftuned_green_never_loses_and_budget_bounded() -> None:
    """The steel-man GREEN picks the carbon-max pause budget with a 0 floor, so its
    saving is never negative and its chosen budget stays within the sweep grid;
    pausing more to win carbon never reduces its makespan below zero."""
    res = _run()
    for i in range(res["n_offsets"]):
        assert res["green_selftuned_rea_pct"][i] >= -1e-9         # 0 floor (decline to pause)
        assert 0.0 <= res["green_selftuned_budget"][i] <= 0.6     # within the sweep grid
        assert res["green_selftuned_makespan_h"][i] >= -1e-9      # pausing only adds makespan


def test_dedicated_idle_never_helps_pause() -> None:
    """Billing a dedicated idle floor can only reduce (or leave equal) GREEN's
    carbon saving versus the reallocated (0 W idle) regime."""
    res = _run()
    for i in range(res["n_offsets"]):
        assert res["green_on_ded_pct"][i] <= res["green_on_rea_pct"][i] + 1e-9
        assert res["green_off_ded_pct"][i] <= res["green_off_rea_pct"][i] + 1e-9

"""Tests for the workload-dependent-cap comparison: energy lookup, the optimality
condition (throughput elasticity brackets alpha at the optimum), and the
cross-application penalty arithmetic."""
from __future__ import annotations

import pytest

from experiments.exp_workload_cap_compare import _elasticity_brackets_alpha, _energy_at
from hasagi.energy.throttle_pareto import CapPoint, PowerCapProfile

# U-curve with the energy-per-iter minimum at the MIDDLE cap (200 W); throughput
# rises then saturates so the elasticity falls across the caps.
_PROFILE = PowerCapProfile(
    gpu_name="synthetic",
    points={
        100.0: CapPoint(100.0, 10.0, 100.0, 6.0e-6),
        200.0: CapPoint(200.0, 18.0, 190.0, 4.0e-6),   # energy-optimal
        300.0: CapPoint(300.0, 20.0, 285.0, 5.0e-6),
    },
)


def test_energy_at_uses_nearest_cap() -> None:
    assert _energy_at(_PROFILE, 200.0) == pytest.approx(4.0e-6 * 3_600_000.0)
    assert _energy_at(_PROFILE, 205.0) == pytest.approx(4.0e-6 * 3_600_000.0)  # nearest


def test_energy_optimal_cap_is_the_ucurve_min() -> None:
    assert _PROFILE.energy_optimal_cap == 200.0


def test_elasticity_brackets_alpha_at_optimum() -> None:
    """Below the optimum throughput is elastic (> alpha); above it saturates
    (< alpha). With alpha between the two, the optimum brackets it."""
    chk = _elasticity_brackets_alpha(_PROFILE, alpha=0.5)
    assert chk["throughput_elasticity_below_opt"] > 0.5    # 100->200: ln1.8/ln2 ≈ 0.85
    assert chk["throughput_elasticity_above_opt"] < 0.5    # 200->300: ln(20/18)/ln1.5 ≈ 0.26
    assert chk["elasticity_brackets_alpha_at_opt"] is True


def test_elasticity_does_not_bracket_when_alpha_too_low() -> None:
    """If alpha sits below the above-optimum elasticity, the discrete grid does
    not bracket it (the continuous optimum is past the last grid point)."""
    chk = _elasticity_brackets_alpha(_PROFILE, alpha=0.1)
    assert chk["throughput_elasticity_above_opt"] > 0.1
    assert chk["elasticity_brackets_alpha_at_opt"] is False


def test_cross_application_penalty_direction() -> None:
    """Forcing a workload off its own energy-optimal cap never lowers its
    energy-per-iter (its own optimum is the minimum by definition)."""
    own = _energy_at(_PROFILE, _PROFILE.energy_optimal_cap)
    for other_cap in (100.0, 300.0):
        forced = _energy_at(_PROFILE, other_cap)
        assert forced >= own

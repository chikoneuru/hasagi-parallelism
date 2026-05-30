"""Tests for the workload-dependent-cap comparison: energy lookup, the
interior-minimum check (the measured optimum is inside the swept caps — which,
since E/iter = power/throughput, is identical to the elasticity bracket and is
NOT an independent optimality law), and the cross-application penalty arithmetic."""
from __future__ import annotations

import pytest

from experiments.exp_workload_cap_compare import (
    _energy_at,
    _optimality_check,
    _plateau,
)
from hasagi.energy.throttle_pareto import CapPoint, PowerCapProfile


def _cp(cap: float, thru: float, power: float) -> CapPoint:
    """CapPoint with energy-per-iter made internally consistent (= power/throughput)."""
    return CapPoint(cap, thru, power, (power / thru) / 3_600_000.0)


# Internally-consistent U-curve (E/iter = power/throughput) with the minimum at
# the MIDDLE cap (200 W): throughput rises then saturates, power keeps climbing.
_PROFILE = PowerCapProfile(
    gpu_name="synthetic",
    points={
        100.0: _cp(100.0, 10.0, 100.0),   # E/iter 10.0 J
        200.0: _cp(200.0, 25.0, 190.0),   # E/iter  7.6 J  ← energy-optimal
        300.0: _cp(300.0, 28.0, 285.0),   # E/iter 10.18 J
    },
)


def test_energy_at_uses_nearest_cap() -> None:
    own = (190.0 / 25.0) / 3_600_000.0 * 3_600_000.0
    assert _energy_at(_PROFILE, 200.0) == pytest.approx(own)
    assert _energy_at(_PROFILE, 205.0) == pytest.approx(own)  # nearest


def test_energy_optimal_cap_is_the_ucurve_min() -> None:
    assert _PROFILE.energy_optimal_cap == 200.0


def test_interior_minimum_check_equals_argmin_interior() -> None:
    """The elasticity bracket is identical to "the energy argmin is interior" (since
    E/iter = power/throughput exactly). For this profile the min is at the middle
    cap, so it is interior; the elasticities are reported as descriptive detail."""
    chk = _optimality_check(_PROFILE)
    # descriptive: below the optimum throughput outgrows power, above power outgrows throughput
    assert chk["throughput_elasticity_below_opt"] > chk["power_elasticity_below_opt"]
    assert chk["throughput_elasticity_above_opt"] < chk["power_elasticity_above_opt"]
    assert chk["optimum_is_interior"] is True
    # the check is exactly the interior-argmin condition on the stored E/iter
    e = {c: _PROFILE.point(c).energy_per_iter_kwh for c in _PROFILE.caps}
    eo = _PROFILE.energy_optimal_cap
    interior = e[eo] < min(e[c] for c in _PROFILE.caps if c != eo)
    assert chk["optimum_is_interior"] == interior


def test_not_interior_when_optimum_at_endpoint() -> None:
    """If energy keeps falling to the top cap, the optimum sits at the last grid
    point; the above-segment is undefined so it is not interior."""
    prof = PowerCapProfile(
        gpu_name="x",
        points={100.0: _cp(100.0, 10.0, 100.0), 200.0: _cp(200.0, 30.0, 190.0)},  # min at 200 (top)
    )
    chk = _optimality_check(prof)
    assert chk["optimum_is_interior"] is False


def test_plateau_reports_flat_bottom() -> None:
    """A near-flat U-curve bottom is reported as a plateau range, not a point."""
    prof = PowerCapProfile(
        gpu_name="x",
        points={
            100.0: CapPoint(100.0, 10.0, 100.0, 10.0e-6),
            200.0: CapPoint(200.0, 25.0, 190.0, 4.00e-6),   # min
            300.0: CapPoint(300.0, 28.0, 285.0, 4.05e-6),   # within 2% of min → same plateau
        },
    )
    lo, hi = _plateau(prof, tol_frac=0.02)
    assert (lo, hi) == (200.0, 300.0)


def test_cross_application_penalty_direction() -> None:
    """Forcing a workload off its own energy-optimal cap never lowers its
    energy-per-iter (its own optimum is the minimum by definition)."""
    own = _energy_at(_PROFILE, _PROFILE.energy_optimal_cap)
    for other_cap in (100.0, 300.0):
        forced = _energy_at(_PROFILE, other_cap)
        assert forced >= own

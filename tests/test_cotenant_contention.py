"""Tests for the co-tenant contention helper (pure math; no GPU)."""
from __future__ import annotations

from experiments.exp_cotenant_contention import _contention_factors


def test_contention_factors_no_contention() -> None:
    # Perfect spatial sharing: per-tenant throughput unchanged as tenants are added.
    f = _contention_factors({1: 100.0, 2: 100.0, 3: 100.0})
    assert f[1]["contention_factor"] == 1.0
    assert f[2]["contention_factor"] == 1.0
    assert abs(f[3]["aggregate_scaling"] - 3.0) < 1e-9  # aggregate scales linearly


def test_contention_factors_time_slice_saturation() -> None:
    # Pure time-slicing: per-tenant throughput halves at N=2, thirds at N=3,
    # so aggregate throughput stays flat (saturated GPU).
    f = _contention_factors({1: 120.0, 2: 60.0, 3: 40.0})
    assert abs(f[2]["contention_factor"] - 0.5) < 1e-9
    assert abs(f[3]["contention_factor"] - (1.0 / 3.0)) < 1e-9
    assert abs(f[2]["aggregate_scaling"] - 1.0) < 1e-9
    assert abs(f[3]["aggregate_scaling"] - 1.0) < 1e-9


def test_contention_factors_aggregate_is_n_times_per_tenant() -> None:
    f = _contention_factors({1: 100.0, 2: 70.0})
    assert abs(f[2]["aggregate_iters_per_s"] - 140.0) < 1e-9
    assert abs(f[2]["aggregate_scaling"] - 1.4) < 1e-9

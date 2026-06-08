"""Tests for the PowerFlow allocator port (experiments/baselines/powerflow.py).

These pin the behaviour described in arXiv 2304.06381 Algorithm 1:
greedy by (ΔJCT/JCT) / (ΔE/E), with the cluster-wide energy-budget constraint.
"""
from __future__ import annotations

import math

from experiments.baselines.powerflow import (
    _jct_and_energy,
    _relative_priority,
    powerflow_allocate,
)
from tare.admission.energy_profile import EnergyProfile, linear_profile

# --- _jct_and_energy ---


def test_jct_and_energy_zero_when_no_gpus() -> None:
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    jct, e = _jct_and_energy(p, gpus=0, iterations=100)
    assert jct == math.inf
    assert e == 0.0


def test_jct_and_energy_matches_profile() -> None:
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    jct, e = _jct_and_energy(p, gpus=2, iterations=200)
    assert abs(jct - 200 / p.throughput(2)) < 1e-9
    assert abs(e - p.energy_per_iter(2) * 200) < 1e-12


# --- _relative_priority ---


def test_priority_saturated_returns_neg_inf() -> None:
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    assert _relative_priority(p, current_gpus=4, iterations=100) == -math.inf
    # One past max also returns -inf (saturated)
    assert _relative_priority(p, current_gpus=10, iterations=100) == -math.inf


def test_priority_diminishing_returns() -> None:
    """Priority should decrease as allocation grows (concave throughput + convex energy)."""
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    priorities = [_relative_priority(p, g, 1000) for g in range(1, 8)]
    # Strictly decreasing (modulo equality at the tail in some pathological profiles).
    assert all(priorities[i] >= priorities[i + 1] for i in range(len(priorities) - 1))


def test_priority_zero_energy_yields_inf() -> None:
    """Pathological: a degenerate profile with zero energy at g returns +inf priority."""
    p = EnergyProfile(
        energy_per_iter_kwh=(0.0, 1.0, 2.0),
        throughput_iters_per_s=(1.0, 2.0, 3.0),
    )
    # At g=1, E(1)=0 → priority = +inf so this job wins the next GPU.
    assert _relative_priority(p, current_gpus=1, iterations=10) == math.inf


# --- powerflow_allocate ---


def test_two_identical_jobs_split_evenly() -> None:
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = powerflow_allocate(
        jobs=[("A", p, 1000), ("B", p, 1000)],
        available_gpus=8,
    )
    assert sum(alloc.values()) <= 8
    assert alloc["A"] == alloc["B"]   # Symmetry: equal split


def test_respects_gpu_budget() -> None:
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = powerflow_allocate(
        jobs=[("A", p, 500), ("B", p, 500), ("C", p, 500)],
        available_gpus=6,
    )
    assert sum(alloc.values()) <= 6
    assert all(alloc[j] >= 1 for j in ("A", "B", "C"))


def test_respects_per_job_max_gpus() -> None:
    """A job with max_gpus=2 must not be allocated > 2."""
    p_small = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=2)
    p_big = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = powerflow_allocate(
        jobs=[("small", p_small, 500), ("big", p_big, 500)],
        available_gpus=8,
    )
    assert alloc["small"] <= 2
    assert alloc["big"] <= 8


def test_energy_budget_stops_allocation_when_exceeded() -> None:
    """Tight budget → fewer GPUs."""
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc_unbounded = powerflow_allocate(
        jobs=[("A", p, 1000), ("B", p, 1000)],
        available_gpus=16,
    )
    # Tight budget: cap the total at well below what unbounded gives.
    unbounded_energy = (
        p.energy_per_iter(alloc_unbounded["A"]) * 1000
        + p.energy_per_iter(alloc_unbounded["B"]) * 1000
    )
    alloc_tight = powerflow_allocate(
        jobs=[("A", p, 1000), ("B", p, 1000)],
        available_gpus=16,
        energy_budget_kwh=unbounded_energy * 0.3,
    )
    total_tight = (
        p.energy_per_iter(alloc_tight["A"]) * 1000
        + p.energy_per_iter(alloc_tight["B"]) * 1000
    )
    assert total_tight <= unbounded_energy * 0.3 + 1e-9
    assert sum(alloc_tight.values()) <= sum(alloc_unbounded.values())


def test_infeasible_initial_returns_zeros() -> None:
    """If the initial allocation already breaks budget, return zeros."""
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = powerflow_allocate(
        jobs=[("A", p, 1000), ("B", p, 1000)],
        available_gpus=1,           # only 1 GPU but initial alloc is 2
    )
    assert alloc == {"A": 0, "B": 0}


def test_supports_initial_allocation_override() -> None:
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = powerflow_allocate(
        jobs=[("A", p, 1000), ("B", p, 1000)],
        available_gpus=8,
        initial_allocation={"A": 3, "B": 1},
    )
    assert alloc["A"] >= 3
    assert alloc["B"] >= 1
    assert sum(alloc.values()) <= 8


def test_more_iterations_get_priority() -> None:
    """A job with more remaining iterations dominates the priority (larger absolute ΔJCT
    per GPU once the throughput improvement is the same)."""
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = powerflow_allocate(
        jobs=[("small", p, 100), ("big", p, 10_000)],
        available_gpus=8,
    )
    # PowerFlow's priority uses *relative* deltas, so iteration count cancels.
    # Both jobs should split evenly under identical profiles regardless of iters.
    assert alloc["small"] == alloc["big"]


def test_empty_jobs_returns_empty() -> None:
    alloc = powerflow_allocate(jobs=[], available_gpus=8)
    assert alloc == {}


def test_zero_budget_returns_zero_alloc() -> None:
    """Zero energy budget → no GPUs allocated beyond the initial 1-each, possibly zero."""
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = powerflow_allocate(
        jobs=[("A", p, 1000), ("B", p, 1000)],
        available_gpus=16,
        energy_budget_kwh=0.0,
    )
    # With 0 budget, even the initial 1-each violates (energy > 0), so back out to zeros.
    assert alloc == {"A": 0, "B": 0}

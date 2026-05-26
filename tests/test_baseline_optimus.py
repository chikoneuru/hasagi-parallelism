"""Tests for the Optimus allocator port (experiments/baselines/optimus.py).

These pin the remaining-time-min behaviour described in EuroSys'18 §4:
greedy by absolute marginal JCT reduction, no admission control, no energy
awareness.
"""
from __future__ import annotations

import math

from experiments.baselines.optimus import (
    _marginal_time_reduction,
    optimus_allocate,
    project_cluster_average_jct_s,
    project_total_energy_kwh,
    remaining_time_s,
)
from hise.admission.energy_profile import linear_profile

# --- remaining_time_s ---


def test_remaining_time_zero_iters() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    assert remaining_time_s(prof, iterations=0, gpus=4) == 0.0


def test_remaining_time_zero_gpus_is_inf() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    assert remaining_time_s(prof, iterations=100, gpus=0) == math.inf


def test_remaining_time_strictly_decreasing_in_gpus() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    times = [remaining_time_s(prof, iterations=1000, gpus=g) for g in range(1, 9)]
    assert all(times[i] > times[i + 1] for i in range(len(times) - 1))


# --- _marginal_time_reduction ---


def test_marginal_saturated_returns_neg_inf() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    assert _marginal_time_reduction(prof, iterations=1000, current_gpus=4) == -math.inf


def test_marginal_diminishing_returns() -> None:
    """Concave throughput ⇒ absolute time-reduction shrinks as allocation grows."""
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    deltas = [_marginal_time_reduction(prof, iterations=1000, current_gpus=g) for g in range(1, 8)]
    assert all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))


# --- optimus_allocate ---


def test_two_identical_jobs_split_evenly() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = optimus_allocate(
        jobs=[("A", prof, 1000), ("B", prof, 1000)],
        available_gpus=8,
    )
    assert sum(alloc.values()) <= 8
    assert alloc["A"] == alloc["B"]


def test_longer_job_gets_more_gpus() -> None:
    """Absolute-time priority ⇒ the 10× longer job dominates allocation."""
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = optimus_allocate(
        jobs=[("short", prof, 100), ("long", prof, 10_000)],
        available_gpus=8,
    )
    assert alloc["long"] > alloc["short"]


def test_respects_gpu_budget() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = optimus_allocate(
        jobs=[("A", prof, 1000), ("B", prof, 1000), ("C", prof, 1000)],
        available_gpus=6,
    )
    assert sum(alloc.values()) <= 6
    assert all(alloc[j] >= 1 for j in ("A", "B", "C"))


def test_respects_per_job_max_gpus() -> None:
    p_small = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=2)
    p_big = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = optimus_allocate(
        jobs=[("small", p_small, 1000), ("big", p_big, 1000)],
        available_gpus=8,
    )
    assert alloc["small"] <= 2
    assert alloc["big"] <= 8


def test_empty_jobs_returns_empty() -> None:
    assert optimus_allocate(jobs=[], available_gpus=8) == {}


def test_infeasible_initial_returns_zeros() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = optimus_allocate(
        jobs=[("A", prof, 1000), ("B", prof, 1000)],
        available_gpus=1,
    )
    assert alloc == {"A": 0, "B": 0}


def test_supports_initial_allocation_override() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    alloc = optimus_allocate(
        jobs=[("A", prof, 1000), ("B", prof, 1000)],
        available_gpus=8,
        initial_allocation={"A": 3, "B": 1},
    )
    assert alloc["A"] >= 3
    assert alloc["B"] >= 1


# --- projections ---


def test_project_cluster_jct_matches_manual() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [("A", prof, 1000), ("B", prof, 2000)]
    alloc = {"A": 2, "B": 4}
    avg = project_cluster_average_jct_s(jobs, alloc)
    expected = (1000 / prof.throughput(2) + 2000 / prof.throughput(4)) / 2
    assert abs(avg - expected) < 1e-9


def test_project_total_energy_matches_manual() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [("A", prof, 1000), ("B", prof, 2000)]
    alloc = {"A": 2, "B": 4}
    e = project_total_energy_kwh(jobs, alloc)
    expected = prof.energy_per_iter(2) * 1000 + prof.energy_per_iter(4) * 2000
    assert abs(e - expected) < 1e-12


def test_project_total_energy_skips_zero_gpus() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    jobs = [("A", prof, 1000), ("B", prof, 1000)]
    e = project_total_energy_kwh(jobs, {"A": 0, "B": 2})
    assert e == prof.energy_per_iter(2) * 1000

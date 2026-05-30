"""Tests for the Zeus per-job optimiser port (experiments/baselines/zeus.py)."""
from __future__ import annotations

import math

import pytest

from experiments.baselines.zeus import zeus_optimize_one_job, zeus_schedule
from hasagi.admission.energy_profile import linear_profile

# --- zeus_optimize_one_job ---


def _zeus_profile():
    return linear_profile(
        power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8,
    )


def test_eta_zero_picks_min_energy() -> None:
    """η=0 → pure energy minimization → the GPU count with the lowest E_per_iter."""
    p = _zeus_profile()
    g, ttc, e = zeus_optimize_one_job(p, iterations=1000, eta=0.0)
    # Linear profile with α=0.05 is monotone-increasing in E_per_iter; min energy
    # is at g=1.
    assert g == 1
    assert ttc == pytest.approx(1000 / p.throughput(1))
    assert e == pytest.approx(p.energy_per_iter(1) * 1000)


def test_eta_one_picks_min_time() -> None:
    """η=1 → pure time minimization → the maximum GPU count."""
    p = _zeus_profile()
    g, ttc, e = zeus_optimize_one_job(p, iterations=1000, eta=1.0)
    assert g == p.max_gpus
    assert ttc == pytest.approx(1000 / p.throughput(p.max_gpus))


def test_eta_intermediate_picks_pareto_middle() -> None:
    """η=0.5 should pick a GPU count strictly between the η=0 and η=1 extremes."""
    p = _zeus_profile()
    g_min_e = zeus_optimize_one_job(p, iterations=1000, eta=0.0)[0]
    g_min_t = zeus_optimize_one_job(p, iterations=1000, eta=1.0)[0]
    g_mid = zeus_optimize_one_job(p, iterations=1000, eta=0.5)[0]
    assert g_min_e < g_mid < g_min_t


def test_monotonicity_in_eta() -> None:
    """As η increases from 0 to 1, the chosen GPU count must be non-decreasing."""
    p = _zeus_profile()
    last_g = 0
    for eta in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        g, _, _ = zeus_optimize_one_job(p, iterations=1000, eta=eta)
        assert g >= last_g
        last_g = g


def test_deadline_filters_infeasible_choices() -> None:
    """A tight deadline restricts Zeus to allocations whose TTC fits."""
    p = _zeus_profile()
    # At g=1, TTC = 1000/10 = 100s. At g=8, TTC ≈ 17.1s. Deadline=20s rules out g≤4.
    g, ttc, _ = zeus_optimize_one_job(
        p, iterations=1000, eta=0.0, deadline_seconds=20.0,
    )
    assert ttc <= 20.0
    assert g >= 5    # must use ≥5 GPUs to fit


def test_deadline_unmeetable_falls_back_to_min_cost() -> None:
    """If no allocation meets the deadline, return the unconstrained min-cost pick."""
    p = _zeus_profile()
    g, ttc, _ = zeus_optimize_one_job(
        p, iterations=10_000, eta=0.5, deadline_seconds=1.0,
    )
    assert g >= 1                  # something is returned
    assert ttc > 1.0               # deadline is breached


def test_rejects_invalid_eta() -> None:
    p = _zeus_profile()
    with pytest.raises(ValueError, match="eta"):
        zeus_optimize_one_job(p, iterations=100, eta=-0.1)
    with pytest.raises(ValueError, match="eta"):
        zeus_optimize_one_job(p, iterations=100, eta=1.5)


def test_zero_iterations_returns_zero() -> None:
    p = _zeus_profile()
    g, ttc, e = zeus_optimize_one_job(p, iterations=0, eta=0.5)
    assert (g, ttc, e) == (0, 0.0, 0.0)


# --- zeus_schedule ---


def test_schedule_two_jobs_fits_in_cluster() -> None:
    p = _zeus_profile()
    alloc, rejected = zeus_schedule(
        jobs=[("A", p, 1000), ("B", p, 1000)],
        available_gpus=8,
        eta=0.5,
    )
    assert sum(alloc.values()) <= 8
    assert rejected == ()
    assert alloc["A"] == alloc["B"]


def test_schedule_evicts_largest_first_on_overflow() -> None:
    """When sum of Zeus picks exceeds budget, eviction drops the biggest pick."""
    p = _zeus_profile()
    # η=1 → each job wants max_gpus=8. Available budget=10 < 16 → one must be evicted.
    alloc, rejected = zeus_schedule(
        jobs=[("A", p, 1000), ("B", p, 1000)],
        available_gpus=10,
        eta=1.0,
    )
    assert len(rejected) == 1
    assert sum(alloc.values()) <= 10


def test_schedule_respects_per_job_deadline() -> None:
    p = _zeus_profile()
    alloc, _ = zeus_schedule(
        jobs=[("fast", p, 1000), ("slow", p, 1000)],
        available_gpus=16,
        eta=0.0,
        deadlines={"fast": 20.0, "slow": math.inf},
    )
    # fast must use ≥5 GPUs (TTC ≤ 20s), slow stays at the energy minimum (1 GPU)
    assert alloc["fast"] >= 5
    assert alloc["slow"] == 1

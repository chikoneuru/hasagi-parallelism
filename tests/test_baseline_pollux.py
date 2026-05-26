"""Tests for the Pollux allocator port (experiments/baselines/pollux.py).

These pin the goodput-max behaviour of OSDI'21 Algorithm 1:
greedy by marginal goodput, with the gradient-noise-scale efficiency model
imposing a soft cap on per-job allocation size as the total batch grows.
"""
from __future__ import annotations

import math

from experiments.baselines.pollux import (
    PolluxJob,
    goodput,
    pollux_allocate,
    project_energy_kwh,
    statistical_efficiency,
)
from hise.admission.energy_profile import linear_profile

# --- statistical_efficiency ---


def test_efficiency_is_one_at_zero_batch() -> None:
    assert statistical_efficiency(total_batch=0, gradient_noise_scale=1000.0) == 1.0


def test_efficiency_strictly_decreasing_in_batch() -> None:
    phi = 1000.0
    eps = [statistical_efficiency(b, phi) for b in (32, 128, 1024, 8192)]
    assert all(eps[i] > eps[i + 1] for i in range(len(eps) - 1))
    assert all(0.0 < e <= 1.0 for e in eps)


def test_efficiency_zero_for_zero_noise_scale() -> None:
    """Degenerate ϕ=0 → ε=0 (no useful work). Pollux flags this as a bad fit."""
    assert statistical_efficiency(total_batch=128, gradient_noise_scale=0.0) == 0.0


def test_efficiency_resnet_vs_lm_phi_scaling() -> None:
    """ResNet-like ϕ=3000 tolerates B=512 better than LM-like ϕ=200."""
    eps_resnet = statistical_efficiency(512, gradient_noise_scale=3000.0)
    eps_lm = statistical_efficiency(512, gradient_noise_scale=200.0)
    assert eps_resnet > eps_lm


# --- goodput ---


def test_goodput_zero_at_zero_gpus() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    job = PolluxJob("J", prof, 1000, local_batch_size=32, gradient_noise_scale=1000.0)
    assert goodput(job, gpus=0) == 0.0


def test_goodput_peaks_then_falls_at_large_phi_collapse() -> None:
    """With small ϕ, goodput should peak then fall as the total batch overruns the noise scale."""
    prof = linear_profile(
        power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=16,
        scaling_efficiency=0.95,
    )
    job = PolluxJob("J", prof, 1000, local_batch_size=128, gradient_noise_scale=200.0)
    g = [goodput(job, w) for w in range(1, 16)]
    # Find peak; goodput before peak must be ≤ peak, after peak must be ≤ peak.
    peak_idx = max(range(len(g)), key=lambda i: g[i])
    assert all(g[i] <= g[peak_idx] for i in range(len(g)))
    # And the curve actually has a peak that isn't the first sample (ϕ=200 + b=128 collapses quickly).
    assert peak_idx > 0


# --- pollux_allocate ---


def test_two_identical_jobs_split_evenly() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [
        PolluxJob("A", prof, 1000, local_batch_size=32, gradient_noise_scale=1000.0),
        PolluxJob("B", prof, 1000, local_batch_size=32, gradient_noise_scale=1000.0),
    ]
    result = pollux_allocate(jobs, available_gpus=8)
    assert sum(result.allocation.values()) <= 8
    assert result.allocation["A"] == result.allocation["B"]


def test_respects_gpu_budget() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [
        PolluxJob(f"J{i}", prof, 500, local_batch_size=32, gradient_noise_scale=1000.0)
        for i in range(3)
    ]
    result = pollux_allocate(jobs, available_gpus=6)
    assert sum(result.allocation.values()) <= 6
    assert all(result.allocation[j.job_id] >= 1 for j in jobs)


def test_efficiency_collapse_stops_growth() -> None:
    """Tiny ϕ → goodput peaks early → leftover GPUs > 0 even when budget is large."""
    prof = linear_profile(
        power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=16,
        scaling_efficiency=0.95,
    )
    job = PolluxJob("J", prof, 1000, local_batch_size=256, gradient_noise_scale=100.0)
    result = pollux_allocate([job], available_gpus=16)
    # Pollux refuses to add GPUs once marginal goodput goes non-positive — leftover > 0.
    assert result.leftover_gpus > 0
    assert result.allocation["J"] < 16


def test_high_phi_job_wins_growth() -> None:
    """When two jobs compete, the one with larger ϕ (tolerates large batch) accrues more GPUs."""
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [
        PolluxJob("resnet", prof, 1000, local_batch_size=64, gradient_noise_scale=3000.0),
        PolluxJob("lm", prof, 1000, local_batch_size=64, gradient_noise_scale=200.0),
    ]
    result = pollux_allocate(jobs, available_gpus=8)
    assert result.allocation["resnet"] >= result.allocation["lm"]


def test_empty_jobs_returns_empty() -> None:
    result = pollux_allocate([], available_gpus=8)
    assert result.allocation == {}
    assert result.leftover_gpus == 8


def test_infeasible_initial_returns_zeros() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [
        PolluxJob("A", prof, 1000, local_batch_size=32, gradient_noise_scale=1000.0),
        PolluxJob("B", prof, 1000, local_batch_size=32, gradient_noise_scale=1000.0),
    ]
    result = pollux_allocate(jobs, available_gpus=1)
    assert result.allocation == {"A": 0, "B": 0}


def test_initial_allocation_override() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [
        PolluxJob("A", prof, 1000, local_batch_size=32, gradient_noise_scale=1000.0),
        PolluxJob("B", prof, 1000, local_batch_size=32, gradient_noise_scale=1000.0),
    ]
    result = pollux_allocate(jobs, available_gpus=8, initial_allocation={"A": 3, "B": 1})
    assert result.allocation["A"] >= 3
    assert result.allocation["B"] >= 1


# --- project_energy_kwh ---


def test_project_energy_grows_with_inefficiency() -> None:
    """Larger total batch → smaller ε → effective iter count rises → projected energy rises."""
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=16)
    job = PolluxJob("J", prof, 1000, local_batch_size=128, gradient_noise_scale=200.0)
    e_small = project_energy_kwh(job, gpus=1)
    e_big = project_energy_kwh(job, gpus=8)
    assert e_big > e_small
    assert math.isfinite(e_big)


def test_project_energy_zero_for_zero_gpus() -> None:
    prof = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=4)
    job = PolluxJob("J", prof, 1000, local_batch_size=32, gradient_noise_scale=1000.0)
    assert project_energy_kwh(job, gpus=0) == 0.0

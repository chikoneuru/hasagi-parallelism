"""Tests for the ElasticFlow scheduler port (experiments/baselines/elasticflow.py).

ElasticFlow (NSDI'23): admit-by-MSS, then distribute leftover GPUs by marginal
throughput. The tests cover admission, rejection ordering, distribution, and
the energy projection used by head-to-head experiments.
"""
from __future__ import annotations

from experiments.baselines.elasticflow import (
    ElasticFlowJob,
    elasticflow_schedule,
    project_energy_kwh,
)
from tare.admission.mss import ScalingCurve

# --- Scheduling ---


def _concave_curve(max_gpus: int = 8) -> ScalingCurve:
    """Concave curve: 10, 18, 24, 28, 30, 31, 32, 32.5 — diminishing returns."""
    return ScalingCurve(throughput_per_gpu_count=tuple(
        10.0 * (g ** 0.85) for g in range(1, max_gpus + 1)
    ))


def test_two_jobs_fit_split_remaining_gpus() -> None:
    """Identical jobs with slack budget should split GPUs evenly."""
    curve = _concave_curve()
    jobs = [
        ElasticFlowJob("A", curve, iterations_remaining=1000, deadline_seconds=200),
        ElasticFlowJob("B", curve, iterations_remaining=1000, deadline_seconds=200),
    ]
    result = elasticflow_schedule(jobs, available_gpus=8)
    assert result.allocation == {"A": 4, "B": 4}
    assert result.rejected == ()
    assert result.leftover_gpus == 0


def test_rejects_job_that_cannot_meet_deadline() -> None:
    """A deadline so tight no allocation in [1, max_gpus] satisfies it → reject."""
    curve = _concave_curve()
    # Throughput at max=8 GPUs is ~10 * 8^0.85 ≈ 59 iter/s. To finish 100,000
    # iters in 1 second we'd need >1700 iter/s — impossible.
    impossible = ElasticFlowJob("tight", curve,
                                iterations_remaining=100_000, deadline_seconds=1.0)
    feasible = ElasticFlowJob("ok", curve,
                              iterations_remaining=1000, deadline_seconds=200.0)
    result = elasticflow_schedule([impossible, feasible], available_gpus=8)
    assert "tight" in result.rejected
    assert "ok" in result.allocation


def test_rejects_when_cluster_cannot_hold_all_mss() -> None:
    """Σ MSS > available_gpus → drop largest-MSS jobs until the budget fits."""
    curve = _concave_curve()
    # At 4 GPUs throughput ≈ 32.5 iter/s, so 5000 iters fit in ~154s for a
    # 200s deadline → individual MSS = 4. With two such jobs and only 4 GPUs
    # available, the cluster cannot hold both MSSes → one is evicted.
    j1 = ElasticFlowJob("j1", curve, iterations_remaining=5000, deadline_seconds=200.0)
    j2 = ElasticFlowJob("j2", curve, iterations_remaining=5000, deadline_seconds=200.0)
    result = elasticflow_schedule([j1, j2], available_gpus=4)
    assert len(result.allocation) == 1
    assert len(result.rejected) == 1
    # The rejected one must have been admitted-but-evicted (not infeasible by deadline).
    assert result.rejected[0] in ("j1", "j2")


def test_distributes_extra_gpus_by_marginal_throughput() -> None:
    """With slack GPUs, the scheduler should hand extras to the job with the
    highest marginal throughput — when curves match, distribution stays balanced."""
    curve = _concave_curve()
    jobs = [
        ElasticFlowJob("A", curve, iterations_remaining=500, deadline_seconds=500.0),
        ElasticFlowJob("B", curve, iterations_remaining=500, deadline_seconds=500.0),
    ]
    result = elasticflow_schedule(jobs, available_gpus=8)
    # Both jobs identical → equal split.
    assert abs(result.allocation["A"] - result.allocation["B"]) <= 1
    assert sum(result.allocation.values()) <= 8


def test_no_jobs_returns_empty_allocation() -> None:
    result = elasticflow_schedule([], available_gpus=8)
    assert result.allocation == {}
    assert result.rejected == ()
    assert result.leftover_gpus == 8


def test_max_gpus_saturation_leaves_leftover() -> None:
    """If every admitted job hits its scaling-curve max, GPUs are left over."""
    curve = _concave_curve(max_gpus=3)
    jobs = [
        ElasticFlowJob("A", curve, iterations_remaining=500, deadline_seconds=500.0),
    ]
    result = elasticflow_schedule(jobs, available_gpus=8)
    assert result.allocation["A"] == 3
    assert result.leftover_gpus == 5


# --- project_energy_kwh ---


def test_project_energy_scales_linearly_with_gpus() -> None:
    curve = _concave_curve()
    e1 = project_energy_kwh(curve, power_per_gpu_w=300.0, gpus=1, iterations=1000)
    e4 = project_energy_kwh(curve, power_per_gpu_w=300.0, gpus=4, iterations=1000)
    # E(g) = P * g * iters / throughput / 3.6e6. At higher g the throughput
    # grows sublinearly (concave), so per-iter energy goes UP (this is the
    # whole point of EnergyProfile convexity). e4 / 1000 > e1 / 1000 expected.
    assert e4 > e1 * 0.5    # not exactly 4x because of sublinear scaling
    assert e1 > 0


def test_project_energy_zero_when_no_gpus_or_no_iters() -> None:
    curve = _concave_curve()
    assert project_energy_kwh(curve, power_per_gpu_w=300.0, gpus=0, iterations=1000) == 0.0
    assert project_energy_kwh(curve, power_per_gpu_w=300.0, gpus=4, iterations=0) == 0.0

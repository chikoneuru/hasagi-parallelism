"""ElasticFlow scheduler port (He et al., NSDI'23) — throughput-max baseline.

ElasticFlow's two-stage scheduler:

    1. **Admission**: for each job, compute its Minimum Satisfactory Share (MSS) —
       the smallest GPU count whose throughput satisfies the deadline. Reject if
       no allocation in `[1, max_gpus]` meets the deadline.
    2. **Distribution**: hand the remaining GPUs to the job with the highest
       marginal throughput gain (ΔT per added GPU). Continue until the GPU pool
       is exhausted or every admitted job has saturated.

Both stages are already implemented as standalone primitives in
``hasagi.admission.mss`` (`minimum_satisfactory_share`, `greedy_marginal_allocation`).
This module composes them into the canonical ElasticFlow scheduler interface so
it can be invoked side-by-side with the PowerFlow port for head-to-head experiments.

Reference:
    He, Choi, Mao, Mei, Stoica, "ElasticFlow: An Elastic Serverless Training
    Platform for Distributed Deep Learning," NSDI 2023.

Compared to HASAGI EB:
    - ElasticFlow MAXIMISES throughput under deadline.
    - HASAGI EB MINIMISES energy under deadline + energy budget.

The two are duals on the throughput–energy Pareto frontier, but with no shared
energy budget ElasticFlow walks toward the high-throughput / high-energy end.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hasagi.admission.mss import (
    ScalingCurve,
    greedy_marginal_allocation,
    minimum_satisfactory_share,
)


@dataclass(frozen=True)
class ElasticFlowJob:
    """One job admitted to the ElasticFlow scheduler."""

    job_id: str
    curve: ScalingCurve
    iterations_remaining: int
    deadline_seconds: float


@dataclass(frozen=True)
class ElasticFlowResult:
    """Outcome of one ``elasticflow_schedule`` call."""

    allocation: dict[str, int]                 # admitted job_id → GPU count
    rejected: tuple[str, ...]                  # job_ids whose deadline cannot be met
    leftover_gpus: int                         # GPUs not handed out (max-cap or saturation)


def elasticflow_schedule(
    jobs: Sequence[ElasticFlowJob],
    available_gpus: int,
) -> ElasticFlowResult:
    """Two-stage ElasticFlow scheduling: admit-by-MSS → distribute by marginal throughput.

    Args:
        jobs: jobs competing for GPUs. Each must have ``curve`` (concave throughput
            scaling), ``iterations_remaining``, and ``deadline_seconds``.
        available_gpus: total cluster GPU budget.

    Returns:
        ``ElasticFlowResult`` with the admitted allocation, the rejected job_ids,
        and the count of GPUs not allocated. Rejected jobs do not appear in the
        allocation dict.
    """
    mss_per_job: dict[str, int] = {}
    rejected: list[str] = []
    for job in jobs:
        mss = minimum_satisfactory_share(
            iterations_remaining=job.iterations_remaining,
            deadline_seconds=job.deadline_seconds,
            curve=job.curve,
        )
        if mss == 0:
            rejected.append(job.job_id)
        else:
            mss_per_job[job.job_id] = mss

    if sum(mss_per_job.values()) > available_gpus:
        # Cluster cannot satisfy the union of admitted MSS — ElasticFlow's
        # admission control rejects in order until the budget fits. Reject
        # largest MSS first (drop the most expensive job).
        ordered = sorted(mss_per_job.items(), key=lambda item: -item[1])
        for jid, _ in ordered:
            if sum(mss_per_job.values()) <= available_gpus:
                break
            rejected.append(jid)
            mss_per_job.pop(jid)

    if not mss_per_job:
        return ElasticFlowResult(allocation={}, rejected=tuple(rejected),
                                 leftover_gpus=available_gpus)

    curves = {j.job_id: j.curve for j in jobs if j.job_id in mss_per_job}
    admitted_list: list[tuple[str, ScalingCurve, int]] = [
        (jid, curves[jid], mss_per_job[jid]) for jid in mss_per_job
    ]
    final_alloc = greedy_marginal_allocation(admitted=admitted_list,
                                             available_gpus=available_gpus)
    leftover = available_gpus - sum(final_alloc.values())
    return ElasticFlowResult(
        allocation=final_alloc,
        rejected=tuple(rejected),
        leftover_gpus=max(0, leftover),
    )


def project_energy_kwh(
    curve: ScalingCurve,
    power_per_gpu_w: float,
    gpus: int,
    iterations: int,
) -> float:
    """Linear power-model projection: ``P_per_gpu × gpus × iters / throughput / 3.6e6``.

    The ElasticFlow paper has no energy notion, so we measure the energy
    *consequence* of its allocation under HASAGI's same linear power model — the
    fair-comparison baseline against ``hasagi.admission.mss.EnergyBudgetMSS``.
    """
    if gpus <= 0 or iterations <= 0:
        return 0.0
    throughput = curve.throughput(gpus)
    if throughput <= 0:
        return math.inf
    duration_s = iterations / throughput
    return (power_per_gpu_w * gpus * duration_s) / 3_600_000.0

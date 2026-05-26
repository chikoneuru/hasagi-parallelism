"""Pollux scheduler port (Qiao et al., OSDI'21) — goodput-max baseline.

Pollux observes that raw throughput is a poor optimisation target for DNN
training because the *useful* progress per second depends on the statistical
efficiency of each gradient step, which is degraded by very large total batch
sizes. The cluster-level objective is therefore **goodput**:

    goodput(w, B) = throughput(w, B) · statistical_efficiency(B)

where ``w`` is worker count and ``B`` the total batch size. Under data
parallelism with a fixed per-GPU local batch ``b0``, ``B(w) = b0 · w``, so the
statistical-efficiency term is a closed-form function of ``w``:

    ε(B) = ϕ / (ϕ + B)                  (gradient-noise-scale model, §3.2)

Larger ``ϕ`` (noisy gradients, ResNet-like) tolerates large batches well;
smaller ``ϕ`` (sharp landscapes, large LMs at small token counts) sees ε
collapse quickly. The Pollux scheduler picks ``w`` per job to maximise
cluster-wide ``Σ goodput_j``.

Port shape: greedy marginal goodput allocator over an HISE ``EnergyProfile``
so it can be compared side-by-side with PowerFlow, ElasticFlow, Zeus, and
HISE EB-MSS. Energy is *not* in Pollux's objective; we report it as a
downstream consequence using the same profile, the same way the ElasticFlow
port does.

Reference:
    Qiao, Choe, Subramanya, Neiswanger, Ho, Zhang, Ganger, Xing, "Pollux:
    Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning,"
    OSDI 2021. Goodput definition is §3.2; the scheduler is Algorithm 1 §4.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hise.admission.energy_profile import EnergyProfile


@dataclass(frozen=True)
class PolluxJob:
    """One job admitted to the Pollux scheduler.

    Args:
        job_id: stable identifier.
        profile: ``EnergyProfile`` indexed by GPU count.
        iterations_remaining: nominal (efficiency=1) iterations still to compute.
        local_batch_size: per-GPU mini-batch; total batch is ``local_batch · w``.
        gradient_noise_scale: ``ϕ`` in the goodput model. ResNet-50 fits ϕ≈3000
            on ImageNet per the Pollux paper Table 1; LMs fit ϕ≈100-500.
    """

    job_id: str
    profile: EnergyProfile
    iterations_remaining: int
    local_batch_size: int
    gradient_noise_scale: float


@dataclass(frozen=True)
class PolluxResult:
    """Outcome of one ``pollux_allocate`` call."""

    allocation: dict[str, int]
    goodput_per_job: dict[str, float]
    leftover_gpus: int


def statistical_efficiency(total_batch: int, gradient_noise_scale: float) -> float:
    """Pollux Eq. (1): ε(B) = ϕ / (ϕ + B). Always in (0, 1]."""
    if total_batch <= 0:
        return 1.0
    if gradient_noise_scale <= 0:
        return 0.0
    return gradient_noise_scale / (gradient_noise_scale + total_batch)


def goodput(job: PolluxJob, gpus: int) -> float:
    """Useful iterations per second at allocation ``gpus``.

    ``goodput = throughput · ε(B(gpus))``. The throughput comes from the
    energy profile (so allreduce overhead is baked in); the efficiency
    discounts iterations whose effective gradient contribution shrinks as the
    total batch grows.
    """
    if gpus <= 0:
        return 0.0
    raw = job.profile.throughput(gpus)
    if raw <= 0:
        return 0.0
    total_batch = job.local_batch_size * gpus
    return raw * statistical_efficiency(total_batch, job.gradient_noise_scale)


def _marginal_goodput(job: PolluxJob, current_gpus: int) -> float:
    """Goodput delta from adding one GPU; -inf when saturated or negative."""
    if current_gpus >= job.profile.max_gpus:
        return -math.inf
    return goodput(job, current_gpus + 1) - goodput(job, current_gpus)


def pollux_allocate(
    jobs: Sequence[PolluxJob],
    available_gpus: int,
    *,
    initial_allocation: dict[str, int] | None = None,
) -> PolluxResult:
    """Greedy goodput-maximising allocator over a fixed GPU budget.

    Algorithm: start each job at 1 GPU (or the supplied ``initial_allocation``),
    then repeatedly hand the next GPU to the job whose marginal goodput delta is
    largest. Stops when GPUs run out or every job has non-positive marginal
    goodput (the efficiency-collapse regime where adding a worker actively
    hurts cluster goodput).

    Args:
        jobs: jobs competing for GPUs. Each ``PolluxJob`` bundles the energy
            profile, iteration count, local batch, and gradient noise scale.
        available_gpus: total cluster GPU budget.
        initial_allocation: optional per-job starting allocation (default 1 per job).

    Returns:
        ``PolluxResult`` with the per-job allocation, per-job goodput, and the
        leftover GPU count (>0 when efficiency-collapse halts allocation early).
    """
    if not jobs:
        return PolluxResult(allocation={}, goodput_per_job={}, leftover_gpus=available_gpus)

    job_by_id = {j.job_id: j for j in jobs}
    if initial_allocation is None:
        alloc = {jid: 1 for jid in job_by_id}
    else:
        alloc = dict(initial_allocation)
        for jid in job_by_id:
            alloc.setdefault(jid, 1)

    if sum(alloc.values()) > available_gpus:
        return PolluxResult(
            allocation={jid: 0 for jid in job_by_id},
            goodput_per_job={jid: 0.0 for jid in job_by_id},
            leftover_gpus=available_gpus,
        )

    remaining = available_gpus - sum(alloc.values())
    while remaining > 0:
        best_jid: str | None = None
        best_gain = 0.0
        for jid, job in job_by_id.items():
            gain = _marginal_goodput(job, alloc[jid])
            if gain > best_gain:
                best_gain = gain
                best_jid = jid
        if best_jid is None or best_gain <= 0:
            break
        alloc[best_jid] += 1
        remaining -= 1

    goodput_map = {jid: goodput(job, alloc[jid]) for jid, job in job_by_id.items()}
    return PolluxResult(
        allocation=alloc,
        goodput_per_job=goodput_map,
        leftover_gpus=remaining,
    )


def project_energy_kwh(
    job: PolluxJob, gpus: int,
) -> float:
    """Total energy to complete the *useful* work at ``gpus``.

    Pollux's accounting trick: at smaller statistical efficiency, the job needs
    more nominal iterations to converge. The fair energy projection is therefore
    ``energy_per_iter(gpus) · (nominal_iters / ε(B))`` — adding workers may
    raise raw throughput while the *effective* energy budget swells.
    """
    if gpus <= 0:
        return 0.0
    eps = statistical_efficiency(job.local_batch_size * gpus, job.gradient_noise_scale)
    if eps <= 0:
        return math.inf
    effective_iters = job.iterations_remaining / eps
    return job.profile.energy_per_iter(gpus) * effective_iters

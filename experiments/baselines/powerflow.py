"""PowerFlow allocator (Gu et al., arXiv 2304.06381) — port for head-to-head comparison.

PowerFlow's Algorithm 1 is a greedy multi-job GPU allocator under a cluster-wide
energy budget. The priority each iteration is

    priority_j = (ΔJCT / JCT) / (ΔE / E)

i.e. *relative job-completion-time reduction per relative energy increase* — the
allocator hands the next GPU to the job whose next GPU yields the largest such
ratio. PowerFlow also varies DVFS frequency; this port omits the freq knob and
keeps only the GPU-count one so the comparison vs HISE EB-MSS is apples-to-apples
(HISE controls the NVML power cap externally and does not modulate freq directly).

Inputs use HISE's ``EnergyProfile`` so per-iter throughput and energy come from the
same fitted curve EB-MSS sees. Output is ``dict[job_id, allocated_gpus]``.

Reference:
    Gu, Xie, Huang, Jin, Liu, "Energy-Efficient GPU Clusters Scheduling for Deep
    Learning," arXiv 2304.06381v2, May 2023. Algorithm 1 is in §5.2.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from hise.admission.energy_profile import EnergyProfile


def _jct_and_energy(profile: EnergyProfile, gpus: int, iterations: int) -> tuple[float, float]:
    """Job completion time (seconds) and total energy (kWh) at allocation ``gpus``."""
    if gpus <= 0 or iterations <= 0:
        return math.inf, 0.0
    throughput = profile.throughput(gpus)
    if throughput <= 0:
        return math.inf, 0.0
    jct = iterations / throughput
    energy = profile.energy_per_iter(gpus) * iterations
    return jct, energy


def _relative_priority(
    profile: EnergyProfile, current_gpus: int, iterations: int,
) -> float:
    """PowerFlow Eq. (16-17) priority for adding one GPU to a job at allocation `current_gpus`.

    Returns -infinity if the job is saturated (no room to grow) or if adding a GPU
    yields zero / negative throughput delta. Returns +infinity if `ΔE / E` is zero
    (an "always pick this" condition — extra GPU is effectively free in energy terms).
    """
    if current_gpus >= profile.max_gpus:
        return -math.inf
    jct_now, e_now = _jct_and_energy(profile, current_gpus, iterations)
    jct_next, e_next = _jct_and_energy(profile, current_gpus + 1, iterations)
    if not math.isfinite(jct_now) or not math.isfinite(jct_next):
        return -math.inf
    delta_jct_rel = (jct_now - jct_next) / jct_now if jct_now > 0 else 0.0
    if delta_jct_rel <= 0:
        return -math.inf
    if e_now <= 0:
        return math.inf
    delta_e_rel = (e_next - e_now) / e_now
    if delta_e_rel <= 0:
        return math.inf
    return delta_jct_rel / delta_e_rel


def powerflow_allocate(
    jobs: Sequence[tuple[str, EnergyProfile, int]],
    available_gpus: int,
    *,
    energy_budget_kwh: float = math.inf,
    initial_allocation: dict[str, int] | None = None,
) -> dict[str, int]:
    """Greedy GPU allocator following PowerFlow Algorithm 1.

    Each ``jobs`` entry is ``(job_id, energy_profile, iterations_remaining)``.

    Args:
        jobs: jobs competing for GPUs.
        available_gpus: total GPU budget across all jobs.
        energy_budget_kwh: cluster-wide energy budget (sum of projected E_j).
            Default ``inf`` disables the constraint, matching the GPU-only variant
            used when comparing against allocators with deadlines.
        initial_allocation: per-job starting allocation (default 1 each). Used to
            mimic PowerFlow's behaviour of re-allocating from a running state.

    Returns:
        ``dict[job_id, allocated_gpus]`` with ``Σ alloc[j] ≤ available_gpus`` and
        ``Σ E_j(alloc[j]) ≤ energy_budget_kwh``.

    Notes:
        - Jobs whose ``EnergyProfile`` does not satisfy ``validate_convexity()``
          still work — PowerFlow's algorithm makes no convexity assumption, the
          priority is computed pointwise.
        - Ties broken by job_id (stable insertion order).
    """
    profiles = {jid: prof for jid, prof, _iters in jobs}
    iters_remaining = {jid: it for jid, _prof, it in jobs}

    if initial_allocation is None:
        alloc = {jid: 1 for jid in profiles}
    else:
        alloc = dict(initial_allocation)
        for jid in profiles:
            alloc.setdefault(jid, 1)

    def _total_energy_kwh(a: dict[str, int]) -> float:
        return sum(profiles[j].energy_per_iter(a[j]) * iters_remaining[j] for j in a)

    # If the initial allocation already exceeds either constraint, no allocation is feasible
    # — return zeros and let the caller decide what to do.
    if sum(alloc.values()) > available_gpus:
        return {jid: 0 for jid in profiles}
    if _total_energy_kwh(alloc) > energy_budget_kwh:
        return {jid: 0 for jid in profiles}

    remaining_gpus = available_gpus - sum(alloc.values())
    while remaining_gpus > 0:
        best_jid: str | None = None
        best_priority = -math.inf
        for jid in profiles:
            p = _relative_priority(profiles[jid], alloc[jid], iters_remaining[jid])
            if p > best_priority:
                best_priority = p
                best_jid = jid
        if best_jid is None or best_priority <= 0:
            break

        # Tentatively assign; back out if budget violated.
        alloc[best_jid] += 1
        if _total_energy_kwh(alloc) > energy_budget_kwh:
            alloc[best_jid] -= 1
            break
        remaining_gpus -= 1

    return alloc

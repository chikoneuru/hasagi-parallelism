"""Optimus scheduler port (Peng et al., EuroSys'18) — remaining-time-min baseline.

Optimus fits a per-job training-speed model from history, predicts the
remaining-time as a function of allocation, and greedily hands GPUs to the
job whose absolute remaining-time decreases most per added GPU. The cluster
objective is average job completion time:

    minimise   (1 / |J|) · Σ_j  iterations_remaining_j / throughput_j(w_j)

The marginal priority for adding one GPU to job ``j`` at allocation ``w`` is

    Δ_j(w) = iters_j · (1 / T_j(w) - 1 / T_j(w+1))

which is the *absolute* (not relative) time reduction. Long jobs and jobs on
the steep part of their scaling curve dominate the priority order — Optimus
will (perhaps unfairly) starve a short job in favour of a long one if the
long one's throughput keeps improving.

Differences from neighbouring baselines:
    - ElasticFlow: admits by MSS first, distributes by marginal *throughput*;
      Optimus has no admission control and uses marginal *remaining time*.
    - PowerFlow: marginal priority is *relative* (ΔJCT/JCT)/(ΔE/E) and includes
      an energy term; Optimus is energy-blind and absolute.
    - Pollux: marginal priority is goodput (throughput × statistical efficiency);
      Optimus uses raw throughput.

Port operates over HASAGI's ``EnergyProfile`` so the energy *consequence* of the
Optimus allocation can be compared against energy-aware baselines on a level
playing field.

Reference:
    Peng, Bao, Zhao, Chen, Wu, Guo, "Optimus: An Efficient Dynamic Resource
    Scheduler for Deep Learning Clusters," EuroSys 2018.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from hasagi.admission.energy_profile import EnergyProfile


def remaining_time_s(profile: EnergyProfile, iterations: int, gpus: int) -> float:
    """Wall-clock seconds to complete ``iterations`` at ``gpus`` allocation."""
    if gpus <= 0 or iterations <= 0:
        return 0.0 if iterations <= 0 else math.inf
    rate = profile.throughput(gpus)
    if rate <= 0:
        return math.inf
    return iterations / rate


def _marginal_time_reduction(
    profile: EnergyProfile, iterations: int, current_gpus: int,
) -> float:
    """Optimus priority: ``time(current) - time(current+1)`` in seconds.

    Returns -inf if the job is saturated (cannot grow) or if adding a GPU
    yields zero / negative throughput delta.
    """
    if current_gpus >= profile.max_gpus:
        return -math.inf
    t_now = remaining_time_s(profile, iterations, current_gpus)
    t_next = remaining_time_s(profile, iterations, current_gpus + 1)
    if not math.isfinite(t_now) or not math.isfinite(t_next):
        return -math.inf
    delta = t_now - t_next
    if delta <= 0:
        return -math.inf
    return delta


def optimus_allocate(
    jobs: Sequence[tuple[str, EnergyProfile, int]],
    available_gpus: int,
    *,
    initial_allocation: dict[str, int] | None = None,
) -> dict[str, int]:
    """Greedy allocator following Optimus §4.

    Args:
        jobs: ``(job_id, energy_profile, iterations_remaining)`` triples.
        available_gpus: cluster GPU budget.
        initial_allocation: optional per-job starting allocation (default 1 each).

    Returns:
        ``dict[job_id, allocated_gpus]`` with ``Σ alloc[j] ≤ available_gpus``.

    Notes:
        - Ties are broken by job_id ordering (stable, deterministic).
        - Optimus has no energy notion; the per-allocation energy can be
          recovered by ``profile.energy_per_iter(alloc[j]) · iters[j]`` for
          comparison plots.
    """
    profiles = {jid: prof for jid, prof, _it in jobs}
    iters = {jid: it for jid, _prof, it in jobs}

    if not profiles:
        return {}

    if initial_allocation is None:
        alloc = {jid: 1 for jid in profiles}
    else:
        alloc = dict(initial_allocation)
        for jid in profiles:
            alloc.setdefault(jid, 1)

    if sum(alloc.values()) > available_gpus:
        return {jid: 0 for jid in profiles}

    remaining = available_gpus - sum(alloc.values())
    while remaining > 0:
        best_jid: str | None = None
        best_delta = 0.0
        for jid in profiles:
            d = _marginal_time_reduction(profiles[jid], iters[jid], alloc[jid])
            if d > best_delta:
                best_delta = d
                best_jid = jid
        if best_jid is None or best_delta <= 0:
            break
        alloc[best_jid] += 1
        remaining -= 1

    return alloc


def project_cluster_average_jct_s(
    jobs: Sequence[tuple[str, EnergyProfile, int]],
    allocation: dict[str, int],
) -> float:
    """Mean per-job JCT under an allocation — the metric Optimus minimises."""
    if not jobs:
        return 0.0
    times = [
        remaining_time_s(prof, iters, allocation.get(jid, 0))
        for jid, prof, iters in jobs
    ]
    finite = [t for t in times if math.isfinite(t)]
    if not finite:
        return math.inf
    return sum(finite) / len(finite)


def project_total_energy_kwh(
    jobs: Sequence[tuple[str, EnergyProfile, int]],
    allocation: dict[str, int],
) -> float:
    """Sum of per-job projected energy at the allocation — the metric Optimus *ignores*."""
    return sum(
        prof.energy_per_iter(allocation.get(jid, 0)) * iters
        for jid, prof, iters in jobs
        if allocation.get(jid, 0) > 0
    )

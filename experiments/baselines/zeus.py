"""Zeus per-job energy/time optimiser port (You et al., NSDI'23).

Zeus picks one operating point per job on the energy / time Pareto frontier by
minimising the convex combination

    cost(η, g) = η · TTC(g) / TTC_max  +  (1-η) · E(g) / E_max

where ``g`` is the operating knob (GPU count here; in the original paper it is a
DVFS power cap), ``η ∈ [0, 1]`` is the user-set "knob-position" — η=0 is pure
energy minimisation, η=1 is pure time minimisation.

The port differs from the paper in one place: the original Zeus scans NVML
power caps on a fixed GPU count; this port scans GPU counts under the same
EnergyProfile abstraction HASAGI EB and PowerFlow consume. The trade-off curve
is preserved — adding GPUs trades energy (more parallel power draw + allreduce
overhead) for time (lower JCT) — so the per-job optimiser remains a clean
analog suitable for a head-to-head experiment.

Reference:
    You, Chung, Chowdhury, "Zeus: Understanding and Optimizing GPU Energy
    Consumption of DNN Training," NSDI 2023.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hasagi.admission.energy_profile import EnergyProfile


@dataclass(frozen=True)
class ZeusChoice:
    """One Zeus operating-point selection for a single job."""

    job_id: str
    gpus: int
    ttc_s: float
    energy_kwh: float


def zeus_optimize_one_job(
    profile: EnergyProfile,
    iterations: int,
    eta: float,
    *,
    deadline_seconds: float = math.inf,
) -> tuple[int, float, float]:
    """Pick the GPU count for one job that minimises the η-weighted cost.

    Args:
        profile: per-GPU-count energy + throughput curve.
        iterations: iterations remaining for this job.
        eta: knob ∈ [0, 1]. 0 = min energy, 1 = min time, 0.5 = balanced.
        deadline_seconds: optional deadline; allocations that overshoot are
            penalised by ``+infty`` (ignored if no allocation meets it).

    Returns:
        ``(gpus, ttc_s, energy_kwh)``. If no allocation meets the deadline,
        falls back to the deadline-violating min-cost choice so the caller can
        still see what Zeus would pick — JCT will exceed the deadline.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    if iterations <= 0:
        return 0, 0.0, 0.0

    candidates: list[tuple[int, float, float]] = []
    for g in range(1, profile.max_gpus + 1):
        t = profile.throughput(g)
        if t <= 0:
            continue
        ttc = iterations / t
        e = profile.energy_per_iter(g) * iterations
        candidates.append((g, ttc, e))
    if not candidates:
        return 0, 0.0, 0.0

    ttc_max = max(c[1] for c in candidates)
    energy_max = max(c[2] for c in candidates) or 1.0

    def _cost(c: tuple[int, float, float]) -> float:
        g, ttc, e = c
        deadline_penalty = math.inf if ttc > deadline_seconds else 0.0
        return eta * (ttc / ttc_max) + (1 - eta) * (e / energy_max) + deadline_penalty

    feasible = [c for c in candidates if c[1] <= deadline_seconds]
    pool = feasible if feasible else candidates
    best = min(pool, key=_cost)
    return best


def zeus_schedule(
    jobs: Sequence[tuple[str, EnergyProfile, int]],
    available_gpus: int,
    eta: float,
    *,
    deadlines: dict[str, float] | None = None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Multi-job Zeus scheduler — per-job optimiser + eviction on cluster overflow.

    Args:
        jobs: ``(job_id, energy_profile, iterations_remaining)`` triples.
        available_gpus: cluster GPU budget.
        eta: shared knob applied to every job (Zeus is per-user; we use the same
            η for all jobs in a comparison run).
        deadlines: optional per-job deadlines. Jobs whose Zeus-optimal allocation
            overshoots its deadline are kept but flagged in the JCT report.

    Returns:
        ``(allocation, rejected)`` where ``allocation`` is the per-job GPU count
        and ``rejected`` lists jobs evicted because the cluster cannot hold the
        sum of Zeus picks.
    """
    deadlines = deadlines or {}
    picks: dict[str, ZeusChoice] = {}
    for jid, profile, iters in jobs:
        g, ttc, e = zeus_optimize_one_job(
            profile, iters, eta,
            deadline_seconds=deadlines.get(jid, math.inf),
        )
        picks[jid] = ZeusChoice(job_id=jid, gpus=g, ttc_s=ttc, energy_kwh=e)

    # Eviction: drop largest-GPU picks first when the sum exceeds budget.
    rejected: list[str] = []
    if sum(p.gpus for p in picks.values()) > available_gpus:
        ordered = sorted(picks.items(), key=lambda kv: -kv[1].gpus)
        for jid, _ in ordered:
            if sum(p.gpus for p in picks.values()) <= available_gpus:
                break
            rejected.append(jid)
            picks.pop(jid)

    return {jid: p.gpus for jid, p in picks.items()}, tuple(rejected)

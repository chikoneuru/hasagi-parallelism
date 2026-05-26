"""Six-allocator head-to-head: HISE EB vs PowerFlow, ElasticFlow, Zeus, Pollux, Optimus.

Adds the two scheduler-level closest-competitor baselines (Pollux OSDI'21
goodput-max and Optimus EuroSys'18 remaining-time-min) to the prior four-way
sweep, so the head-to-head matrix matches the typical "why not X?" reviewer
checklist for an energy-aware serverless scheduler. The workload distribution
matches ``exp_h1e_head_to_head.py`` plus the per-job Pollux fields
``(local_batch_size, gradient_noise_scale)``.

Each seed:
  - draws one workload (n jobs × asymmetric profile shapes)
  - runs all six allocators on the same draw
  - records (energy, max JCT, deadlines met) per allocator

Aggregate output:
  - per-allocator mean ± sd on energy, max JCT, deadlines met
  - Cohen's d of HISE EB vs each baseline on the energy axis with
    Bonferroni-corrected significance threshold

Usage:
    python -m experiments.exp_scheduler_head_to_head --seeds 10
    python -m experiments.exp_scheduler_head_to_head --seeds 10 --asymmetric
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.baselines.elasticflow import ElasticFlowJob, elasticflow_schedule
from experiments.baselines.optimus import optimus_allocate
from experiments.baselines.pollux import PolluxJob, pollux_allocate
from experiments.baselines.powerflow import powerflow_allocate
from experiments.baselines.zeus import zeus_schedule
from hise.admission.energy_profile import EnergyProfile, linear_profile
from hise.admission.mss import (
    EnergyBudgetMSS,
    ScalingCurve,
    greedy_marginal_energy_allocation,
)


@dataclass
class WorkloadJob:
    job_id: str
    profile: EnergyProfile
    iterations_remaining: int
    deadline_seconds: float
    energy_budget_kwh: float
    local_batch_size: int
    gradient_noise_scale: float


@dataclass
class TrialResult:
    seed: int
    allocator: str
    total_energy_kwh: float
    max_jct_s: float
    deadlines_met: int
    n_jobs: int


def _profile_to_curve(p: EnergyProfile) -> ScalingCurve:
    return ScalingCurve(throughput_per_gpu_count=tuple(p.throughput_iters_per_s))


def _summarise(alloc: dict[str, int], jobs: list[WorkloadJob]) -> tuple[float, float, int]:
    """Project (energy_kwh, max JCT s, deadlines met) for an allocation.

    All allocators are scored on the same energy/throughput projection (the
    workload's ``EnergyProfile``) so the comparison is apples-to-apples — even
    when the allocator did not optimise for energy or for deadlines.
    """
    e = 0.0
    max_jct = 0.0
    met = 0
    for j in jobs:
        g = alloc.get(j.job_id, 0)
        if g <= 0:
            continue
        t = j.profile.throughput(g)
        if t <= 0:
            continue
        e += j.profile.energy_per_iter(g) * j.iterations_remaining
        jct = j.iterations_remaining / t
        max_jct = max(max_jct, jct)
        if jct <= j.deadline_seconds:
            met += 1
    return e, max_jct, met


def draw_workload(rng: random.Random, n_jobs: int, asymmetric: bool) -> list[WorkloadJob]:
    """Sample a workload from the experiment distribution.

    Pollux fields draw from a published gradient-noise-scale range:
        ϕ ∈ [200, 3000]  (LM-like at the low end, ResNet-like at the high end)
        local_batch ∈ {32, 64, 128}
    """
    jobs = []
    for i in range(n_jobs):
        if asymmetric:
            eta = rng.uniform(0.55, 0.95)
            alpha = rng.uniform(0.02, 0.20)
            base_t = rng.uniform(6.0, 12.0)
            power_w = rng.uniform(220.0, 360.0)
        else:
            eta = rng.uniform(0.80, 0.90)
            alpha = rng.uniform(0.04, 0.08)
            base_t = rng.uniform(9.0, 11.0)
            power_w = rng.uniform(280.0, 320.0)
        p = linear_profile(
            power_per_gpu_w=power_w,
            base_throughput_iters_per_s=base_t,
            max_gpus=8,
            scaling_efficiency=eta,
            allreduce_coefficient=alpha,
        )
        iters = rng.choice([1500, 2000, 3000])
        deadline = rng.uniform(180.0, 360.0)
        e_budget = rng.uniform(0.6, 1.2)
        local_batch = rng.choice([32, 64, 128])
        phi = rng.uniform(200.0, 3000.0)
        jobs.append(WorkloadJob(
            job_id=f"job-{i}",
            profile=p,
            iterations_remaining=iters,
            deadline_seconds=deadline,
            energy_budget_kwh=e_budget,
            local_batch_size=local_batch,
            gradient_noise_scale=phi,
        ))
    return jobs


def run_hise_eb(jobs: list[WorkloadJob], available_gpus: int) -> dict[str, int]:
    admitted: list[tuple[str, EnergyProfile, int]] = []
    for j in jobs:
        eb = EnergyBudgetMSS(
            curve=_profile_to_curve(j.profile),
            power_per_gpu_w=0.0,
            energy_budget_kwh=j.energy_budget_kwh,
            energy_profile=j.profile,
        )
        decision = eb.find(j.iterations_remaining, j.deadline_seconds)
        if decision.admitted:
            admitted.append((j.job_id, j.profile, decision.gpus))
    if not admitted:
        return {}
    if sum(g for _j, _p, g in admitted) > available_gpus:
        admitted.sort(key=lambda t: -t[2])
        while admitted and sum(g for _j, _p, g in admitted) > available_gpus:
            admitted.pop(0)
        if not admitted:
            return {}
    return greedy_marginal_energy_allocation(admitted=admitted, available_gpus=available_gpus)


def run_powerflow(jobs: list[WorkloadJob], available_gpus: int) -> dict[str, int]:
    return powerflow_allocate(
        jobs=[(j.job_id, j.profile, j.iterations_remaining) for j in jobs],
        available_gpus=available_gpus,
    )


def run_elasticflow(jobs: list[WorkloadJob], available_gpus: int) -> dict[str, int]:
    ef_jobs = [
        ElasticFlowJob(
            job_id=j.job_id,
            curve=_profile_to_curve(j.profile),
            iterations_remaining=j.iterations_remaining,
            deadline_seconds=j.deadline_seconds,
        ) for j in jobs
    ]
    return elasticflow_schedule(ef_jobs, available_gpus).allocation


def run_zeus(jobs: list[WorkloadJob], available_gpus: int, eta: float = 0.5) -> dict[str, int]:
    alloc, _ = zeus_schedule(
        jobs=[(j.job_id, j.profile, j.iterations_remaining) for j in jobs],
        available_gpus=available_gpus,
        eta=eta,
        deadlines={j.job_id: j.deadline_seconds for j in jobs},
    )
    return alloc


def run_pollux(jobs: list[WorkloadJob], available_gpus: int) -> dict[str, int]:
    pollux_jobs = [
        PolluxJob(
            job_id=j.job_id,
            profile=j.profile,
            iterations_remaining=j.iterations_remaining,
            local_batch_size=j.local_batch_size,
            gradient_noise_scale=j.gradient_noise_scale,
        ) for j in jobs
    ]
    return pollux_allocate(pollux_jobs, available_gpus).allocation


def run_optimus(jobs: list[WorkloadJob], available_gpus: int) -> dict[str, int]:
    return optimus_allocate(
        jobs=[(j.job_id, j.profile, j.iterations_remaining) for j in jobs],
        available_gpus=available_gpus,
    )


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("inf") if ma != mb else 0.0
    return (ma - mb) / pooled


def _effect_size_tag(d: float) -> str:
    if math.isnan(d):
        return "n/a"
    if abs(d) < 0.2:
        return "negligible"
    if abs(d) < 0.5:
        return "small"
    if abs(d) < 0.8:
        return "medium"
    if abs(d) < 1.5:
        return "large"
    return "very large"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--available-gpus", type=int, default=10)
    parser.add_argument("--asymmetric", action="store_true",
                        help="Draw asymmetric EnergyProfiles (mimics heterogeneous serverless pool).")
    parser.add_argument("--output", default="artifacts/scheduler_head_to_head.json")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[bold]Six-allocator head-to-head[/]: {args.seeds} seeds × {args.n_jobs} jobs × "
        f"{args.available_gpus} GPUs, asymmetric={args.asymmetric}"
    )

    results: list[TrialResult] = []
    allocators: list[tuple[str, Callable[[list[WorkloadJob], int], dict[str, int]]]] = [
        ("PowerFlow", run_powerflow),
        ("ElasticFlow", run_elasticflow),
        ("Zeus(η=0.5)", lambda jobs, g: run_zeus(jobs, g, eta=0.5)),
        ("Pollux", run_pollux),
        ("Optimus", run_optimus),
        ("HISE EB", run_hise_eb),
    ]

    for seed in range(args.seeds):
        rng = random.Random(seed)
        jobs = draw_workload(rng, args.n_jobs, args.asymmetric)
        for name, fn in allocators:
            alloc = fn(jobs, args.available_gpus)
            e, jct, met = _summarise(alloc, jobs)
            results.append(TrialResult(
                seed=seed, allocator=name,
                total_energy_kwh=e, max_jct_s=jct,
                deadlines_met=met, n_jobs=args.n_jobs,
            ))

    by_alloc: dict[str, list[TrialResult]] = {}
    for r in results:
        by_alloc.setdefault(r.allocator, []).append(r)

    table = Table(title=f"Six-allocator head-to-head ({args.seeds} seeds, asymmetric={args.asymmetric})")
    table.add_column("allocator")
    table.add_column("mean kWh", justify="right")
    table.add_column("stddev kWh", justify="right")
    table.add_column("mean max JCT (s)", justify="right")
    table.add_column("deadlines met (avg)", justify="right")
    for name, _ in allocators:
        rs = by_alloc[name]
        es = [r.total_energy_kwh for r in rs]
        jcts = [r.max_jct_s for r in rs]
        met = [r.deadlines_met for r in rs]
        table.add_row(
            name,
            f"{statistics.mean(es):.4f}",
            f"{statistics.stdev(es) if len(es) > 1 else 0.0:.4f}",
            f"{statistics.mean(jcts):.1f}",
            f"{statistics.mean(met):.2f}/{args.n_jobs}",
        )
    console.print(table)

    hise = [r.total_energy_kwh for r in by_alloc["HISE EB"]]
    others = [name for name, _ in allocators if name != "HISE EB"]
    bonferroni_alpha = 0.05 / max(1, len(others))
    pairwise = Table(title=f"HISE EB vs baselines — Cohen's d on energy "
                            f"(Bonferroni α = {bonferroni_alpha:.4f})")
    pairwise.add_column("baseline")
    pairwise.add_column("Δ HISE − baseline (kWh)", justify="right")
    pairwise.add_column("Δ %", justify="right")
    pairwise.add_column("Cohen's d", justify="right")
    pairwise.add_column("effect size", justify="center")
    for name in others:
        other = [r.total_energy_kwh for r in by_alloc[name]]
        delta_abs = statistics.mean(hise) - statistics.mean(other)
        delta_pct = 100.0 * delta_abs / max(statistics.mean(other), 1e-9)
        d = cohens_d(hise, other)
        pairwise.add_row(
            name,
            f"{delta_abs:+.4f}",
            f"{delta_pct:+.1f}%",
            f"{d:+.2f}",
            _effect_size_tag(d),
        )
    console.print(pairwise)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "results": [r.__dict__ for r in results],
        "summary": {
            name: {
                "mean_kwh": statistics.mean([r.total_energy_kwh for r in by_alloc[name]]),
                "stddev_kwh": (
                    statistics.stdev([r.total_energy_kwh for r in by_alloc[name]])
                    if len(by_alloc[name]) > 1 else 0.0
                ),
                "mean_max_jct_s": statistics.mean([r.max_jct_s for r in by_alloc[name]]),
                "mean_deadlines_met": statistics.mean([r.deadlines_met for r in by_alloc[name]]),
            } for name in by_alloc
        },
    }, indent=2))
    console.print(f"\n[dim]Results saved: {out}[/]")


if __name__ == "__main__":
    main()

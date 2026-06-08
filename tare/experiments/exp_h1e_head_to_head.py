"""Multi-seed head-to-head: HASAGI EnergyBudgetMSS vs PowerFlow, ElasticFlow, Zeus.

Claim under test: on workloads with asymmetric EnergyProfiles and a hard
cluster-wide energy budget, HASAGI EB sits on or below the energy frontier
traced by prior allocators, with a deadline-meeting rate at least
matching the throughput-max family.

Each seed draws a fresh workload (per-job iter count + per-job
EnergyProfile shape parameters) from the same distribution, runs each
allocator on the same draw, and records (energy, max JCT, deadlines met).
The aggregator then reports mean ± stddev per allocator, plus Cohen's d
of HASAGI EB vs each baseline on the energy axis with a Bonferroni-
corrected significance threshold.

Usage:
    python -m experiments.exp_h1e_head_to_head --seeds 5
    python -m experiments.exp_h1e_head_to_head --seeds 5 --asymmetric
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
from experiments.baselines.powerflow import powerflow_allocate
from experiments.baselines.zeus import zeus_schedule
from tare.admission.energy_profile import EnergyProfile, linear_profile
from tare.admission.mss import (
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
    """Sample a workload from the experiment distribution."""
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
        jobs.append(WorkloadJob(
            job_id=f"job-{i}",
            profile=p,
            iterations_remaining=iters,
            deadline_seconds=deadline,
            energy_budget_kwh=e_budget,
        ))
    return jobs


def run_tare_eb(jobs: list[WorkloadJob], available_gpus: int) -> dict[str, int]:
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


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("inf") if ma != mb else 0.0
    return (ma - mb) / pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--available-gpus", type=int, default=10)
    parser.add_argument("--asymmetric", action="store_true",
                        help="Draw asymmetric EnergyProfiles (mimics heterogeneous serverless pool).")
    parser.add_argument("--output", default="artifacts/h1e_results.json")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[bold]H1-E head-to-head[/]: {args.seeds} seeds × {args.n_jobs} jobs × "
        f"{args.available_gpus} GPUs, asymmetric={args.asymmetric}"
    )

    results: list[TrialResult] = []
    allocators: list[tuple[str, Callable[[list[WorkloadJob], int], dict[str, int]]]] = [
        ("PowerFlow", run_powerflow),
        ("ElasticFlow", run_elasticflow),
        ("Zeus(η=0.5)", lambda jobs, g: run_zeus(jobs, g, eta=0.5)),
        ("HASAGI EB", run_tare_eb),
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

    table = Table(title=f"H1-E head-to-head ({args.seeds} seeds, asymmetric={args.asymmetric})")
    table.add_column("allocator")
    table.add_column("mean kWh", justify="right")
    table.add_column("stddev", justify="right")
    table.add_column("mean max JCT (s)", justify="right")
    table.add_column("deadlines met (avg)", justify="right")
    for name in [a[0] for a in allocators]:
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

    # Cohen's d: HASAGI EB vs each other allocator on the energy axis.
    tare = [r.total_energy_kwh for r in by_alloc["HASAGI EB"]]
    bonferroni_alpha = 0.05 / max(1, len([a for a in allocators if a[0] != "HASAGI EB"]))
    pairwise = Table(title=f"HASAGI EB vs baselines — Cohen's d on energy "
                            f"(Bonferroni α = {bonferroni_alpha:.4f})")
    pairwise.add_column("baseline")
    pairwise.add_column("Δ HASAGI − baseline (kWh)", justify="right")
    pairwise.add_column("Δ %", justify="right")
    pairwise.add_column("Cohen's d", justify="right")
    pairwise.add_column("effect size", justify="center")
    for name in [a[0] for a in allocators]:
        if name == "HASAGI EB":
            continue
        other = [r.total_energy_kwh for r in by_alloc[name]]
        delta_abs = statistics.mean(tare) - statistics.mean(other)
        delta_pct = 100.0 * delta_abs / max(statistics.mean(other), 1e-9)
        d = cohens_d(tare, other)
        if math.isnan(d):
            tag = "n/a"
        elif abs(d) < 0.2:
            tag = "negligible"
        elif abs(d) < 0.5:
            tag = "small"
        elif abs(d) < 0.8:
            tag = "medium"
        elif abs(d) < 1.5:
            tag = "large"
        else:
            tag = "very large"
        pairwise.add_row(
            name,
            f"{delta_abs:+.4f}",
            f"{delta_pct:+.1f}%",
            f"{d:+.2f}",
            tag,
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

"""Six-allocator head-to-head on a *measured* ResNet-18 / CIFAR-10 EnergyProfile.

Closes Extension 1 of the pre-paper review: the original
``exp_scheduler_head_to_head.py`` drives the allocators with the synthetic
``linear_profile``; this variant replaces that with the measured RTX 3080 Ti
power-cap-Pareto curve from ``profile_resnet18_real.py``. Each of the 5
power-cap operating points (150/200/250/300/350 W) becomes one "allocation
size" on the Pareto, indexed 1..5.

The semantic interpretation: in HASAGI's serverless setting an "allocation
size" = the energy/throughput point the worker pool is currently operating
at; in the simplest case that maps to a power-cap level (no need for multiple
GPUs). The allocator picks the operating point that minimises its objective
(energy / throughput / goodput / etc) for each job.

This is the apples-to-apples comparison the synthetic head-to-head could not
make: every allocator sees the actual measured (energy, throughput) trade-off
on real ResNet-18 training, not a synthesised curve.

Usage::

    python -m experiments.exp_scheduler_real_workload --seeds 10 \\
        --profile-json artifacts/resnet18_real_profile.json \\
        --out artifacts/scheduler_real_workload.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.baselines.elasticflow import ElasticFlowJob, elasticflow_schedule
from experiments.baselines.optimus import optimus_allocate
from experiments.baselines.pollux import PolluxJob, pollux_allocate
from experiments.baselines.powerflow import powerflow_allocate
from experiments.baselines.zeus import zeus_schedule
from hasagi.admission.energy_profile import EnergyProfile
from hasagi.admission.mss import (
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


def load_measured_profile(path: Path) -> EnergyProfile:
    """Convert the profile_resnet18_real.py JSON to an EnergyProfile.

    Order points by *ascending throughput* (smallest = "low-allocation",
    largest = "high-allocation") so index 0 corresponds to the lowest
    power-cap (slowest, often most energy-efficient) and the last index
    to the highest cap. This matches the EnergyProfile convention where
    larger indices = larger allocations.
    """
    data = json.loads(path.read_text())
    points = sorted(data["points"], key=lambda p: p["throughput_iter_per_s"])
    energy = tuple(p["energy_per_iter_j"] / 3_600_000.0 for p in points)   # J -> kWh
    throughput = tuple(p["throughput_iter_per_s"] for p in points)
    return EnergyProfile(
        energy_per_iter_kwh=energy,
        throughput_iters_per_s=throughput,
    )


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


def draw_workload(
    rng: random.Random, n_jobs: int, profile: EnergyProfile,
) -> list[WorkloadJob]:
    """All jobs share the measured ResNet-18 profile; per-job vary iter count + deadline.

    Iterations + deadlines are drawn so a mix of tight + slack deadlines is exercised
    (matches the multi-tenant serverless scenario the allocator targets).
    """
    jobs = []
    for i in range(n_jobs):
        iters = rng.choice([400, 800, 1200, 1600, 2000])
        # Tight enough that the energy-optimum (low cap) often *misses* deadline,
        # forcing the allocator to make a trade-off.
        deadline = rng.uniform(15.0, 60.0)   # seconds
        e_budget = rng.uniform(0.002, 0.010)   # kWh — tight enough to bind
        jobs.append(WorkloadJob(
            job_id=f"job-{i}",
            profile=profile,
            iterations_remaining=iters,
            deadline_seconds=deadline,
            energy_budget_kwh=e_budget,
            local_batch_size=128,   # measured profile is at batch 128
            gradient_noise_scale=rng.uniform(1500.0, 3000.0),   # ResNet-style
        ))
    return jobs


# --- Allocator drivers (identical interface to exp_scheduler_head_to_head) ---


def run_hasagi_eb(jobs: list[WorkloadJob], available_gpus: int) -> dict[str, int]:
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


def _effect_tag(d: float) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--available-gpus", type=int, default=10,
                        help="Total operating-points budget across jobs (each integer = one cap-level slot).")
    parser.add_argument("--profile-json", default="artifacts/resnet18_real_profile.json")
    parser.add_argument("--out", default="artifacts/scheduler_real_workload.json")
    args = parser.parse_args()

    console = Console()
    profile = load_measured_profile(Path(args.profile_json))
    console.print(
        f"[bold]Real-workload head-to-head[/]: profile has {profile.max_gpus} measured "
        f"operating points; throughput range "
        f"[{min(profile.throughput_iters_per_s):.2f}, {max(profile.throughput_iters_per_s):.2f}] iter/s, "
        f"E/iter range [{min(profile.energy_per_iter_kwh)*3.6e6:.2f}, "
        f"{max(profile.energy_per_iter_kwh)*3.6e6:.2f}] J"
    )
    console.print(
        f"{args.seeds} seeds × {args.n_jobs} jobs × {args.available_gpus}-slot budget"
    )

    results: list[TrialResult] = []
    allocators: list[tuple[str, Callable[[list[WorkloadJob], int], dict[str, int]]]] = [
        ("PowerFlow", run_powerflow),
        ("ElasticFlow", run_elasticflow),
        ("Zeus(η=0.5)", lambda jobs, g: run_zeus(jobs, g, eta=0.5)),
        ("Pollux", run_pollux),
        ("Optimus", run_optimus),
        ("HASAGI EB", run_hasagi_eb),
    ]

    for seed in range(args.seeds):
        rng = random.Random(seed)
        jobs = draw_workload(rng, args.n_jobs, profile)
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

    table = Table(title=f"Real ResNet-18 profile — {args.seeds} seeds × {args.n_jobs} jobs")
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
            f"{statistics.mean(es)*1000:.4f}",   # display as Wh for readability
            f"{(statistics.stdev(es)*1000 if len(es) > 1 else 0.0):.4f}",
            f"{statistics.mean(jcts):.1f}",
            f"{statistics.mean(met):.2f}/{args.n_jobs}",
        )
    console.print("[dim]kWh shown × 1000 (= Wh) for readability[/]")
    console.print(table)

    hasagi = [r.total_energy_kwh for r in by_alloc["HASAGI EB"]]
    others = [name for name, _ in allocators if name != "HASAGI EB"]
    bonf_alpha = 0.05 / max(1, len(others))
    pairwise = Table(title=f"HASAGI EB vs baselines — energy axis (Bonferroni α = {bonf_alpha:.4f})")
    pairwise.add_column("baseline")
    pairwise.add_column("Δ HASAGI − baseline (Wh)", justify="right")
    pairwise.add_column("Δ %", justify="right")
    pairwise.add_column("Cohen's d", justify="right")
    pairwise.add_column("effect size", justify="center")
    for name in others:
        other = [r.total_energy_kwh for r in by_alloc[name]]
        d_abs = statistics.mean(hasagi) - statistics.mean(other)
        d_pct = 100.0 * d_abs / max(statistics.mean(other), 1e-15)
        d = cohens_d(hasagi, other)
        pairwise.add_row(
            name,
            f"{d_abs*1000:+.4f}",
            f"{d_pct:+.1f}%",
            f"{d:+.2f}",
            _effect_tag(d),
        )
    console.print(pairwise)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "profile_path": str(args.profile_json),
        "n_operating_points": profile.max_gpus,
        "results": [asdict(r) for r in results],
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
    console.print(f"\n[dim]wrote {out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

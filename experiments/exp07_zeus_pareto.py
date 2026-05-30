"""Zeus η-sweep Pareto frontier vs HASAGI EB single operating point.

Zeus selects one (energy, time) operating point per job by minimising a convex
combination weighted by η ∈ [0, 1]. Sweeping η traces the Pareto frontier on the
(Σ energy, max JCT) plane. HASAGI EB returns a single operating point determined
by the deadline + energy budget combination — the comparison places that point
on the Pareto plot and shows how far inside / outside Zeus's frontier it lies.

This run lights up the modelling-gap finding from the ElasticFlow comparison
plus the Pareto-locus claim from the PowerFlow comparison in a single picture.

Usage:
    python -m experiments.exp07_zeus_pareto
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from experiments.baselines.zeus import zeus_schedule
from hasagi.admission.energy_profile import EnergyProfile, linear_profile
from hasagi.admission.mss import (
    EnergyBudgetMSS,
    ScalingCurve,
    greedy_marginal_energy_allocation,
)


@dataclass
class Job:
    job_id: str
    profile: EnergyProfile
    iterations_remaining: int
    deadline_seconds: float
    energy_budget_kwh: float


def _profile_to_curve(p: EnergyProfile) -> ScalingCurve:
    return ScalingCurve(throughput_per_gpu_count=tuple(p.throughput_iters_per_s))


def _summary(alloc: dict[str, int], jobs: list[Job]) -> tuple[float, float, int]:
    """Return (Σ energy kWh, max JCT s, deadlines met)."""
    energy = 0.0
    max_jct = 0.0
    met = 0
    for j in jobs:
        g = alloc.get(j.job_id, 0)
        if g == 0:
            continue
        energy += j.profile.energy_per_iter(g) * j.iterations_remaining
        jct = j.iterations_remaining / j.profile.throughput(g)
        max_jct = max(max_jct, jct)
        if jct <= j.deadline_seconds:
            met += 1
    return energy, max_jct, met


def schedule_hasagi_eb(jobs: list[Job], available_gpus: int) -> dict[str, int]:
    admitted: list[tuple[str, EnergyProfile, int]] = []
    for job in jobs:
        eb = EnergyBudgetMSS(
            curve=_profile_to_curve(job.profile),
            power_per_gpu_w=0.0,
            energy_budget_kwh=job.energy_budget_kwh,
            energy_profile=job.profile,
        )
        decision = eb.find(
            iterations_remaining=job.iterations_remaining,
            deadline_seconds=job.deadline_seconds,
        )
        if decision.admitted:
            admitted.append((job.job_id, job.profile, decision.gpus))
    if not admitted:
        return {}
    return greedy_marginal_energy_allocation(admitted, available_gpus)


def run(jobs: list[Job], available_gpus: int, console: Console) -> None:
    deadlines = {j.job_id: j.deadline_seconds for j in jobs}
    zeus_jobs = [(j.job_id, j.profile, j.iterations_remaining) for j in jobs]

    eta_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    pareto_rows: list[tuple[float, dict[str, int], float, float, int]] = []
    for eta in eta_grid:
        alloc, _rej = zeus_schedule(zeus_jobs, available_gpus, eta=eta, deadlines=deadlines)
        e, jct, met = _summary(alloc, jobs)
        pareto_rows.append((eta, alloc, e, jct, met))

    table = Table(title="Zeus η-sweep — Pareto frontier")
    table.add_column("η", justify="right")
    table.add_column("alloc", overflow="fold")
    table.add_column("Σ kWh", justify="right")
    table.add_column("max JCT (s)", justify="right")
    table.add_column("deadlines met", justify="right")
    for eta, alloc, e, jct, met in pareto_rows:
        table.add_row(
            f"{eta:.2f}",
            ", ".join(f"{k}:{v}" for k, v in sorted(alloc.items())),
            f"{e:.4f}",
            f"{jct:.1f}",
            f"{met}/{len(jobs)}",
        )
    console.print(table)

    hasagi_alloc = schedule_hasagi_eb(jobs, available_gpus)
    he, hjct, hmet = _summary(hasagi_alloc, jobs)
    console.print(
        f"\n[bold]HASAGI EB operating point[/]: "
        f"alloc={hasagi_alloc}, Σ kWh={he:.4f}, max JCT={hjct:.1f}s, "
        f"deadlines met={hmet}/{len(jobs)}"
    )

    # Compare Pareto-dominance only against Zeus points that completed every job.
    # A point that rejected jobs trivially "wins" on energy/JCT by doing less work,
    # which is not a meaningful Pareto comparison.
    n_jobs = len(jobs)
    feasible_zeus = [
        (eta, e, jct) for eta, _a, e, jct, met in pareto_rows if met == n_jobs
    ]
    dominated_by = [
        f"η={eta:.2f}" for eta, e, jct in feasible_zeus
        if e <= he and jct <= hjct and (e < he or jct < hjct)
    ]
    dominates = [
        f"η={eta:.2f}" for eta, e, jct in feasible_zeus
        if e >= he and jct >= hjct and (e > he or jct > hjct)
    ]
    rejecting_zeus = [
        f"η={eta:.2f}" for eta, _a, _e, _jct, met in pareto_rows if met < n_jobs
    ]
    if rejecting_zeus:
        console.print(
            f"[dim]Zeus rejects ≥1 job at: {', '.join(rejecting_zeus)} — excluded "
            "from the Pareto comparison (rejecting jobs trivially saves energy).[/]"
        )
    if dominated_by:
        console.print(
            f"[yellow]HASAGI EB is dominated by Zeus at: {', '.join(dominated_by)} "
            "(among the deadline-meeting points).[/]"
        )
    elif dominates:
        console.print(
            f"[green]HASAGI EB Pareto-dominates Zeus at: {', '.join(dominates)}[/]"
        )
    else:
        console.print(
            "[dim]Among the deadline-meeting Zeus points, HASAGI EB neither "
            "dominates nor is dominated — it lies on the same Pareto frontier.[/]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--available-gpus", type=int, default=12)
    parser.add_argument("--iters-per-job", type=int, default=2000)
    parser.add_argument("--deadline-s", type=float, default=400.0)
    parser.add_argument("--energy-budget-kwh", type=float, default=0.05)
    args = parser.parse_args()

    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [
        Job(
            job_id=f"job-{i}",
            profile=p,
            iterations_remaining=args.iters_per_job,
            deadline_seconds=args.deadline_s,
            energy_budget_kwh=args.energy_budget_kwh,
        )
        for i in range(args.jobs)
    ]

    console = Console()
    run(jobs, args.available_gpus, console)

    console.print(
        "\n[dim]Zeus traces a (Σ energy, max JCT) Pareto frontier as η sweeps "
        "from 0 (energy-min) to 1 (time-min). HASAGI EnergyBudgetMSS, given a "
        "specific deadline + energy budget, returns a single operating point. "
        "If HASAGI lies on Zeus's frontier, both schemes agree on the Pareto "
        "trade-off; if HASAGI Pareto-dominates a Zeus point, the deadline-first "
        "formulation has found a strict improvement over η-driven balancing.[/]"
    )


if __name__ == "__main__":
    main()

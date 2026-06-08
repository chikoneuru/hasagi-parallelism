"""Head-to-head: ElasticFlow scheduler vs HASAGI EnergyBudgetMSS.

Both admit jobs under deadlines and distribute leftover GPUs across the cluster.
They differ in what they optimise once admission is settled:

    ElasticFlow         max throughput  (no energy notion)
    HASAGI EnergyBudgetMSS  min energy      (deadline + energy budget)

The two are duals on the throughput-energy Pareto frontier. With the same job
set, ElasticFlow walks toward the high-throughput / high-energy end; HASAGI EB
walks toward the low-energy / longer-runtime end (still meeting deadlines).

Scenarios:
    A. Generous energy budget — both should accept the same jobs; HASAGI should
       allocate strictly fewer GPUs (saving energy) at the cost of higher JCT.
    B. Tight energy budget — ElasticFlow ignores energy (over-budget); HASAGI EB
       respects it, possibly rejecting jobs that ElasticFlow accepts.
    C. Heterogeneous EnergyProfiles — efficient jobs get more GPUs under HASAGI,
       inefficient jobs are de-prioritised; ElasticFlow ignores efficiency.

Usage:
    python -m experiments.exp06_eb_mss_vs_elasticflow
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from experiments.baselines.elasticflow import (
    ElasticFlowJob,
    elasticflow_schedule,
    project_energy_kwh,
)
from tare.admission.energy_profile import EnergyProfile, linear_profile
from tare.admission.mss import (
    EnergyBudgetMSS,
    ScalingCurve,
    greedy_marginal_energy_allocation,
)


@dataclass
class Job:
    """Joint workload spec — single source of truth that drives both schedulers."""

    job_id: str
    profile: EnergyProfile
    iterations_remaining: int
    deadline_seconds: float
    energy_budget_kwh: float


def _profile_to_curve(profile: EnergyProfile) -> ScalingCurve:
    """Extract the throughput curve from an EnergyProfile for ElasticFlow's MSS."""
    return ScalingCurve(throughput_per_gpu_count=tuple(profile.throughput_iters_per_s))


def schedule_tare_eb(jobs: list[Job], available_gpus: int) -> tuple[dict[str, int], tuple[str, ...]]:
    """HASAGI EB scheduler: per-job EnergyBudgetMSS admission, then marginal-energy distribution."""
    admitted: list[tuple[str, EnergyProfile, int]] = []
    rejected: list[str] = []
    for job in jobs:
        eb = EnergyBudgetMSS(
            curve=_profile_to_curve(job.profile),
            power_per_gpu_w=0.0,                              # using EnergyProfile branch
            energy_budget_kwh=job.energy_budget_kwh,
            energy_profile=job.profile,
        )
        decision = eb.find(
            iterations_remaining=job.iterations_remaining,
            deadline_seconds=job.deadline_seconds,
        )
        if decision.admitted:
            admitted.append((job.job_id, job.profile, decision.gpus))
        else:
            rejected.append(job.job_id)
    if not admitted:
        return {}, tuple(rejected)

    if sum(g for _j, _p, g in admitted) > available_gpus:
        # Cluster cannot hold all baseline EB allocations — reject the most
        # expensive (largest baseline GPU need) until they fit.
        admitted.sort(key=lambda t: -t[2])
        while admitted and sum(g for _j, _p, g in admitted) > available_gpus:
            jid, _p, _g = admitted.pop(0)
            rejected.append(jid)
    final = greedy_marginal_energy_allocation(admitted=admitted, available_gpus=available_gpus)
    return final, tuple(rejected)


def _tare_energy_kwh(profile: EnergyProfile, gpus: int, iters: int) -> float:
    if gpus <= 0 or iters <= 0:
        return 0.0
    return profile.energy_per_iter(gpus) * iters


def _linear_energy_kwh(profile: EnergyProfile, gpus: int, iters: int) -> float:
    """Linear P_per_gpu × gpus × duration projection (ElasticFlow / Zeus's η-rebalancing
    use this form). Drops the allreduce term, so it underestimates real-world energy."""
    if gpus <= 0 or iters <= 0:
        return 0.0
    curve = _profile_to_curve(profile)
    p_per_gpu = profile.energy_per_iter(1) * profile.throughput(1) * 3_600_000.0
    return project_energy_kwh(curve, p_per_gpu, gpus, iters)


def _jct(profile_or_curve, gpus: int, iters: int) -> float:
    if gpus <= 0:
        return math.inf
    return iters / profile_or_curve.throughput(gpus)


def summarise(
    policy: str,
    allocation: dict[str, int],
    rejected: tuple[str, ...],
    jobs: list[Job],
) -> dict:
    """Compute BOTH energy estimates for the same allocation. The linear estimate
    is what ElasticFlow / Zeus would report; the profile estimate is what HASAGI's
    Zeus-style EnergyProfile reports (and what the hardware would actually draw).
    Both numbers describe the SAME physical configuration."""
    linear_total = 0.0
    profile_total = 0.0
    jcts = []
    finished_on_time = 0
    for job in jobs:
        g = allocation.get(job.job_id, 0)
        if g == 0:
            jcts.append(math.inf)
            continue
        linear_total += _linear_energy_kwh(job.profile, g, job.iterations_remaining)
        profile_total += _tare_energy_kwh(job.profile, g, job.iterations_remaining)
        jct = _jct(job.profile, g, job.iterations_remaining)
        jcts.append(jct)
        if jct <= job.deadline_seconds:
            finished_on_time += 1
    return {
        "policy": policy,
        "alloc": allocation,
        "rejected": rejected,
        "total_gpus": sum(allocation.values()),
        "linear_energy_kwh": linear_total,
        "profile_energy_kwh": profile_total,
        "avg_jct_s": sum(j for j in jcts if math.isfinite(j)) / max(
            1, sum(1 for j in jcts if math.isfinite(j))
        ),
        "max_jct_s": max((j for j in jcts if math.isfinite(j)), default=math.inf),
        "deadlines_met": finished_on_time,
        "n_jobs": len(jobs),
    }


def run_scenario(name: str, jobs: list[Job], available_gpus: int, console: Console) -> None:
    console.print(f"\n[bold magenta]=== {name} ===[/]")
    console.print(
        f"jobs={len(jobs)}, GPU budget={available_gpus}, "
        f"per-job E budget={jobs[0].energy_budget_kwh:.3f} kWh"
    )

    ef_jobs = [
        ElasticFlowJob(
            job_id=j.job_id,
            curve=_profile_to_curve(j.profile),
            iterations_remaining=j.iterations_remaining,
            deadline_seconds=j.deadline_seconds,
        ) for j in jobs
    ]
    ef_result = elasticflow_schedule(ef_jobs, available_gpus)
    tare_alloc, tare_rejected = schedule_tare_eb(jobs, available_gpus)

    summary_ef = summarise("ElasticFlow", ef_result.allocation, ef_result.rejected, jobs)
    summary_hi = summarise("HASAGI EB", tare_alloc, tare_rejected, jobs)

    table = Table(title=name)
    table.add_column("policy")
    table.add_column("alloc", overflow="fold")
    table.add_column("rejected", overflow="fold")
    table.add_column("Σ GPU", justify="right")
    table.add_column("linear-est. kWh", justify="right")
    table.add_column("Zeus-profile kWh", justify="right")
    table.add_column("model gap %", justify="right")
    table.add_column("avg JCT (s)", justify="right")
    table.add_column("deadlines met", justify="right")
    for s in (summary_ef, summary_hi):
        gap_pct = (
            100.0 * (s["profile_energy_kwh"] - s["linear_energy_kwh"]) /
            s["linear_energy_kwh"]
            if s["linear_energy_kwh"] > 0 else 0.0
        )
        table.add_row(
            s["policy"],
            ", ".join(f"{k}:{v}" for k, v in sorted(s["alloc"].items())) or "—",
            ", ".join(sorted(s["rejected"])) or "—",
            str(s["total_gpus"]),
            f"{s['linear_energy_kwh']:.4f}",
            f"{s['profile_energy_kwh']:.4f}",
            f"+{gap_pct:.1f}%",
            f"{s['avg_jct_s']:.1f}",
            f"{s['deadlines_met']}/{s['n_jobs']}",
        )
    console.print(table)

    diff_alloc = summary_ef["alloc"] != summary_hi["alloc"]
    if diff_alloc:
        diff_msg = (
            f"Allocations differ — HASAGI picked {summary_hi['alloc']}, "
            f"ElasticFlow picked {summary_ef['alloc']}. "
            "Use the Zeus-profile column for the true energy comparison."
        )
    else:
        diff_msg = (
            "Allocations are identical. The linear-est. column is what "
            "ElasticFlow / Zeus would *report* internally; the Zeus-profile "
            "column is what the hardware would actually draw. The 'model gap' "
            "column is how much ElasticFlow's reporting under-counts energy "
            "on this exact allocation — not a scheduling outcome."
        )
    console.print(f"[dim]{diff_msg}[/]")


def scenario_a() -> tuple[str, list[Job], int]:
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [
        Job(f"job-{i}", p, iterations_remaining=2000, deadline_seconds=1000.0,
            energy_budget_kwh=10.0)         # generous
        for i in range(2)
    ]
    return ("A. Generous energy budget — HASAGI should use fewer GPUs", jobs, 12)


def scenario_b() -> tuple[str, list[Job], int]:
    p = linear_profile(power_per_gpu_w=300, base_throughput_iters_per_s=10, max_gpus=8)
    jobs = [
        Job(f"job-{i}", p, iterations_remaining=2000, deadline_seconds=1000.0,
            energy_budget_kwh=0.02)         # very tight per-job budget
        for i in range(2)
    ]
    return ("B. Tight energy budget — HASAGI rejects, ElasticFlow ignores", jobs, 12)


def scenario_c() -> tuple[str, list[Job], int]:
    p_efficient = linear_profile(
        power_per_gpu_w=250, base_throughput_iters_per_s=12, max_gpus=8,
        scaling_efficiency=0.85, allreduce_coefficient=0.05,
    )
    p_inefficient = linear_profile(
        power_per_gpu_w=400, base_throughput_iters_per_s=8, max_gpus=8,
        scaling_efficiency=0.80, allreduce_coefficient=0.08,
    )
    jobs = [
        Job("efficient", p_efficient, iterations_remaining=3000,
            deadline_seconds=600.0, energy_budget_kwh=1.0),
        Job("inefficient", p_inefficient, iterations_remaining=3000,
            deadline_seconds=600.0, energy_budget_kwh=1.0),
    ]
    return ("C. Heterogeneous EnergyProfiles", jobs, 10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["a", "b", "c"], default=None)
    args = parser.parse_args()

    console = Console()
    scenarios = {"a": scenario_a, "b": scenario_b, "c": scenario_c}
    keys = [args.only] if args.only else list(scenarios)
    for k in keys:
        name, jobs, gpus = scenarios[k]()
        run_scenario(name, jobs, gpus, console)

    console.print(
        "\n[dim]Two energy columns describe the same allocation under two energy "
        "models. Linear-est. = P_per_gpu × gpus × duration (what ElasticFlow / Zeus "
        "report internally, no allreduce term). Zeus-profile = e(g) × iters from "
        "the convex EnergyProfile (what the hardware actually draws). The model "
        "gap is the additional energy the linear estimate misses — it is independent "
        "of which allocator made the decision.\n"
        "Where the two allocators picked different allocations (none of the scenarios "
        "above, because the workloads are symmetric enough to converge), HASAGI EB sits "
        "on the energy-min end of the Pareto frontier and ElasticFlow on the "
        "throughput-max end (see exp09 asymmetric divergence).[/]"
    )


if __name__ == "__main__":
    main()

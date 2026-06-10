"""Tight-budget sensitivity sweep for HASAGI EnergyBudgetMSS.

The flat-budget head-to-head in ``exp_scheduler_head_to_head.py`` draws
per-job ``energy_budget_kwh`` uniformly from [0.6, 1.2] kWh, which is
~29× the actual per-job energy under any sensible allocation. That
makes the EB admission gate non-binding at the default and PowerFlow
wins by 9.3% on raw kWh while HASAGI EB wins on max JCT and deadlines
met (Pareto trade rather than a strict loss).

This harness sweeps a **cluster-relative** budget: for each seed we
first run PowerFlow to compute its total energy ``E_pf`` on that
workload, then set the HASAGI EB per-job budget to
``(multiplier × E_pf) / n_jobs``. At ``multiplier = 1.0`` HASAGI EB has
just enough budget to match PowerFlow's energy footprint; at < 1.0 it
must reject jobs (or fit them in fewer GPUs); at > 1.0 the gate is
non-binding.

At each multiplier we run HASAGI EB, PowerFlow, and ElasticFlow on the
SAME workload and record ``(energy_kwh, max_jct_s, deadlines_met,
admitted_count)``. PowerFlow and ElasticFlow ignore the budget knob
entirely — that is precisely the contribution HASAGI EB exposes.

Two complementary views are reported:

  - **Per-multiplier table**: mean energy / JCT / admitted across N
    seeds for each allocator at each budget multiplier.
  - **Energy-axis Cohen's d**: HASAGI EB vs PowerFlow at each
    multiplier, with Bonferroni-corrected α across the multiplier
    sweep.

Expected behaviour: at multipliers < 1.0 HASAGI EB rejects infeasible
jobs and uses strictly less cluster energy than PowerFlow but finishes
fewer jobs; at multiplier ≥ 1.0 the gate is non-binding and the
default Pareto-trade behaviour reasserts. The headline is the explicit
energy-budget knob HASAGI EB exposes that PowerFlow lacks.

Usage::

    python -m experiments.exp_tare_eb_tight_budget --seeds 10 \\
        --out artifacts/tare_eb_tight_budget.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.exp_scheduler_head_to_head import (
    WorkloadJob,
    _summarise,
    cohens_d,
    draw_workload,
    run_elasticflow,
    run_powerflow,
    run_tare_eb,
)


@dataclass
class TightBudgetTrial:
    seed: int
    budget_multiplier: float
    allocator: str
    energy_kwh: float
    max_jct_s: float
    deadlines_met: int
    admitted_count: int
    n_jobs: int


def _admitted_count(alloc: dict[str, int]) -> int:
    return sum(1 for g in alloc.values() if g > 0)


def _set_cluster_relative_budget(
    jobs: list[WorkloadJob],
    pf_total_energy_kwh: float,
    multiplier: float,
) -> list[WorkloadJob]:
    """Set per-job ``energy_budget_kwh`` to ``(multiplier × E_pf) / n_jobs``.

    Infinity multiplier disables the budget gate entirely (HASAGI EB never
    rejects on energy grounds).
    """
    n = max(1, len(jobs))
    if math.isinf(multiplier):
        new_budget = float("inf")
    else:
        new_budget = multiplier * pf_total_energy_kwh / n
    return [
        WorkloadJob(
            job_id=j.job_id,
            profile=j.profile,
            iterations_remaining=j.iterations_remaining,
            deadline_seconds=j.deadline_seconds,
            energy_budget_kwh=new_budget,
            local_batch_size=j.local_batch_size,
            gradient_noise_scale=j.gradient_noise_scale,
        )
        for j in jobs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--available-gpus", type=int, default=10)
    parser.add_argument("--asymmetric", action="store_true", default=True,
                        help="Asymmetric EnergyProfiles (matches scheduler_head_to_head_asymmetric).")
    parser.add_argument(
        "--multipliers", type=float, nargs="+",
        default=[0.5, 0.7, 0.9, 1.0, 1.2, 1.5, float("inf")],
        help=(
            "Cluster-relative budget multipliers. Per-job budget = "
            "(multiplier × PowerFlow total energy on the same workload) / n_jobs."
        ),
    )
    parser.add_argument("--out", default="artifacts/tare_eb_tight_budget.json")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[bold]HASAGI EB tight-budget sweep[/]: {args.seeds} seeds × "
        f"{len(args.multipliers)} multipliers × {args.n_jobs} jobs × "
        f"{args.available_gpus} GPUs (asymmetric={args.asymmetric})"
    )

    trials: list[TightBudgetTrial] = []
    allocators = [
        ("PowerFlow", run_powerflow),
        ("ElasticFlow", run_elasticflow),
        ("HASAGI EB", run_tare_eb),
    ]

    for seed in range(args.seeds):
        rng = random.Random(seed)
        base_jobs = draw_workload(rng, args.n_jobs, args.asymmetric)
        # Calibrate budget against PowerFlow's actual cluster energy on this seed.
        pf_alloc = run_powerflow(base_jobs, args.available_gpus)
        pf_energy, _, _ = _summarise(pf_alloc, base_jobs)
        for mult in args.multipliers:
            jobs = _set_cluster_relative_budget(base_jobs, pf_energy, mult)
            for name, fn in allocators:
                alloc = fn(jobs, args.available_gpus)
                e, jct, met = _summarise(alloc, jobs)
                trials.append(TightBudgetTrial(
                    seed=seed,
                    budget_multiplier=mult,
                    allocator=name,
                    energy_kwh=e,
                    max_jct_s=jct,
                    deadlines_met=met,
                    admitted_count=_admitted_count(alloc),
                    n_jobs=args.n_jobs,
                ))

    by_key: dict[tuple[float, str], list[TightBudgetTrial]] = {}
    for t in trials:
        by_key.setdefault((t.budget_multiplier, t.allocator), []).append(t)

    table = Table(title="Tight-budget sweep — energy, JCT, admitted (mean ± sd across seeds)")
    table.add_column("budget ×")
    table.add_column("allocator")
    table.add_column("mean kWh", justify="right")
    table.add_column("sd kWh", justify="right")
    table.add_column("mean JCT s", justify="right")
    table.add_column("mean met", justify="right")
    table.add_column("mean admitted", justify="right")
    for mult in args.multipliers:
        for name, _ in allocators:
            rs = by_key[(mult, name)]
            es = [r.energy_kwh for r in rs]
            jcts = [r.max_jct_s for r in rs]
            mets = [r.deadlines_met for r in rs]
            adms = [r.admitted_count for r in rs]
            label = "∞" if math.isinf(mult) else f"{mult:.1f}"
            table.add_row(
                label,
                name,
                f"{statistics.mean(es):.4f}",
                f"{statistics.stdev(es) if len(es) > 1 else 0.0:.4f}",
                f"{statistics.mean(jcts):.1f}",
                f"{statistics.mean(mets):.2f}/{args.n_jobs}",
                f"{statistics.mean(adms):.2f}/{args.n_jobs}",
            )
    console.print(table)

    # Pairwise HASAGI EB vs PowerFlow at each multiplier.
    n_compare = len(args.multipliers)
    bonferroni_alpha = 0.05 / max(1, n_compare)
    pairwise = Table(
        title=f"HASAGI EB vs PowerFlow at each multiplier — Cohen's d on energy "
              f"(Bonferroni α = {bonferroni_alpha:.4f})"
    )
    pairwise.add_column("budget ×")
    pairwise.add_column("Δ HASAGI − PF (kWh)", justify="right")
    pairwise.add_column("Δ %", justify="right")
    pairwise.add_column("Cohen's d", justify="right")
    pairwise.add_column("HASAGI met − PF met", justify="right")
    for mult in args.multipliers:
        tare = [r.energy_kwh for r in by_key[(mult, "HASAGI EB")]]
        pf = [r.energy_kwh for r in by_key[(mult, "PowerFlow")]]
        delta_abs = statistics.mean(tare) - statistics.mean(pf)
        delta_pct = 100.0 * delta_abs / max(statistics.mean(pf), 1e-9)
        d = cohens_d(tare, pf)
        h_met = statistics.mean(r.deadlines_met for r in by_key[(mult, "HASAGI EB")])
        p_met = statistics.mean(r.deadlines_met for r in by_key[(mult, "PowerFlow")])
        label = "∞" if math.isinf(mult) else f"{mult:.1f}"
        pairwise.add_row(
            label,
            f"{delta_abs:+.4f}",
            f"{delta_pct:+.1f}%",
            f"{d:+.2f}",
            f"{h_met - p_met:+.2f}",
        )
    console.print(pairwise)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "trials": [t.__dict__ for t in trials],
    }, indent=2))
    console.print(f"\n[dim]wrote {out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

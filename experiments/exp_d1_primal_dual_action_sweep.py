"""D1 — primal-dual sweep across multiple action sets.

The existing :mod:`experiments.exp_online_primal_dual_sweep` validates
the T8 regret bound on a single hand-tuned 4-action Pareto frontier.
Reviewer concern (closes Tier 3 D1 of the supervisor audit): one
action set is too narrow — a regret bound that holds on one config but
not others is uninformative. This sweep stress-tests the bound across
5 qualitatively different action sets, plus the standard 5 noise
levels × 10 seeds × T = 24h horizon:

  - **standard-4** (default 4-point Pareto frontier)
  - **fine-8** (finer DVFS granularity, 8 actions)
  - **coarse-3** (only slow / medium / rush)
  - **wide-range** (4 actions with hardware-realistic energy span 30-180 J)
  - **non-convex** (middle action dominated, stresses the inner argmin)

For each (action_set, noise, seed) cell we record the primal-dual gap
vs the offline LP optimum, the analytical envelope value, and a
within-envelope flag. Holm-Bonferroni controls the family-wise error
rate across the action-set comparisons.

Usage::

    python -m experiments.exp_d1_primal_dual_action_sweep \\
        --hours 24 --num-seeds 10 \\
        --out artifacts/d1_primal_dual_action_sweep.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.exp_online_deadline import Action
from experiments.exp_online_primal_dual_sweep import run_trial
from hasagi.stats import (
    bootstrap_mean_ci,
    cohens_d,
    effect_size_tag,
    holm_bonferroni,
    paired_permutation_pvalue,
)

# ---------------------------------------------------------------------------
# Action set library
# ---------------------------------------------------------------------------


def _standard_4() -> list[Action]:
    return [
        Action(name="slow",   energy_per_iter=40.0,  throughput=1.0),
        Action(name="medium", energy_per_iter=50.0,  throughput=2.0),
        Action(name="fast",   energy_per_iter=70.0,  throughput=4.0),
        Action(name="rush",   energy_per_iter=110.0, throughput=8.0),
    ]


def _fine_8() -> list[Action]:
    return [
        Action(name="r0", energy_per_iter=38.0,  throughput=1.0),
        Action(name="r1", energy_per_iter=44.0,  throughput=1.5),
        Action(name="r2", energy_per_iter=50.0,  throughput=2.0),
        Action(name="r3", energy_per_iter=58.0,  throughput=2.8),
        Action(name="r4", energy_per_iter=68.0,  throughput=3.8),
        Action(name="r5", energy_per_iter=80.0,  throughput=5.0),
        Action(name="r6", energy_per_iter=95.0,  throughput=6.4),
        Action(name="r7", energy_per_iter=115.0, throughput=8.0),
    ]


def _coarse_3() -> list[Action]:
    return [
        Action(name="slow",   energy_per_iter=40.0,  throughput=1.0),
        Action(name="medium", energy_per_iter=60.0,  throughput=3.0),
        Action(name="rush",   energy_per_iter=110.0, throughput=8.0),
    ]


def _wide_range() -> list[Action]:
    """Hardware-realistic span: 150 W → 350 W on a 3080 Ti shifts
    energy-per-iter ≈ 30 J → 180 J at constant batch."""
    return [
        Action(name="ultra-eco", energy_per_iter=30.0,  throughput=0.8),
        Action(name="eco",       energy_per_iter=55.0,  throughput=2.5),
        Action(name="boost",     energy_per_iter=110.0, throughput=6.0),
        Action(name="max",       energy_per_iter=180.0, throughput=9.0),
    ]


def _non_convex() -> list[Action]:
    """The middle action is dominated (worse on both axes than a convex
    interpolation of its neighbours). Stresses the inner argmin: the
    primal-dual policy should never select it at any λ.
    """
    return [
        Action(name="slow",     energy_per_iter=40.0,  throughput=1.0),
        Action(name="dominated", energy_per_iter=85.0, throughput=2.5),  # off the convex hull
        Action(name="fast",     energy_per_iter=70.0,  throughput=4.0),
        Action(name="rush",     energy_per_iter=110.0, throughput=8.0),
    ]


ACTION_SETS: dict[str, list[Action]] = {
    "standard-4": _standard_4(),
    "fine-8": _fine_8(),
    "coarse-3": _coarse_3(),
    "wide-range": _wide_range(),
    "non-convex": _non_convex(),
}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSummary:
    action_set: str
    noise_pct: float
    hours: int
    n_seeds: int
    pd_gap_mean: float
    pd_gap_ci_lo: float
    pd_gap_ci_hi: float
    coverage_pct: float
    regret_mean: float
    envelope_mean: float
    pd_beats_mpc_count: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-sets", nargs="+",
                        default=list(ACTION_SETS), choices=list(ACTION_SETS))
    parser.add_argument(
        "--noise-levels",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[0.05, 0.10, 0.15, 0.20, 0.30],
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--sample-minutes", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--deadline-multiplier", type=float, default=1.0)
    parser.add_argument("--mpc-horizon", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="artifacts/d1_primal_dual_action_sweep.json")
    args = parser.parse_args()

    console = Console()
    n_cells = len(args.action_sets) * len(args.noise_levels)
    n_trials = n_cells * args.num_seeds
    console.print(
        f"[bold]D1 primal-dual action-set sweep[/]: "
        f"{len(args.action_sets)} sets × {len(args.noise_levels)} noise × "
        f"{args.num_seeds} seeds = {n_trials} trials"
    )

    summaries: list[CellSummary] = []
    raw_gaps: dict[tuple[str, float], list[float]] = {}
    raw_coverage: dict[tuple[str, float], list[int]] = {}
    rng_for_stats = random.Random(0)

    for as_name in args.action_sets:
        actions = ACTION_SETS[as_name]
        for noise in args.noise_levels:
            trials = []
            for seed in range(args.num_seeds):
                trial = run_trial(
                    noise_pct=noise,
                    hours=args.hours,
                    sample_minutes=args.sample_minutes,
                    seed=seed,
                    actions=actions,
                    deadline_multiplier=args.deadline_multiplier,
                    mpc_horizon=args.mpc_horizon,
                )
                trials.append(trial)
            gaps = [t.primal_dual_gap_pct for t in trials]
            within = [1 if t.within_envelope else 0 for t in trials]
            mean, lo, hi = bootstrap_mean_ci(gaps, n_boot=5000, rng=rng_for_stats)
            beats_mpc = sum(1 for t in trials if t.primal_dual_cost < t.mpc_cost)
            summaries.append(CellSummary(
                action_set=as_name,
                noise_pct=noise,
                hours=args.hours,
                n_seeds=len(trials),
                pd_gap_mean=mean,
                pd_gap_ci_lo=lo,
                pd_gap_ci_hi=hi,
                coverage_pct=100.0 * sum(within) / len(within),
                regret_mean=statistics.mean(t.primal_dual_regret for t in trials),
                envelope_mean=statistics.mean(t.envelope for t in trials),
                pd_beats_mpc_count=beats_mpc,
            ))
            raw_gaps[(as_name, noise)] = gaps
            raw_coverage[(as_name, noise)] = within

    # Per-cell table.
    table = Table(title="D1 — primal-dual gap (95% bootstrap CI) per (action set, noise)")
    table.add_column("action set", justify="left")
    table.add_column("noise%", justify="right")
    table.add_column("PD gap % mean", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("coverage %", justify="right")
    table.add_column("PD < MPC", justify="right")
    for s in summaries:
        table.add_row(
            s.action_set,
            f"{s.noise_pct * 100:.0f}",
            f"{s.pd_gap_mean:+.2f}",
            f"[{s.pd_gap_ci_lo:+.2f}, {s.pd_gap_ci_hi:+.2f}]",
            f"{s.coverage_pct:.0f}",
            f"{s.pd_beats_mpc_count}/{s.n_seeds}",
        )
    console.print(table)

    # Per-action-set rollup (pooled over noise levels).
    rollup = Table(title="Per-action-set rollup (pooled over noise levels)")
    rollup.add_column("action set")
    rollup.add_column("n actions", justify="right")
    rollup.add_column("pooled PD gap % mean", justify="right")
    rollup.add_column("95% CI", justify="right")
    rollup.add_column("envelope coverage %", justify="right")
    by_set: dict[str, list[float]] = {a: [] for a in args.action_sets}
    cov_by_set: dict[str, list[int]] = {a: [] for a in args.action_sets}
    for s in summaries:
        by_set[s.action_set].extend(raw_gaps[(s.action_set, s.noise_pct)])
        cov_by_set[s.action_set].extend(raw_coverage[(s.action_set, s.noise_pct)])
    for as_name in args.action_sets:
        gaps = by_set[as_name]
        cov = cov_by_set[as_name]
        mean, lo, hi = bootstrap_mean_ci(gaps, n_boot=5000, rng=rng_for_stats)
        rollup.add_row(
            as_name,
            str(len(ACTION_SETS[as_name])),
            f"{mean:+.2f}",
            f"[{lo:+.2f}, {hi:+.2f}]",
            f"{100.0 * sum(cov) / len(cov):.0f}",
        )
    console.print(rollup)

    # Pairwise comparison standard-4 vs each other set (Holm-Bonferroni
    # across the alternatives).
    baseline = "standard-4"
    if baseline in args.action_sets:
        pairwise = Table(
            title=f"Pairwise vs {baseline} (paired-permutation; Holm-Bonferroni α = {args.alpha})",
        )
        pairwise.add_column("alt set")
        pairwise.add_column("Δ pooled PD gap %", justify="right")
        pairwise.add_column("Cohen's d", justify="right")
        pairwise.add_column("effect", justify="left")
        pairwise.add_column("p-value", justify="right")
        pairwise.add_column("Holm α", justify="right")
        pairwise.add_column("verdict", justify="left")

        alt_sets = [a for a in args.action_sets if a != baseline]
        pvalues: list[float] = []
        diffs: list[float] = []
        ds: list[float] = []
        for alt in alt_sets:
            alt_gaps = by_set[alt]
            base_gaps = by_set[baseline]
            d = cohens_d(alt_gaps, base_gaps)
            p = paired_permutation_pvalue(
                alt_gaps, base_gaps, n_perm=5000, rng=random.Random(0),
            )
            diffs.append(statistics.mean(alt_gaps) - statistics.mean(base_gaps))
            ds.append(d)
            pvalues.append(p)
        holm = holm_bonferroni(pvalues, alpha=args.alpha)
        for alt, dlt, d, (rej, p, adj) in zip(alt_sets, diffs, ds, holm, strict=True):
            pairwise.add_row(
                alt,
                f"{dlt:+.2f}",
                f"{d:+.2f}",
                effect_size_tag(d),
                f"{p:.4f}",
                f"{adj:.4f}",
                "reject H0 (differs)" if rej else "accept H0",
            )
        console.print(pairwise)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "cells": [asdict(s) for s in summaries],
    }, indent=2))
    console.print(f"\n[dim]wrote {out}[/]")

    # Acceptance: every action set's pooled coverage ≥ 90% of envelope.
    all_pass = True
    for as_name in args.action_sets:
        cov = cov_by_set[as_name]
        cov_pct = 100.0 * sum(cov) / len(cov)
        if cov_pct < 90.0:
            console.print(f"[red]✗ {as_name}: coverage {cov_pct:.0f}% < 90%[/]")
            all_pass = False
        else:
            console.print(f"[green]✓ {as_name}: coverage {cov_pct:.0f}% ≥ 90%[/]")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

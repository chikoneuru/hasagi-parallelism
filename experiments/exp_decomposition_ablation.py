"""Decomposition ablation — attribute the HISE energy gap to its three mechanisms.

The joint partition-and-throttle plan combines two decisions:

  1. **Partition** — pick layer cuts that minimise energy (vs the bottleneck-
     balanced partition that maximises throughput).
  2. **Throttle** — apply Perseus-style stage throttling to remove slack on the
     non-bottleneck stages.

The joint optimum is *not* the sum of the two contributions when applied
sequentially — there is a "joint synergy" term that captures the extra savings
the DP finds by co-designing both knobs. This harness decomposes the gap into
its four pieces so reviewers can see exactly where the savings come from:

    partition-only contribution =  E[bottleneck-only]  −  E[energy-only]
    throttle-only contribution  =  E[bottleneck-only]  −  E[bottleneck+Perseus]
    sequential combined         =  E[bottleneck-only]  −  E[energy+Perseus]
    joint full                  =  E[bottleneck-only]  −  E[joint]
    joint synergy               =  E[joint full]       −  E[sequential combined]
    synergy fraction            =  joint synergy        /  joint full

All five required allocators are already implemented in
``experiments.exp_joint_vs_stacked.evaluate_workload``; this harness sweeps the
same workload grid as ``exp_joint_real_workloads.py`` (3 models × 3 hardware
profiles × T_floor multipliers) and reports per-cell attribution + aggregate
mean / median over the grid.

The empirical-curve mode (``--pareto-json``) routes all allocators through the
real-hardware Pareto frontier instead of the parametric ``r^α`` model — useful
for an apples-to-apples decomposition on the measured RTX 3080 Ti curve.

Usage::

    python -m experiments.exp_decomposition_ablation
    python -m experiments.exp_decomposition_ablation --pareto-json artifacts/hardware-pareto-3080ti.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass

from rich.console import Console
from rich.table import Table

from experiments.exp_joint_real_workloads import (
    HARDWARE_PROFILES,
    MODELS,
)
from experiments.exp_joint_vs_stacked import evaluate_workload
from hise.parallel.joint_partitioner import ThrottleCurve
from hise.parallel.partitioner import LinkSpec, partition_pipeline


@dataclass(frozen=True)
class AttributionCell:
    """Per-cell decomposition of the joint energy gap into its three mechanisms.

    Each cell carries both the strict view (NaN when any allocator misses
    the T_floor deadline) and the relaxed view (uses each allocator's
    best-effort raw energy regardless of deadline feasibility). The
    relaxed view is always defined for the partition baselines and Perseus
    variants; it remains undefined only when the joint DP itself reports
    no plan (truly infeasible regime).
    """

    model: str
    hardware: str
    t_floor_multiplier: float
    # Strict allocator energies; inf when infeasible at this T_floor.
    e_bottleneck: float
    e_energy_partition: float
    e_bottleneck_perseus: float
    e_energy_perseus: float
    e_joint: float
    # Relaxed (deadline-ignored) allocator energies; inf only when no
    # partition or projection exists. Falls back to unit-throttle energy
    # when Perseus has no admissible throttle vector.
    raw_e_bottleneck: float
    raw_e_energy_partition: float
    raw_e_bottleneck_perseus: float
    raw_e_energy_perseus: float
    raw_e_joint: float
    # Strict attribution (as fraction of strict bottleneck baseline).
    # NaN when any contributing strict energy is inf.
    partition_only_pct: float
    throttle_only_pct: float
    sequential_combined_pct: float
    joint_full_pct: float
    joint_synergy_pct: float
    synergy_fraction: float
    # Relaxed attribution against the raw bottleneck-only energy.
    # Always defined for partition_only/throttle_only/sequential; defined
    # for joint only when the joint DP returned a plan.
    partition_only_pct_relaxed: float
    throttle_only_pct_relaxed: float
    sequential_combined_pct_relaxed: float
    joint_full_pct_relaxed: float
    joint_synergy_pct_relaxed: float
    synergy_fraction_relaxed: float
    joint_feasible: bool


def _pct(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0:
        return math.nan
    if not math.isfinite(candidate):
        return math.nan
    return 100.0 * (baseline - candidate) / baseline


def _cell_attribution(
    model_label: str,
    hardware_label: str,
    layers,
    stages,
    t_floor_multiplier: float,
    *,
    voltage_alpha: float,
    throttle_min: float,
    throttle_granularity: int,
    throttle_curve: ThrottleCurve | None,
) -> AttributionCell:
    links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(len(stages) - 1)]
    bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    t_max_bot = max(bot.stage_exec_time.values())
    t_floor = t_max_bot * t_floor_multiplier
    alloc_results = evaluate_workload(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / t_floor,
        voltage_alpha=voltage_alpha,
        throttle_min=throttle_min,
        throttle_granularity=throttle_granularity,
        throttle_curve=throttle_curve,
    )
    by_name = {r.name: r for r in alloc_results}
    e_bot = by_name["bottleneck-only"].energy if by_name["bottleneck-only"].feasible else math.inf
    e_en = by_name["energy-only"].energy if by_name["energy-only"].feasible else math.inf
    e_bot_pers = by_name["bottleneck + Perseus"].energy if by_name["bottleneck + Perseus"].feasible else math.inf
    e_en_pers = by_name["energy + Perseus"].energy if by_name["energy + Perseus"].feasible else math.inf
    e_joint = by_name["joint"].energy if by_name["joint"].feasible else math.inf

    # Relaxed allocator energies — defined whenever the partition exists,
    # ignoring T_floor feasibility.
    r_bot = by_name["bottleneck-only"].raw_energy
    r_en = by_name["energy-only"].raw_energy
    r_bot_pers = by_name["bottleneck + Perseus"].raw_energy
    r_en_pers = by_name["energy + Perseus"].raw_energy
    r_joint = by_name["joint"].raw_energy

    partition_only_pct = _pct(e_bot, e_en)
    throttle_only_pct = _pct(e_bot, e_bot_pers)
    sequential_combined_pct = _pct(e_bot, e_en_pers)
    joint_full_pct = _pct(e_bot, e_joint)
    joint_synergy_pct = (
        joint_full_pct - sequential_combined_pct
        if not math.isnan(joint_full_pct) and not math.isnan(sequential_combined_pct)
        else math.nan
    )
    synergy_fraction = (
        joint_synergy_pct / joint_full_pct
        if not math.isnan(joint_synergy_pct) and joint_full_pct > 0
        else math.nan
    )

    partition_only_pct_r = _pct(r_bot, r_en)
    throttle_only_pct_r = _pct(r_bot, r_bot_pers)
    sequential_combined_pct_r = _pct(r_bot, r_en_pers)
    joint_full_pct_r = _pct(r_bot, r_joint)
    joint_synergy_pct_r = (
        joint_full_pct_r - sequential_combined_pct_r
        if not math.isnan(joint_full_pct_r) and not math.isnan(sequential_combined_pct_r)
        else math.nan
    )
    synergy_fraction_r = (
        joint_synergy_pct_r / joint_full_pct_r
        if not math.isnan(joint_synergy_pct_r) and joint_full_pct_r > 0
        else math.nan
    )

    return AttributionCell(
        model=model_label, hardware=hardware_label,
        t_floor_multiplier=t_floor_multiplier,
        e_bottleneck=e_bot, e_energy_partition=e_en,
        e_bottleneck_perseus=e_bot_pers, e_energy_perseus=e_en_pers,
        e_joint=e_joint,
        raw_e_bottleneck=r_bot, raw_e_energy_partition=r_en,
        raw_e_bottleneck_perseus=r_bot_pers, raw_e_energy_perseus=r_en_pers,
        raw_e_joint=r_joint,
        partition_only_pct=partition_only_pct,
        throttle_only_pct=throttle_only_pct,
        sequential_combined_pct=sequential_combined_pct,
        joint_full_pct=joint_full_pct,
        joint_synergy_pct=joint_synergy_pct,
        synergy_fraction=synergy_fraction,
        partition_only_pct_relaxed=partition_only_pct_r,
        throttle_only_pct_relaxed=throttle_only_pct_r,
        sequential_combined_pct_relaxed=sequential_combined_pct_r,
        joint_full_pct_relaxed=joint_full_pct_r,
        joint_synergy_pct_relaxed=joint_synergy_pct_r,
        synergy_fraction_relaxed=synergy_fraction_r,
        joint_feasible=by_name["joint"].feasible,
    )


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:+.2f}%"


def _mean_finite(xs: list[float]) -> float:
    finite = [x for x in xs if not math.isnan(x)]
    if not finite:
        return math.nan
    return statistics.mean(finite)


def _median_finite(xs: list[float]) -> float:
    finite = [x for x in xs if not math.isnan(x)]
    if not finite:
        return math.nan
    return statistics.median(finite)


def run(args: argparse.Namespace) -> int:
    console = Console()
    model_keys = args.models or list(MODELS)
    hw_keys = args.hardware or list(HARDWARE_PROFILES)

    throttle_curve: ThrottleCurve | None = None
    if args.pareto_json:
        throttle_curve = ThrottleCurve.from_pareto_json(args.pareto_json)
        console.print(
            f"[bold]Empirical mode[/]: loaded {len(throttle_curve.points)} "
            f"throttle points from {args.pareto_json}"
        )

    cells: list[AttributionCell] = []
    for mk in model_keys:
        if mk not in MODELS:
            raise ValueError(f"unknown model {mk!r}; options: {list(MODELS)}")
        m_label, layers = MODELS[mk]
        for hk in hw_keys:
            if hk not in HARDWARE_PROFILES:
                raise ValueError(f"unknown hardware {hk!r}; options: {list(HARDWARE_PROFILES)}")
            hw_label, stages = HARDWARE_PROFILES[hk]
            for mult in args.t_floor_multipliers:
                cells.append(_cell_attribution(
                    m_label, hw_label, layers, stages, mult,
                    voltage_alpha=args.voltage_alpha,
                    throttle_min=args.throttle_min,
                    throttle_granularity=args.throttle_granularity,
                    throttle_curve=throttle_curve,
                ))

    per_cell = Table(title="Per-cell attribution — strict (deadline-feasible) % of bottleneck baseline")
    per_cell.add_column("model")
    per_cell.add_column("hw")
    per_cell.add_column("Tfloor×", justify="right")
    per_cell.add_column("partition", justify="right")
    per_cell.add_column("throttle", justify="right")
    per_cell.add_column("seq", justify="right")
    per_cell.add_column("joint", justify="right")
    per_cell.add_column("synergy", justify="right")
    per_cell.add_column("syn frac", justify="right")
    for c in cells:
        per_cell.add_row(
            c.model, c.hardware, f"{c.t_floor_multiplier:.2f}",
            _fmt_pct(c.partition_only_pct),
            _fmt_pct(c.throttle_only_pct),
            _fmt_pct(c.sequential_combined_pct),
            _fmt_pct(c.joint_full_pct),
            _fmt_pct(c.joint_synergy_pct),
            f"{c.synergy_fraction*100:.1f}%" if not math.isnan(c.synergy_fraction) else "—",
        )
    console.print(per_cell)

    per_cell_r = Table(title="Per-cell attribution — RELAXED (deadline-agnostic) % of raw bottleneck baseline")
    per_cell_r.add_column("model")
    per_cell_r.add_column("hw")
    per_cell_r.add_column("Tfloor×", justify="right")
    per_cell_r.add_column("partition", justify="right")
    per_cell_r.add_column("throttle", justify="right")
    per_cell_r.add_column("seq", justify="right")
    per_cell_r.add_column("joint", justify="right")
    per_cell_r.add_column("synergy", justify="right")
    per_cell_r.add_column("syn frac", justify="right")
    for c in cells:
        per_cell_r.add_row(
            c.model, c.hardware, f"{c.t_floor_multiplier:.2f}",
            _fmt_pct(c.partition_only_pct_relaxed),
            _fmt_pct(c.throttle_only_pct_relaxed),
            _fmt_pct(c.sequential_combined_pct_relaxed),
            _fmt_pct(c.joint_full_pct_relaxed),
            _fmt_pct(c.joint_synergy_pct_relaxed),
            f"{c.synergy_fraction_relaxed*100:.1f}%" if not math.isnan(c.synergy_fraction_relaxed) else "—",
        )
    console.print(per_cell_r)

    # Aggregate per-mechanism mean / median across the grid for both views.
    def _agg_table(title: str, getter) -> Table:
        tbl = Table(title=title)
        tbl.add_column("mechanism")
        tbl.add_column("mean", justify="right")
        tbl.add_column("median", justify="right")
        tbl.add_column("max", justify="right")
        tbl.add_column("n (finite)", justify="right")
        n_total = len(cells)
        for label, attr in (
            ("partition-only", "partition_only_pct"),
            ("throttle-only", "throttle_only_pct"),
            ("sequential combined", "sequential_combined_pct"),
            ("joint full", "joint_full_pct"),
            ("joint synergy", "joint_synergy_pct"),
        ):
            field = attr if getter == "strict" else attr.replace("_pct", "_pct_relaxed")
            vals = [getattr(c, field) for c in cells]
            finite = [v for v in vals if not math.isnan(v)]
            if not finite:
                tbl.add_row(label, "—", "—", "—", f"0/{n_total}")
                continue
            tbl.add_row(
                label,
                f"{_mean_finite(vals):+.2f}%",
                f"{_median_finite(vals):+.2f}%",
                f"{max(finite):+.2f}%",
                f"{len(finite)}/{n_total}",
            )
        frac_field = "synergy_fraction" if getter == "strict" else "synergy_fraction_relaxed"
        frac_vals = [getattr(c, frac_field) for c in cells]
        finite_frac = [v for v in frac_vals if not math.isnan(v)]
        if finite_frac:
            tbl.add_row(
                "synergy fraction (of joint full)",
                f"{statistics.mean(finite_frac)*100:.1f}%",
                f"{statistics.median(finite_frac)*100:.1f}%",
                f"{max(finite_frac)*100:.1f}%",
                f"{len(finite_frac)}/{n_total}",
            )
        return tbl

    console.print(_agg_table(
        f"Aggregate decomposition — strict view over {len(cells)} cells",
        "strict",
    ))
    console.print(_agg_table(
        f"Aggregate decomposition — RELAXED view over {len(cells)} cells",
        "relaxed",
    ))

    if args.out:
        from pathlib import Path
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "args": {k: v for k, v in vars(args).items() if k != "models" or v},
            "cells": [asdict(c) for c in cells],
        }, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--hardware", nargs="*", default=None)
    parser.add_argument(
        "--t-floor-multipliers", nargs="*", type=float,
        default=[1.00, 1.25, 1.50, 2.00],
    )
    parser.add_argument("--voltage-alpha", type=float, default=2.0)
    parser.add_argument("--throttle-min", type=float, default=0.5)
    parser.add_argument("--throttle-granularity", type=int, default=8)
    parser.add_argument(
        "--pareto-json", default=None,
        help="Optional real-hardware Pareto JSON (from exp_hardware_pareto.py).",
    )
    parser.add_argument("--out", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

"""Joint partition-and-throttle on three reference models across three hardware profiles.

Extends exp_joint_vs_stacked from the synthetic equal-FLOPS witness to
realistic-shape layer distributions and a 3-way hardware-skew sweep:

    models:       ResNet-18, ViT-B/16, GPT-2-small  (synthetic-but-shaped)
    hardware:     uniform, mild-skew, strong-skew   (K=4 stages)
    T_floor:      multipliers 1.00, 1.25, 1.50, 2.00 × T_max(c_B^*)

For each (model, hardware, T_floor) cell, all five allocators are run on
the same workload and the joint-vs-best-sequential gap is reported.
Acceptance criterion: ≥1 cell per (model, hardware) pair with strict
gap > 5% somewhere in the T_floor sweep.

Usage:
    python -m experiments.exp_joint_real_workloads
    python -m experiments.exp_joint_real_workloads --models resnet18 vit_b16
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from experiments.exp_joint_vs_stacked import evaluate_workload
from hasagi.parallel.joint_partitioner import ThrottleCurve
from hasagi.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    partition_pipeline,
)

# ---------------------------------------------------------------------------
# Model layer profiles — synthetic-but-shaped after the canonical models.
#
# FLOPS values are in arbitrary units consistent across stages (the joint
# DP is scale-invariant in (FLOPS, stage throughput) — what matters is the
# *shape* of the per-layer FLOPS distribution and the *ratio* between
# stage capacities).
# ---------------------------------------------------------------------------

def resnet18_layers() -> list[LayerProfile]:
    """18 layer groups; first block carries large spatial activations,
    later blocks have more channels but smaller spatial extent. Net FLOPS
    distribution is mildly increasing; activation footprint decreases."""
    layers: list[LayerProfile] = []
    for i in range(18):
        # FLOPS: ramp up gently with depth (more channels later).
        f = 1.0 + 0.04 * i
        # Activation: large at the front (high spatial), shrinks deeper.
        a = max(64, int(4096 * (18 - i) / 18))
        layers.append(LayerProfile(index=i, fwd_flops=f, bwd_flops=2.0 * f, activation_bytes=a))
    return layers


def vit_b16_layers() -> list[LayerProfile]:
    """12 transformer blocks, very uniform FLOPS, identical token-tensor
    shapes throughout the network (no spatial reduction)."""
    layers: list[LayerProfile] = []
    for i in range(12):
        # FLOPS: nearly flat; tiny per-block variation for the attention/MLP alternation.
        f = 1.5 + 0.05 * (i % 2)
        a = 196 * 768 // 64  # ~ token tensor proxy (196 tokens × 768 dim / scale)
        layers.append(LayerProfile(index=i, fwd_flops=f, bwd_flops=2.0 * f, activation_bytes=a))
    return layers


def gpt2_small_layers() -> list[LayerProfile]:
    """12 transformer blocks at GPT-2-small dimensions. FLOPS distribution
    is highly uniform; activation shape constant (seq_len fixed)."""
    layers: list[LayerProfile] = []
    for i in range(12):
        f = 1.2
        a = 1024 * 768 // 64
        layers.append(LayerProfile(index=i, fwd_flops=f, bwd_flops=2.0 * f, activation_bytes=a))
    return layers


MODELS: dict[str, tuple[str, list[LayerProfile]]] = {
    "resnet18": ("ResNet-18", resnet18_layers()),
    "vit_b16": ("ViT-B/16", vit_b16_layers()),
    "gpt2_small": ("GPT-2-small", gpt2_small_layers()),
}


# ---------------------------------------------------------------------------
# Hardware profiles — K=4 stage capacity patterns.
# ---------------------------------------------------------------------------

def hardware_uniform(n_stages: int = 4) -> list[StageSpec]:
    """All stages identical."""
    return [
        StageSpec(stage_id=s, throughput_flops=1.0, memory_bytes=10**18, power_draw_w=1.0)
        for s in range(n_stages)
    ]


def hardware_mild_skew(n_stages: int = 4) -> list[StageSpec]:
    """Slow head stage, faster tail (Jetson + GPU edge case)."""
    caps = [1.0] + [2.0] * (n_stages - 1)
    return [
        StageSpec(stage_id=s, throughput_flops=caps[s], memory_bytes=10**18, power_draw_w=1.0)
        for s in range(n_stages)
    ]


def hardware_strong_skew(n_stages: int = 4) -> list[StageSpec]:
    """Monotone capacity ramp from slow head to fast tail."""
    caps = [1.0 + s for s in range(n_stages)]  # (1, 2, 3, 4) at K=4
    return [
        StageSpec(stage_id=s, throughput_flops=caps[s], memory_bytes=10**18, power_draw_w=1.0)
        for s in range(n_stages)
    ]


HARDWARE_PROFILES: dict[str, tuple[str, list[StageSpec]]] = {
    "uniform": ("uniform", hardware_uniform()),
    "mild_skew": ("mild-skew", hardware_mild_skew()),
    "strong_skew": ("strong-skew", hardware_strong_skew()),
}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellResult:
    model: str
    hardware: str
    t_floor_multiplier: float
    t_floor_seconds: float
    e_best_seq: float
    e_joint: float
    gap_pct: float
    best_seq_name: str
    joint_feasible: bool


def _ratio(stages: list[StageSpec]) -> float:
    caps = [s.throughput_flops for s in stages]
    return max(caps) / min(caps)


def _entropy(stages: list[StageSpec]) -> float:
    caps = [s.throughput_flops for s in stages]
    total = sum(caps)
    h = 0.0
    for c in caps:
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def sweep_cell(
    model_name: str,
    layers: list[LayerProfile],
    hardware_name: str,
    stages: list[StageSpec],
    t_floor_multipliers: list[float],
    *,
    voltage_alpha: float,
    throttle_min: float,
    throttle_granularity: int,
    throttle_curve: ThrottleCurve | None = None,
) -> list[CellResult]:
    links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(len(stages) - 1)]
    bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    t_max_bot = max(bot.stage_exec_time.values())

    results: list[CellResult] = []
    for mult in t_floor_multipliers:
        t_floor = t_max_bot * mult
        alloc_results = evaluate_workload(
            layers, stages, links,
            throughput_floor_iters_per_s=1.0 / t_floor,
            voltage_alpha=voltage_alpha,
            throttle_min=throttle_min,
            throttle_granularity=throttle_granularity,
            throttle_curve=throttle_curve,
        )
        feasible = {r.name: (r.energy, r) for r in alloc_results if r.feasible}
        if "joint" not in feasible:
            results.append(CellResult(
                model=model_name, hardware=hardware_name,
                t_floor_multiplier=mult, t_floor_seconds=t_floor,
                e_best_seq=math.inf, e_joint=math.inf,
                gap_pct=0.0, best_seq_name="—", joint_feasible=False,
            ))
            continue
        e_joint = feasible["joint"][0]
        seq = [(n, e) for n, (e, _) in feasible.items() if n != "joint"]
        if not seq:
            results.append(CellResult(
                model=model_name, hardware=hardware_name,
                t_floor_multiplier=mult, t_floor_seconds=t_floor,
                e_best_seq=math.inf, e_joint=e_joint,
                gap_pct=math.inf, best_seq_name="—", joint_feasible=True,
            ))
            continue
        best_name, e_best = min(seq, key=lambda nv: nv[1])
        gap_pct = 100.0 * (e_best - e_joint) / e_best if e_best > 0 else 0.0
        results.append(CellResult(
            model=model_name, hardware=hardware_name,
            t_floor_multiplier=mult, t_floor_seconds=t_floor,
            e_best_seq=e_best, e_joint=e_joint,
            gap_pct=gap_pct, best_seq_name=best_name, joint_feasible=True,
        ))
    return results


def run(args: argparse.Namespace) -> None:
    console = Console()
    model_keys = args.models or list(MODELS)
    hw_keys = args.hardware or list(HARDWARE_PROFILES)
    t_floor_mults = args.t_floor_multipliers

    throttle_curve: ThrottleCurve | None = None
    if args.pareto_json:
        throttle_curve = ThrottleCurve.from_pareto_json(args.pareto_json)
        console.print(
            f"[bold]Empirical mode[/]: loaded {len(throttle_curve.points)} "
            f"throttle points from {args.pareto_json}"
        )

    all_results: list[CellResult] = []
    for mk in model_keys:
        if mk not in MODELS:
            raise ValueError(f"unknown model {mk!r}; options: {list(MODELS)}")
        m_label, layers = MODELS[mk]
        for hk in hw_keys:
            if hk not in HARDWARE_PROFILES:
                raise ValueError(f"unknown hardware {hk!r}; options: {list(HARDWARE_PROFILES)}")
            hw_label, stages = HARDWARE_PROFILES[hk]
            rows = sweep_cell(
                m_label, layers, hw_label, stages, t_floor_mults,
                voltage_alpha=args.voltage_alpha,
                throttle_min=args.throttle_min,
                throttle_granularity=args.throttle_granularity,
                throttle_curve=throttle_curve,
            )
            all_results.extend(rows)

    # Wide table — one row per (model, hardware), one column per T_floor.
    wide = Table(title="Gap % vs T_floor multiplier — joint vs best sequential")
    wide.add_column("model")
    wide.add_column("hardware")
    wide.add_column("ratio θ", justify="right")
    wide.add_column("entropy", justify="right")
    for mult in t_floor_mults:
        wide.add_column(f"×{mult:.2f}", justify="right")
    wide.add_column("max gap %", justify="right")

    # Aggregate stats
    cell_acceptance: dict[tuple[str, str], float] = {}

    seen_pairs: list[tuple[str, str]] = []
    for r in all_results:
        if (r.model, r.hardware) not in seen_pairs:
            seen_pairs.append((r.model, r.hardware))

    for model, hw in seen_pairs:
        cell_rows = [r for r in all_results if r.model == model and r.hardware == hw]
        if not cell_rows:
            continue
        # Find the stages for this hardware to compute heterogeneity stats.
        hw_key = next(k for k, v in HARDWARE_PROFILES.items() if v[0] == hw)
        stages = HARDWARE_PROFILES[hw_key][1]
        ratio = _ratio(stages)
        ent = _entropy(stages)
        gaps_by_mult = {r.t_floor_multiplier: r.gap_pct if r.joint_feasible else -math.inf
                        for r in cell_rows}
        max_gap = max((g for g in gaps_by_mult.values() if math.isfinite(g)), default=0.0)
        cell_acceptance[(model, hw)] = max_gap

        cells: list[str] = []
        for mult in t_floor_mults:
            g = gaps_by_mult.get(mult, -math.inf)
            cells.append("infeas" if not math.isfinite(g) else f"{g:+.2f}%")
        wide.add_row(model, hw, f"{ratio:.2f}", f"{ent:.3f}",
                     *cells, f"{max_gap:+.2f}%")

    console.print("")
    console.print(wide)

    # Aggregate stats across all cells
    gaps = [r.gap_pct for r in all_results if r.joint_feasible and math.isfinite(r.gap_pct)]
    if gaps:
        agg = Table(title="Aggregate gap statistics across all cells")
        agg.add_column("metric")
        agg.add_column("value", justify="right")
        agg.add_row("# feasible cells", str(len(gaps)))
        agg.add_row("min gap %", f"{min(gaps):+.2f}%")
        agg.add_row("mean gap %", f"{sum(gaps)/len(gaps):+.2f}%")
        agg.add_row("median gap %", f"{sorted(gaps)[len(gaps)//2]:+.2f}%")
        agg.add_row("max gap %", f"{max(gaps):+.2f}%")
        agg.add_row("# cells with gap > 5%", str(sum(1 for g in gaps if g > 5.0)))
        agg.add_row("# cells with gap > 10%", str(sum(1 for g in gaps if g > 10.0)))
        console.print(agg)

    # Acceptance check: ≥1 cell per (model, hardware) pair with gap > 5%
    threshold = args.acceptance_gap_pct
    failures: list[tuple[str, str, float]] = []
    for (model, hw), max_gap in cell_acceptance.items():
        if max_gap < threshold:
            failures.append((model, hw, max_gap))

    if not failures:
        console.print(
            f"\n[bold green]Acceptance passed[/]: every (model, hardware) cell achieves "
            f"gap > {threshold}% on at least one T_floor setting."
        )
    else:
        console.print(
            f"\n[bold red]Acceptance failed[/]: {len(failures)} cells below the "
            f"{threshold}% threshold."
        )
        for model, hw, g in failures:
            console.print(f"  - {model} on {hw}: max gap {g:+.2f}%")

    console.print(
        "\n[dim]Each cell runs five allocators on the same workload + T_floor. "
        "Gap = (E_best_sequential - E_joint) / E_best_sequential × 100. The "
        "T_floor multiplier sets T_floor = m × T_max(c_B*); m=1 is the tight "
        "regime (no slack), m>1 opens the slack window the joint optimiser "
        "throttles into. ResNet-18 has 18 layers, ViT-B/16 and GPT-2-small "
        "have 12 transformer blocks. Hardware profiles model uniform / "
        "mild-skew / strong-skew capacity distributions across K=4 stages.[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        choices=list(MODELS))
    parser.add_argument("--hardware", type=str, nargs="*", default=None,
                        choices=list(HARDWARE_PROFILES))
    parser.add_argument("--t-floor-multipliers", type=float, nargs="+",
                        default=[1.00, 1.25, 1.50, 2.00])
    parser.add_argument("--voltage-alpha", type=float, default=2.0)
    parser.add_argument("--throttle-min", type=float, default=0.5)
    parser.add_argument("--throttle-granularity", type=int, default=11)
    parser.add_argument(
        "--pareto-json",
        type=str,
        default=None,
        help=(
            "Path to an exp_hardware_pareto.py JSON file. When given, the sweep "
            "uses the empirical throttle curve instead of the parametric "
            "voltage-alpha model; the alpha/min/granularity flags become ignored."
        ),
    )
    parser.add_argument("--acceptance-gap-pct", type=float, default=5.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

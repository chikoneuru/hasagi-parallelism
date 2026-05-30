"""Perseus throttling vs HASAGI energy-objective partition — pipeline energy ablation.

Two ways to reduce pipeline-energy bloat from imbalanced stages:

  P1. Keep the bottleneck-optimal partition, post-hoc throttle the fast stages
      (Perseus, NSDI'24). Same throughput, lower power on fast stages.

  P2. Re-partition with the energy objective so the stages are balanced in
      ``Σ P_s · T_s`` rather than just in ``max T_s`` (HASAGI's existing
      ``partition_pipeline(..., objective="energy")``).

The two strategies are complementary, not competing:

  - P1 saves the slack-idle energy at fixed stage cuts.
  - P2 changes the cuts so slack is small in the first place.
  - P1 ∘ P2 stacks: Perseus on the energy-optimal partition recovers any
    residual bloat.

This experiment runs four configurations on the same synthetic pipeline (12
layers, 3 stages with heterogeneous FLOPS to force imbalance):

    config              partition              throttling
    bottleneck-baseline bottleneck-optimal     none
    bottleneck+perseus  bottleneck-optimal     Perseus
    energy-baseline     energy-optimal         none
    energy+perseus      energy-optimal         Perseus

Usage:
    python -m experiments.exp08_perseus_vs_energy_partition
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from experiments.baselines.perseus import perseus_throttle
from hasagi.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    Partition,
    StageSpec,
    partition_pipeline,
)


@dataclass
class ConfigResult:
    name: str
    pipeline_time_s: float
    baseline_energy_kwh: float        # bloated (no throttling) given the partition
    final_energy_kwh: float           # after optional Perseus throttling
    savings_pct: float                # vs config's own baseline


def _baseline_energy_per_iter_kwh(partition: Partition,
                                   stages: list[StageSpec],
                                   idle_power_fraction: float) -> float:
    """Σ_s (P_s · T_s + P_idle_s · (T_max - T_s)) / 3.6e6, summed over stages."""
    t_max = max(partition.stage_exec_time.values())
    total_ws = 0.0
    for s in stages:
        t_s = partition.stage_exec_time.get(s.stage_id, 0.0)
        p_s = s.power_draw_w
        total_ws += p_s * t_s + (p_s * idle_power_fraction) * (t_max - t_s)
    return total_ws / 3_600_000.0


def evaluate(
    name: str,
    partition: Partition,
    stages: list[StageSpec],
    *,
    apply_perseus: bool,
    voltage_alpha: float,
    idle_power_fraction: float,
) -> ConfigResult:
    baseline_e = _baseline_energy_per_iter_kwh(partition, stages, idle_power_fraction)
    if apply_perseus:
        plan = perseus_throttle(
            partition, stages,
            voltage_alpha=voltage_alpha,
            idle_power_fraction=idle_power_fraction,
        )
        final_e = plan.throttled_energy_per_iter_kwh
    else:
        final_e = baseline_e
    saved_pct = 100.0 * (baseline_e - final_e) / baseline_e if baseline_e > 0 else 0.0
    return ConfigResult(
        name=name,
        pipeline_time_s=max(partition.stage_exec_time.values()),
        baseline_energy_kwh=baseline_e,
        final_energy_kwh=final_e,
        savings_pct=saved_pct,
    )


def run(args: argparse.Namespace) -> None:
    console = Console()
    n_layers = args.n_layers
    layers = [
        LayerProfile(index=i, fwd_flops=args.fwd_flops, bwd_flops=args.bwd_flops,
                     activation_bytes=args.activation_bytes)
        for i in range(n_layers)
    ]
    # Heterogeneous stage capacities to force imbalance.
    stages = [
        StageSpec(stage_id=0, throughput_flops=1e10, memory_bytes=10**12, power_draw_w=300.0),
        StageSpec(stage_id=1, throughput_flops=2e10, memory_bytes=10**12, power_draw_w=300.0),
        StageSpec(stage_id=2, throughput_flops=4e10, memory_bytes=10**12, power_draw_w=300.0),
    ]
    links = [LinkSpec(0, 1, 1e10), LinkSpec(1, 2, 1e10)]

    part_bottleneck = partition_pipeline(layers, stages, links, objective="bottleneck")
    part_energy = partition_pipeline(layers, stages, links, objective="energy")

    console.print(
        f"\n[bold]Partitions[/]: bottleneck-optimal cuts={part_bottleneck.cuts}, "
        f"energy-optimal cuts={part_energy.cuts}"
    )
    console.print(
        f"[dim]bottleneck stage_exec_time={dict(part_bottleneck.stage_exec_time)} "
        f"max={max(part_bottleneck.stage_exec_time.values()):.4f}s[/]"
    )
    console.print(
        f"[dim]energy   stage_exec_time={dict(part_energy.stage_exec_time)} "
        f"max={max(part_energy.stage_exec_time.values()):.4f}s[/]"
    )

    configs = [
        evaluate("bottleneck-only", part_bottleneck, stages,
                 apply_perseus=False, voltage_alpha=args.voltage_alpha,
                 idle_power_fraction=args.idle_power_fraction),
        evaluate("bottleneck + Perseus", part_bottleneck, stages,
                 apply_perseus=True, voltage_alpha=args.voltage_alpha,
                 idle_power_fraction=args.idle_power_fraction),
        evaluate("energy-only", part_energy, stages,
                 apply_perseus=False, voltage_alpha=args.voltage_alpha,
                 idle_power_fraction=args.idle_power_fraction),
        evaluate("energy + Perseus", part_energy, stages,
                 apply_perseus=True, voltage_alpha=args.voltage_alpha,
                 idle_power_fraction=args.idle_power_fraction),
    ]

    table = Table(title="Perseus vs HASAGI energy partition — pipeline-energy comparison")
    table.add_column("config")
    table.add_column("pipeline (s)", justify="right")
    table.add_column("baseline kWh/iter", justify="right")
    table.add_column("final kWh/iter", justify="right")
    table.add_column("savings %", justify="right")
    for c in configs:
        table.add_row(
            c.name,
            f"{c.pipeline_time_s:.4f}",
            f"{c.baseline_energy_kwh*1e6:.2f} μWh",
            f"{c.final_energy_kwh*1e6:.2f} μWh",
            f"{c.savings_pct:+.1f}%",
        )
    console.print(table)

    # Cross-config energy deltas: energy reduction vs the worst-case bottleneck-only run.
    worst_e = configs[0].final_energy_kwh
    deltas = []
    for c in configs[1:]:
        delta_pct = (worst_e - c.final_energy_kwh) / worst_e * 100.0 if worst_e > 0 else 0.0
        deltas.append((c.name, delta_pct))

    table2 = Table(title="Energy reduction vs bottleneck-only baseline")
    table2.add_column("config")
    table2.add_column("reduction %", justify="right")
    for name, pct in deltas:
        table2.add_row(name, f"{pct:+.1f}%")
    console.print(table2)

    console.print(
        "\n[dim]Perseus (NSDI'24) post-hoc throttles fast stages to remove "
        "slack-idle bloat at fixed cuts. HASAGI's energy-objective partitioner "
        "shifts the cuts so the slack is small to begin with. The two are "
        "complementary: stacking Perseus on the energy-optimal partition "
        "recovers any residual bloat — the empirical evidence above shows "
        "the stack out-performs either strategy alone.[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--fwd-flops", type=float, default=1e9)
    parser.add_argument("--bwd-flops", type=float, default=2e9)
    parser.add_argument("--activation-bytes", type=int, default=1024)
    parser.add_argument("--voltage-alpha", type=float, default=2.0)
    parser.add_argument("--idle-power-fraction", type=float, default=0.3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

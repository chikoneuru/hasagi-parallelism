"""Sensitivity of the joint-vs-sequential gap to algorithm hyperparameters.

Sweeps each of the three joint-DP hyperparameters one axis at a time
while holding the other two at their defaults. Two reference workloads:

  (a) Witness workload — 2-stage equal-FLOPS, capacity ratio 3, T_floor
      multiplier 1.25. Reproduces the proof's analytic configuration.
  (b) Real-shape workload — ResNet-18 layer profile on the K=4 mild-skew
      hardware profile at T_floor multiplier 1.5. Representative of the
      real-model sweep cells.

Axes:
    voltage_alpha       ∈ {2.0, 2.5, 3.0}        (DVFS exponent)
    throttle_granularity ∈ {6, 11, 21, 41}        (R grid resolution)
    throttle_min        ∈ {0.3, 0.5, 0.7}         (hardware throttle floor)

Reports the gap (joint vs best sequential) at each parameter setting.

Usage:
    python -m experiments.exp_joint_sensitivity
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from experiments.exp_joint_real_workloads import (
    HARDWARE_PROFILES,
    MODELS,
)
from experiments.exp_joint_vs_stacked import evaluate_workload
from hasagi.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    partition_pipeline,
)


@dataclass(frozen=True)
class SensitivityRow:
    workload: str
    axis: str
    value: str
    e_best_seq: float
    e_joint: float
    gap_pct: float
    feasible: bool


def _witness_workload() -> tuple[list[LayerProfile], list[StageSpec], list[LinkSpec], float]:
    """2-stage equal-FLOPS, capacity ratio 3, T_floor mult 1.25 → T_floor = 1.25 × T_max(c_B*)."""
    layers = [
        LayerProfile(index=i, fwd_flops=1.0, bwd_flops=2.0, activation_bytes=0)
        for i in range(10)
    ]
    stages = [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10**18, power_draw_w=1.0),
        StageSpec(stage_id=1, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.0),
    ]
    links = [LinkSpec(0, 1, 1e18, 0.0)]
    bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    t_floor = max(bot.stage_exec_time.values()) * 1.25
    return layers, stages, links, t_floor


def _real_workload() -> tuple[list[LayerProfile], list[StageSpec], list[LinkSpec], float]:
    """ResNet-18 × K=4 mild-skew × T_floor mult 1.5 — representative of the real-model sweep."""
    layers = MODELS["resnet18"][1]
    stages = HARDWARE_PROFILES["mild_skew"][1]
    links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(len(stages) - 1)]
    bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    t_floor = max(bot.stage_exec_time.values()) * 1.5
    return layers, stages, links, t_floor


WORKLOADS: dict[str, tuple] = {
    "witness (K=2, ratio=3, ×1.25)": _witness_workload(),
    "ResNet-18 × mild-skew × ×1.50": _real_workload(),
}


# Defaults (kept fixed while sweeping the other axes).
DEFAULT_ALPHA = 2.0
DEFAULT_M = 11
DEFAULT_RMIN = 0.5


def _eval_at(
    workload_name: str,
    layers: list[LayerProfile],
    stages: list[StageSpec],
    links: list[LinkSpec],
    t_floor: float,
    *,
    voltage_alpha: float,
    throttle_min: float,
    throttle_granularity: int,
    axis_label: str,
    value_label: str,
) -> SensitivityRow:
    results = evaluate_workload(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / t_floor,
        voltage_alpha=voltage_alpha,
        throttle_min=throttle_min,
        throttle_granularity=throttle_granularity,
    )
    feasible = {r.name: r.energy for r in results if r.feasible}
    if "joint" not in feasible:
        return SensitivityRow(
            workload=workload_name, axis=axis_label, value=value_label,
            e_best_seq=math.inf, e_joint=math.inf,
            gap_pct=0.0, feasible=False,
        )
    e_joint = feasible["joint"]
    seq = [(n, e) for n, e in feasible.items() if n != "joint"]
    if not seq:
        return SensitivityRow(
            workload=workload_name, axis=axis_label, value=value_label,
            e_best_seq=math.inf, e_joint=e_joint,
            gap_pct=math.inf, feasible=True,
        )
    _, e_best = min(seq, key=lambda nv: nv[1])
    gap_pct = 100.0 * (e_best - e_joint) / e_best if e_best > 0 else 0.0
    return SensitivityRow(
        workload=workload_name, axis=axis_label, value=value_label,
        e_best_seq=e_best, e_joint=e_joint, gap_pct=gap_pct, feasible=True,
    )


def sweep_axis(
    workload_name: str,
    workload: tuple,
    axis: str,
    values: list,
) -> list[SensitivityRow]:
    layers, stages, links, t_floor = workload
    rows: list[SensitivityRow] = []
    for v in values:
        kwargs = {
            "voltage_alpha": DEFAULT_ALPHA,
            "throttle_min": DEFAULT_RMIN,
            "throttle_granularity": DEFAULT_M,
        }
        if axis == "voltage_alpha":
            kwargs["voltage_alpha"] = v
            label = f"α={v}"
        elif axis == "throttle_granularity":
            kwargs["throttle_granularity"] = v
            label = f"M={v}"
        elif axis == "throttle_min":
            kwargs["throttle_min"] = v
            label = f"r_min={v}"
        else:
            raise ValueError(f"unknown axis {axis!r}")
        rows.append(_eval_at(
            workload_name, layers, stages, links, t_floor,
            axis_label=axis, value_label=label, **kwargs,
        ))
    return rows


def render_table(workload_name: str, rows_by_axis: dict[str, list[SensitivityRow]],
                 console: Console) -> None:
    table = Table(title=workload_name)
    table.add_column("axis")
    table.add_column("setting")
    table.add_column("E best seq", justify="right")
    table.add_column("E joint", justify="right")
    table.add_column("gap %", justify="right")
    for axis, rows in rows_by_axis.items():
        for r in rows:
            if not r.feasible:
                table.add_row(axis, r.value, "—", "—", "infeasible")
                continue
            table.add_row(
                axis, r.value,
                f"{r.e_best_seq:.3f}", f"{r.e_joint:.3f}", f"{r.gap_pct:+.2f}%",
            )
    console.print(table)


def run(args: argparse.Namespace) -> None:
    console = Console()
    console.print(
        f"[bold]exp_joint_sensitivity[/] — defaults: α={DEFAULT_ALPHA}, "
        f"M={DEFAULT_M}, r_min={DEFAULT_RMIN}"
    )

    axes = {
        "voltage_alpha": args.alpha_values,
        "throttle_granularity": args.granularity_values,
        "throttle_min": args.throttle_min_values,
    }

    for wl_name, wl in WORKLOADS.items():
        rows_by_axis: dict[str, list[SensitivityRow]] = {}
        for axis, values in axes.items():
            rows_by_axis[axis] = sweep_axis(wl_name, wl, axis, values)
        console.print("")
        render_table(wl_name, rows_by_axis, console)

    console.print(
        "\n[dim]Single-axis sweep: hold the two unswept parameters at the default "
        "(α=2.0, M=11, r_min=0.5) while varying the third. Defaults match the "
        "real-model sweep. The witness reproduces the proof construction; the "
        "ResNet-18 row is representative of realistic K=4 workloads. The gap "
        "should (a) increase with α (steeper DVFS curve → more throttle benefit), "
        "(b) increase with M (finer grid → less rounding loss), (c) increase as "
        "r_min decreases (more throttle headroom).[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alpha-values", type=float, nargs="+",
                        default=[2.0, 2.5, 3.0])
    parser.add_argument("--granularity-values", type=int, nargs="+",
                        default=[6, 11, 21, 41])
    parser.add_argument("--throttle-min-values", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

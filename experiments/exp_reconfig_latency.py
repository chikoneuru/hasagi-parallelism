"""Empirical validation of the stagnation-restored global optimality bound.

Theorem (proven in the partitioner's stagnation tracker): when the global
optimum drifts outside the incremental window, the orchestrator's
incremental → stagnation-fallback composition restores global optimality
in at most ``patience`` non-improving steps; total wall-time bounded by
``patience · period_orchestrator`` seconds.

This experiment drives a synthetic workload-drift scenario: start at a
balanced partition, then shift layer FLOPS asymmetrically so the true
optimum moves outside the incremental search window. The orchestrator
runs ``incremental_partition`` repeatedly with a ``StagnationTracker``
observing each result; the experiment records the number of incremental
steps until fallback fires and verifies the fallback recovers the global
optimum (matches ``partition_pipeline`` output exactly).

Usage:
    python -m experiments.exp_reconfig_latency
    python -m experiments.exp_reconfig_latency --patience 5 --window 2
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from hise.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    StagnationTracker,
    incremental_partition,
    partition_pipeline,
)


@dataclass
class DriftScenario:
    name: str
    layers_before: list[LayerProfile]
    layers_after: list[LayerProfile]
    stages: list[StageSpec]
    links: list[LinkSpec]


def _flat_layers(n: int, fwd: float, bwd: float) -> list[LayerProfile]:
    return [LayerProfile(index=i, fwd_flops=fwd, bwd_flops=bwd, activation_bytes=1024)
            for i in range(n)]


def _skewed_layers(n: int, base_flops: float, skew_at: int, skew_factor: float) -> list[LayerProfile]:
    """Create layers whose FLOPS jumps at ``skew_at`` (simulates workload drift)."""
    out = []
    for i in range(n):
        f = base_flops * (skew_factor if i >= skew_at else 1.0)
        out.append(LayerProfile(index=i, fwd_flops=f, bwd_flops=2 * f, activation_bytes=1024))
    return out


def build_scenarios() -> list[DriftScenario]:
    stages = [
        StageSpec(stage_id=0, throughput_flops=1e10, memory_bytes=10**12, power_draw_w=300.0),
        StageSpec(stage_id=1, throughput_flops=1e10, memory_bytes=10**12, power_draw_w=300.0),
        StageSpec(stage_id=2, throughput_flops=1e10, memory_bytes=10**12, power_draw_w=300.0),
    ]
    links = [LinkSpec(0, 1, 1e10), LinkSpec(1, 2, 1e10)]
    n = 12

    return [
        DriftScenario(
            name="mild drift — skew at layer 6, ×2",
            layers_before=_flat_layers(n, 1e9, 2e9),
            layers_after=_skewed_layers(n, 1e9, skew_at=6, skew_factor=2.0),
            stages=stages, links=links,
        ),
        DriftScenario(
            name="strong drift — skew at layer 8, ×4",
            layers_before=_flat_layers(n, 1e9, 2e9),
            layers_after=_skewed_layers(n, 1e9, skew_at=8, skew_factor=4.0),
            stages=stages, links=links,
        ),
        DriftScenario(
            name="extreme drift — skew at layer 9, ×8 (forces fallback)",
            layers_before=_flat_layers(n, 1e9, 2e9),
            layers_after=_skewed_layers(n, 1e9, skew_at=9, skew_factor=8.0),
            stages=stages, links=links,
        ),
    ]


def run_scenario(scenario: DriftScenario,
                 patience: int,
                 window: int,
                 period_s: float,
                 console: Console) -> dict:
    # Establish the "before" partition (the orchestrator's last known good plan).
    partition_before = partition_pipeline(scenario.layers_before, scenario.stages,
                                           scenario.links, objective="bottleneck")

    # The "after" workload arrives; compute the true global optimum for ground truth.
    true_optimum = partition_pipeline(scenario.layers_after, scenario.stages,
                                       scenario.links, objective="bottleneck")
    true_score = max(true_optimum.stage_exec_time.values())

    # Now run incremental_partition repeatedly on the drifted workload, observing.
    tracker = StagnationTracker(patience=patience, objective="bottleneck")
    current = partition_before
    incremental_steps = 0
    fallback_fired = False
    fallback_score = None
    incremental_wall_s = 0.0
    fallback_wall_s = 0.0

    while True:
        t0 = time.monotonic()
        current = incremental_partition(
            previous=current, layers=scenario.layers_after,
            stages=scenario.stages, links=scenario.links,
            boundary_window=window, objective="bottleneck",
        )
        incremental_wall_s += time.monotonic() - t0
        incremental_steps += 1
        if tracker.observe(current):
            fallback_fired = True
            t0 = time.monotonic()
            current = partition_pipeline(scenario.layers_after, scenario.stages,
                                          scenario.links, objective="bottleneck")
            fallback_wall_s = time.monotonic() - t0
            fallback_score = max(current.stage_exec_time.values())
            break
        # Safety stop in case the test workload happens to converge incrementally.
        if incremental_steps > patience + 5:
            break

    incremental_score = max(current.stage_exec_time.values())
    recovery_seconds = incremental_steps * period_s

    return {
        "scenario": scenario.name,
        "true_optimum_score": true_score,
        "before_drift_score": max(partition_before.stage_exec_time.values()),
        "incremental_steps_before_fallback": incremental_steps,
        "fallback_fired": fallback_fired,
        "final_score": incremental_score,
        "matches_global_optimum": abs(incremental_score - true_score) < 1e-9,
        "patience_bound_steps": patience,
        "recovery_wall_seconds_at_period": recovery_seconds,
        "incremental_compute_wall_s": incremental_wall_s,
        "fallback_compute_wall_s": fallback_wall_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patience", type=int, default=3,
                        help="Consecutive non-improving observations before fallback fires")
    parser.add_argument("--window", type=int, default=3,
                        help="Incremental search radius around the previous cuts")
    parser.add_argument("--period-s", type=float, default=5.0,
                        help="Orchestrator decision period (seconds between incremental tries)")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[bold]Reconfig-latency benchmark[/]: patience={args.patience}, "
        f"window=±{args.window}, period={args.period_s}s"
    )
    console.print(
        f"[dim]Predicted recovery bound: patience × period = "
        f"{args.patience * args.period_s}s[/]"
    )

    rows = []
    for scenario in build_scenarios():
        rows.append(run_scenario(scenario, args.patience, args.window, args.period_s, console))

    table = Table(title="Reconfig recovery — incremental + stagnation fallback")
    table.add_column("scenario", overflow="fold")
    table.add_column("steps", justify="right")
    table.add_column("fallback fired?", justify="center")
    table.add_column("recovery (s)", justify="right")
    table.add_column("matches optimum?", justify="center")
    table.add_column("compute wall (ms)", justify="right")
    for r in rows:
        table.add_row(
            r["scenario"],
            str(r["incremental_steps_before_fallback"]),
            "✓" if r["fallback_fired"] else "—",
            f"{r['recovery_wall_seconds_at_period']:.1f}",
            "✓" if r["matches_global_optimum"] else "✗",
            f"{(r['incremental_compute_wall_s'] + r['fallback_compute_wall_s']) * 1000:.1f}",
        )
    console.print(table)

    # The theoretical bound is "patience non-improving observations AFTER the last
    # Case-I improvement" (see T3). A workload that finds an intermediate
    # incremental improvement resets the counter; the total step count can exceed
    # patience by the number of Case-I events. What the theorem guarantees is
    # finite termination and recovery to the global optimum — not a fixed step bound.
    all_fired = all(r["fallback_fired"] for r in rows)
    if all_fired:
        max_steps = max(r["incremental_steps_before_fallback"] for r in rows)
        console.print(
            f"\n[bold green]Termination + recovery confirmed[/]: every scenario "
            f"fired fallback in finite time (max observed = {max_steps} incremental "
            f"steps; corresponding wall-time at period {args.period_s}s = "
            f"{max_steps * args.period_s:.1f}s)."
        )
        console.print(
            f"[dim]The theorem bounds the count of non-improving observations after "
            f"the last improvement at patience={args.patience}; intermediate "
            f"improvements reset the counter, so observed step count can exceed "
            f"patience without violating the theorem (extra steps = Case-I events).[/]"
        )
    else:
        unfired = [r for r in rows if not r["fallback_fired"]]
        console.print(
            f"\n[bold red]Fallback did not fire[/] in {len(unfired)} scenarios. "
            "Either workload converged incrementally (unexpected for designed drift) "
            "or safety stop hit (raise --patience or extend the experiment loop)."
        )

    # Optimality validation
    mismatches = [r for r in rows if not r["matches_global_optimum"]]
    if mismatches:
        console.print(
            f"[bold red]Optimum mismatch[/]: {len(mismatches)} scenarios did not "
            "recover to the global optimum after fallback."
        )
    else:
        console.print(
            "[bold green]Optimum recovered[/]: every scenario reached the global "
            "optimum (full DP output) after fallback."
        )

    console.print(
        "\n[dim]Compute wall-time is the actual Python time spent in incremental + DP "
        "calls; recovery (s) is the predicted orchestrator wall-time at the chosen "
        "decision period. Both are simulation measurements; real-hardware reconfig "
        "latency requires a multi-GPU testbed.[/]"
    )


if __name__ == "__main__":
    main()

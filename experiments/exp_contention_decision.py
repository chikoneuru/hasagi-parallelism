"""Decision quality of the pipeline partitioner under co-tenant contention.

A serverless burst pool is multi-tenant: a co-located job can land on a subset of
the GPUs backing some pipeline stages and steal compute from them (the kind of
interference NVIDIA MPS / time-slicing produces). This is a SIMULATION study on
the decision layer — there is no distributed execution. It asks whether the
partitioner's decisions stay good when its per-stage throughput estimates are
perturbed by contention, and whether the implemented incremental re-planner
recovers the loss cheaply.

Two regimes, isolating the effect with a communication-light link (small
activations) so the result is about compute balance, not bandwidth — bandwidth
sensitivity is the separate concern of ``exp_comm_sensitivity``:

  1. UNIFORM contention. Every stage's effective throughput is scaled by the same
     factor c. With negligible comm the bottleneck partition is scale-invariant —
     argmin_s (flops_s / (c·thru)) = argmin_s (flops_s / thru) — so the optimal
     cuts do NOT change and a contention-blind plan carries ~0 regret. The
     partitioner correctly does nothing; this is the robustness boundary, and it
     means uniform slowdown (e.g. a global power cap) needs no re-partition.

  2. ASYMMETRIC contention. A co-tenant degrades only a subset of stages, so the
     load balance shifts materially: the contended stages should hold fewer
     layers. A contention-blind plan (cuts chosen on nominal throughput) leaves
     real regret that grows with severity; a contention-aware re-plan recovers it.

For regime 2 we exercise the implemented online machinery: starting from the
nominal-optimal cuts, repeatedly call ``incremental_partition`` (sliding cuts
within ±window) under a ``StagnationTracker``, and report the regret trajectory,
the step at which the cheap incremental walk reaches the contention-aware optimum,
and whether the tracker escalates to a full ``partition_pipeline`` re-solve. A
larger drift needs more steps at window ±1 than at ±3 — the cost of cheap local
repair versus a full re-plan.

The contention factor c is swept parametrically here; the measured 2-/3-tenant
degradation from ``exp_cotenant_contention`` lands within this swept range and
anchors which c values are realistic.

Usage::

    python -m experiments.exp_contention_decision --out artifacts/contention_decision.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hasagi.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    Partition,
    StageSpec,
    StagnationTracker,
    incremental_partition,
    partition_pipeline,
)

_THRU = 3.5e13      # nominal aggregate stage throughput (FLOPS), ~3080 Ti-class
_BW = 4.8e12        # NVLink-class link; with the small activations below, comm ~ 0


def _uniform_layers(n: int) -> list[LayerProfile]:
    """A uniform layer chain with tiny activations, so the bottleneck partition is
    driven by compute balance and comm is negligible. Labelled synthetic."""
    return [
        LayerProfile(index=i, fwd_flops=2.0e9, bwd_flops=4.0e9, activation_bytes=2_000_000)
        for i in range(n)
    ]


def _stages(k: int, thru: float = _THRU, power_w: float = 250.0) -> list[StageSpec]:
    return [
        StageSpec(stage_id=s, throughput_flops=thru,
                  memory_bytes=10_000_000_000, power_draw_w=power_w)
        for s in range(k)
    ]


def _links(k: int, bw: float = _BW) -> list[LinkSpec]:
    return [LinkSpec(src_stage=s, dst_stage=s + 1, bandwidth_bps=bw) for s in range(k - 1)]


def _apply_contention(stages: list[StageSpec], factors: dict[int, float]) -> list[StageSpec]:
    """Scale each stage's effective throughput by factors[stage_id] (default 1.0).
    A co-tenant stealing half a stage's GPU -> factor 0.5."""
    return [
        StageSpec(
            stage_id=s.stage_id,
            throughput_flops=s.throughput_flops * factors.get(s.stage_id, 1.0),
            memory_bytes=s.memory_bytes,
            power_cap_w=s.power_cap_w,
            power_draw_w=s.power_draw_w,
        )
        for s in stages
    ]


def _score(part: Partition, objective: str) -> float:
    """Bottleneck = max stage exec time (microbatch-count independent); energy =
    energy_per_iter. Both are what the partitioner minimises under each objective."""
    if objective == "energy":
        return part.energy_per_iter
    return max(part.stage_exec_time.values())


def _eval_at_cuts(layers: list[LayerProfile], stages: list[StageSpec],
                  links: list[LinkSpec], cuts: tuple[int, ...]) -> Partition:
    """Evaluate a FIXED cut set under the given stages/links without re-optimising
    (incremental_partition with window 0 rebuilds exactly ``cuts``)."""
    prev = Partition(cuts=tuple(cuts), num_stages=len(stages))
    return incremental_partition(prev, layers, stages, links, boundary_window=0)


def _blind_vs_aware_regret(layers: list[LayerProfile], stages_nominal: list[StageSpec],
                           links: list[LinkSpec], factors: dict[int, float],
                           objective: str) -> dict:
    """Regret of the contention-BLIND plan (cuts chosen on nominal throughput,
    then run under contention) versus the contention-AWARE optimum (re-planned
    with the degraded throughputs)."""
    contended = _apply_contention(stages_nominal, factors)
    blind_cuts = partition_pipeline(layers, stages_nominal, links, objective=objective).cuts
    aware = partition_pipeline(layers, contended, links, objective=objective)
    blind_under_contention = _eval_at_cuts(layers, contended, links, blind_cuts)
    aware_s = _score(aware, objective)
    blind_s = _score(blind_under_contention, objective)
    regret = blind_s / aware_s - 1.0 if aware_s > 0 else 0.0
    return {
        "blind_cuts": list(blind_cuts),
        "aware_cuts": list(aware.cuts),
        "cuts_changed": tuple(blind_cuts) != tuple(aware.cuts),
        "regret": regret,
    }


def _incremental_recovery(layers: list[LayerProfile], stages_nominal: list[StageSpec],
                          links: list[LinkSpec], factors: dict[int, float], objective: str,
                          window: int = 1, patience: int = 3, max_steps: int = 30) -> dict:
    """Start from the nominal-optimal cuts, apply contention, and run incremental
    repartition under a StagnationTracker. Report the per-step regret toward the
    contention-aware optimum, the step at which regret first falls below 1%, and
    whether/when the tracker escalated to a full DP re-solve."""
    contended = _apply_contention(stages_nominal, factors)
    aware = partition_pipeline(layers, contended, links, objective=objective)
    aware_s = _score(aware, objective)

    prev = partition_pipeline(layers, stages_nominal, links, objective=objective)
    tracker = StagnationTracker(patience=patience, objective=objective)
    tracker.reset_all()

    trajectory: list[float] = []
    recovered_step: int | None = None
    fallback_step: int | None = None
    for step in range(max_steps):
        incr = incremental_partition(prev, layers, contended, links,
                                     boundary_window=window, objective=objective)
        if tracker.observe(incr):
            # Window stalled -> escalate to full DP (the orchestrator's fallback).
            incr = partition_pipeline(layers, contended, links, objective=objective)
            if fallback_step is None:
                fallback_step = step
            tracker.reset()
        regret = _score(incr, objective) / aware_s - 1.0 if aware_s > 0 else 0.0
        trajectory.append(regret)
        if recovered_step is None and regret <= 0.01:
            recovered_step = step
        prev = incr
        if regret <= 1e-9:
            break

    return {
        "initial_regret": trajectory[0] if trajectory else 0.0,
        "final_regret": trajectory[-1] if trajectory else 0.0,
        "recovered_step": recovered_step,
        "fallback_step": fallback_step,
        "window": window,
        "steps": len(trajectory),
        "trajectory": trajectory,
    }


def _scenario_table(console: Console, title: str, rows: list[dict]) -> None:
    t = Table(title=title)
    t.add_column("c", justify="right")
    t.add_column("blind cuts")
    t.add_column("aware cuts")
    t.add_column("moved?", justify="center")
    t.add_column("regret %", justify="right")
    for r in rows:
        t.add_row(f"{r['c']:.2f}", str(r["blind_cuts"]), str(r["aware_cuts"]),
                  "yes" if r["cuts_changed"] else "no", f"{r['regret'] * 100:.2f}")
    console.print(t)


def run(args: argparse.Namespace) -> int:
    console = Console()
    layers = _uniform_layers(args.layers)
    k = args.stages
    links = _links(k)
    stages = _stages(k)
    c_grid = [round(0.3 + 0.1 * i, 2) for i in range(7)]  # 0.3 .. 0.9
    summary: dict = {"layers": args.layers, "stages": k, "objective": args.objective,
                     "c_grid": c_grid, "regimes": {}, "recovery": {}}

    # --- Regime 1: uniform contention (scale-invariant) ---
    rows1 = []
    for c in c_grid:
        factors = dict.fromkeys(range(k), c)
        r = _blind_vs_aware_regret(layers, stages, links, factors, args.objective)
        rows1.append({"c": c, **r})
    _scenario_table(console, "Regime 1 — UNIFORM contention "
                             "(expected: scale-invariant, ~0 regret)", rows1)
    summary["regimes"]["uniform"] = rows1
    console.print(f"  max regret across c: {max(r['regret'] for r in rows1) * 100:.3f}% "
                  "— decision robust; uniform slowdown needs no re-partition.\n")

    # --- Regime 2: asymmetric contention (co-tenant on the first half of stages) ---
    affected = list(range(max(1, k // 2)))
    rows2 = []
    for c in c_grid:
        factors = dict.fromkeys(affected, c)
        r = _blind_vs_aware_regret(layers, stages, links, factors, args.objective)
        rows2.append({"c": c, **r})
    _scenario_table(console, f"Regime 2 — ASYMMETRIC contention on stages {affected} "
                             "(blind plan mis-balances load)", rows2)
    summary["regimes"]["asymmetric"] = rows2
    console.print(f"  max regret across c: {max(r['regret'] for r in rows2) * 100:.2f}% "
                  "— a contention-aware re-plan recovers this.\n")

    # --- Incremental recovery for the worst asymmetric case ---
    worst = max(rows2, key=lambda r: r["regret"])
    c_worst = worst["c"]
    factors = dict.fromkeys(affected, c_worst)
    console.print(f"[bold]Incremental recovery[/] (asymmetric c={c_worst}, "
                  f"affected stages {affected}, objective={args.objective}; "
                  f"nominal cuts {worst['blind_cuts']} -> aware {worst['aware_cuts']}):")
    for window in (1, 3):
        rec = _incremental_recovery(layers, stages, links, factors, args.objective,
                                    window=window, max_steps=args.max_steps)
        summary["recovery"][f"window_{window}"] = rec
        rec_str = (f"recovered to <1% at step {rec['recovered_step']}"
                   if rec["recovered_step"] is not None else "did NOT reach <1% within budget")
        fb_str = (f"; full-DP fallback fired at step {rec['fallback_step']}"
                  if rec["fallback_step"] is not None else "; no fallback needed")
        console.print(f"  window ±{window}: initial regret {rec['initial_regret']*100:.2f}% -> "
                      f"final {rec['final_regret']*100:.3f}%; {rec_str}{fb_str}")
    console.print()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--objective", choices=("bottleneck", "energy"), default="bottleneck")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

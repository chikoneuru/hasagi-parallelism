"""When does carbon-driven REPARTITION pay for a serverless hybrid-parallel job?

This is the break-even study for the one genuinely-unoccupied cell in the prior
art (carbon-signal -> intra-job repartition of the hybrid-parallel LAYOUT, vs the
carbon-blind reshard of Tenplex/DynaTrain/ResiHP and the worker-COUNT scaling of
CarbonScaler). It is a SIMULATION on the decision layer (no torch.distributed);
real multi-GPU is the future validation.

The question the field has not posed: a carbon-aware controller could respond to a
dirty grid window by (a) THROTTLING the power cap (DVFS; effectively free, no state
moves) or (b) REPARTITIONING to a lower-power layout (changing the parallel layout;
pays a serverless state-migration + cold-start cost every switch). Throttle is the
cheap lever HASAGI already measured (~15% energy at ~+30% latency). Repartition is
the expensive structural lever. So: *is the carbon signal ever worth routing into
the structural lever instead of the free one, once migration is charged?*

Two honest groundings, both pushing the answer toward "rarely":
  1. PP cut-point repartition alone has almost NO energy lever. Running the
     partitioner's energy objective vs its bottleneck objective on a uniform-power
     model yields the SAME energy-per-iter at lower throughput (see
     ``_cutpoint_energy_gap``) -- moving cut-points trades throughput, not energy.
     So a repartition that actually saves energy must change the layout's POWER
     (fewer/again-different replicas), which is parametrised here as the eco layout.
  2. We give repartition its BEST CASE: the eco layout is granted a real
     energy reduction (default 20%) -- if it still loses to free throttle once
     migration is charged, the negative result is strong.

Output: the break-even surface over (carbon swing x migrated state size x migration
bandwidth) -- where, if anywhere, carbon-driven repartition's net carbon (including
migration) beats carbon-aware throttle, and at what makespan cost.

Usage::

    python -m experiments.exp_carbon_repartition_breakeven \
        --out artifacts/carbon_repartition_breakeven.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hasagi.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    partition_pipeline,
)

_J_PER_KWH = 3.6e6


@dataclass(frozen=True)
class Layout:
    name: str
    energy_per_iter_j: float
    throughput_iter_s: float


# ---------------------------------------------------------------------------
# Grounding: PP cut-point repartition barely moves energy
# ---------------------------------------------------------------------------

def _cutpoint_energy_gap() -> dict:
    """Run the partitioner's energy vs bottleneck objective on a uniform-power
    hybrid-parallel model and report the energy-per-iter ratio. Demonstrates that
    moving cut-points alone trades throughput, not energy (ratio ~ 1.0), so the
    structural energy lever must come from a power/layout change, not cut placement."""
    layers = [LayerProfile(index=i, fwd_flops=1.0e9, bwd_flops=2.0e9, activation_bytes=4_000_000)
              for i in range(16)]
    stages = [StageSpec(stage_id=s, throughput_flops=3.5e13, memory_bytes=10_000_000_000,
                        power_draw_w=250.0) for s in range(4)]
    links = [LinkSpec(src_stage=s, dst_stage=s + 1, bandwidth_bps=1.0e11) for s in range(3)]
    fast = partition_pipeline(layers, stages, links, num_microbatches=8, objective="bottleneck")
    eco = partition_pipeline(layers, stages, links, num_microbatches=8, objective="energy")
    return {
        "fast_cuts": list(fast.cuts), "eco_cuts": list(eco.cuts),
        "fast_energy_per_iter": fast.energy_per_iter,
        "eco_energy_per_iter": eco.energy_per_iter,
        "energy_ratio_eco_over_fast": eco.energy_per_iter / fast.energy_per_iter,
        "fast_throughput": 1.0 / max(fast.stage_exec_time.values()),
        "eco_throughput": 1.0 / max(eco.stage_exec_time.values()),
    }


# ---------------------------------------------------------------------------
# Carbon trace (parametric so swing can be swept)
# ---------------------------------------------------------------------------

def parametric_trace(hours: int, swing: float, mean_g: float = 400.0, period_h: int = 24) -> list[float]:
    """Diurnal carbon-intensity trace. ``swing`` in [0,1] is the peak fractional
    deviation from the mean (0 = flat, 0.8 = +/-80%). Clamped to stay positive."""
    out: list[float] = []
    for h in range(hours):
        val = mean_g * (1.0 + swing * math.sin(2.0 * math.pi * h / period_h))
        out.append(max(val, 1.0))
    return out


def _threshold(trace: list[float], q: float) -> float:
    s = sorted(trace)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx]


# ---------------------------------------------------------------------------
# Migration cost (the price of the structural lever)
# ---------------------------------------------------------------------------

def migration_cost(state_gb: float, bw_gbps: float, cold_start_s: float, power_w: float) -> tuple[float, float]:
    """Return (time_s, energy_j) to repartition: move ``state_gb`` of model+optimiser
    state over ``bw_gbps`` plus a fixed cold-start, drawing ``power_w`` meanwhile."""
    move_s = (state_gb * 8.0) / max(bw_gbps, 1e-9)   # GB*8 = Gb; / Gbps = s
    t = move_s + cold_start_s
    return t, power_w * t


# ---------------------------------------------------------------------------
# Policy simulation
# ---------------------------------------------------------------------------

def simulate(trace: list[float], fast: Layout, eco: Layout, policy: str, *,
             iters_per_window: int, threshold_q: float,
             throttle_energy_frac: float, throttle_tput_frac: float,
             migration_energy_j: float = 0.0, migration_time_s: float = 0.0) -> dict:
    """Run ``len(trace)`` work windows under one policy. Each window does
    ``iters_per_window`` iters on the active layout at that window's intensity.

    Policies:
      static_fast / static_eco : fixed layout.
      throttle : fast layout; in dirty windows scale energy by throttle_energy_frac
                 and throughput by throttle_tput_frac (free, no migration).
      repartition : eco layout in dirty windows, fast in clean; pay migration cost
                    on every layout CHANGE.
    """
    thr = _threshold(trace, threshold_q)
    carbon_g = 0.0
    energy_j = 0.0
    makespan_s = 0.0
    switches = 0
    cur = "fast"
    for intensity in trace:
        dirty = intensity > thr
        if policy == "static_fast":
            e, t = fast.energy_per_iter_j, 1.0 / fast.throughput_iter_s
        elif policy == "static_eco":
            e, t = eco.energy_per_iter_j, 1.0 / eco.throughput_iter_s
        elif policy == "throttle":
            if dirty:
                e = fast.energy_per_iter_j * throttle_energy_frac
                t = (1.0 / fast.throughput_iter_s) / throttle_tput_frac
            else:
                e, t = fast.energy_per_iter_j, 1.0 / fast.throughput_iter_s
        elif policy == "repartition":
            want = "eco" if dirty else "fast"
            if want != cur:
                switches += 1
                carbon_g += (migration_energy_j / _J_PER_KWH) * intensity
                energy_j += migration_energy_j
                makespan_s += migration_time_s
                cur = want
            lay = eco if want == "eco" else fast
            e, t = lay.energy_per_iter_j, 1.0 / lay.throughput_iter_s
        else:
            raise ValueError(f"unknown policy {policy!r}")
        win_energy = e * iters_per_window
        carbon_g += (win_energy / _J_PER_KWH) * intensity
        energy_j += win_energy
        makespan_s += t * iters_per_window
    return {"policy": policy, "carbon_g": carbon_g, "energy_j": energy_j,
            "makespan_h": makespan_s / 3600.0, "switches": switches}


# ---------------------------------------------------------------------------
# Break-even sweep
# ---------------------------------------------------------------------------

def breakeven_sweep(fast: Layout, eco: Layout, *, swings: list[float], state_gbs: list[float],
                    bws: list[float], hours: int, iters_per_window: int, threshold_q: float,
                    throttle_energy_frac: float, throttle_tput_frac: float,
                    cold_start_s: float, migrate_power_w: float) -> list[dict]:
    """For each (swing, state_gb, bw): does carbon-driven repartition beat carbon-aware
    throttle on net carbon (including migration)? Returns one row per cell."""
    rows: list[dict] = []
    for swing in swings:
        trace = parametric_trace(hours, swing)
        thr = simulate(trace, fast, eco, "throttle", iters_per_window=iters_per_window,
                       threshold_q=threshold_q, throttle_energy_frac=throttle_energy_frac,
                       throttle_tput_frac=throttle_tput_frac)
        for state_gb in state_gbs:
            for bw in bws:
                mt, me = migration_cost(state_gb, bw, cold_start_s, migrate_power_w)
                rep = simulate(trace, fast, eco, "repartition", iters_per_window=iters_per_window,
                               threshold_q=threshold_q, throttle_energy_frac=throttle_energy_frac,
                               throttle_tput_frac=throttle_tput_frac,
                               migration_energy_j=me, migration_time_s=mt)
                delta = rep["carbon_g"] - thr["carbon_g"]   # <0 => repartition wins
                rows.append({
                    "swing": swing, "state_gb": state_gb, "bw_gbps": bw,
                    "carbon_throttle_g": thr["carbon_g"], "carbon_repartition_g": rep["carbon_g"],
                    "delta_g": delta, "repartition_wins": delta < 0,
                    "repartition_makespan_h": rep["makespan_h"], "throttle_makespan_h": thr["makespan_h"],
                    "switches": rep["switches"],
                })
    return rows


def breakeven_eco_strength(fast: Layout, *, swing: float, hours: int, iters_per_window: int,
                           threshold_q: float, throttle_energy_frac: float, throttle_tput_frac: float,
                           eco_tput_frac: float, state_gb: float, bw_gbps: float, cold_start_s: float,
                           migrate_power_w: float, fracs: list[float]) -> dict:
    """How strong must the structural (eco) lever be to beat free throttle? Sweep the
    eco layout's energy fraction; return the crossover frac below which carbon-driven
    repartition's net carbon (incl. migration) beats carbon-aware throttle."""
    trace = parametric_trace(hours, swing)
    thr = simulate(trace, fast, fast, "throttle", iters_per_window=iters_per_window,
                   threshold_q=threshold_q, throttle_energy_frac=throttle_energy_frac,
                   throttle_tput_frac=throttle_tput_frac)["carbon_g"]
    mt, me = migration_cost(state_gb, bw_gbps, cold_start_s, migrate_power_w)
    rows = []
    crossover = None
    for f in fracs:
        eco = Layout("eco", energy_per_iter_j=f, throughput_iter_s=eco_tput_frac)
        rep = simulate(trace, fast, eco, "repartition", iters_per_window=iters_per_window,
                       threshold_q=threshold_q, throttle_energy_frac=throttle_energy_frac,
                       throttle_tput_frac=throttle_tput_frac, migration_energy_j=me, migration_time_s=mt)["carbon_g"]
        wins = rep < thr
        rows.append({"eco_energy_frac": f, "carbon_repartition_g": rep, "wins": wins})
        if wins and crossover is None:
            crossover = f
    return {"throttle_carbon_g": thr, "throttle_energy_frac": throttle_energy_frac,
            "crossover_eco_frac": crossover, "rows": rows}


def run(args: argparse.Namespace) -> int:
    console = Console()

    gap = _cutpoint_energy_gap()
    console.print("[bold]Grounding: does moving pipeline cut-points save energy?[/]")
    console.print(f"  fast cuts {gap['fast_cuts']} vs eco cuts {gap['eco_cuts']}; "
                  f"energy ratio eco/fast = [bold]{gap['energy_ratio_eco_over_fast']:.3f}[/] "
                  f"(throughput {gap['eco_throughput']:.0f} vs {gap['fast_throughput']:.0f} it/s). "
                  "Cut-point repartition trades THROUGHPUT, not energy -> the energy lever must come "
                  "from a power/layout change, modelled below as the eco layout.\n")

    # Fast vs eco layout. eco is GRANTED a real energy reduction (best case for repartition).
    fast = Layout("fast", energy_per_iter_j=1.0, throughput_iter_s=1.0)
    eco = Layout("eco", energy_per_iter_j=args.eco_energy_frac,
                 throughput_iter_s=args.eco_tput_frac)

    # Headline scenario.
    trace = parametric_trace(args.hours, args.swing)
    mt, me = migration_cost(args.state_gb, args.bw_gbps, args.cold_start_s, args.migrate_power_w)
    common = dict(iters_per_window=args.iters_per_window, threshold_q=args.threshold_q,
                  throttle_energy_frac=args.throttle_energy_frac, throttle_tput_frac=args.throttle_tput_frac)
    pols = {p: simulate(trace, fast, eco, p, **common) for p in ("static_fast", "static_eco", "throttle")}
    pols["repartition"] = simulate(trace, fast, eco, "repartition", **common,
                                   migration_energy_j=me, migration_time_s=mt)

    t = Table(title=f"Policies @ swing={args.swing}, state={args.state_gb} GB, bw={args.bw_gbps} Gbps "
                    f"(eco={args.eco_energy_frac:.2f}E/{args.eco_tput_frac:.2f}T, throttle="
                    f"{args.throttle_energy_frac:.2f}E/{args.throttle_tput_frac:.2f}T)")
    t.add_column("policy")
    t.add_column("carbon (g)", justify="right")
    t.add_column("makespan (h)", justify="right")
    t.add_column("switches", justify="right")
    base = pols["static_fast"]["carbon_g"]
    for p, r in pols.items():
        t.add_row(p, f"{r['carbon_g']:.1f} ({100*(r['carbon_g']/base-1):+.1f}%)",
                  f"{r['makespan_h']:.2f}", str(r["switches"]))
    console.print(t)
    rep_vs_thr = pols["repartition"]["carbon_g"] - pols["throttle"]["carbon_g"]
    console.print(f"  repartition vs throttle: {rep_vs_thr:+.1f} g carbon "
                  f"({'REPARTITION WINS' if rep_vs_thr < 0 else 'throttle wins'}), "
                  f"at {pols['repartition']['makespan_h']:.2f}h vs {pols['throttle']['makespan_h']:.2f}h makespan.\n")

    # Break-even surface.
    rows = breakeven_sweep(
        fast, eco,
        swings=[0.2, 0.4, 0.6, 0.8], state_gbs=[1.0, 5.0, 20.0], bws=[10.0, 100.0, 600.0],
        hours=args.hours, iters_per_window=args.iters_per_window, threshold_q=args.threshold_q,
        throttle_energy_frac=args.throttle_energy_frac, throttle_tput_frac=args.throttle_tput_frac,
        cold_start_s=args.cold_start_s, migrate_power_w=args.migrate_power_w,
    )
    wins = [r for r in rows if r["repartition_wins"]]
    console.print(f"[bold]Break-even sweep[/]: repartition beats throttle on carbon in "
                  f"[bold]{len(wins)}/{len(rows)}[/] cells.")
    if wins:
        for r in wins:
            console.print(f"  WIN @ swing={r['swing']}, state={r['state_gb']}GB, bw={r['bw_gbps']}Gbps: "
                          f"{r['delta_g']:+.1f} g (makespan {r['repartition_makespan_h']:.1f}h)")
    else:
        console.print("  [yellow]Repartition NEVER beats free throttle in the swept range — "
                      "the carbon signal is not worth routing into the structural lever once migration "
                      "is charged. (Honest negative; this is the first characterisation of when it would.)[/]")

    # How strong must the structural lever be to beat free throttle?
    eco_be = breakeven_eco_strength(
        fast, swing=args.swing, hours=args.hours, iters_per_window=args.iters_per_window,
        threshold_q=args.threshold_q, throttle_energy_frac=args.throttle_energy_frac,
        throttle_tput_frac=args.throttle_tput_frac, eco_tput_frac=args.eco_tput_frac,
        state_gb=args.state_gb, bw_gbps=args.bw_gbps, cold_start_s=args.cold_start_s,
        migrate_power_w=args.migrate_power_w,
        fracs=[round(0.85 - 0.05 * i, 2) for i in range(12)],  # 0.85 .. 0.30
    )
    co = eco_be["crossover_eco_frac"]
    if co is None:
        console.print("[bold]Eco-lever strength break-even[/]: repartition does NOT beat throttle even "
                      "when the eco layout uses [bold]70% less[/] energy/iter — migration (cold-start "
                      f"floor {args.cold_start_s}s x {pols['repartition']['switches']} switches) dominates.\n")
    else:
        console.print(f"[bold]Eco-lever strength break-even[/]: repartition beats throttle only once the "
                      f"eco layout uses <= [bold]{co:.2f}x[/] fast energy/iter (i.e. saves "
                      f">= {100*(1-co):.0f}%, vs throttle's {100*(1-args.throttle_energy_frac):.0f}%) — "
                      "the structural lever must clear DVFS by a wide margin to justify migration.\n")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "cutpoint_energy_gap": gap,
            "headline": {p: r for p, r in pols.items()},
            "breakeven_rows": rows,
            "repartition_win_cells": len(wins),
            "total_cells": len(rows),
            "eco_strength_breakeven": eco_be,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=168)             # 7-day horizon
    p.add_argument("--iters-per-window", type=int, default=1000)
    p.add_argument("--swing", type=float, default=0.6)
    p.add_argument("--threshold-q", type=float, default=0.6)
    p.add_argument("--eco-energy-frac", type=float, default=0.80,  # eco layout saves 20% energy (best case)
                   help="eco layout energy-per-iter as a fraction of fast (lower = more saving)")
    p.add_argument("--eco-tput-frac", type=float, default=0.6,     # at 40% less throughput
                   help="eco layout throughput as a fraction of fast")
    p.add_argument("--throttle-energy-frac", type=float, default=0.85,  # measured eco-cap ~15% saving
                   help="throttle energy-per-iter fraction in dirty windows")
    p.add_argument("--throttle-tput-frac", type=float, default=0.7)
    p.add_argument("--state-gb", type=float, default=5.0)
    p.add_argument("--bw-gbps", type=float, default=100.0)
    p.add_argument("--cold-start-s", type=float, default=4.7)     # measured H3 cold start
    p.add_argument("--migrate-power-w", type=float, default=100.0)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

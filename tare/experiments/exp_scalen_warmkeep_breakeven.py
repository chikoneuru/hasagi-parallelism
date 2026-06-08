"""Zero-cost kill-gate: can carbon-aware elastic device-count scaling beat carbon
pausing on the (carbon, makespan) Pareto, NET of the mesh-resize cost, for a
single sharded training job -- using only data already on disk?

Motivation and the physics it hard-codes
-----------------------------------------
A single real multi-GPU measurement already showed that changing the parallel
LAYOUT at a fixed device count cannot cut compute energy (energy/iter ratio
1.04-1.10 vs the data-parallel layout): at fixed work and fixed resources the
compute joules are invariant and a layout change only adds communication. This
gate therefore refuses to model any energy saving from changing the layout or
the device count. It hard-codes three facts so it cannot smuggle one back in:

  1. always-on, reduced-count, pause, and scale-count all do the SAME total
     work, so they burn the SAME compute joules (only the wall-clock HOURS the
     joules land in differ, and that is the only thing that can move carbon).
  2. pause and scale-count both shift work out of dirty windows; pause shifts
     ALL of it (idle in dirty hours), scale-count shifts only the high-rate
     fraction (it keeps doing reduced-rate work in dirty hours).
  3. the only per-transition difference between pause and scale-count is the
     tax paid on each dirty/clean boundary: pause pays a full-state cold reload
     (state-size dependent, the measured resume curve), scale-count pays a mesh
     reshard (entered here as an analytical BRACKET, since the real number is
     unmeasured -- but published reshard systems put it in the seconds range
     even at hundreds of GB).

What the gate computes, per real zone over staggered diurnal start offsets:
  - always_on  : full count throughout (carbon-blind).
  - reduced_n  : reduced count throughout (the "just use fewer GPUs" efficiency
                 knob; carbon-blind).
  - pause      : full count in clean hours, idle in dirty hours; full reload on
                 each resume.
  - scale_n    : full count in clean hours, reduced count in dirty hours; mesh
                 reshard on each boundary (tax = a chosen bracket).

It then decomposes the carbon move into an EFFICIENCY component
(always_on -> reduced_n; NOT a contribution) and an AWARENESS component
(reduced_n -> scale_n; the carbon-tracking part), and asks whether scale_n is a
non-dominated point against pause -- specifically whether it beats pause on
carbon by more than its own reshard tax. Inference is the project's
zone-clustered bootstrap + exact sign-flip permutation over the real zones.

Pre-registered PASS (evaluated on the OPTIMISTIC reshard bracket first):
  (i)  scale_n's carbon is strictly BELOW pause's, clustered CI excluding zero;
  (ii) the awareness component exceeds 0.50 percentage points (deliberately well
       above the measured throttle awareness-null of ~0.33pp), CI excluding zero;
  (iii) construct guard: total compute energy is identical (to rounding) across
       always_on / reduced_n / pause / scale_n, confirming no energy was
       smuggled from the dead count/layout lever.
FAIL (ship the honest negative; do NOT rent hardware) if the optimistic bracket
cannot satisfy (i)+(ii) at ANY state size -- no real reshard number can then
rescue it. GO-MEASURE only if the optimistic bracket passes but the full-reload
(upper) bracket fails: the real reshard cost is then the single sign-flipping
unknown worth a tightly-scoped rental.

Pure analysis on cached traces; no GPU. Reuses the resume-cost model, the
zone-clustered statistics, and the real-trace loader.

Usage::

    python -m experiments.exp_scaleN_warmkeep_breakeven --out artifacts/scaleN_warmkeep_breakeven.json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.exp_resume_collapse import resume_cost
from tare.energy.carbon_trace import CarbonTrace
from tare.energy.trace_schedule import (
    GRID_ZONE_IDS,
    load_zone_traces,
    quantile_threshold,
    trace_to_hourly,
)
from tare.stats.bootstrap import (
    clustered_bootstrap_ci,
    clustered_permutation_pvalue,
)

_J_PER_KWH = 3.6e6
_HOUR_S = 3600.0


@dataclass(frozen=True)
class Tax:
    """One transition's cost: wall-clock seconds + energy (kWh)."""

    time_s: float
    energy_kwh: float


def reload_tax(state_gb: float, *, n: int, cold_start_s: float, resume_power_per_gpu_w: float,
               write_bw_gbps: float, reload_bw_gbps: float, warmup_s: float) -> Tax:
    """Pause's per-resume tax: a full-state cold reload of the whole mesh.

    Reuses the measured single-GPU resume curve (cold start + state-size write +
    state-size reload + warmup) and bills the held N-way mesh's power across it."""
    t, _ = resume_cost(state_gb, stateless=False, cold_start_s=cold_start_s,
                       resume_power_w=resume_power_per_gpu_w, write_bw_gbps=write_bw_gbps,
                       reload_bw_gbps=reload_bw_gbps, warmup_s=warmup_s)
    return Tax(t, resume_power_per_gpu_w * n * t / _J_PER_KWH)


def reshard_tax(state_gb: float, *, n: int, m: int, bracket: str, collective_bw_gbps: float,
                dynatrain_const_s: float, resume_power_per_gpu_w: float, reload_kwargs: dict) -> Tax:
    """scale-count's per-boundary tax, entered as a BRACKET (the real number is unmeasured):

      - "lower"     : optimistic. Only the moved fraction (|n-m|/max(n,m)) of the
                      sharded state crosses a fast collective interconnect; the
                      surviving ranks keep their shard resident, the job never
                      exits (no cold start, no warmup).
      - "dynatrain" : a published-fast-switch anchor -- an in-place P2P reshard
                      that stays in the seconds range even at very large state
                      (treated as a small constant, state-independent).
      - "upper"     : pessimistic. A full teardown identical to pause's reload
                      (no warm-keep advantage at all)."""
    if bracket == "upper":
        return reload_tax(state_gb, n=max(n, m), **reload_kwargs)
    if bracket == "dynatrain":
        t = dynatrain_const_s
    elif bracket == "lower":
        moved_frac = abs(n - m) / float(max(n, m))
        t = (moved_frac * state_gb * 8.0) / max(collective_bw_gbps, 1e-9)  # GB*8=Gb; /Gbps=s
    else:
        raise ValueError(f"unknown reshard bracket {bracket!r}")
    return Tax(t, resume_power_per_gpu_w * max(n, m) * t / _J_PER_KWH)


def replay(hourly: list[float], policy: str, *, start_hour: int, h_job_fngh: float, threshold: float,
           n: int, m: int, e_fngh_kwh: float, reload_t: Tax, reshard_t: Tax) -> dict:
    """Hour-stepped wall-clock replay of an equal-WORK job under one policy.

    Work is measured in full-count-GPU-hours (FNGH); the job needs ``h_job_fngh``
    of them. Each wall-clock hour does ``rate`` FNGH of work at the policy's
    device count, billed the grid intensity of the hour it occupies; the energy
    per FNGH is fixed (``e_fngh_kwh``), so no policy can cut compute joules. Mesh
    boundaries pay the appropriate tax, billed at the boundary hour's intensity.

    ``hourly`` is the zone's per-hour intensity (cycled); intensity lookups index
    it in O(1), consistent with the one-hour simulation step."""
    nh = len(hourly)

    def at(seconds: float) -> float:
        return hourly[int(seconds // _HOUR_S) % nh]

    carbon_g = 0.0
    compute_kwh = 0.0   # joules of actual work (must be identical across policies: construct guard)
    tax_kwh = 0.0       # transition (reload/reshard) energy; legitimately differs across policies
    work = 0.0
    clock = float(start_hour) * _HOUR_S
    start_s = clock
    switches = 0
    mode = None  # "N" (full) | "M" (reduced) | "P" (paused)
    while work < h_job_fngh - 1e-9:
        intensity = at(clock)
        dirty = intensity > threshold
        if policy == "always_on":
            want, rate = "N", 1.0
        elif policy == "reduced_n":
            want, rate = "M", m / float(n)
        elif policy == "pause":
            want, rate = ("P", 0.0) if dirty else ("N", 1.0)
        elif policy == "scale_n":
            want, rate = ("M", m / float(n)) if dirty else ("N", 1.0)
        else:
            raise ValueError(f"unknown policy {policy!r}")

        # pay the transition tax (if any) at the current hour's intensity
        if mode is not None and want != mode:
            if policy == "pause" and want == "N":   # resume: full reload
                tax = reload_t
            elif policy == "scale_n":                # mesh reshard on each boundary
                tax = reshard_t
            else:
                tax = Tax(0.0, 0.0)
            if tax.time_s or tax.energy_kwh:
                switches += 1
                carbon_g += tax.energy_kwh * at(clock)
                tax_kwh += tax.energy_kwh
                clock += tax.time_s
        mode = want

        if rate <= 0.0:        # idle (paused) hour: advance clock, no work, no job energy
            clock += _HOUR_S
            continue
        do = min(rate, h_job_fngh - work)
        win_energy = do * e_fngh_kwh
        dt = _HOUR_S * (do / rate)
        carbon_g += win_energy * at(clock + dt / 2.0)
        compute_kwh += win_energy
        work += do
        clock += dt
    return {"policy": policy, "carbon_g": carbon_g,
            "compute_kwh": compute_kwh, "tax_kwh": tax_kwh, "energy_kwh": compute_kwh + tax_kwh,
            "makespan_h": (clock - start_s) / _HOUR_S, "switches": switches}


def zone_metrics(trace: CarbonTrace, *, h_job_fngh: float, threshold_q: float, n: int, m: int,
                 e_fngh_kwh: float, reload_t: Tax, reshard_t: Tax, offset_stride_h: int) -> dict:
    """Per-offset efficiency/awareness/pause-delta for one zone, tiled over the trace."""
    hourly = trace_to_hourly(trace)
    thr = quantile_threshold(hourly, threshold_q)
    n_hours = len(hourly)
    # keep the whole pause makespan inside the trace (worst case: every dirty hour idles)
    dirty_frac = sum(1 for h in hourly if h > thr) / max(n_hours, 1)
    max_makespan_h = h_job_fngh / max(1.0 - dirty_frac, 1e-3) + h_job_fngh * reload_t.time_s / _HOUR_S

    eff_pp: list[float] = []
    awr_pp: list[float] = []
    pause_delta_g: list[float] = []     # pause_carbon - scale_carbon (>0 => scale beats pause)
    pause_makespan_delta_h: list[float] = []
    energy_spread: list[float] = []     # max-min total energy across structural policies (construct guard)
    start_hour = 0
    while start_hour + max_makespan_h <= n_hours + 1.0:
        common = dict(start_hour=start_hour, h_job_fngh=h_job_fngh, threshold=thr, n=n, m=m,
                      e_fngh_kwh=e_fngh_kwh, reload_t=reload_t, reshard_t=reshard_t)
        ao = replay(hourly, "always_on", **common)
        rn = replay(hourly, "reduced_n", **common)
        pa = replay(hourly, "pause", **common)
        sc = replay(hourly, "scale_n", **common)
        base = ao["carbon_g"]
        eff_pp.append(100.0 * (ao["carbon_g"] - rn["carbon_g"]) / base if base else 0.0)
        awr_pp.append(100.0 * (rn["carbon_g"] - sc["carbon_g"]) / base if base else 0.0)
        pause_delta_g.append(pa["carbon_g"] - sc["carbon_g"])
        pause_makespan_delta_h.append(pa["makespan_h"] - sc["makespan_h"])
        cc = [ao["compute_kwh"], rn["compute_kwh"], pa["compute_kwh"], sc["compute_kwh"]]
        energy_spread.append(max(cc) - min(cc))
        start_hour += offset_stride_h
    n_off = max(len(eff_pp), 1)
    return {
        "n_offsets": len(eff_pp),
        "efficiency_pp": eff_pp, "awareness_pp": awr_pp,
        "pause_delta_g": pause_delta_g, "pause_makespan_delta_h": pause_makespan_delta_h,
        "mean_efficiency_pp": sum(eff_pp) / n_off,
        "mean_awareness_pp": sum(awr_pp) / n_off,
        "mean_pause_delta_g": sum(pause_delta_g) / n_off,
        "mean_pause_makespan_delta_h": sum(pause_makespan_delta_h) / n_off,
        "max_energy_spread_kwh": max(energy_spread) if energy_spread else 0.0,
    }


def _clustered(per_zone: dict, key: str, rng: random.Random) -> dict:
    clusters = [d[key] for d in per_zone.values() if d[key]]
    if not clusters:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p": 1.0, "ci_excludes_zero": False}
    mean, lo, hi = clustered_bootstrap_ci(clusters, rng=rng)
    p = clustered_permutation_pvalue(clusters, rng=rng)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "p": p,
            "ci_excludes_zero": (lo > 0.0 or hi < 0.0)}


def evaluate_cell(ztraces: dict, *, state_gb: float, n: int, m: int, bracket: str,
                  h_job_fngh: float, threshold_q: float, e_fngh_kwh: float, offset_stride_h: int,
                  reload_kwargs: dict, collective_bw_gbps: float, dynatrain_const_s: float,
                  resume_power_per_gpu_w: float, awareness_pp_bar: float, rng: random.Random) -> dict:
    """One (state, degree-pair, reshard-bracket) cell: per-zone metrics + clustered verdict."""
    reload_t = reload_tax(state_gb, n=n, **reload_kwargs)
    reshard_t = reshard_tax(state_gb, n=n, m=m, bracket=bracket, collective_bw_gbps=collective_bw_gbps,
                            dynatrain_const_s=dynatrain_const_s,
                            resume_power_per_gpu_w=resume_power_per_gpu_w, reload_kwargs=reload_kwargs)
    per_zone = {}
    for z, zt in ztraces.items():
        if zt.source != "real-csv":
            continue
        per_zone[z] = zone_metrics(zt.trace, h_job_fngh=h_job_fngh, threshold_q=threshold_q, n=n, m=m,
                                   e_fngh_kwh=e_fngh_kwh, reload_t=reload_t, reshard_t=reshard_t,
                                   offset_stride_h=offset_stride_h)
    awareness = _clustered(per_zone, "awareness_pp", rng)
    pause_delta = _clustered(per_zone, "pause_delta_g", rng)
    efficiency = _clustered(per_zone, "efficiency_pp", rng)
    max_energy_spread = max((d["max_energy_spread_kwh"] for d in per_zone.values()), default=0.0)
    # construct guard: structural policies must burn identical compute joules
    construct_ok = max_energy_spread < 1e-6 * max(e_fngh_kwh * h_job_fngh, 1e-9)
    prong_i = pause_delta["mean"] > 0.0 and pause_delta["ci_lo"] > 0.0
    prong_ii = awareness["mean"] > awareness_pp_bar and awareness["ci_lo"] > 0.0
    return {
        "state_gb": state_gb, "n": n, "m": m, "bracket": bracket,
        "reload_tax_s": reload_t.time_s, "reshard_tax_s": reshard_t.time_s,
        "n_real_zones": len(per_zone),
        "efficiency_pp": efficiency, "awareness_pp": awareness, "pause_delta_g": pause_delta,
        "max_energy_spread_kwh": max_energy_spread, "construct_guard_ok": construct_ok,
        "prong_i_beats_pause_carbon": prong_i,
        "prong_ii_awareness_over_bar": prong_ii,
        "pass": prong_i and prong_ii and construct_ok,
    }


def run(args: argparse.Namespace, ztraces: dict) -> int:
    console = Console()
    n_full = args.n_full
    e_fngh_kwh = n_full * args.power_per_gpu_w / 1000.0     # 1 full-count-hour of work
    reload_kwargs = dict(cold_start_s=args.cold_start_s, resume_power_per_gpu_w=args.resume_power_per_gpu_w,
                         write_bw_gbps=args.write_bw_gbps, reload_bw_gbps=args.reload_bw_gbps,
                         warmup_s=args.warmup_s)
    state_gbs = [float(x) for x in args.state_gbs.split(",")]
    degree_pairs = [(n_full, n_full // 2), (n_full // 2, max(1, n_full // 4))]
    rng = random.Random(0)

    n_real = sum(1 for zt in ztraces.values() if zt.source == "real-csv")
    out: dict = {
        "premise": "fixed work => fixed compute joules across always_on/reduced_n/pause/scale_n; "
                   "carbon moves only with the wall-clock hours energy occupies, plus per-transition tax",
        "n_full": n_full, "h_job_full_count_hours": args.h_job, "threshold_q": args.threshold_q,
        "power_per_gpu_w": args.power_per_gpu_w, "n_real_zones": n_real,
        "awareness_pp_bar": args.awareness_pp_bar,
        "reshard_brackets": ["lower", "dynatrain", "upper"], "cells": [],
    }
    for bracket in ("lower", "dynatrain", "upper"):
        for (n, m) in degree_pairs:
            for s in state_gbs:
                out["cells"].append(evaluate_cell(
                    ztraces, state_gb=s, n=n, m=m, bracket=bracket, h_job_fngh=float(args.h_job),
                    threshold_q=args.threshold_q, e_fngh_kwh=e_fngh_kwh, offset_stride_h=args.offset_stride_h,
                    reload_kwargs=reload_kwargs, collective_bw_gbps=args.collective_bw_gbps,
                    dynatrain_const_s=args.dynatrain_const_s, resume_power_per_gpu_w=args.resume_power_per_gpu_w,
                    awareness_pp_bar=args.awareness_pp_bar, rng=rng))

    # the gate fires on the OPTIMISTIC bracket (best case for scale_n)
    optimistic = [c for c in out["cells"] if c["bracket"] in ("lower", "dynatrain")]
    any_pass = any(c["pass"] for c in optimistic)
    out["gate_pass"] = any_pass

    t = Table(title=f"scale-N vs pause kill-gate ({n_real} real zones; degree {n_full}->{n_full//2}; "
                    f"awareness bar {args.awareness_pp_bar:.2f}pp)")
    for c in ("bracket", "state GB", "reshard s", "reload s", "awareness pp [CI]",
              "scale-vs-pause Δg [CI]", "i", "ii", "guard"):
        t.add_column(c, justify="right" if c != "bracket" else "left")
    for c in out["cells"]:
        if (c["n"], c["m"]) != (n_full, n_full // 2):
            continue
        a, pd = c["awareness_pp"], c["pause_delta_g"]
        t.add_row(c["bracket"], f"{c['state_gb']:.3g}", f"{c['reshard_tax_s']:.1f}", f"{c['reload_tax_s']:.1f}",
                  f"{a['mean']:+.2f} [{a['ci_lo']:+.2f},{a['ci_hi']:+.2f}]",
                  f"{pd['mean']:+.2f} [{pd['ci_lo']:+.2f},{pd['ci_hi']:+.2f}]",
                  "Y" if c["prong_i_beats_pause_carbon"] else "n",
                  "Y" if c["prong_ii_awareness_over_bar"] else "n",
                  "ok" if c["construct_guard_ok"] else "BAD")
    console.print(t)

    console.print(
        f"\n[bold]KILL-GATE[/]: {'PASS' if any_pass else 'FAIL'} — "
        + ("a cell wins on the optimistic reshard bracket (scale-N beats pause on carbon AND clears the "
           "awareness bar); the real reshard cost is then worth a tightly-scoped measurement."
           if any_pass else
           "even the optimistic (warm / fast-switch) reshard bracket cannot make carbon-aware scale-N beat "
           "carbon pausing on carbon while clearing the awareness bar, at ANY state size. Pause already "
           "evacuates dirty windows entirely; scale-N keeps doing dirty-hour work, so it cannot undercut "
           "pause on carbon. Ship the honest negative; do NOT rent hardware."))
    # construct-guard sanity
    if not all(c["construct_guard_ok"] for c in out["cells"]):
        console.print("[red]construct guard FAILED: a policy moved compute joules — model bug.[/]")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        console.print(f"[dim]wrote {p}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default="data_cache/real_traces")
    p.add_argument("--real-dir-winter", default="data_cache/real_traces_winter")
    p.add_argument("--season", default="summer", choices=["summer", "winter"])
    p.add_argument("--n-full", type=int, default=8)
    p.add_argument("--h-job", type=int, default=48, help="total work in full-count-GPU-hours")
    p.add_argument("--offset-stride-h", type=int, default=24)
    p.add_argument("--threshold-q", type=float, default=0.6)
    p.add_argument("--power-per-gpu-w", type=float, default=300.0)
    p.add_argument("--state-gbs", default="1,20,80,160")
    p.add_argument("--awareness-pp-bar", type=float, default=0.50)   # >> the ~0.33pp throttle awareness-null
    # transition-cost model (reused/extrapolated from the measured resume curve)
    p.add_argument("--cold-start-s", type=float, default=4.7)
    p.add_argument("--resume-power-per-gpu-w", type=float, default=100.0)
    p.add_argument("--write-bw-gbps", type=float, default=3.9)
    p.add_argument("--reload-bw-gbps", type=float, default=7.7)
    p.add_argument("--warmup-s", type=float, default=0.761)
    p.add_argument("--collective-bw-gbps", type=float, default=200.0, help="fast interconnect for the warm reshard lower bound")
    p.add_argument("--dynatrain-const-s", type=float, default=4.36, help="published fast in-place reshard anchor (state-independent)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    real_dir = args.real_dir if args.season == "summer" else args.real_dir_winter
    ztraces = load_zone_traces(real_dir, GRID_ZONE_IDS)
    return run(args, ztraces)


if __name__ == "__main__":
    raise SystemExit(main())

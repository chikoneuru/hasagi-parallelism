"""Sensitivity sweep for the real-trace carbon panel.

The headline real-trace numbers come from a single scenario (one summer-2024
fortnight, dirty-quantile q=0.6, a 24 h job, dedicated vs reallocated idle).
A reviewer-flagged risk is that the "carbon-aware deferral barely helps / the
mechanisms are carbon-comparable" verdict is an artifact of that scenario. This
sweep re-runs the SAME reviewed core (:func:`_savings_over_offsets` + clustered
inference over the real zones) across a grid of the cheap, no-new-data knobs —
the dirty-window quantile ``q`` and the job length — and reports how the
load-bearing quantities move:

  - eco endpoint (should stay ~constant: it is a carbon-blind hardware ratio);
  - carbon-SIGNAL value (throttle minus same-budget blind throttle);
  - fair mechanism gap (throttle vs GREEN pause on identical oracle windows,
    reallocated idle) and its zone-dependent sign split;
  - GREEN's online+dedicated cell (the pessimistic "GREEN loses on flat grids"
    number) vs GREEN's offline+reallocated cell (GREEN at its own best).

If the carbon-signal value and the gap stay in the same place across q and job
length, the verdict is robust; if they swing, the carbon story is
scenario-dependent and must be reported as such. The SEASON knob (winter vs
summer) needs new data — set ``$ELECTRICITYMAPS_TOKEN`` and fetch another
fortnight via ``fetch_electricitymaps_traces`` into a second ``--real-dir``.

Usage::

    python -m experiments.exp_realtrace_sensitivity \
        --real-dir data_cache/real_traces \
        --profile artifacts/hardware-pareto-3080ti.json \
        --quantiles 0.5 0.6 0.7 0.8 --job-hours 12 24 48 96 \
        --out artifacts/realtrace_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.exp_realtrace_pareto import (
    _dirty_excess,
    _mean,
    _pearson,
    _savings_over_offsets,
)
from tare.energy.throttle_pareto import PowerCapProfile
from tare.energy.trace_schedule import (
    GRID_ZONE_IDS,
    load_zone_traces,
    quantile_threshold,
    trace_to_hourly,
)
from tare.stats.bootstrap import clustered_bootstrap_ci, clustered_permutation_pvalue


def _one_config(profile, ztraces, *, q, job_hours, throttle_cap, resume_kwh, dedicated_idle_w,
                stride_hours, green_window, rng):
    """Run the reviewed core for one (q, job_hours) and return cross-zone summary."""
    full = profile.max_throughput_cap
    total_iters = int(job_hours * 3600.0 * profile.point(full).throughput_iters_s)
    green_pause_fraction = max(0.0, min(1.0, 1.0 - q))
    clusters: dict[str, list[list[float]]] = {}
    per_zone_gap: list[float] = []
    per_zone_excess: list[float] = []
    for zt in ztraces.values():
        if zt.source != "real-csv":
            continue
        hourly = trace_to_hourly(zt.trace)
        threshold = quantile_threshold(hourly, q)
        res = _savings_over_offsets(
            profile, hourly, total_iters=total_iters, threshold=threshold,
            throttle_cap=throttle_cap, resume_kwh=resume_kwh, dedicated_idle_w=dedicated_idle_w,
            stride_hours=stride_hours, span_hours=int(job_hours * 2),
            green_pause_fraction=green_pause_fraction, green_window=green_window,
        )
        if res["n_offsets"] == 0:
            continue
        for k in ("eco_pct", "signal_pp", "gap_fair_pp", "green_on_ded_pct", "green_off_rea_pct"):
            clusters.setdefault(k, []).append(res[k])
        per_zone_gap.append(_mean(res["gap_fair_pp"]))
        per_zone_excess.append(_dirty_excess(hourly, threshold))

    def cz(k):
        return clustered_bootstrap_ci(clusters[k], rng=rng)

    pause_wins = sum(1 for g in per_zone_gap if g < 0)
    return {
        "q": q,
        "job_hours": job_hours,
        "n_real_zones": len(per_zone_gap),
        "eco_pct": list(cz("eco_pct")),
        "signal_pp": list(cz("signal_pp")),
        "signal_pvalue": clustered_permutation_pvalue(clusters["signal_pp"], rng=rng),
        "fair_gap_pp": list(cz("gap_fair_pp")),
        "fair_gap_pvalue": clustered_permutation_pvalue(clusters["gap_fair_pp"], rng=rng),
        "green_online_dedicated_pct": list(cz("green_on_ded_pct")),
        "green_offline_reallocated_pct": list(cz("green_off_rea_pct")),
        "corr_gap_vs_dirty_excess": _pearson(per_zone_gap, per_zone_excess),
        "pause_wins_carbon_n": pause_wins,
    }


def run(args: argparse.Namespace) -> int:
    console = Console()
    profile = PowerCapProfile.from_json(args.profile)
    throttle_cap = args.throttle_cap_w or profile.energy_optimal_cap
    resume_kwh = args.resume_energy_wh / 1000.0
    rng = random.Random(args.seed)
    ztraces = load_zone_traces(args.real_dir, GRID_ZONE_IDS)
    n_real = sum(1 for z in ztraces.values() if z.source == "real-csv")
    console.print(
        f"[bold]Real-trace sensitivity sweep[/] — {n_real} real zones; "
        f"q in {args.quantiles}; job-hours in {args.job_hours}; eco/throttle {throttle_cap:.0f} W; "
        f"idle dedicated {args.dedicated_idle_w:.0f} W vs reallocated 0 W (both in every cell)."
    )

    rows = [
        _one_config(
            profile, ztraces, q=q, job_hours=jh, throttle_cap=throttle_cap, resume_kwh=resume_kwh,
            dedicated_idle_w=args.dedicated_idle_w, stride_hours=args.stride_hours,
            green_window=args.green_window_hours, rng=rng,
        )
        for q in args.quantiles
        for jh in args.job_hours
    ]

    table = Table(title="Carbon-story robustness across q x job-length (cross-zone clustered means; 16 real zones)")
    table.add_column("q", justify="right")
    table.add_column("job h", justify="right")
    table.add_column("eco%", justify="right")
    table.add_column("signal pp [CI] (p)", justify="right")
    table.add_column("fair gap pp [CI] (p)", justify="right")
    table.add_column("GREEN on+ded%", justify="right")
    table.add_column("GREEN off+rea%", justify="right")
    table.add_column("corr / pause-wins", justify="right")
    for r in rows:
        s, g = r["signal_pp"], r["fair_gap_pp"]
        table.add_row(
            f"{r['q']:.2f}", f"{r['job_hours']:.0f}",
            f"{r['eco_pct'][0]:+.1f}",
            f"{s[0]:+.2f}[{s[1]:+.2f},{s[2]:+.2f}] ({r['signal_pvalue']:.3f})",
            f"{g[0]:+.2f}[{g[1]:+.2f},{g[2]:+.2f}] ({r['fair_gap_pvalue']:.3f})",
            f"{r['green_online_dedicated_pct'][0]:+.1f}",
            f"{r['green_offline_reallocated_pct'][0]:+.1f}",
            f"{r['corr_gap_vs_dirty_excess']:+.2f} / {r['pause_wins_carbon_n']}/{r['n_real_zones']}",
        )
    console.print(table)

    # Robustness verdict on the two load-bearing claims.
    sig_means = [r["signal_pp"][0] for r in rows]
    sig_sig = [r for r in rows if r["signal_pp"][1] > 0]   # CI lower bound > 0
    gap_means = [r["fair_gap_pp"][0] for r in rows]
    gap_sig = [r for r in rows if r["fair_gap_pvalue"] <= 0.05]
    console.print(
        f"\n[bold]Robustness[/]: carbon-signal value across the grid "
        f"= [{min(sig_means):+.2f}, {max(sig_means):+.2f}] pp, "
        f"significant (CI excludes 0) in {len(sig_sig)}/{len(rows)} cells. "
        f"Fair mechanism gap = [{min(gap_means):+.2f}, {max(gap_means):+.2f}] pp, "
        f"significant in {len(gap_sig)}/{len(rows)} cells."
    )
    console.print(
        "[dim]If signal value stays positive+significant and the fair gap stays near zero across the "
        "grid, the carbon verdict (modest signal value; throttle/pause carbon-comparable) is robust to "
        "q and job length. Season (winter swing) needs new data — set $ELECTRICITYMAPS_TOKEN and fetch.[/]"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "profile": args.profile,
            "eco_throttle_cap_w": throttle_cap,
            "quantiles": args.quantiles,
            "job_hours": args.job_hours,
            "dedicated_idle_w": args.dedicated_idle_w,
            "n_real_zones": n_real,
            "note": "season knob (winter vs summer) requires fetching another fortnight; not swept here",
            "cells": rows,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default="data_cache/real_traces")
    p.add_argument("--profile", default="artifacts/hardware-pareto-3080ti.json")
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    p.add_argument("--job-hours", type=float, nargs="+", default=[12.0, 24.0, 48.0, 96.0])
    p.add_argument("--throttle-cap-w", type=float, default=None)
    p.add_argument("--stride-hours", type=int, default=12)
    p.add_argument("--green-window-hours", type=int, default=24)
    p.add_argument("--resume-energy-wh", type=float, default=0.07)
    p.add_argument("--dedicated-idle-w", type=float, default=26.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

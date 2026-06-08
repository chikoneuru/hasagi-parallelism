"""How much of the large carbon-shifting win survives for a LARGE (must-shard) model,
once the state-size-dependent resume tax is paid on every pause/resume cycle?

The big carbon win in this project is carbon-aware PAUSING (run in clean hours, idle in
dirty hours): a large temporal-shift reduction vs an always-on, carbon-blind baseline.
That win is real and large for a stateless / small-state job. A LARGE training job pays,
on every dirty->clean resume, a checkpoint-write + state-reload + cold-start + warmup that
grows with model+optimiser state size (the wedge a stateless-FaaS carbon scheduler omits).
This experiment asks the headline question for serverless training of must-shard models:

    achievable_reduction(state, zone) = 100 * (always_on_carbon - pause_carbon_with_tax)
                                              / always_on_carbon

as a function of model state size, over the real 16-zone x 2-season traces, with the resume
tax charged per cycle at the resume hour's grid intensity. At ~0 state it recovers the pure
shifting win (the ceiling); as state grows the per-cycle tax erodes it. The verdict is the
state size at which the large win stops surviving.

Pure analysis on cached traces; no GPU. Reuses the wall-clock replay + state-dependent
reload-tax model + the zone-clustered statistics.

Usage::

    python -m experiments.exp_largemodel_carbon_ledger --out artifacts/largemodel_carbon_ledger.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.exp_scalen_warmkeep_breakeven import reload_tax, replay
from tare.energy.trace_schedule import (
    GRID_ZONE_IDS,
    load_zone_traces,
    quantile_threshold,
    trace_to_hourly,
)
from tare.stats.bootstrap import clustered_bootstrap_ci, clustered_permutation_pvalue

_HOUR_S = 3600.0


def zone_reductions(trace, *, state_gb: float, h_job: float, threshold_q: float, n_full: int,
                    e_fngh_kwh: float, reload_kwargs: dict, offset_stride_h: int) -> dict:
    """Per-offset carbon reduction of carbon-aware pause (with the state-size resume tax)
    vs always-on, for one zone."""
    hourly = trace_to_hourly(trace)
    thr = quantile_threshold(hourly, threshold_q)
    n_hours = len(hourly)
    dirty_frac = sum(1 for h in hourly if h > thr) / max(n_hours, 1)
    reload_t = reload_tax(state_gb, n=n_full, **reload_kwargs)
    max_makespan_h = h_job / max(1.0 - dirty_frac, 1e-3) + h_job * reload_t.time_s / _HOUR_S

    red: list[float] = []
    mk_factor: list[float] = []
    resumes: list[int] = []
    start_hour = 0
    while start_hour + max_makespan_h <= n_hours + 1.0:
        common = dict(start_hour=start_hour, h_job_fngh=h_job, threshold=thr, n=n_full,
                      m=n_full, e_fngh_kwh=e_fngh_kwh, reload_t=reload_t,
                      reshard_t=reload_t)  # reshard_t unused by always_on/pause
        ao = replay(hourly, "always_on", **common)
        pa = replay(hourly, "pause", **common)
        base = ao["carbon_g"]
        red.append(100.0 * (base - pa["carbon_g"]) / base if base else 0.0)
        mk_factor.append(pa["makespan_h"] / ao["makespan_h"] if ao["makespan_h"] else 1.0)
        resumes.append(pa["switches"])
        start_hour += offset_stride_h
    n = max(len(red), 1)
    return {"reductions": red, "n_offsets": len(red),
            "mean_reduction_pct": sum(red) / n,
            "mean_makespan_factor": sum(mk_factor) / n,
            "mean_resumes": sum(resumes) / n,
            "reload_tax_s": reload_t.time_s}


def run(args: argparse.Namespace, ztraces: dict) -> int:
    console = Console()
    n_full = args.n_full
    e_fngh_kwh = n_full * args.power_per_gpu_w / 1000.0
    reload_kwargs = dict(cold_start_s=args.cold_start_s, resume_power_per_gpu_w=args.resume_power_per_gpu_w,
                         write_bw_gbps=args.write_bw_gbps, reload_bw_gbps=args.reload_bw_gbps,
                         warmup_s=args.warmup_s)
    state_gbs = [float(x) for x in args.state_gbs.split(",")]
    rng = random.Random(0)

    n_real = sum(1 for zt in ztraces.values() if zt.source == "real-csv")
    out: dict = {"n_full": n_full, "h_job_full_count_hours": args.h_job, "threshold_q": args.threshold_q,
                 "power_per_gpu_w": args.power_per_gpu_w, "n_real_zones": n_real,
                 "policy": "carbon-aware pause (scale-to-zero in dirty hours) vs always-on, "
                           "with per-resume state-size reload tax", "rows": []}
    for s in state_gbs:
        per_zone = {}
        for z, zt in ztraces.items():
            if zt.source != "real-csv":
                continue
            per_zone[z] = zone_reductions(zt.trace, state_gb=s, h_job=float(args.h_job),
                                          threshold_q=args.threshold_q, n_full=n_full,
                                          e_fngh_kwh=e_fngh_kwh, reload_kwargs=reload_kwargs,
                                          offset_stride_h=args.offset_stride_h)
        clusters = [d["reductions"] for d in per_zone.values() if d["reductions"]]
        mean, lo, hi = clustered_bootstrap_ci(clusters, rng=rng) if clusters else (0.0, 0.0, 0.0)
        p = clustered_permutation_pvalue(clusters, rng=rng) if clusters else 1.0
        mk = sum(d["mean_makespan_factor"] for d in per_zone.values()) / max(len(per_zone), 1)
        res = sum(d["mean_resumes"] for d in per_zone.values()) / max(len(per_zone), 1)
        tax = next(iter(per_zone.values()))["reload_tax_s"] if per_zone else 0.0
        out["rows"].append({
            "state_gb": s, "mean_reduction_pct": mean, "ci_lo": lo, "ci_hi": hi, "permutation_p": p,
            "mean_makespan_factor": mk, "mean_resumes_per_run": res, "reload_tax_s_per_resume": tax,
            "ci_excludes_zero": (lo > 0.0 or hi < 0.0),
        })

    t = Table(title=f"Achievable carbon reduction of carbon-aware pause vs always-on, "
                    f"net of the state-size resume tax ({n_real} real zones, threshold_q={args.threshold_q})")
    for c in ("model state (GB)", "reduction % [95% CI]", "makespan x", "resumes/run", "reload tax (s)"):
        t.add_column(c, justify="right" if c != "model state (GB)" else "left")
    for r in out["rows"]:
        t.add_row(f"{r['state_gb']:.3g}",
                  f"{r['mean_reduction_pct']:.1f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]",
                  f"{r['mean_makespan_factor']:.2f}", f"{r['mean_resumes_per_run']:.1f}",
                  f"{r['reload_tax_s_per_resume']:.1f}")
    console.print(t)

    ceiling = out["rows"][0]["mean_reduction_pct"]
    big = out["rows"][-1]
    console.print(
        f"\nCeiling (state ~{out['rows'][0]['state_gb']:.3g} GB) = [bold]{ceiling:.1f}%[/] reduction "
        f"(the pure carbon-shifting win). At {big['state_gb']:.3g} GB it is "
        f"[bold]{big['mean_reduction_pct']:.1f}%[/] (makespan {big['mean_makespan_factor']:.2f}x, "
        f"{big['reload_tax_s_per_resume']:.0f}s reload/resume). "
        f"Erosion {ceiling - big['mean_reduction_pct']:.1f} pp.")
    survives = big["mean_reduction_pct"] > args.large_win_bar and big["ci_excludes_zero"]
    console.print(
        f"[bold]Large-win-survives at max state[/]: {survives} "
        f"(bar = {args.large_win_bar:.0f}% reduction, well above the ~16% free-DVFS/throttle reference).")

    out["ceiling_reduction_pct"] = ceiling
    out["max_state_reduction_pct"] = big["mean_reduction_pct"]
    out["large_win_survives_at_max_state"] = survives
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
    p.add_argument("--h-job", type=int, default=48)
    p.add_argument("--offset-stride-h", type=int, default=24)
    p.add_argument("--threshold-q", type=float, default=0.5, help="pause hours dirtier than this quantile")
    p.add_argument("--power-per-gpu-w", type=float, default=300.0)
    p.add_argument("--state-gbs", default="0.045,1,5,20,80,160")
    p.add_argument("--large-win-bar", type=float, default=30.0)
    p.add_argument("--cold-start-s", type=float, default=4.7)
    p.add_argument("--resume-power-per-gpu-w", type=float, default=100.0)
    p.add_argument("--write-bw-gbps", type=float, default=3.9)
    p.add_argument("--reload-bw-gbps", type=float, default=7.7)
    p.add_argument("--warmup-s", type=float, default=0.761)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    real_dir = args.real_dir if args.season == "summer" else args.real_dir_winter
    ztraces = load_zone_traces(real_dir, GRID_ZONE_IDS)
    return run(args, ztraces)


if __name__ == "__main__":
    raise SystemExit(main())

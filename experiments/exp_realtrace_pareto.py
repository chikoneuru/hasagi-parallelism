"""Carbon-policy Pareto over REAL grid traces (ElectricityMaps DE / NO).

Replaces the synthetic 200/900 pattern with real hourly carbon-intensity traces
and feeds them into the WS1 throttle-vs-pause-vs-always-on Pareto (driven by the
measured power-cap profile). For each zone the job is run from many staggered
diurnal start offsets — genuinely different real-day realisations — so the start
offset is the replication unit, decoupled from a single fixed diurnal phase
(this is what the earlier synthetic seeds lacked). A within-zone percentile
bootstrap over offsets gives the CI.

Honest scope: this ships with two real zones (DE, dirty + swingy; NO, clean +
flat), so it reports a per-zone result with a within-zone CI over real days, NOT
a cross-zone significance claim — two zones cannot support that. The clean/dirty
contrast is the point: deferral and throttling buy real carbon on a swingy grid
and almost nothing on an already-clean one. Cross-zone inference awaits more
licensed zones.

Usage::

    python -m experiments.exp_realtrace_pareto \
        --de data_cache/real_traces/de_2024-07-01_2024-07-15_hourly.csv \
        --no data_cache/real_traces/no_2024-07-01_2024-07-15_hourly.csv \
        --profile artifacts/hardware-pareto-3080ti.json \
        --job-hours 24 --threshold-quantile 0.6 --out artifacts/realtrace_pareto.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hasagi.energy.carbon_trace import load_electricitymaps_csv
from hasagi.energy.throttle_pareto import PowerCapProfile, simulate_policy
from hasagi.energy.trace_schedule import (
    diurnal_offsets,
    quantile_threshold,
    rotate,
    trace_to_hourly,
    zone_stats,
)
from hasagi.stats.bootstrap import bootstrap_mean_ci


def _savings_over_offsets(
    profile: PowerCapProfile,
    hourly: list[float],
    *,
    total_iters: int,
    threshold: float,
    throttle_cap: float,
    resume_kwh: float,
    idle_w: float,
    stride_hours: int,
    span_hours: int,
) -> dict:
    """Run all three policies from each diurnal offset; collect per-offset savings."""
    full = profile.max_throughput_cap
    offsets = diurnal_offsets(len(hourly), stride_hours=stride_hours, span_hours=span_hours)
    throttle_pct: list[float] = []
    pause_pct: list[float] = []
    pause_makespan_h: list[float] = []
    throttle_makespan_h: list[float] = []
    for off in offsets:
        sched = rotate(hourly, off)
        common = dict(
            total_iters=total_iters, window_s=3600.0, schedule_g=sched, threshold_g=threshold,
        )
        base = simulate_policy(profile, name="always-on", clean_cap_w=full, dirty_cap_w=full, **common)
        thr = simulate_policy(profile, name="throttle", clean_cap_w=full, dirty_cap_w=throttle_cap, **common)
        pau = simulate_policy(
            profile, name="pause", clean_cap_w=full, dirty_cap_w=None,
            resume_energy_kwh=resume_kwh, idle_power_w=idle_w, **common,
        )
        if base.total_carbon_g <= 0:
            continue
        throttle_pct.append(100.0 * (base.total_carbon_g - thr.total_carbon_g) / base.total_carbon_g)
        pause_pct.append(100.0 * (base.total_carbon_g - pau.total_carbon_g) / base.total_carbon_g)
        throttle_makespan_h.append((thr.makespan_s - base.makespan_s) / 3600.0)
        pause_makespan_h.append((pau.makespan_s - base.makespan_s) / 3600.0)
    return {
        "n_offsets": len(throttle_pct),
        "throttle_pct": throttle_pct,
        "pause_pct": pause_pct,
        "throttle_makespan_h": throttle_makespan_h,
        "pause_makespan_h": pause_makespan_h,
    }


def _ci(values: list[float], rng: random.Random) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` (matching ``bootstrap_mean_ci``)."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v, v
    return bootstrap_mean_ci(values, n_boot=10_000, alpha=0.05, rng=rng)


def run(args: argparse.Namespace) -> int:
    console = Console()
    profile = PowerCapProfile.from_json(args.profile)
    full = profile.max_throughput_cap
    throttle_cap = args.throttle_cap_w or profile.energy_optimal_cap
    total_iters = int(args.job_hours * 3600.0 * profile.point(full).throughput_iters_s)
    resume_kwh = args.resume_energy_wh / 1000.0
    rng = random.Random(args.seed)

    zones = {"DE": args.de, "NO": args.no}
    console.print(
        f"[bold]Real-trace carbon Pareto[/] — full {full:.0f} W, throttle {throttle_cap:.0f} W; "
        f"job ≈ {args.job_hours} h ({total_iters} iters); pause when intensity > "
        f"q{args.threshold_quantile:.2f} of the zone."
    )

    out_zones: dict[str, dict] = {}
    table = Table(title="Carbon savings vs always-on over real grid traces (within-zone 95% CI over diurnal offsets)")
    table.add_column("zone")
    table.add_column("intensity gCO2/kWh", justify="right")
    table.add_column("n", justify="right")
    table.add_column("throttle save %", justify="right")
    table.add_column("pause save %", justify="right")
    table.add_column("pause +makespan (h)", justify="right")

    for zone, path in zones.items():
        if not path or not Path(path).exists():
            console.print(f"[yellow]skip {zone}: trace not found ({path})[/]")
            continue
        hourly = trace_to_hourly(load_electricitymaps_csv(path))
        zs = zone_stats(hourly)
        threshold = quantile_threshold(hourly, args.threshold_quantile)
        res = _savings_over_offsets(
            profile, hourly, total_iters=total_iters, threshold=threshold,
            throttle_cap=throttle_cap, resume_kwh=resume_kwh, idle_w=args.dedicated_idle_w,
            stride_hours=args.stride_hours, span_hours=int(args.job_hours * 2),
        )
        t_mean, t_lo, t_hi = _ci(res["throttle_pct"], rng)
        p_mean, p_lo, p_hi = _ci(res["pause_pct"], rng)
        mk = sum(res["pause_makespan_h"]) / max(1, len(res["pause_makespan_h"]))
        table.add_row(
            zone,
            f"{zs['mean']:.0f} (swing {zs['swing']:.0f})",
            str(res["n_offsets"]),
            f"{t_mean:+.1f} [{t_lo:+.1f},{t_hi:+.1f}]",
            f"{p_mean:+.1f} [{p_lo:+.1f},{p_hi:+.1f}]",
            f"+{mk:.1f}",
        )
        out_zones[zone] = {
            "intensity_stats": zs,
            "threshold_g_per_kwh": threshold,
            "n_offsets": res["n_offsets"],
            "throttle_save_pct_ci_mean_lo_hi": [t_mean, t_lo, t_hi],
            "pause_save_pct_ci_mean_lo_hi": [p_mean, p_lo, p_hi],
            "pause_makespan_h_mean": mk,
        }

    console.print(table)
    console.print(
        "[dim]Two real zones → per-zone within-zone CI (resampling diurnal offsets), "
        "not a cross-zone significance claim. Offsets are distinct real-day phases, "
        "so this is not the eff-N≈3 single-phase pitfall.[/]"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "profile": args.profile,
            "full_cap_w": full,
            "throttle_cap_w": throttle_cap,
            "job_hours": args.job_hours,
            "total_iters": total_iters,
            "threshold_quantile": args.threshold_quantile,
            "resume_energy_wh": args.resume_energy_wh,
            "dedicated_idle_w": args.dedicated_idle_w,
            "trace_source": "real-csv",
            "energy_source": "measured-power-cap-profile",
            "replication_unit": "diurnal-start-offset (within zone)",
            "zones": out_zones,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--de", default="data_cache/real_traces/de_2024-07-01_2024-07-15_hourly.csv")
    p.add_argument("--no", default="data_cache/real_traces/no_2024-07-01_2024-07-15_hourly.csv")
    p.add_argument("--profile", default="artifacts/hardware-pareto-3080ti.json")
    p.add_argument("--job-hours", type=float, default=24.0,
                   help="Always-on makespan target in hours (sizes the job).")
    p.add_argument("--threshold-quantile", type=float, default=0.6,
                   help="Pause/throttle when intensity exceeds this within-zone quantile.")
    p.add_argument("--throttle-cap-w", type=float, default=None)
    p.add_argument("--stride-hours", type=int, default=12,
                   help="Spacing between diurnal start offsets (replication samples).")
    p.add_argument("--resume-energy-wh", type=float, default=0.07)
    p.add_argument("--dedicated-idle-w", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

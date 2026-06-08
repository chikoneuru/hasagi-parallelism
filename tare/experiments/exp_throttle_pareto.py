"""Carbon-throttle vs pause vs always-on Pareto, on the measured power-cap profile.

Feeds the real per-cap sweep (``artifacts/hardware-pareto-3080ti.json``: throughput,
power, energy-per-iter at each cap) into a forward simulation over a clean/dirty
grid schedule. No live power-capping is done — capping is device-wide and would
throttle a co-tenant — so the throttle numbers come from the measured profile.

Reports the carbon-vs-makespan trade-off for the three policies, with the pause
policy evaluated under each freed-GPU regime (its idle floor is the only
regime-dependent term; throttle and always-on never idle).

Usage::

    python -m experiments.exp_throttle_pareto \
        --profile artifacts/hardware-pareto-3080ti.json \
        --total-iters 20000 --window-s 600 --intensities 200,200,900,900 \
        --threshold 800 --resume-energy-wh 0.07 --dedicated-idle-w 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from tare.energy.throttle_pareto import PowerCapProfile, simulate_policy


def run(args: argparse.Namespace) -> int:
    console = Console()
    profile = PowerCapProfile.from_json(args.profile)
    schedule = [float(x) for x in args.intensities.split(",")]
    full = profile.max_throughput_cap
    throttle_cap = args.throttle_cap_w if args.throttle_cap_w else profile.energy_optimal_cap
    resume_kwh = args.resume_energy_wh / 1000.0
    common = dict(
        total_iters=args.total_iters, window_s=args.window_s,
        schedule_g=schedule, threshold_g=args.threshold,
    )
    console.print(
        f"[bold]Throttle Pareto[/] — {profile.gpu_name}; full cap {full:.0f} W, "
        f"throttle cap {throttle_cap:.0f} W (energy-optimal), {args.total_iters} iters, "
        f"window {args.window_s:.0f}s; schedule={schedule}"
    )

    results: dict[str, dict] = {}

    always_on = simulate_policy(
        profile, name="always-on", clean_cap_w=full, dirty_cap_w=full, **common,
    )
    # Carbon-BLIND efficiency baseline: energy-optimal cap in every window.
    eco = simulate_policy(
        profile, name=f"always-on@{throttle_cap:.0f}W(eco)", clean_cap_w=throttle_cap,
        dirty_cap_w=throttle_cap, **common,
    )
    throttle = simulate_policy(
        profile, name=f"throttle@{throttle_cap:.0f}W", clean_cap_w=full,
        dirty_cap_w=throttle_cap, **common,
    )
    # pause is regime-dependent through its idle floor.
    pause_by_regime = {
        "reallocated": simulate_policy(
            profile, name="pause/reallocated", clean_cap_w=full, dirty_cap_w=None,
            resume_energy_kwh=resume_kwh, idle_power_w=0.0, **common,
        ),
        "dedicated": simulate_policy(
            profile, name="pause/dedicated", clean_cap_w=full, dirty_cap_w=None,
            resume_energy_kwh=resume_kwh, idle_power_w=args.dedicated_idle_w, **common,
        ),
    }

    def line(r) -> None:
        rel = 100.0 * (r.total_carbon_g - always_on.total_carbon_g) / always_on.total_carbon_g
        dmk = r.makespan_s - always_on.makespan_s
        console.print(
            f"  {r.name:22s} carbon {r.total_carbon_g:8.2f} g ({rel:+6.1f}%)  "
            f"makespan {r.makespan_s/60:6.1f} min ({dmk/60:+.1f})"
        )
        results[r.name] = {
            "total_carbon_g": r.total_carbon_g, "makespan_s": r.makespan_s,
            "iters": r.iters, "idle_carbon_g": r.idle_carbon_g,
            "resume_carbon_g": r.resume_carbon_g,
            "rel_carbon_pct_vs_alwayson": rel, "makespan_delta_s": dmk,
        }

    console.print("[bold]Policy comparison (vs always-on@full)[/]:")
    line(always_on)
    line(eco)
    line(throttle)
    line(pause_by_regime["reallocated"])
    line(pause_by_regime["dedicated"])
    console.print(
        "[dim]throttle keeps training at a leaner cap during dirty windows "
        "(no pause/resume); pause defers work and pays resume + regime idle.[/]"
    )

    # Decompose the throttle saving into a carbon-BLIND efficiency part (just
    # using the energy-optimal cap) and the carbon-AWARENESS part (timing the
    # leanness to dirty windows). If awareness is small, most of the "carbon-
    # aware" win is really energy efficiency a carbon-unaware operator gets too.
    base_g = always_on.total_carbon_g
    if base_g > 0:
        eff_pct = 100.0 * (base_g - eco.total_carbon_g) / base_g
        throttle_total_pct = 100.0 * (base_g - throttle.total_carbon_g) / base_g
        aware_pct = throttle_total_pct - eff_pct
        console.print(
            f"[bold]Decomposition of throttle's {throttle_total_pct:+.1f}% vs always-on@full[/]: "
            f"efficiency (eco cap, carbon-blind) {eff_pct:+.1f}% + carbon-awareness "
            f"(dirty-window timing) {aware_pct:+.1f}%."
        )
        results["_decomposition"] = {
            "throttle_total_pct": throttle_total_pct,
            "efficiency_pct_carbon_blind": eff_pct,
            "carbon_awareness_pct": aware_pct,
        }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "gpu": profile.gpu_name,
            "profile": args.profile,
            "full_cap_w": full,
            "throttle_cap_w": throttle_cap,
            "schedule_g_per_kwh": schedule,
            "threshold_g_per_kwh": args.threshold,
            "total_iters": args.total_iters,
            "window_s": args.window_s,
            "resume_energy_wh": args.resume_energy_wh,
            "dedicated_idle_w": args.dedicated_idle_w,
            "trace_source": "synthetic-parametric",
            "energy_source": "measured-power-cap-profile",
            "policies": results,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="artifacts/hardware-pareto-3080ti.json")
    p.add_argument("--total-iters", type=int, default=20000)
    p.add_argument("--window-s", type=float, default=600.0)
    p.add_argument("--intensities", default="200,200,900,900")
    p.add_argument("--threshold", type=float, default=800.0)
    p.add_argument("--throttle-cap-w", type=float, default=None,
                   help="Dirty-window cap; default = profile's energy-optimal cap.")
    p.add_argument("--resume-energy-wh", type=float, default=0.07,
                   help="Per-wakeup resume energy (checkpoint reload + warmup), Wh.")
    p.add_argument("--dedicated-idle-w", type=float, default=26.0)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

"""WS6 Stage 0 (zero-cost kill-gate): does carbon-driven repartition ever beat free
throttle on the REAL cached grid traces under honest WALL-CLOCK charging?

The break-even study (exp_carbon_repartition_breakeven) samples grid intensity
PER WORK-WINDOW, so a slower eco layout is not charged the extra dirty wall-clock
it actually runs through (disclosed there as generous-to-repartition). This Stage 0
removes that approximation: it replays the SAME policies over the 16 real
ElectricityMaps zones x 2 seasons on a WALL-CLOCK timeline -- each window of equal
WORK takes longer wall-clock on a lower-throughput layout and is billed the grid
intensity at the wall-clock hour it actually occupies (CarbonTrace.intensity_at).
This is the GATE 0 kill-switch before any hardware spend or distributed build:

  - If NO real zone shows repartition beating throttle even at the favourable
    break-even params (eco saves 25%, switch 2.0s), the structural lever cannot win
    on real traces -> ship the simulation surface as the contribution, build nothing.
  - It also runs the statistical-power calc: given the measured per-zone delta
    distribution, how many zone-seasons a zone-clustered CI would need to exclude
    zero (the plan's other kill condition: if n_needed exceeds an affordable GPU
    budget, the A* significance claim is unreachable regardless of engineering).

Pure analysis on cached data; no GPU, no hardware. Reuses the same Layout /
migration_cost model and the project's zone-clustered bootstrap.

Usage::

    python -m experiments.exp_ws6_stage0_realtrace --out artifacts/ws6_stage0_realtrace.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.exp_carbon_repartition_breakeven import Layout, migration_cost
from tare.energy.carbon_trace import CarbonTrace
from tare.energy.trace_schedule import GRID_ZONE_IDS, load_zone_traces
from tare.stats.bootstrap import (
    clustered_bootstrap_ci,
    clustered_permutation_pvalue,
)

_J_PER_KWH = 3.6e6
_HOUR_S = 3600.0


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def wallclock_replay(trace: CarbonTrace, fast: Layout, eco: Layout, policy: str, *,
                     start_s: float, n_windows: int, iters_per_window: int,
                     threshold: float, throttle_energy_frac: float, throttle_tput_frac: float,
                     migration_energy_j: float = 0.0, migration_time_s: float = 0.0) -> dict:
    """Replay an equal-WORK job (n_windows x iters_per_window iters) under one policy on
    a WALL-CLOCK timeline starting at ``start_s`` into the trace. Each window's energy is
    billed the grid intensity at the wall-clock instant it runs, so a slower layout is
    charged the dirty hours it actually occupies."""
    carbon_g = 0.0
    energy_j = 0.0
    clock = start_s
    switches = 0
    cur = "fast"
    for _ in range(n_windows):
        intensity = trace.intensity_at(clock)
        dirty = intensity > threshold
        if policy == "static_fast":
            e, tput = fast.energy_per_iter_j, fast.throughput_iter_s
        elif policy == "static_eco":
            e, tput = eco.energy_per_iter_j, eco.throughput_iter_s
        elif policy == "throttle":
            if dirty:
                e, tput = fast.energy_per_iter_j * throttle_energy_frac, fast.throughput_iter_s * throttle_tput_frac
            else:
                e, tput = fast.energy_per_iter_j, fast.throughput_iter_s
        elif policy == "repartition":
            want = "eco" if dirty else "fast"
            if want != cur:
                switches += 1
                # pay the switch: wall-clock + migration energy billed at the current hour
                carbon_g += (migration_energy_j / _J_PER_KWH) * intensity
                energy_j += migration_energy_j
                clock += migration_time_s
                cur = want
            lay = eco if want == "eco" else fast
            e, tput = lay.energy_per_iter_j, lay.throughput_iter_s
        else:
            raise ValueError(f"unknown policy {policy!r}")
        dt = iters_per_window / tput
        win_energy = e * iters_per_window
        # bill at the wall-clock midpoint of the window
        carbon_g += (win_energy / _J_PER_KWH) * trace.intensity_at(clock + dt / 2.0)
        energy_j += win_energy
        clock += dt
    return {"policy": policy, "carbon_g": carbon_g, "energy_j": energy_j,
            "makespan_h": (clock - start_s) / _HOUR_S, "switches": switches}


def zone_deltas(trace: CarbonTrace, fast: Layout, eco: Layout, *, n_windows: int,
                iters_per_window: int, threshold_q: float, throttle_energy_frac: float,
                throttle_tput_frac: float, migration_energy_j: float, migration_time_s: float,
                offset_stride_h: int) -> dict:
    """Per-offset throttle-minus-repartition carbon deltas for one zone (delta>0 =>
    repartition WINS, lower carbon). Offsets tile the trace so the job fits before the end."""
    hourly = trace.intensities
    thr = _quantile(hourly, threshold_q)
    dur_s = trace.duration_seconds
    # worst-case makespan (eco) to keep the whole job inside the trace
    max_makespan_s = n_windows * iters_per_window / min(fast.throughput_iter_s, eco.throughput_iter_s) \
        + n_windows * migration_time_s
    deltas: list[float] = []
    rep_wins = 0
    rep_c: list[float] = []
    thr_c: list[float] = []
    eco_c: list[float] = []
    start = 0.0
    while start + max_makespan_s <= dur_s + 1.0:
        common = dict(start_s=start, n_windows=n_windows, iters_per_window=iters_per_window,
                      threshold=thr, throttle_energy_frac=throttle_energy_frac,
                      throttle_tput_frac=throttle_tput_frac)
        t = wallclock_replay(trace, fast, eco, "throttle", **common)
        r = wallclock_replay(trace, fast, eco, "repartition", **common,
                             migration_energy_j=migration_energy_j, migration_time_s=migration_time_s)
        ec = wallclock_replay(trace, fast, eco, "static_eco", **common)
        d = t["carbon_g"] - r["carbon_g"]
        deltas.append(d)
        rep_wins += int(d > 0)
        rep_c.append(r["carbon_g"])
        thr_c.append(t["carbon_g"])
        eco_c.append(ec["carbon_g"])
        start += offset_stride_h * _HOUR_S
    n = max(len(deltas), 1)
    return {"deltas": deltas, "n_offsets": len(deltas), "rep_win_offsets": rep_wins,
            "mean_delta_g": sum(deltas) / n, "mean_throttle_g": sum(thr_c) / n,
            "mean_repartition_g": sum(rep_c) / n, "mean_static_eco_g": sum(eco_c) / n}


def _n_needed_to_exclude_zero(mean: float, sd: float) -> float:
    """Rough number of zone-clusters needed for a 95% CI on the mean to exclude zero on
    the WIN side (normal approx). A non-positive mean can never demonstrate a win, so
    return inf; a positive mean with zero spread needs only one."""
    if mean <= 0.0:
        return float("inf")
    if sd == 0.0:
        return 1.0
    return (1.96 * sd / mean) ** 2


def run(args: argparse.Namespace, ztraces: dict) -> int:
    console = Console()
    fast = Layout("fast", energy_per_iter_j=1.0, throughput_iter_s=1.0)
    # iters_per_window chosen so one fast window = 1 wall-clock hour (tput 1.0 iter/s)
    iters_per_window = int(_HOUR_S)
    n_windows = args.job_fast_hours

    regimes = {
        "realistic": dict(eco=Layout("eco", args.eco_energy_frac, args.eco_tput_frac),
                          mig=migration_cost(args.state_gb, args.bw_gbps, args.cold_start_s, args.migrate_power_w)),
        "best_case": dict(eco=Layout("eco", 0.75, args.eco_tput_frac),       # eco saves 25% (the crossover)
                          mig=(2.0, args.migrate_power_w * 2.0)),            # t_switch forced to 2.0s
    }

    rng = random.Random(0)
    out: dict = {"job_fast_hours": args.job_fast_hours, "threshold_q": args.threshold_q,
                 "throttle_energy_frac": args.throttle_energy_frac, "zones": {}, "regimes": {}}

    for rk in ("realistic", "best_case"):
        eco = regimes[rk]["eco"]
        mt, me = regimes[rk]["mig"]
        per_zone: dict[str, dict] = {}
        for z, zt in ztraces.items():
            zd = zone_deltas(zt.trace, fast, eco, n_windows=n_windows, iters_per_window=iters_per_window,
                             threshold_q=args.threshold_q, throttle_energy_frac=args.throttle_energy_frac,
                             throttle_tput_frac=args.throttle_tput_frac, migration_energy_j=me,
                             migration_time_s=mt, offset_stride_h=args.offset_stride_h)
            zd["source"] = zt.source
            per_zone[z] = zd

        real = {z: d for z, d in per_zone.items() if d["source"] == "real-csv"}
        clusters = [d["deltas"] for d in real.values() if d["deltas"]]
        zone_means = [d["mean_delta_g"] for d in real.values()]
        ci = clustered_bootstrap_ci(clusters, rng=rng) if clusters else (0.0, 0.0, 0.0)
        pval = clustered_permutation_pvalue(clusters, rng=rng) if clusters else 1.0
        sd = (statistics_stdev(zone_means))
        n_needed = _n_needed_to_exclude_zero(ci[0], sd)
        win_zones = sum(1 for d in real.values() if d["mean_delta_g"] > 0)
        out["regimes"][rk] = {
            "eco_energy_frac": eco.energy_per_iter_j, "t_switch_s": mt,
            "n_real_zones": len(real), "repartition_win_zones": win_zones,
            "cross_zone_mean_delta_g": ci[0], "ci_lo": ci[1], "ci_hi": ci[2],
            "permutation_p": pval, "zone_mean_sd": sd,
            "n_zones_needed_to_exclude_zero": n_needed,
            "ci_excludes_zero": (ci[1] > 0.0 or ci[2] < 0.0),
        }
        if rk == "realistic":
            out["zones"] = {z: {k: d[k] for k in ("source", "n_offsets", "rep_win_offsets",
                                                  "mean_delta_g", "mean_throttle_g",
                                                  "mean_repartition_g", "mean_static_eco_g")}
                            for z, d in per_zone.items()}

    # ---- report ----
    for rk in ("realistic", "best_case"):
        r = out["regimes"][rk]
        console.print(f"[bold]{rk}[/] (eco={r['eco_energy_frac']:.2f}E, t_switch={r['t_switch_s']:.1f}s): "
                      f"repartition beats throttle in [bold]{r['repartition_win_zones']}/{r['n_real_zones']}[/] "
                      f"real zones; cross-zone mean delta {r['cross_zone_mean_delta_g']:+.3f} g "
                      f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], p={r['permutation_p']:.3f}; "
                      f"CI excludes zero: {r['ci_excludes_zero']}; "
                      f"zones needed to exclude zero ~ {r['n_zones_needed_to_exclude_zero']:.0f}")
    t = Table(title="Per-zone wall-clock carbon (realistic regime; delta>0 => repartition wins)")
    for c in ("zone", "src", "throttle g", "repart g", "delta g", "static_eco g", "win/offsets"):
        t.add_column(c, justify="right" if c != "zone" else "left")
    for z, d in sorted(out["zones"].items(), key=lambda kv: kv[1]["mean_delta_g"], reverse=True):
        t.add_row(z, d["source"][:4], f"{d['mean_throttle_g']:.2f}", f"{d['mean_repartition_g']:.2f}",
                  f"{d['mean_delta_g']:+.3f}", f"{d['mean_static_eco_g']:.2f}",
                  f"{d['rep_win_offsets']}/{d['n_offsets']}")
    console.print(t)

    bc = out["regimes"]["best_case"]
    gate0_pass = bc["repartition_win_zones"] > 0 and bc["ci_excludes_zero"]
    console.print(f"\n[bold]GATE 0[/]: {'PASS — a real zone wins at the favourable params AND the CI can exclude zero' if gate0_pass else 'FAIL — no winning regime / CI cannot exclude zero on real traces under wall-clock charging'}")
    if not gate0_pass:
        console.print("  [yellow]Do NOT commit hardware: ship the simulation break-even surface as the "
                      "contribution. The structural lever does not flip the negative on real traces even "
                      "at the favourable break-even params under honest wall-clock charging.[/]")

    out["gate0_pass"] = gate0_pass
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        console.print(f"[dim]wrote {p}[/]")
    return 0


def statistics_stdev(xs: list[float]) -> float:
    import statistics as _s
    return _s.stdev(xs) if len(xs) >= 2 else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default="data_cache/real_traces")
    p.add_argument("--real-dir-winter", default="data_cache/real_traces_winter")
    p.add_argument("--job-fast-hours", type=int, default=24)
    p.add_argument("--offset-stride-h", type=int, default=24)
    p.add_argument("--threshold-q", type=float, default=0.6)
    p.add_argument("--eco-energy-frac", type=float, default=0.80)
    p.add_argument("--eco-tput-frac", type=float, default=0.6)
    p.add_argument("--throttle-energy-frac", type=float, default=0.85)
    p.add_argument("--throttle-tput-frac", type=float, default=0.7)
    p.add_argument("--state-gb", type=float, default=5.0)
    p.add_argument("--bw-gbps", type=float, default=100.0)
    p.add_argument("--cold-start-s", type=float, default=4.7)
    p.add_argument("--migrate-power-w", type=float, default=100.0)
    p.add_argument("--season", default="summer", choices=["summer", "winter"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    real_dir = args.real_dir if args.season == "summer" else args.real_dir_winter
    ztraces = load_zone_traces(real_dir, GRID_ZONE_IDS)
    return run(args, ztraces)


if __name__ == "__main__":
    raise SystemExit(main())

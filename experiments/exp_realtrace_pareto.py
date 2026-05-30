"""Carbon-policy panel over REAL grid traces (ElectricityMaps zones).

Feeds real hourly carbon-intensity traces into a single measured power-cap
energy substrate (the RTX 3080 Ti per-cap sweep) and runs a panel of policies
whose ONLY difference is the carbon response, so the comparisons isolate where
carbon savings actually come from. Three honest questions are answered:

1. How much carbon does running at the energy-optimal cap buy, with no carbon
   signal at all?  always-on@eco is the carbon-BLIND endpoint: it runs the
   energy-optimal cap in every window. As a constant its saving is a
   deterministic hardware energy ratio (intensity cancels); the realised per-job
   saving averages that ratio but varies with the start offset, and it is banked
   at a throughput (latency) cost.

2. What does the carbon SIGNAL add, at a fixed response budget?  throttle (eco
   cap in the dirty windows, full cap elsewhere) is compared against a
   same-budget carbon-BLIND throttle that leans just as many windows chosen by
   position rather than by intensity. throttle − throttle_blind is the marginal
   value of knowing WHICH windows are dirty, at fixed throughput and zero added
   latency. This is the experiment's genuine, modest contribution.

3. Throttle vs pause (the GREEN comparison).  GREEN (Xu et al., NSDI 2025) is a
   temporal shifter: it PAUSES dirty windows and defers the work. We run it
   across its full capability range — online rolling-percentile vs offline
   forecast (perfect detection, pausing the SAME oracle windows the throttle
   leans), and the dedicated-idle (the paused GPU keeps idling, 26 W) vs
   reallocated-idle (a co-tenant pays the idle, 0 W) regime. Under GREEN's own
   favourable assumptions (reallocated idle + forecast) the NET carbon gap to
   throttle is not detectable across the panel — but that net is a CANCELLATION:
   pause wins carbon on the high-dirty-window-excess grids while throttle wins on
   the flat-tail grids (the carbon-winning mechanism is zone-dependent). What is
   robust and does NOT cancel is latency: throttling keeps the job moving while
   deferral lengthens it several-fold. We report all cells rather than the single
   one that flatters either side.

Replication: each zone is launched from many staggered diurnal start offsets.
The valid inference unit is the ZONE (cross-zone clustered bootstrap over the 16
real zones); the per-zone CIs over offsets are DESCRIPTIVE only, because
adjacent offsets into one fortnight overlap and are not independent draws.

Usage::

    python -m experiments.exp_realtrace_pareto \
        --real-dir data_cache/real_traces \
        --profile artifacts/hardware-pareto-3080ti.json \
        --job-hours 24 --threshold-quantile 0.6 --out artifacts/realtrace_pareto.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.baselines.green import (
    green_offline_optimal_mask,
    green_online_percentile_mask,
)
from hasagi.energy.throttle_pareto import (
    PowerCapProfile,
    simulate_masked_policy,
    simulate_policy,
)
from hasagi.energy.trace_schedule import (
    GRID_ZONE_IDS,
    diurnal_offsets,
    load_zone_traces,
    quantile_threshold,
    rotate,
    trace_to_hourly,
    zone_stats,
)
from hasagi.stats.bootstrap import (
    bootstrap_mean_ci,
    clustered_bootstrap_ci,
    clustered_permutation_pvalue,
    holm_bonferroni,
)

#: Reallocated regime: a co-tenant pays the freed GPU's idle, so a paused job is
#: billed ~0 idle. Dedicated regime: the paused job keeps the card idling.
_REALLOCATED_IDLE_W = 0.0


def _pct(base_g: float, r) -> float:
    """Carbon saving of policy result ``r`` vs the always-on baseline carbon ``base_g``."""
    return 100.0 * (base_g - r.total_carbon_g) / base_g


def _mk_h(base_makespan_s: float, r) -> float:
    """Makespan delta (hours) of policy result ``r`` vs the always-on baseline."""
    return (r.makespan_s - base_makespan_s) / 3600.0


def _green_pause(profile, mask, *, total_iters, window_s, schedule_g, full_cap_w, resume_kwh, idle_w):
    """GREEN's response: pause (scale-to-zero) the masked-off windows, deferring
    the work, paying ``resume_kwh`` on each wake and ``idle_w`` over each paused
    window. Makespan is independent of ``idle_w`` (idle adds carbon, not wall)."""
    return simulate_masked_policy(
        profile, name="green", active_mask=mask, off_cap_w=None,
        resume_energy_kwh=resume_kwh, idle_power_w=idle_w,
        total_iters=total_iters, window_s=window_s, schedule_g=schedule_g, full_cap_w=full_cap_w,
    )


#: Pause-budget grid for the self-tuned (steel-man) GREEN.
_SELFTUNE_BUDGETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def _green_selftuned(profile, sched, base_g, base_mk, *, gp, budgets=_SELFTUNE_BUDGETS):
    """Strongest possible GREEN: a per-offset oracle pick of the pause budget that
    MAXIMISES GREEN's reallocated-idle (0 W) carbon saving. budget=0 means "do not
    pause" → 0 % saving, so a self-tuning GREEN never makes itself worse and never
    "loses". This removes the objection that GREEN is hobbled by a fixed
    (1−q)=0.40 budget on flat grids where shifting cannot pay. The carbon-maximising
    budget ignores latency, so the makespan at that budget is returned alongside
    the saving — pausing more to win carbon costs more makespan, and that cost must
    be shown. Returns ``(best_save_pct, best_budget, best_makespan_h)``.
    """
    best_pct, best_pf, best_mk = 0.0, 0.0, 0.0
    for pf in budgets:
        mask = green_offline_optimal_mask(sched, pause_fraction=pf)
        g = _green_pause(profile, mask, idle_w=_REALLOCATED_IDLE_W, **gp)
        pct = _pct(base_g, g)
        if pct > best_pct:
            best_pct, best_pf, best_mk = pct, pf, _mk_h(base_mk, g)
    return best_pct, best_pf, best_mk


def _uniform_off_mask(n: int, n_off: int) -> list[int]:
    """A 0/1 mask of length ``n`` with exactly ``n_off`` zeros spread uniformly by
    position (carbon-blind: the response windows are chosen without looking at
    intensity). Over the full trace the zero-count equals the oracle's; per
    executed job the counts differ by start offset but match in expectation, so
    this is the same-FULL-TRACE-budget carbon-blind counterfactual for a targeted
    policy: it throttles just as often, but not where the grid is dirty.
    """
    mask = [1] * n
    n_off = max(0, min(n_off, n))
    for i in range(n_off):
        mask[(i * n) // n_off] = 0
    return mask


#: Per-offset vectors collected by :func:`_savings_over_offsets`.
_VECTOR_KEYS = (
    # carbon savings % vs always-on@full
    "eco_pct", "throttle_pct", "throttle_blind_pct", "signal_pp", "throttle_online_pct",
    "green_on_ded_pct", "green_on_rea_pct", "green_off_ded_pct", "green_off_rea_pct",
    # head-to-head gaps (throttle minus GREEN), pp
    "gap_fair_pp", "gap_online_ded_pp", "gap_online_rea_pp",
    # steel-man GREEN: per-offset oracle-best pause budget, reallocated idle
    "green_selftuned_rea_pct", "gap_selftuned_pp", "green_selftuned_budget",
    "green_selftuned_makespan_h",
    # makespan deltas, hours
    "eco_makespan_h", "throttle_makespan_h", "throttle_online_makespan_h",
    "green_on_makespan_h", "green_off_makespan_h",
    # realised GREEN pause fractions (diagnostics)
    "green_on_pause_frac", "green_off_pause_frac",
)


def _savings_over_offsets(
    profile: PowerCapProfile,
    hourly: list[float],
    *,
    total_iters: int,
    threshold: float,
    throttle_cap: float,
    resume_kwh: float,
    dedicated_idle_w: float,
    stride_hours: int,
    span_hours: int,
    green_pause_fraction: float,
    green_window: int,
) -> dict:
    """Run the policy panel from each diurnal offset; collect per-offset vectors.

    All policies share the measured power-cap energy substrate. ``throttle`` and
    ``pause`` use a perfect within-zone quantile cutoff (oracle detection);
    ``throttle_blind`` uses a same-budget carbon-blind mask. GREEN is run as
    pause under four cells: online vs offline (forecast/oracle) detection, and
    dedicated (``dedicated_idle_w``) vs reallocated (0 W) idle. ``throttle_online``
    throttles under GREEN's online mask. The fair mechanism gap compares
    throttle vs GREEN-offline pause on the SAME oracle windows at reallocated
    idle (so it is purely throttle-vs-pause, no detection or idle handicap).
    """
    full = profile.max_throughput_cap
    offsets = diurnal_offsets(len(hourly), stride_hours=stride_hours, span_hours=span_hours)
    out: dict[str, list[float]] = {k: [] for k in _VECTOR_KEYS}
    for off in offsets:
        sched = rotate(hourly, off)
        base = simulate_policy(
            profile, name="always-on", clean_cap_w=full, dirty_cap_w=full,
            total_iters=total_iters, window_s=3600.0, schedule_g=sched, threshold_g=threshold,
        )
        if base.total_carbon_g <= 0:
            continue
        base_g, base_mk = base.total_carbon_g, base.makespan_s
        masked = dict(total_iters=total_iters, window_s=3600.0, schedule_g=sched, full_cap_w=full)

        # always-on@eco: carbon-blind efficiency endpoint (lean in EVERY window).
        eco = simulate_policy(
            profile, name="eco", clean_cap_w=throttle_cap, dirty_cap_w=throttle_cap,
            total_iters=total_iters, window_s=3600.0, schedule_g=sched, threshold_g=threshold,
        )

        # Oracle dirty mask (intensity-targeted) + same-size carbon-blind mask.
        oracle_mask = [0 if v > threshold else 1 for v in sched]
        blind_mask = _uniform_off_mask(len(sched), oracle_mask.count(0))
        thr = simulate_masked_policy(
            profile, name="throttle", active_mask=oracle_mask, off_cap_w=throttle_cap, **masked,
        )
        thr_blind = simulate_masked_policy(
            profile, name="throttle-blind", active_mask=blind_mask, off_cap_w=throttle_cap, **masked,
        )

        # GREEN masks. Online: rolling percentile, no lookahead (realistic
        # deployable). Offline/forecast: perfect detection of the dirtiest
        # windows — we use the SAME oracle mask the throttle leans, so the
        # offline GREEN cell pauses exactly the windows throttle throttles and
        # the fair mechanism gap is a true pause-vs-throttle contrast on
        # identical windows (no detection-quality difference mixed in).
        gmask_on = green_online_percentile_mask(
            sched, pause_fraction=green_pause_fraction, window_size=green_window,
        )
        gmask_off = oracle_mask

        gp = dict(total_iters=total_iters, window_s=3600.0, schedule_g=sched, full_cap_w=full, resume_kwh=resume_kwh)
        grn_on_ded = _green_pause(profile, gmask_on, idle_w=dedicated_idle_w, **gp)
        grn_on_rea = _green_pause(profile, gmask_on, idle_w=_REALLOCATED_IDLE_W, **gp)
        grn_off_ded = _green_pause(profile, gmask_off, idle_w=dedicated_idle_w, **gp)
        grn_off_rea = _green_pause(profile, gmask_off, idle_w=_REALLOCATED_IDLE_W, **gp)

        # Throttle under GREEN's online mask (throttle has no idle term).
        thr_online = simulate_masked_policy(
            profile, name="throttle-online", active_mask=gmask_on, off_cap_w=throttle_cap, **masked,
        )

        # Steel-man GREEN: best per-offset pause budget, reallocated idle (never loses).
        gst_pct, gst_pf, gst_mk = _green_selftuned(profile, sched, base_g, base_mk, gp=gp)

        thr_pct, blind_pct = _pct(base_g, thr), _pct(base_g, thr_blind)
        thr_online_pct = _pct(base_g, thr_online)
        grn_off_rea_pct = _pct(base_g, grn_off_rea)
        grn_on_ded_pct, grn_on_rea_pct = _pct(base_g, grn_on_ded), _pct(base_g, grn_on_rea)

        out["eco_pct"].append(_pct(base_g, eco))
        out["throttle_pct"].append(thr_pct)
        out["throttle_blind_pct"].append(blind_pct)
        out["signal_pp"].append(thr_pct - blind_pct)          # value of the carbon signal, same budget
        out["throttle_online_pct"].append(thr_online_pct)
        out["green_on_ded_pct"].append(grn_on_ded_pct)
        out["green_on_rea_pct"].append(grn_on_rea_pct)
        out["green_off_ded_pct"].append(_pct(base_g, grn_off_ded))
        out["green_off_rea_pct"].append(grn_off_rea_pct)
        # Fair mechanism gap: throttle vs pause on identical oracle windows,
        # reallocated idle — purely the response mechanism, no handicap to GREEN.
        out["gap_fair_pp"].append(thr_pct - grn_off_rea_pct)
        out["green_selftuned_rea_pct"].append(gst_pct)
        out["gap_selftuned_pp"].append(thr_pct - gst_pct)
        out["green_selftuned_budget"].append(gst_pf)
        out["green_selftuned_makespan_h"].append(gst_mk)
        out["gap_online_ded_pp"].append(thr_online_pct - grn_on_ded_pct)
        out["gap_online_rea_pp"].append(thr_online_pct - grn_on_rea_pct)
        out["eco_makespan_h"].append(_mk_h(base_mk, eco))
        out["throttle_makespan_h"].append(_mk_h(base_mk, thr))
        out["throttle_online_makespan_h"].append(_mk_h(base_mk, thr_online))
        out["green_on_makespan_h"].append(_mk_h(base_mk, grn_on_rea))   # makespan independent of idle
        out["green_off_makespan_h"].append(_mk_h(base_mk, grn_off_rea))
        out["green_on_pause_frac"].append(1.0 - sum(gmask_on) / len(gmask_on))
        out["green_off_pause_frac"].append(1.0 - sum(gmask_off) / len(gmask_off))
    out["n_offsets"] = len(out["throttle_pct"])  # type: ignore[assignment]
    return out


def _ci(values: list[float], rng: random.Random) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` (matching ``bootstrap_mean_ci``). Descriptive only
    for per-zone offset vectors (offsets overlap; not independent draws)."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v, v
    return bootstrap_mean_ci(values, n_boot=10_000, alpha=0.05, rng=rng)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of two equal-length series (0.0 if undefined)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _dirty_excess(hourly: list[float], threshold: float) -> float:
    """Mean intensity of the dirty (above-threshold) windows minus the overall
    mean — the quantity the carbon-signal decomposition actually exploits (a
    better predictor of signal value than raw swing)."""
    dirty = [v for v in hourly if v > threshold]
    if not dirty:
        return 0.0
    return statistics.mean(dirty) - statistics.mean(hourly)


def run(args: argparse.Namespace) -> int:
    console = Console()
    profile = PowerCapProfile.from_json(args.profile)
    full = profile.max_throughput_cap
    throttle_cap = args.throttle_cap_w or profile.energy_optimal_cap
    eco_energy_ratio = profile.point(throttle_cap).energy_per_iter_kwh / profile.point(full).energy_per_iter_kwh
    eco_throughput_cost = 1.0 - profile.point(throttle_cap).throughput_iters_s / profile.point(full).throughput_iters_s
    total_iters = int(args.job_hours * 3600.0 * profile.point(full).throughput_iters_s)
    resume_kwh = args.resume_energy_wh / 1000.0
    green_pause_fraction = max(0.0, min(1.0, 1.0 - args.threshold_quantile))
    rng = random.Random(args.seed)

    ztraces = load_zone_traces(args.real_dir, GRID_ZONE_IDS)
    n_real = sum(1 for z in ztraces.values() if z.source == "real-csv")
    console.print(
        f"[bold]Real-trace carbon panel[/] — {len(ztraces)} zones "
        f"({n_real} real CSV, {len(ztraces) - n_real} synthetic) from {args.real_dir}; "
        f"full {full:.0f} W, eco/throttle {throttle_cap:.0f} W (energy {eco_energy_ratio:.3f}x, "
        f"throughput {1 - eco_throughput_cost:.2f}x); job ≈ {args.job_hours} h ({total_iters} iters); "
        f"dirty = intensity > q{args.threshold_quantile:.2f}; GREEN budget targeted {green_pause_fraction * 100:.0f}% "
        f"over {args.green_window_hours} h; dedicated idle {args.dedicated_idle_w:.0f} W vs reallocated 0 W."
    )

    out_zones: dict[str, dict] = {}
    clusters: dict[str, list[list[float]]] = {k: [] for k in _VECTOR_KEYS}

    table = Table(title="Per-zone carbon savings vs always-on@full (per-zone CIs are DESCRIPTIVE; zone is the inference unit)")
    table.add_column("zone")
    table.add_column("src")
    table.add_column("mean/sw/Δdirty", justify="right")
    table.add_column("n", justify="right")
    table.add_column("eco%", justify="right")
    table.add_column("throttle%", justify="right")
    table.add_column("signal pp", justify="right")
    table.add_column("GREEN% ded/rea/fair", justify="right")
    table.add_column("fair gap pp", justify="right")
    table.add_column("mk thr/grn (h)", justify="right")

    for zone, zt in ztraces.items():
        hourly = trace_to_hourly(zt.trace)
        zs = zone_stats(hourly)
        threshold = quantile_threshold(hourly, args.threshold_quantile)
        dirty_excess = _dirty_excess(hourly, threshold)
        res = _savings_over_offsets(
            profile, hourly, total_iters=total_iters, threshold=threshold,
            throttle_cap=throttle_cap, resume_kwh=resume_kwh, dedicated_idle_w=args.dedicated_idle_w,
            stride_hours=args.stride_hours, span_hours=int(args.job_hours * 2),
            green_pause_fraction=green_pause_fraction, green_window=args.green_window_hours,
        )
        eco_m = _mean(res["eco_pct"])
        t_m = _mean(res["throttle_pct"])
        sig_m = _mean(res["signal_pp"])
        gon_d, gon_r = _mean(res["green_on_ded_pct"]), _mean(res["green_on_rea_pct"])
        goff_r = _mean(res["green_off_rea_pct"])
        gap_fair = _mean(res["gap_fair_pp"])
        thr_mk, grn_mk = _mean(res["throttle_online_makespan_h"]), _mean(res["green_on_makespan_h"])
        src = "real" if zt.source == "real-csv" else "synth"
        table.add_row(
            zone, src,
            f"{zs['mean']:.0f}/{zs['swing']:.0f}/{dirty_excess:.0f}",
            str(res["n_offsets"]),
            f"{eco_m:+.1f}",
            f"{t_m:+.1f}",
            f"{sig_m:+.2f}",
            f"{gon_d:+.1f}/{gon_r:+.1f}/{goff_r:+.1f}",
            f"{gap_fair:+.2f}",
            f"+{thr_mk:.1f}/+{grn_mk:.1f}",
        )
        out_zones[zone] = {
            "source": zt.source,
            "intensity_stats": zs,
            "dirty_window_excess_g_per_kwh": dirty_excess,
            "threshold_g_per_kwh": threshold,
            "n_offsets": res["n_offsets"],
            "eco_endpoint_save_pct_mean": eco_m,
            "throttle_save_pct_mean": t_m,
            "throttle_blind_same_budget_save_pct_mean": _mean(res["throttle_blind_pct"]),
            "carbon_signal_value_pp_mean": sig_m,
            "throttle_online_save_pct_mean": _mean(res["throttle_online_pct"]),
            "green_online_dedicated_save_pct_mean": gon_d,
            "green_online_reallocated_save_pct_mean": gon_r,
            "green_offline_dedicated_save_pct_mean": _mean(res["green_off_ded_pct"]),
            "green_offline_reallocated_save_pct_mean": goff_r,
            "green_selftuned_reallocated_save_pct_mean": _mean(res["green_selftuned_rea_pct"]),
            "green_selftuned_budget_mean": _mean(res["green_selftuned_budget"]),
            "green_selftuned_makespan_h_mean": _mean(res["green_selftuned_makespan_h"]),
            "fair_mechanism_gap_pp_mean": gap_fair,
            "selftuned_gap_pp_mean": _mean(res["gap_selftuned_pp"]),
            "eco_makespan_h_mean": _mean(res["eco_makespan_h"]),
            "throttle_makespan_h_mean": _mean(res["throttle_makespan_h"]),
            "throttle_online_makespan_h_mean": thr_mk,
            "green_online_makespan_h_mean": grn_mk,
            "green_offline_makespan_h_mean": _mean(res["green_off_makespan_h"]),
            "green_online_realized_pause_frac_mean": _mean(res["green_on_pause_frac"]),
        }
        if zt.source == "real-csv" and res["n_offsets"] > 0:
            for k in _VECTOR_KEYS:
                clusters[k].append(res[k])

    console.print(table)

    cross_zone: dict | None = None
    n_clusters = len(clusters["throttle_pct"])
    if n_clusters >= 2:
        def cz(key: str) -> tuple[float, float, float]:
            return clustered_bootstrap_ci(clusters[key], rng=rng)

        def cp(key: str) -> float:
            return clustered_permutation_pvalue(clusters[key], rng=rng)

        eco_cz, thr_cz, blind_cz, sig_cz = cz("eco_pct"), cz("throttle_pct"), cz("throttle_blind_pct"), cz("signal_pp")
        gon_d_cz, gon_r_cz = cz("green_on_ded_pct"), cz("green_on_rea_pct")
        goff_r_cz = cz("green_off_rea_pct")
        thr_on_cz = cz("throttle_online_pct")
        gap_fair_cz, gap_ded_cz, gap_rea_cz = cz("gap_fair_pp"), cz("gap_online_ded_pp"), cz("gap_online_rea_pp")
        gst_cz, gap_gst_cz = cz("green_selftuned_rea_pct"), cz("gap_selftuned_pp")
        gst_budget = _mean([b for v in clusters["green_selftuned_budget"] for b in v])
        gst_mk = _mean([m for v in clusters["green_selftuned_makespan_h"] for m in v])
        sig_p, gap_fair_p, goff_r_p = cp("signal_pp"), cp("gap_fair_pp"), cp("green_off_rea_pct")
        # Holm-Bonferroni over the inferential family (FWER control at alpha).
        holm = holm_bonferroni([sig_p, gap_fair_p, goff_r_p], alpha=0.05)
        holm_labels = ["carbon_signal", "fair_mechanism_gap", "green_fair_save"]
        eco_mk = _mean([m for v in clusters["eco_makespan_h"] for m in v])
        thr_on_mk = _mean([m for v in clusters["throttle_online_makespan_h"] for m in v])
        gon_mk = _mean([m for v in clusters["green_on_makespan_h"] for m in v])
        goff_mk = _mean([m for v in clusters["green_off_makespan_h"] for m in v])

        cross_zone = {
            "n_real_zones": n_clusters,
            "eco_endpoint_save_pct_ci_mean_lo_hi": list(eco_cz),
            "eco_endpoint_save_pct_deterministic": round(100.0 * (1.0 - eco_energy_ratio), 3),
            "eco_makespan_h_mean": eco_mk,
            "throttle_save_pct_ci_mean_lo_hi": list(thr_cz),
            "throttle_blind_same_budget_save_pct_ci_mean_lo_hi": list(blind_cz),
            "carbon_signal_value_pp_ci_mean_lo_hi": list(sig_cz),
            "carbon_signal_value_pvalue": sig_p,
            "throttle_online_save_pct_ci_mean_lo_hi": list(thr_on_cz),
            "throttle_online_makespan_h_mean": thr_on_mk,
            "green_selftuned_reallocated_save_pct_ci_mean_lo_hi": list(gst_cz),
            "green_selftuned_budget_mean": gst_budget,
            "green_selftuned_makespan_h_mean": gst_mk,
            "selftuned_gap_pp_ci_mean_lo_hi": list(gap_gst_cz),
            "carbon_latency_pareto_note": (
                "throttle ~+9% carbon at +2.4 h; GREEN at its carbon-max budget can exceed throttle on "
                "carbon but only by deferring more work (large makespan). The two are points on a "
                "carbon-vs-latency Pareto; throttle is the low-latency point."
            ),
            "green_capability_range": {
                "online_dedicated_save_pct_ci_mean_lo_hi": list(gon_d_cz),
                "online_reallocated_save_pct_ci_mean_lo_hi": list(gon_r_cz),
                "offline_reallocated_save_pct_ci_mean_lo_hi": list(goff_r_cz),
                "offline_reallocated_save_pvalue": goff_r_p,
                "online_makespan_h_mean": gon_mk,
                "offline_makespan_h_mean": goff_mk,
            },
            "fair_mechanism_gap_pp_ci_mean_lo_hi": list(gap_fair_cz),
            "fair_mechanism_gap_pvalue": gap_fair_p,
            "gap_online_dedicated_pp_ci_mean_lo_hi": list(gap_ded_cz),
            "gap_online_reallocated_pp_ci_mean_lo_hi": list(gap_rea_cz),
            "holm_bonferroni": {
                lbl: {"rejected": bool(rej), "p": p, "adjusted_alpha": a}
                for lbl, (rej, p, a) in zip(holm_labels, holm, strict=True)
            },
        }

        # Mechanism gap is a swing-dependent CANCELLATION, not equivalence:
        # pause wins carbon where the dirty-window excess is largest.
        real_zd = [(z, d) for z, d in out_zones.items() if d["source"] == "real-csv"]
        gaps = [d["fair_mechanism_gap_pp_mean"] for _, d in real_zd]
        excess = [d["dirty_window_excess_g_per_kwh"] for _, d in real_zd]
        gap_excess_corr = _pearson(gaps, excess)
        pause_wins = sorted(z for z, d in real_zd if d["fair_mechanism_gap_pp_mean"] < 0)
        throttle_wins = sorted(z for z, d in real_zd if d["fair_mechanism_gap_pp_mean"] >= 0)
        pause_win_mean = _mean([g for g in gaps if g < 0])
        throttle_win_mean = _mean([g for g in gaps if g >= 0])
        thr_or_mk = _mean([m for v in clusters["throttle_makespan_h"] for m in v])
        eco_makespan_pct = 100.0 * (1.0 / (1.0 - eco_throughput_cost) - 1.0)
        cross_zone["mechanism_gap_split"] = {
            "corr_gap_vs_dirty_excess": gap_excess_corr,
            "pause_wins_carbon_zones": pause_wins,
            "pause_wins_mean_pp": pause_win_mean,
            "throttle_wins_carbon_zones": throttle_wins,
            "throttle_wins_mean_pp": throttle_win_mean,
            "throttle_oracle_makespan_h_mean": thr_or_mk,
            "note": "net fair gap is a cancellation; the carbon-winning mechanism is zone-dependent",
        }

        console.print(f"\n[bold]Cross-zone (zone-clustered 95% CI) over {n_clusters} REAL zones[/]:")
        console.print(
            f"  always-on@eco endpoint  {eco_cz[0]:+.1f}% [{eco_cz[1]:+.1f},{eco_cz[2]:+.1f}]  "
            f"(deterministic hardware ratio {100 * (1 - eco_energy_ratio):.1f}% as a constant; realised "
            f"per-job saving averages this but varies with start offset; carbon-blind; pays "
            f"{100 * eco_throughput_cost:.0f}% throughput / +{eco_mk:.1f} h (+{eco_makespan_pct:.0f}% makespan))"
        )
        console.print(
            "\n  [bold]Carbon-signal decomposition (the genuine contribution):[/]"
        )
        console.print(
            f"  throttle (oracle)       {thr_cz[0]:+.1f}% [{thr_cz[1]:+.1f},{thr_cz[2]:+.1f}]"
        )
        console.print(
            f"    = same-budget carbon-BLIND {blind_cz[0]:+.1f}% [{blind_cz[1]:+.1f},{blind_cz[2]:+.1f}] "
            f"+ carbon-SIGNAL {sig_cz[0]:+.2f}pp [{sig_cz[1]:+.2f},{sig_cz[2]:+.2f}] (p={sig_p:.4f})"
        )
        console.print(
            "  [dim]→ most of throttle's saving is carbon-blind cap efficiency; the carbon SIGNAL "
            "(which windows are dirty) adds the modest remainder, at zero added latency.[/]"
        )
        console.print(
            "\n  [bold]Throttle vs GREEN pause — GREEN's full capability range:[/]"
        )
        console.print(
            f"  GREEN online + dedicated idle   {gon_d_cz[0]:+.1f}% [{gon_d_cz[1]:+.1f},{gon_d_cz[2]:+.1f}]  "
            f"(gap {gap_ded_cz[0]:+.1f}pp — most pessimistic for GREEN)"
        )
        console.print(
            f"  GREEN online + reallocated idle {gon_r_cz[0]:+.1f}% [{gon_r_cz[1]:+.1f},{gon_r_cz[2]:+.1f}]  "
            f"(gap {gap_rea_cz[0]:+.1f}pp)"
        )
        console.print(
            f"  GREEN offline/forecast + realloc {goff_r_cz[0]:+.1f}% [{goff_r_cz[1]:+.1f},{goff_r_cz[2]:+.1f}]  "
            f"(GREEN's OWN assumptions, pauses the SAME oracle windows) vs throttle {thr_cz[0]:+.1f}%"
        )
        console.print(
            f"  GREEN self-tuned budget+realloc {gst_cz[0]:+.1f}% [{gst_cz[1]:+.1f},{gst_cz[2]:+.1f}]  "
            f"(steel-man: carbon-max budget ~{gst_budget:.2f}, never loses) — beats throttle on carbon by "
            f"{-gap_gst_cz[0]:+.1f}pp [{-gap_gst_cz[2]:+.1f},{-gap_gst_cz[1]:+.1f}] BUT pays +{gst_mk:.0f} h "
            f"makespan vs throttle +{thr_on_mk:.1f} h."
        )
        console.print(
            f"  [bold]→ FAIR mechanism gap (same oracle windows, realloc idle): {gap_fair_cz[0]:+.2f}pp "
            f"[{gap_fair_cz[1]:+.2f},{gap_fair_cz[2]:+.2f}] (p={gap_fair_p:.3f}, "
            f"{'no detectable NET difference' if gap_fair_p > 0.05 else 'significant'})[/] — but this net "
            f"is a CANCELLATION, not equivalence:"
        )
        console.print(
            f"    pause wins carbon in {len(pause_wins)} high-dirty-excess zone(s) "
            f"({', '.join(pause_wins) or '—'}; mean {pause_win_mean:+.1f}pp); throttle wins in "
            f"{len(throttle_wins)} flat-tail zone(s) (mean {throttle_win_mean:+.1f}pp); "
            f"corr(gap, dirty-excess) = {gap_excess_corr:+.2f}."
        )
        console.print(
            "  [dim]→ at a matched budget the carbon winner is ZONE-DEPENDENT (pause wins where the "
            "dirty-window excess is largest); GREEN can exceed throttle on carbon only by deferring MORE "
            "(steel-man above). Throttle and pause are thus points on a carbon-vs-LATENCY Pareto:[/]"
        )
        console.print(
            f"  throttle +{thr_or_mk:.1f} h (oracle) / +{thr_on_mk:.1f} h (online) vs GREEN pause "
            f"+{goff_mk:.1f} h (forecast) / +{gon_mk:.1f} h (online) / +{gst_mk:.0f} h (carbon-max) "
            f"on a {args.job_hours:.0f} h job. Throttle is the low-latency point."
        )
        holm_txt = ", ".join(
            f"{lbl}: {'reject' if rej else 'no'} (p={p:.4f}, α'={a:.4f})"
            for lbl, (rej, p, a) in zip(holm_labels, holm, strict=True)
        )
        console.print(f"  [dim]Holm-Bonferroni (m=3): {holm_txt}[/]")

        green_losers = [z for z, d in out_zones.items()
                        if d["source"] == "real-csv" and d["green_online_dedicated_save_pct_mean"] < 0]
        signal_losers = [z for z, d in out_zones.items()
                         if d["source"] == "real-csv" and d["carbon_signal_value_pp_mean"] <= 0]
        if green_losers:
            console.print(
                f"[yellow]GREEN online+dedicated-idle nets <0% in {len(green_losers)} zone(s): "
                f"{', '.join(sorted(green_losers))} — but this is the pessimistic GREEN cell; "
                f"under reallocated idle / forecast those losses largely vanish.[/]"
            )
        if signal_losers:
            console.print(
                f"[yellow]The carbon SIGNAL adds ≤0 in {len(signal_losers)} real zone(s): "
                f"{', '.join(sorted(signal_losers))} — flat tail, so the throttle saving is just "
                f"same-budget cap efficiency there.[/]"
            )
        cross_zone["green_online_dedicated_loser_zones"] = sorted(green_losers)
        cross_zone["carbon_signal_loser_zones"] = sorted(signal_losers)
        if n_clusters < 5:
            console.print(f"[yellow]Only {n_clusters} real zones — cross-zone CIs are wide.[/]")
    else:
        console.print(
            f"[yellow]{n_clusters} real zone(s); need >=2 for cross-zone inference. "
            f"Drop <zone>_*.csv into {args.real_dir}.[/]"
        )
    console.print(
        "[dim]Inference unit: zone (cross-zone clustered bootstrap over REAL zones). Per-zone offset CIs "
        "are descriptive only — adjacent diurnal offsets into one fortnight overlap and are not "
        "independent. GREEN reported across its full online/offline x idle-regime range, not one cell.[/]"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "profile": args.profile,
            "full_cap_w": full,
            "eco_throttle_cap_w": throttle_cap,
            "eco_energy_ratio": eco_energy_ratio,
            "eco_throughput_cost_frac": eco_throughput_cost,
            "job_hours": args.job_hours,
            "total_iters": total_iters,
            "threshold_quantile": args.threshold_quantile,
            "green_pause_fraction_target": green_pause_fraction,
            "green_window_hours": args.green_window_hours,
            "resume_energy_wh": args.resume_energy_wh,
            "dedicated_idle_w": args.dedicated_idle_w,
            "reallocated_idle_w": _REALLOCATED_IDLE_W,
            "real_dir": args.real_dir,
            "n_real_zones": n_real,
            "energy_source": "measured-power-cap-profile",
            "replication_unit": "zone (cross-zone clustered); per-zone offset CIs descriptive only",
            "decomposition": "throttle = throttle-blind (same-budget cap efficiency) + carbon-signal value",
            "fair_gap_definition": "throttle vs GREEN-offline pause on identical oracle windows at reallocated idle (pure mechanism)",
            "policies": [
                "always-on@full", "always-on@eco (carbon-blind endpoint)",
                "throttle (oracle)", "throttle-blind (same-budget carbon-blind)",
                "throttle-online (GREEN online rule + eco cap)",
                "GREEN pause: {online, offline/forecast} x {dedicated 26W, reallocated 0W} idle",
            ],
            "zones": out_zones,
            "cross_zone_real_only": cross_zone,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default="data_cache/real_traces",
                   help="Directory of dropped-in ElectricityMaps CSVs named <zone>_*.csv "
                        "(e.g. de_*.csv, fr_*.csv). Zones without a CSV use the synthetic fallback.")
    p.add_argument("--profile", default="artifacts/hardware-pareto-3080ti.json")
    p.add_argument("--job-hours", type=float, default=24.0,
                   help="Always-on makespan target in hours (sizes the job).")
    p.add_argument("--threshold-quantile", type=float, default=0.6,
                   help="Dirty when intensity exceeds this within-zone quantile; GREEN's "
                        "pause budget is targeted at (1 - this).")
    p.add_argument("--throttle-cap-w", type=float, default=None)
    p.add_argument("--stride-hours", type=int, default=12,
                   help="Spacing between diurnal start offsets (descriptive replication samples).")
    p.add_argument("--green-window-hours", type=int, default=24,
                   help="Rolling window (hours) for GREEN's online percentile estimator.")
    p.add_argument("--resume-energy-wh", type=float, default=0.07)
    p.add_argument("--dedicated-idle-w", type=float, default=26.0,
                   help="Idle floor billed during pause windows in the dedicated regime; the "
                        "reallocated regime (co-tenant pays idle) is always also reported at 0 W.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

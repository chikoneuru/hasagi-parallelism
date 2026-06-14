"""Phase-divergence sensitivity bound for the carbon-signal headline numbers.

The amplitude sweep of ``marginal_intensity_sensitivity`` bounds the exposure of
the signal value to the LARGER SWING of marginal intensity, but it cannot bound
a marginal trace whose dirty hours sit at DIFFERENT CLOCK HOURS than the average
trace's. That phase divergence is the residual the amplitude proxy leaves open,
and it is the realistic failure mode: a deployable detector reads AVERAGE
intensity and throttles the windows that are dirty on average, while the carbon
is actually realized on the MARGINAL intensity whose peak may have moved.

This script bounds that exposure WITHOUT marginal data by a parametric
phase-shift stress test. For each zone and each shift ``d`` (hours), the
marginal proxy is the average series circularly shifted by ``d`` hours (same
amplitude and clean/dirty multiset, peaks relocated by ``d``). The detector
(oracle quantile cutoff and the deployable rolling rule) DECIDES on the
unshifted average series, while every carbon figure is CHARGED on the shifted
marginal series. ``d = 0`` is the identity and reproduces the published
average-trace signal (the sanity check); larger ``d`` measures how fast the
signal decays as the average and marginal diurnal phases pull apart, up to the
anti-phase worst case at ``d = 12``.

The per-window energy accounting is reused verbatim from the published panel
(``simulate_policy`` / ``simulate_masked_policy``); only the decision-vs-account
trace split is added. Cross-zone uncertainty at each ``d`` uses the project's
zone-clustered bootstrap (one per-zone value per cluster).

Usage::

    python -m experiments.marginal_phase_sensitivity \
        --real-dir data_cache/real_traces \
        --profile artifacts/hardware-pareto-3080ti.json \
        --job-hours 24 --threshold-quantile 0.6 \
        --out artifacts/marginal_phase_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from experiments.baselines.green import green_online_percentile_mask
from experiments.exp_realtrace_pareto import _pct, _uniform_off_mask
from tare.energy.throttle_pareto import (
    PowerCapProfile,
    simulate_masked_policy,
    simulate_policy,
)
from tare.energy.trace_schedule import (
    GRID_ZONE_IDS,
    diurnal_offsets,
    load_zone_traces,
    quantile_threshold,
    rotate,
    trace_to_hourly,
)
from tare.stats.bootstrap import clustered_bootstrap_ci

#: Phase shifts in hours between the average (decision) and marginal (accounting)
#: diurnal peaks. d=0 is the identity sanity check; d=12 is anti-phase.
_D_SWEEP = (0, 2, 4, 6, 8, 12)


def _phase_shift(hourly: list[float], d: int) -> list[float]:
    """Circularly shift the hourly series by ``d`` hours (relocates the diurnal
    peak by ``d`` while preserving the value multiset and the diurnal amplitude)."""
    return rotate(hourly, d)


def _signals_at_shift(
    profile: PowerCapProfile,
    base_hourly: list[float],
    marg_hourly: list[float],
    *,
    total_iters: int,
    q: float,
    throttle_cap: float,
    stride_hours: int,
    span_hours: int,
    green_pause_fraction: float,
    green_window: int,
) -> dict:
    """Oracle and deployable signal for one zone with the detector deciding on
    ``base_hourly`` (average) and carbon charged on ``marg_hourly`` (marginal)."""
    full = profile.max_throughput_cap
    offsets = diurnal_offsets(len(base_hourly), stride_hours=stride_hours, span_hours=span_hours)
    oracle_vals: list[float] = []
    deploy_vals: list[float] = []
    for off in offsets:
        dec = rotate(base_hourly, off)      # detector sees the average shape
        acc = rotate(marg_hourly, off)      # carbon is charged on the marginal shape
        threshold = quantile_threshold(dec, q)
        # always-on carbon on the accounting trace
        base = simulate_policy(
            profile, name="always-on", clean_cap_w=full, dirty_cap_w=full,
            total_iters=total_iters, window_s=3600.0, schedule_g=acc, threshold_g=threshold,
        )
        if base.total_carbon_g <= 0:
            continue
        base_g = base.total_carbon_g
        # decisions taken on the average trace
        oracle_mask = [0 if v > threshold else 1 for v in dec]
        blind_mask = _uniform_off_mask(len(dec), oracle_mask.count(0))
        online_mask = green_online_percentile_mask(
            dec, pause_fraction=green_pause_fraction, window_size=green_window,
        )
        masked = dict(total_iters=total_iters, window_s=3600.0, schedule_g=acc, full_cap_w=full)
        thr = simulate_masked_policy(profile, name="throttle", active_mask=oracle_mask, off_cap_w=throttle_cap, **masked)
        thr_blind = simulate_masked_policy(profile, name="throttle-blind", active_mask=blind_mask, off_cap_w=throttle_cap, **masked)
        thr_online = simulate_masked_policy(profile, name="throttle-online", active_mask=online_mask, off_cap_w=throttle_cap, **masked)
        oracle_vals.append(_pct(base_g, thr) - _pct(base_g, thr_blind))
        deploy_vals.append(_pct(base_g, thr_online) - _pct(base_g, thr_blind))
    return {
        "oracle_signal_pp": statistics.mean(oracle_vals) if oracle_vals else 0.0,
        "deployable_signal_pp": statistics.mean(deploy_vals) if deploy_vals else 0.0,
        "n_offsets": len(oracle_vals),
    }


def run(args: argparse.Namespace) -> int:
    profile = PowerCapProfile.from_json(args.profile)
    full = profile.max_throughput_cap
    throttle_cap = args.throttle_cap_w or profile.energy_optimal_cap
    total_iters = int(args.job_hours * 3600.0 * profile.point(full).throughput_iters_s)
    green_pause_fraction = max(0.0, min(1.0, 1.0 - args.threshold_quantile))

    ztraces = load_zone_traces(args.real_dir, GRID_ZONE_IDS)
    per_zone: dict[str, dict] = {}
    oracle_clusters: dict[int, list[list[float]]] = {d: [] for d in _D_SWEEP}
    deploy_clusters: dict[int, list[list[float]]] = {d: [] for d in _D_SWEEP}

    for zone, zt in ztraces.items():
        base_hourly = trace_to_hourly(zt.trace)
        zsig: dict[str, dict[str, float]] = {}
        for d in _D_SWEEP:
            marg = _phase_shift(base_hourly, d)
            sig = _signals_at_shift(
                profile, base_hourly, marg, total_iters=total_iters, q=args.threshold_quantile,
                throttle_cap=throttle_cap, stride_hours=args.stride_hours,
                span_hours=int(args.job_hours * 2), green_pause_fraction=green_pause_fraction,
                green_window=args.green_window_hours,
            )
            zsig[str(d)] = sig
            if zt.source == "real-csv" and sig["n_offsets"] > 0:
                oracle_clusters[d].append([sig["oracle_signal_pp"]])
                deploy_clusters[d].append([sig["deployable_signal_pp"]])
        per_zone[zone] = {"source": zt.source, "by_shift_h": zsig}

    def cz(clusters: dict[int, list[list[float]]]) -> dict:
        rng = random.Random(args.seed)
        out = {}
        for d in _D_SWEEP:
            pt, lo, hi = clustered_bootstrap_ci(clusters[d], n_boot=args.n_boot, rng=rng)
            out[str(d)] = {"mean": pt, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0.0 or hi < 0.0)}
        return out

    oracle_cz = cz(oracle_clusters)
    deploy_cz = cz(deploy_clusters)
    # first shift at which each signal's CI no longer excludes zero / mean turns non-positive
    def first_break(cz_map: dict) -> dict:
        ci_zero = next((d for d in _D_SWEEP if not cz_map[str(d)]["excludes_zero"]), None)
        sign = next((d for d in _D_SWEEP if cz_map[str(d)]["mean"] <= 0.0), None)
        return {"ci_includes_zero_at_h": ci_zero, "mean_non_positive_at_h": sign}

    out = {
        "stress": "phase divergence between average (decision) and marginal (accounting) diurnal peaks",
        "trace_source": "real ElectricityMaps (16 zones, summer fortnight)",
        "n_real_zones": sum(1 for z in per_zone.values() if z["source"] == "real-csv"),
        "shift_hours_swept": list(_D_SWEEP),
        "decision_trace": "average (unshifted)",
        "accounting_trace": "marginal proxy = average circularly shifted by d hours",
        "oracle_signal_pp_by_shift": oracle_cz,
        "deployable_signal_pp_by_shift": deploy_cz,
        "oracle_break": first_break(oracle_cz),
        "deployable_break": first_break(deploy_cz),
        "identity_check_d0": {
            "oracle_pp": oracle_cz["0"]["mean"],
            "deployable_pp": deploy_cz["0"]["mean"],
            "note": "d=0 must reproduce the published average-trace signal (~+1.56 oracle, ~+0.75 deployable)",
        },
        "per_zone": per_zone,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_zone"}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default="data_cache/real_traces")
    p.add_argument("--profile", default="artifacts/hardware-pareto-3080ti.json")
    p.add_argument("--job-hours", type=float, default=24.0)
    p.add_argument("--threshold-quantile", type=float, default=0.6)
    p.add_argument("--throttle-cap-w", type=float, default=None)
    p.add_argument("--stride-hours", type=int, default=1)
    p.add_argument("--green-window-hours", type=int, default=24)
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/marginal_phase_sensitivity.json")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

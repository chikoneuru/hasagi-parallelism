"""Decision-rule margin on the MEASURED cap profile (not a flat energy model).

The flat-energy decision-rule margin (``exp_realtrace_decision_rule``) isolates
which-windows-to-pick on a constant per-tick energy model. This companion puts
the same contrast on the measured RTX 3080 Ti power-cap substrate: both rules
select dirty windows online over the 16 real ElectricityMaps zones, and both
apply their selection as a THROTTLE to the energy-optimal cap (full cap
elsewhere), with energy and carbon read from the measured cap profile rather
than a flat 210 W. The margin is Tare-rule minus GREEN-rule carbon saving
against always-on, so it isolates the decision rule on the measured substrate,
the form a deployed throttling controller would actually run.

Per-zone evaluation reuses ``simulate_masked_policy`` and the diurnal-offset
machinery of the published panel; the cross-zone inference is the project's
zone-clustered bootstrap plus the same small-sample-robust cross-checks the
oracle signal carries.

Usage::

    python -m experiments.exp_realtrace_decision_rule_measured \
        --real-dir data_cache/real_traces \
        --profile artifacts/hardware-pareto-3080ti.json \
        --out artifacts/realtrace_decision_rule_measured.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from experiments.baselines.green import (
    green_online_percentile_mask,
    pause_fraction,
    tare_threshold_online_mask,
)
from experiments.exp_realtrace_pareto import _pct
from experiments.reproduce_paper_robustness import cluster_t_interval, jackknife_range
from tare.energy.throttle_pareto import PowerCapProfile, simulate_masked_policy, simulate_policy
from tare.energy.trace_schedule import (
    GRID_ZONE_IDS,
    diurnal_offsets,
    find_zone_csv,
    load_electricitymaps_csv,
    rotate,
    trace_to_hourly,
)
from tare.stats.bootstrap import clustered_bootstrap_ci, clustered_permutation_pvalue


def _zone_margin(profile, hourly, *, total_iters, full, eco, window, stride_hours, span_hours):
    """Mean (Tare-rule minus GREEN-rule) throttle saving over diurnal offsets, on
    the measured cap profile; both rules throttle their online-selected dirty
    windows to the eco cap, GREEN's budget matched to Tare's per offset."""
    margins, tare_saves, green_saves = [], [], []
    for off in diurnal_offsets(len(hourly), stride_hours=stride_hours, span_hours=span_hours):
        sched = rotate(hourly, off)
        base = simulate_policy(
            profile, name="always-on", clean_cap_w=full, dirty_cap_w=full,
            total_iters=total_iters, window_s=3600.0, schedule_g=sched, threshold_g=0.0,
        )
        if base.total_carbon_g <= 0:
            continue
        base_g = base.total_carbon_g
        # Tare rolling-median rule: 0 = throttle (dirty) window.
        tare_mask = tare_threshold_online_mask(sched, window_size=window)
        tare_pf = pause_fraction(tare_mask)
        # GREEN rolling-percentile rule, budget matched to Tare's emergent fraction.
        green_mask = green_online_percentile_mask(sched, pause_fraction=tare_pf, window_size=window)
        masked = dict(total_iters=total_iters, window_s=3600.0, schedule_g=sched, full_cap_w=full)
        tare_thr = simulate_masked_policy(profile, name="tare-throttle", active_mask=tare_mask, off_cap_w=eco, **masked)
        green_thr = simulate_masked_policy(profile, name="green-throttle", active_mask=green_mask, off_cap_w=eco, **masked)
        ts, gs = _pct(base_g, tare_thr), _pct(base_g, green_thr)
        tare_saves.append(ts)
        green_saves.append(gs)
        margins.append(ts - gs)
    return {
        "decision_rule_gap_pp": statistics.mean(margins) if margins else 0.0,
        "tare_save_pct": statistics.mean(tare_saves) if tare_saves else 0.0,
        "green_save_pct": statistics.mean(green_saves) if green_saves else 0.0,
        "n_offsets": len(margins),
    }


def run(args: argparse.Namespace) -> int:
    profile = PowerCapProfile.from_json(args.profile)
    full = profile.max_throughput_cap
    eco = args.throttle_cap_w or profile.energy_optimal_cap
    total_iters = int(args.job_hours * 3600.0 * profile.point(full).throughput_iters_s)

    per_zone, gaps = [], []
    for z in GRID_ZONE_IDS:
        csv = find_zone_csv(args.real_dir, z)
        if csv is None:
            raise SystemExit(f"no real CSV for zone {z} under {args.real_dir}")
        hourly = trace_to_hourly(load_electricitymaps_csv(csv))
        m = _zone_margin(
            profile, hourly, total_iters=total_iters, full=full, eco=eco,
            window=args.green_window_hours, stride_hours=args.stride_hours,
            span_hours=int(args.job_hours * 2),
        )
        m["zone"] = z
        per_zone.append(m)
        gaps.append(m["decision_rule_gap_pp"])

    clusters = [[g] for g in gaps]
    point, lo, hi = clustered_bootstrap_ci(clusters, n_boot=args.n_boot, rng=random.Random(args.seed))
    pval = clustered_permutation_pvalue(clusters, rng=random.Random(args.seed))
    ct_lo, ct_hi = cluster_t_interval(gaps)
    jk_lo, jk_hi = jackknife_range(gaps)

    out = {
        "contrast": "decision-rule margin (Tare rolling-median minus GREEN rolling-percentile), "
                    "both applied as THROTTLE to the energy-optimal cap, on the MEASURED RTX 3080 Ti profile",
        "energy_model": "measured power-cap profile (eco vs full cap), not flat per-tick",
        "trace_source": "real ElectricityMaps (16 zones, summer fortnight)",
        "full_cap_w": full,
        "eco_cap_w": eco,
        "n_zones": len(per_zone),
        "inference_unit": "zone (cross-zone clustered bootstrap)",
        "cross_zone_decision_rule_gap_pp": {
            "mean": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0.0 or hi < 0.0),
        },
        "cluster_t_95ci_pp": [ct_lo, ct_hi],
        "jackknife_loo_range_pp": [jk_lo, jk_hi],
        "exact_sign_flip_pvalue": pval,
        "zones_positive": sum(1 for g in gaps if g > 0.0),
        "zones_total": len(per_zone),
        "flat_energy_reference_pp": 1.6779,
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
    p.add_argument("--throttle-cap-w", type=float, default=None)
    p.add_argument("--stride-hours", type=int, default=12)
    p.add_argument("--green-window-hours", type=int, default=24)
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/realtrace_decision_rule_measured.json")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

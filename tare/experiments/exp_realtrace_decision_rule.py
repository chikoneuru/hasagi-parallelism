"""Decision-rule margin (rolling-median vs rolling-percentile pause) on REAL traces.

The synthetic-parametric decision-rule study reports a +1.99 pp gap between the
rolling-median pause rule and the rolling-percentile pause rule on a flat
per-tick energy model. That number rides a synthetic multi-harmonic grid whose
diurnal swing is far stronger than real grids show, so it is an upper-bounded
illustration rather than a real-trace result.

This script recomputes the SAME decision-rule contrast (both arms PAUSE, vary
only the rule that picks the hours, identical flat per-tick energy model) on the
16 real ElectricityMaps zones, so the paper carries at least one real-trace
head-to-head against a named carbon-aware pause rule. Per-zone evaluation is
``run_zone`` from ``exp_h5c_real_trace_vs_green`` reused verbatim; the only new
logic is the cross-zone clustered inference (each real zone is one cluster, one
per-zone gap value) matching the rest of the paper.

The gap is HASAGI-online minus GREEN-online savings, in percentage points of
carbon saved against the constant-N (always-on) reference, at a matched pause
budget (GREEN's budget is set to HASAGI-online's emergent pause fraction inside
``run_zone``, so the contrast isolates the decision rule).

Usage::

    python -m experiments.exp_realtrace_decision_rule \
        --real-dir data_cache/real_traces \
        --out artifacts/realtrace_decision_rule.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from experiments.exp_h5c_real_trace_vs_green import _pct_savings, run_zone
from experiments.reproduce_paper_robustness import cluster_t_interval, jackknife_range
from tare.energy.trace_schedule import GRID_ZONE_IDS, find_zone_csv
from tare.stats.bootstrap import clustered_bootstrap_ci, clustered_permutation_pvalue


def _zone_gap(results: list, zone: str) -> dict:
    """HASAGI-online minus GREEN-online savings (pp) for one zone, plus the
    component savings, computed from the per-policy emissions ``run_zone`` returns."""
    by_policy = {r.policy: r for r in results}
    const_em = by_policy["constant-N"].emissions_g
    tare_on = _pct_savings(by_policy["tare-online"].emissions_g, const_em)
    green_on = _pct_savings(by_policy["green-online"].emissions_g, const_em)
    return {
        "zone": zone,
        "tare_online_save_pct": tare_on,
        "green_online_save_pct": green_on,
        "decision_rule_gap_pp": tare_on - green_on,
        "tare_online_pause_frac": by_policy["tare-online"].pause_fraction,
        "n_ticks": by_policy["tare-online"].n_ticks,
    }


def run(args: argparse.Namespace) -> int:
    zones = list(GRID_ZONE_IDS)
    per_zone: list[dict] = []
    missing: list[str] = []
    for z in zones:
        csv = find_zone_csv(args.real_dir, z)
        if csv is None:
            missing.append(z)
            continue
        per_zone.append(_zone_gap(run_zone(csv, z, args.tare_threshold_multiplier), z))

    if missing:
        raise SystemExit(
            f"no real CSV for zones {missing} under {args.real_dir}; "
            "this re-run requires real traces for the full panel."
        )

    gaps = [z["decision_rule_gap_pp"] for z in per_zone]
    clusters = [[g] for g in gaps]
    rng = random.Random(args.seed)
    point, lo, hi = clustered_bootstrap_ci(clusters, n_boot=args.n_boot, rng=rng)
    pval = clustered_permutation_pvalue(clusters, rng=random.Random(args.seed))
    n_pos = sum(1 for g in gaps if g > 0.0)
    # Small-sample-robust cross-checks, matching the oracle signal-value's treatment.
    ct_lo, ct_hi = cluster_t_interval(gaps)
    jk_lo, jk_hi = jackknife_range(gaps)

    out = {
        "contrast": "decision-rule margin (HASAGI rolling-median pause minus "
                    "GREEN rolling-percentile pause), pp of carbon saved vs always-on",
        "energy_model": "flat per-tick (decision-rule isolated; matches the "
                        "synthetic +1.99 pp study's energy model)",
        "trace_source": "real ElectricityMaps (16 zones, summer fortnight)",
        "n_zones": len(per_zone),
        "inference_unit": "zone (cross-zone clustered bootstrap; one gap per zone)",
        "cross_zone_decision_rule_gap_pp": {
            "mean": point, "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": (lo > 0.0 or hi < 0.0),
        },
        "exact_sign_flip_pvalue": pval,
        "cluster_t_95ci_pp": [ct_lo, ct_hi],
        "jackknife_loo_range_pp": [jk_lo, jk_hi],
        "zones_positive": n_pos,
        "zones_total": len(per_zone),
        "per_zone_mean_gap_pp": statistics.mean(gaps),
        "synthetic_reference_pp": 1.99,
        "per_zone": per_zone,
    }
    dest = Path(args.out)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_zone"}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default="data_cache/real_traces")
    p.add_argument("--tare-threshold-multiplier", type=float, default=1.10)
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/realtrace_decision_rule.json")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

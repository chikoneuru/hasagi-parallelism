"""Cross-zone dependence sensitivity for the real-trace carbon-signal headline.

The published carbon-signal intervals (oracle +1.56 pp, deployable +0.75 pp)
treat the 16 ElectricityMaps zones as independent clusters. European zones are
market-coupled and weather-correlated, so the effective number of independent
clusters is below 16 and the zone-clustered intervals may be anti-conservative.
This re-analysis does three things, all from stored artifacts and cached traces
(no re-simulation):

1. Quantifies the dependence directly: pairwise Pearson correlation of the 16
   zones' hourly carbon-intensity series over the shared summer fortnight
   (``data_cache/real_traces``), summarized as mean within-Europe vs
   cross-block correlation.
2. Coarsens the cluster definition to synchronous-grid/continent blocks
   (Europe, Asia, North America, Oceania, South America, Africa), collapses
   each block to the mean of its per-zone signal deltas, and recomputes the
   block-clustered bootstrap CI plus the EXACT sign-flip permutation p over
   blocks for both the oracle signal (throttle - throttle_blind) and the
   deployable signal (throttle_online - throttle_blind).
3. Reproduces the published 16-zone baseline from the same per-zone values so
   the two cluster definitions are directly comparable in one artifact.

Caveat stated up front: with 6 blocks the exact sign-flip permutation test has
a combinatorial floor of 2 / 2**6 = 0.03125, so no effect, however large, can
reach p < 0.03125 under this grouping.

Inputs:  artifacts/realtrace_pareto.json (per-zone signal deltas),
         data_cache/real_traces/*.csv  (hourly intensity, summer window).
Output:  artifacts/crosszone_dependence_sensitivity.json
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tare.stats.bootstrap import (  # noqa: E402
    clustered_bootstrap_ci,
    clustered_permutation_pvalue,
    one_sample_standardized_effect,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.join(ROOT, "artifacts")
TRACES = os.path.join(ROOT, "data_cache", "real_traces")

# Synchronous-grid / continent blocks for the 16-zone pre-registered panel.
# Europe is one block because the EU internal electricity market couples
# day-ahead prices across these bidding zones and a shared synoptic weather
# system drives wind/solar output (GB and NO are AC-asynchronous from the
# Continental grid but DC-interconnected and market-coupled, so they are
# conservatively grouped with it). AE sits on the GCC Gulf grid, electrically
# separate from East/South Asia, but is grouped into the Asia block by
# continent -- the conservative choice, since fewer blocks means a coarser,
# harder test. Zones within the Asia block are NOT mutually synchronous; the
# block is a deliberate upper bound on plausible dependence.
BLOCKS = {
    "europe": ["DE", "FR", "GB", "NO", "PL"],
    "north_america": ["US-CA"],
    "asia": ["AE", "CN", "IN", "JP", "KR", "SG", "VN"],
    "oceania": ["AU"],
    "south_america": ["BR"],
    "africa": ["ZA"],
}

N_BOOT = 20_000


def _load_pareto():
    with open(os.path.join(ART, "realtrace_pareto.json")) as fh:
        return json.load(fh)


def per_zone_signals(pareto):
    """Per-zone oracle and deployable carbon-signal deltas (pp).

    oracle     = throttle - throttle_blind(same budget)  (stored directly)
    deployable = throttle_online - throttle_blind(same budget)
    Both are paired within-zone differences, matching the published headline
    convention (energy_ratio_propagation_ci.py / exp_realtrace_pareto.py).
    """
    out = {}
    for zone, z in pareto["zones"].items():
        if z.get("source") != "real-csv":
            raise SystemExit(f"zone {zone} is not real-csv; refusing to mix sources")
        out[zone] = {
            "oracle": z["carbon_signal_value_pp_mean"],
            "deployable": (
                z["throttle_online_save_pct_mean"]
                - z["throttle_blind_same_budget_save_pct_mean"]
            ),
        }
    return out


def _ci_p(clusters, seed):
    """Cluster-bootstrap CI + exact sign-flip permutation p over the clusters."""
    point, lo, hi = clustered_bootstrap_ci(clusters, n_boot=N_BOOT, rng=random.Random(seed))
    p = clustered_permutation_pvalue(clusters)  # exact for <= 20 clusters
    means = [statistics.mean(c) for c in clusters]
    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": (lo > 0.0 or hi < 0.0),
        "p_exact_signflip": p,
        "standardized_effect": one_sample_standardized_effect(means),
        "n_clusters": len(clusters),
        "permutation_floor": 2.0 / (1 << len(clusters)),
    }


def _read_trace(zone):
    """Hourly {utc-datetime-string: intensity} for one zone's summer CSV."""
    pattern = os.path.join(TRACES, f"{zone.lower()}_*_hourly.csv")
    paths = sorted(glob.glob(pattern))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one summer trace for {zone}, found {paths}")
    series = {}
    with open(paths[0], newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if not row or not row[-1].strip():
                continue
            series[row[0]] = float(row[-1])
    return series


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def correlation_summary(zones, block_of):
    """Pairwise Pearson r of hourly intensity, grouped within/cross block."""
    traces = {z: _read_trace(z) for z in zones}
    shared = set.intersection(*(set(t) for t in traces.values()))
    hours = sorted(shared)
    aligned = {z: [traces[z][h] for h in hours] for z in zones}

    pairs = {}
    within_by_block = {}
    within_block, cross_block = [], []
    for i, za in enumerate(zones):
        for zb in zones[i + 1:]:
            r = _pearson(aligned[za], aligned[zb])
            pairs[f"{za}|{zb}"] = r
            if block_of[za] == block_of[zb]:
                within_block.append(r)
                within_by_block.setdefault(block_of[za], []).append(r)
            else:
                cross_block.append(r)

    def _stats(vals):
        if not vals:
            return None
        return {
            "mean": statistics.mean(vals),
            "min": min(vals),
            "max": max(vals),
            "n_pairs": len(vals),
        }

    return {
        "window_utc": f"{hours[0]} .. {hours[-1]}",
        "n_hours_aligned": len(hours),
        "within_europe": _stats(within_by_block.get("europe", [])),
        "within_block_by_block": {b: _stats(v) for b, v in sorted(within_by_block.items())},
        "within_block_all": _stats(within_block),
        "cross_block": _stats(cross_block),
        "all_pairs": _stats(list(pairs.values())),
        "pairwise_r": {k: round(v, 4) for k, v in sorted(pairs.items())},
    }


def run(_args):
    pareto = _load_pareto()
    signals = per_zone_signals(pareto)
    zones = sorted(signals)

    blocked = sorted(z for zs in BLOCKS.values() for z in zs)
    if blocked != zones:
        raise SystemExit(f"block assignment does not cover the panel: {blocked} vs {zones}")
    block_of = {z: b for b, zs in BLOCKS.items() for z in zs}

    out = {
        "_method": (
            "Sensitivity of the real-trace carbon-signal inference to cross-zone "
            "dependence. Per-zone paired deltas (oracle = throttle - "
            "throttle_blind_same_budget; deployable = throttle_online - "
            "throttle_blind_same_budget) are read from realtrace_pareto.json. "
            "Baseline treats each of the 16 zones as an independent cluster "
            "(percentile cluster bootstrap, 20k draws, seed 0; exact sign-flip "
            "permutation over 2^16 assignments). The sensitivity reassigns zones to 6 "
            "synchronous-grid/continent blocks, collapses each block to the mean of its "
            "per-zone deltas, and redoes both tests at the block level (exact sign-flip "
            "over 2^6 = 64 assignments; combinatorial p floor 2/64 = 0.03125, so "
            "p = 0.03125 is the smallest attainable value and equality with the floor "
            "must not be read as 'p << 0.05'). Dependence is quantified as pairwise "
            "Pearson r of the zones' aligned hourly carbon-intensity series over the "
            "shared summer fortnight (data_cache/real_traces, UTC timestamps). "
            "Reproduce: python -m experiments.crosszone_dependence_sensitivity"
        ),
        "source_artifact": "artifacts/realtrace_pareto.json",
        "trace_dir": pareto.get("real_dir", "data_cache/real_traces"),
        "n_zones": len(zones),
        "zones": zones,
        "block_assignment": BLOCKS,
        "per_zone_signal_pp": signals,
    }

    out["per_block_mean_pp"] = {
        b: {
            "n_zones": len(zs),
            "oracle": statistics.mean(signals[z]["oracle"] for z in zs),
            "deployable": statistics.mean(signals[z]["deployable"] for z in zs),
        }
        for b, zs in BLOCKS.items()
    }

    out["zone_level_baseline_16"] = {
        metric: _ci_p([[signals[z][metric]] for z in zones], seed=0)
        for metric in ("oracle", "deployable")
    }
    out["zone_level_baseline_16"]["published_reference"] = {
        "oracle_ci_mean_lo_hi": pareto["cross_zone_real_only"][
            "carbon_signal_value_pp_ci_mean_lo_hi"
        ],
        "oracle_pvalue": pareto["cross_zone_real_only"]["carbon_signal_value_pvalue"],
        "deployable_source": "artifacts/energy_ratio_propagation_ci.json:deployable_signal_pp",
    }

    out["block_level_6"] = {
        metric: _ci_p([[signals[z][metric] for z in zs] for zs in BLOCKS.values()], seed=0)
        for metric in ("oracle", "deployable")
    }
    out["block_level_6"]["_note_weighting"] = (
        "The block-level grand mean weights every block equally, so the point estimate "
        "shifts relative to the 16-zone mean (deployable 0.75 -> 1.25 pp) by upweighting "
        "the singleton blocks (US-CA, AU) and downweighting the 7-zone Asia block. The "
        "block analysis is a robustness check on the SIGN and significance under coarser "
        "clustering, not a replacement point estimate; the headline point estimate "
        "remains the zone-level mean."
    )

    out["hourly_intensity_correlation"] = correlation_summary(zones, block_of)

    dest = os.path.join(ART, "crosszone_dependence_sensitivity.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)

    brief = {
        "per_block_mean_pp": out["per_block_mean_pp"],
        "zone_level_baseline_16": {
            k: v for k, v in out["zone_level_baseline_16"].items() if k != "published_reference"
        },
        "block_level_6": out["block_level_6"],
        "correlation": {
            k: out["hourly_intensity_correlation"][k]
            for k in ("within_europe", "within_block_all", "cross_block", "all_pairs")
        },
    }
    print(json.dumps(brief, indent=2))
    print(f"wrote {os.path.relpath(dest, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(run(argparse.ArgumentParser().parse_args()))

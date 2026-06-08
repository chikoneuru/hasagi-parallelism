"""Shape-distortion sensitivity bound for the carbon-signal headline numbers.

The carbon-policy panel (``experiments.exp_realtrace_pareto``) computes its
headline savings on AVERAGE (life-cycle-assessment) hourly grid-intensity
traces. The decision-correct quantity for a same-budget throttling response is
the MARGINAL intensity of the displaced generation, which is more volatile than
the average and is not freely available without a paid/licensed feed. This
script bounds the exposure of the three headline carbon numbers to that
average-vs-marginal gap WITHOUT any marginal data, by a parametric
shape-distortion sweep.

For each zone we take its hourly average-intensity series ``a(h)`` with mean
``mu`` and build a synthetic marginal proxy that amplifies the deviation from
the mean::

    m_k(h) = max(0, mu + k * (a(h) - mu))

for a sweep of amplification factors ``k``. ``k = 1`` is the identity and must
reproduce the published average-trace result exactly (it is the sanity check);
``k > 1`` stretches the diurnal swing to emulate the larger volatility of
marginal intensity. This is a PARAMETRIC SHAPE-DISTORTION BOUND, not a
measurement on real marginal data, and must be reported as such.

The savings logic is REUSED, not re-implemented: the script imports
``_savings_over_offsets`` and its helpers from the experiment module so the
methodology is byte-identical to the published panel. Only the per-zone hourly
series fed into the panel is distorted; the within-zone dirty/clean quantile
threshold is recomputed on the distorted series (so the oracle mask remains the
zone-relative q-quantile of whatever series is in play, exactly as the panel
does for the undistorted series).

Three headline numbers are tracked, matching their published definitions:

* oracle carbon signal  = throttle(oracle) - throttle_blind(same budget)
  (stored per-zone as ``carbon_signal_value_pp_mean``; cross-zone mean ~ +1.56)
* deployable carbon signal = throttle_online - throttle_blind(same budget)
  (cross-zone mean ~ +0.75)
* fair mechanism gap    = throttle(oracle) - GREEN_offline_reallocated_pause
  (stored per-zone as ``fair_mechanism_gap_pp_mean``; cross-zone mean ~ +0.57)

Cross-zone uncertainty at each ``k`` uses the project's zone-clustered
bootstrap (each real zone is one cluster, contributing one per-zone-mean value).

Usage::

    python -m experiments.marginal_intensity_sensitivity \
        --real-dir data_cache/real_traces \
        --profile artifacts/hardware-pareto-3080ti.json \
        --job-hours 24 --threshold-quantile 0.6 \
        --out artifacts/marginal_intensity_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from experiments.exp_realtrace_pareto import _savings_over_offsets
from tare.energy.throttle_pareto import PowerCapProfile
from tare.energy.trace_schedule import (
    GRID_ZONE_IDS,
    load_zone_traces,
    quantile_threshold,
    trace_to_hourly,
    zone_stats,
)
from tare.stats.bootstrap import clustered_bootstrap_ci

#: Amplification factors for the deviation-from-mean stretch. k=1 is identity
#: (the published average-trace result and the sanity check).
_K_SWEEP = (1.0, 1.25, 1.5, 2.0, 2.5)

#: Tolerance for the k=1 sanity check against the published per-zone values
#: (the panel itself has no stochastic step at fixed seed for these scalars, so
#: the match should be tight; we allow a small slack for float accumulation).
_SANITY_TOL_PP = 1e-6


def _amplify(hourly: list[float], k: float) -> list[float]:
    """Stretch each hour's deviation from the series mean by ``k`` (floored at 0).

    ``k = 1`` returns the series unchanged. Larger ``k`` widens the diurnal
    swing while keeping the same clean/dirty ordering of hours, emulating the
    higher volatility of marginal relative to average intensity.
    """
    mu = statistics.mean(hourly)
    return [max(0.0, mu + k * (v - mu)) for v in hourly]


def _deployable_signal(zone_vec: dict) -> float:
    """Per-zone deployable signal = mean(throttle_online) - mean(throttle_blind).

    Matches the published +0.75 pp number, which is the online-estimator
    throttle saving minus the same-budget carbon-blind throttle saving.
    """
    online = zone_vec["throttle_online_pct"]
    blind = zone_vec["throttle_blind_pct"]
    return _mean(online) - _mean(blind)


def _oracle_signal(zone_vec: dict) -> float:
    """Per-zone oracle signal = mean(signal_pp) = mean(throttle - throttle_blind)."""
    return _mean(zone_vec["signal_pp"])


def _fair_gap(zone_vec: dict) -> float:
    """Per-zone fair mechanism gap = mean(gap_fair_pp)."""
    return _mean(zone_vec["gap_fair_pp"])


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run(args: argparse.Namespace) -> int:
    profile = PowerCapProfile.from_json(args.profile)
    full = profile.max_throughput_cap
    throttle_cap = args.throttle_cap_w or profile.energy_optimal_cap
    total_iters = int(args.job_hours * 3600.0 * profile.point(full).throughput_iters_s)
    resume_kwh = args.resume_energy_wh / 1000.0
    green_pause_fraction = max(0.0, min(1.0, 1.0 - args.threshold_quantile))
    rng = random.Random(args.seed)

    ztraces = load_zone_traces(args.real_dir, GRID_ZONE_IDS)

    # Per (zone, k): the three signals. Real zones only enter the cross-zone
    # clustered inference (the published panel reports cross-zone over real CSVs).
    per_zone: dict[str, dict] = {}
    # cluster lists keyed by k: each is a list of single-value clusters (one per zone).
    oracle_clusters: dict[float, list[list[float]]] = {k: [] for k in _K_SWEEP}
    deploy_clusters: dict[float, list[list[float]]] = {k: [] for k in _K_SWEEP}
    fair_clusters: dict[float, list[list[float]]] = {k: [] for k in _K_SWEEP}

    for zone, zt in ztraces.items():
        base_hourly = trace_to_hourly(zt.trace)
        zsig: dict[str, dict[str, float]] = {}
        for k in _K_SWEEP:
            hourly_k = _amplify(base_hourly, k)
            threshold_k = quantile_threshold(hourly_k, args.threshold_quantile)
            res = _savings_over_offsets(
                profile, hourly_k, total_iters=total_iters, threshold=threshold_k,
                throttle_cap=throttle_cap, resume_kwh=resume_kwh,
                dedicated_idle_w=args.dedicated_idle_w,
                stride_hours=args.stride_hours, span_hours=int(args.job_hours * 2),
                green_pause_fraction=green_pause_fraction, green_window=args.green_window_hours,
            )
            zsig[f"{k}"] = {
                "oracle_signal_pp": _oracle_signal(res),
                "deployable_signal_pp": _deployable_signal(res),
                "fair_gap_pp": _fair_gap(res),
                "n_offsets": res["n_offsets"],
            }
            if zt.source == "real-csv" and res["n_offsets"] > 0:
                oracle_clusters[k].append([zsig[f"{k}"]["oracle_signal_pp"]])
                deploy_clusters[k].append([zsig[f"{k}"]["deployable_signal_pp"]])
                fair_clusters[k].append([zsig[f"{k}"]["fair_gap_pp"]])
        per_zone[zone] = {
            "source": zt.source,
            "intensity_stats": zone_stats(base_hourly),
            "by_k": zsig,
        }

    # Cross-zone clustered CI per k for each signal.
    def cz(clusters: dict[float, list[list[float]]]) -> dict[str, list[float]]:
        return {f"{k}": list(clustered_bootstrap_ci(clusters[k], rng=rng)) for k in _K_SWEEP}

    oracle_cz = cz(oracle_clusters)
    deploy_cz = cz(deploy_clusters)
    fair_cz = cz(fair_clusters)

    # Envelope (min/max of the cross-zone mean across the k-sweep) per signal.
    def envelope(cz_map: dict[str, list[float]]) -> dict[str, float]:
        means = {kk: vv[0] for kk, vv in cz_map.items()}
        return {
            "min_mean": min(means.values()),
            "max_mean": max(means.values()),
            "argmin_k": float(min(means, key=means.get)),
            "argmax_k": float(max(means, key=means.get)),
        }

    # Per-zone sign-flip counts: how many real zones change the sign of a signal
    # at ANY k relative to the k=1 baseline sign (strict sign change; a value
    # landing on exactly 0.0 from a non-zero baseline counts as a flip).
    def sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    def sign_flip_count(key: str) -> dict:
        flippers = []
        for zone, zd in per_zone.items():
            if zd["source"] != "real-csv":
                continue
            base = zd["by_k"]["1.0"][key]
            base_sign = sign(base)
            ks_flipped = [
                kk for kk in _K_SWEEP
                if kk != 1.0 and sign(zd["by_k"][f"{kk}"][key]) != base_sign
            ]
            if ks_flipped:
                flippers.append({
                    "zone": zone,
                    "k1_value_pp": base,
                    "flipped_at_k": ks_flipped,
                    "values_by_k": {f"{kk}": zd["by_k"][f"{kk}"][key] for kk in _K_SWEEP},
                })
        return {"count": len(flippers), "zones": flippers}

    oracle_flips = sign_flip_count("oracle_signal_pp")
    deploy_flips = sign_flip_count("deployable_signal_pp")
    fair_flips = sign_flip_count("fair_gap_pp")

    # k=1 sanity check against the published artifact (real zones only).
    sanity = {"checked": False}
    pub_path = Path(args.published)
    if pub_path.is_file():
        pub = json.loads(pub_path.read_text())
        pub_zones = pub.get("zones", {})
        diffs_oracle, diffs_deploy, diffs_fair = [], [], []
        examples = []
        for zone, zd in per_zone.items():
            if zd["source"] != "real-csv" or zone not in pub_zones:
                continue
            pz = pub_zones[zone]
            k1 = zd["by_k"]["1.0"]
            pub_oracle = pz["carbon_signal_value_pp_mean"]
            pub_deploy = pz["throttle_online_save_pct_mean"] - pz["throttle_blind_same_budget_save_pct_mean"]
            pub_fair = pz["fair_mechanism_gap_pp_mean"]
            d_o = k1["oracle_signal_pp"] - pub_oracle
            d_d = k1["deployable_signal_pp"] - pub_deploy
            d_f = k1["fair_gap_pp"] - pub_fair
            diffs_oracle.append(abs(d_o))
            diffs_deploy.append(abs(d_d))
            diffs_fair.append(abs(d_f))
            examples.append({
                "zone": zone,
                "oracle_k1": k1["oracle_signal_pp"], "oracle_pub": pub_oracle, "oracle_abs_diff": abs(d_o),
                "deploy_k1": k1["deployable_signal_pp"], "deploy_pub": pub_deploy, "deploy_abs_diff": abs(d_d),
                "fair_k1": k1["fair_gap_pp"], "fair_pub": pub_fair, "fair_abs_diff": abs(d_f),
            })
        max_diff = max(diffs_oracle + diffs_deploy + diffs_fair) if diffs_oracle else float("inf")
        sanity = {
            "checked": True,
            "matched": max_diff <= _SANITY_TOL_PP,
            "tolerance_pp": _SANITY_TOL_PP,
            "max_abs_diff_pp": max_diff,
            "max_abs_diff_oracle_pp": max(diffs_oracle) if diffs_oracle else None,
            "max_abs_diff_deployable_pp": max(diffs_deploy) if diffs_deploy else None,
            "max_abs_diff_fair_pp": max(diffs_fair) if diffs_fair else None,
            "n_zones_compared": len(examples),
            "per_zone": examples,
        }

    result = {
        "method": "parametric-shape-distortion-sensitivity-bound",
        "method_note": (
            "Not a measurement on real marginal-intensity data. m_k(h) = "
            "max(0, mu + k*(a(h)-mu)) amplifies the diurnal swing of each zone's "
            "AVERAGE-intensity series to emulate the higher volatility of MARGINAL "
            "intensity; the savings panel (experiments.exp_realtrace_pareto."
            "_savings_over_offsets) is reused unchanged. Reports an exposure BOUND, "
            "not a marginal-intensity result."
        ),
        "reused_from": "experiments.exp_realtrace_pareto._savings_over_offsets",
        "k_sweep": list(_K_SWEEP),
        "profile": args.profile,
        "full_cap_w": full,
        "eco_throttle_cap_w": throttle_cap,
        "job_hours": args.job_hours,
        "total_iters": total_iters,
        "threshold_quantile": args.threshold_quantile,
        "green_pause_fraction_target": green_pause_fraction,
        "green_window_hours": args.green_window_hours,
        "resume_energy_wh": args.resume_energy_wh,
        "dedicated_idle_w": args.dedicated_idle_w,
        "real_dir": args.real_dir,
        "n_real_zones": sum(1 for z in ztraces.values() if z.source == "real-csv"),
        "signal_definitions": {
            "oracle_signal_pp": "throttle(oracle) - throttle_blind(same budget); published cross-zone ~ +1.56",
            "deployable_signal_pp": "throttle_online - throttle_blind(same budget); published cross-zone ~ +0.75",
            "fair_gap_pp": "throttle(oracle) - GREEN_offline_reallocated_pause; published cross-zone ~ +0.57",
        },
        "k1_sanity_check_vs_published": sanity,
        "cross_zone_clustered_ci_by_k": {
            "note": "each list is [mean, lo, hi]; zone-clustered 95% bootstrap over real zones",
            "oracle_signal_pp": oracle_cz,
            "deployable_signal_pp": deploy_cz,
            "fair_gap_pp": fair_cz,
        },
        "envelope_cross_zone_mean": {
            "oracle_signal_pp": envelope(oracle_cz),
            "deployable_signal_pp": envelope(deploy_cz),
            "fair_gap_pp": envelope(fair_cz),
        },
        "per_zone_sign_flip_counts": {
            "note": "real zones whose signal sign at ANY k differs from its k=1 sign",
            "oracle_signal_pp": oracle_flips,
            "deployable_signal_pp": deploy_flips,
            "fair_gap_pp": fair_flips,
        },
        "zones": per_zone,
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", default="data_cache/real_traces")
    p.add_argument("--profile", default="artifacts/hardware-pareto-3080ti.json")
    p.add_argument("--published", default="artifacts/realtrace_pareto.json",
                   help="Published average-trace artifact, for the k=1 sanity check.")
    p.add_argument("--job-hours", type=float, default=24.0)
    p.add_argument("--threshold-quantile", type=float, default=0.6)
    p.add_argument("--throttle-cap-w", type=float, default=None)
    p.add_argument("--stride-hours", type=int, default=12)
    p.add_argument("--green-window-hours", type=int, default=24)
    p.add_argument("--resume-energy-wh", type=float, default=0.07)
    p.add_argument("--dedicated-idle-w", type=float, default=26.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/marginal_intensity_sensitivity.json")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

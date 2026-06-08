"""Paired average-vs-marginal grid-intensity contrast (Uplift 2 analysis).

Loads two carbon-panel outputs from ``exp_realtrace_pareto`` -- one replayed over
average (ElectricityMaps LCA) traces, one over marginal (WattTime MOER) traces of
the SAME zones and windows -- and reports, per signal, the per-zone divergence and
its zone-clustered bootstrap confidence interval. This is the measured replacement
for the parametric diurnal-amplitude proxy the paper currently uses to bound the
average-vs-marginal exposure.

Two signal-value quantities are contrasted (the ones the central limits rest on):
  * oracle carbon signal  = zones[z]["carbon_signal_value_pp_mean"]
  * deployable signal      = throttle_online_save_pct_mean
                             - throttle_blind_same_budget_save_pct_mean
Only zones that are REAL (source == "real-csv") in BOTH panels are compared, so a
synthetic fallback on either side never contaminates the contrast. The delta is
marginal - average; a delta CI that excludes zero means the marginal signal differs
significantly from the average one.

Usage::

    python -m experiments.compare_avg_vs_marginal \\
        --avg artifacts/realtrace_pareto.json \\
        --marginal artifacts/realtrace_pareto_marginal.json \\
        --out artifacts/avg_vs_marginal_contrast.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tare.stats.bootstrap import clustered_bootstrap_ci  # noqa: E402


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _real_zones(panel: dict) -> dict[str, dict]:
    return {z: d for z, d in panel["zones"].items() if d.get("source") == "real-csv"}


def _oracle(d: dict) -> float:
    return d["carbon_signal_value_pp_mean"]


def _deployable(d: dict) -> float:
    return d["throttle_online_save_pct_mean"] - d["throttle_blind_same_budget_save_pct_mean"]


def _clustered(values: list[float], seed: int) -> dict:
    point, lo, hi = clustered_bootstrap_ci([[v] for v in values], n_boot=20000,
                                           rng=random.Random(seed))
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0.0 or hi < 0.0)}


def _contrast(avg: dict, marg: dict, getter, seed: int) -> dict:
    common = sorted(set(avg) & set(marg))
    avg_vals = [getter(avg[z]) for z in common]
    marg_vals = [getter(marg[z]) for z in common]
    deltas = [m - a for a, m in zip(avg_vals, marg_vals, strict=True)]
    return {
        "n_zones": len(common),
        "average": _clustered(avg_vals, seed),
        "marginal": _clustered(marg_vals, seed + 100),
        "delta_marginal_minus_average": _clustered(deltas, seed + 200),
        "per_zone_delta": dict(zip(common, deltas, strict=True)),
    }


def run(args: argparse.Namespace) -> int:
    avg_panel = _real_zones(_load(args.avg))
    marg_panel = _real_zones(_load(args.marginal))
    common = sorted(set(avg_panel) & set(marg_panel))
    if not common:
        print("ERROR: no zone is real in BOTH panels; nothing to contrast.")
        return 2

    out = {
        "avg_source": args.avg, "marginal_source": args.marginal,
        "n_common_real_zones": len(common), "common_zones": common,
        "oracle_signal_pp": _contrast(avg_panel, marg_panel, _oracle, 0),
        "deployable_signal_pp": _contrast(avg_panel, marg_panel, _deployable, 1),
        "note": ("Delta = marginal - average. A delta CI excluding zero => the marginal "
                 "signal differs significantly from the average one. The energy and latency "
                 "verdicts are intensity-independent and unaffected."),
    }

    print("=" * 72)
    print(f"Average-vs-marginal contrast over {len(common)} zones real in both panels")
    print(f"  avg     : {args.avg}")
    print(f"  marginal: {args.marginal}")
    for key in ("oracle_signal_pp", "deployable_signal_pp"):
        c = out[key]
        a, m, d = c["average"], c["marginal"], c["delta_marginal_minus_average"]
        print(f"  {key}:")
        print(f"    average  {a['point']:+.2f} [{a['ci_lo']:.2f},{a['ci_hi']:.2f}] excl0={a['excludes_zero']}")
        print(f"    marginal {m['point']:+.2f} [{m['ci_lo']:.2f},{m['ci_hi']:.2f}] excl0={m['excludes_zero']}")
        print(f"    delta    {d['point']:+.2f} [{d['ci_lo']:.2f},{d['ci_hi']:.2f}] "
              f"-> {'DIFFERS (CI excludes 0)' if d['excludes_zero'] else 'no significant divergence'}")
    if args.out:
        from pathlib import Path
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"wrote {p}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--avg", required=True, help="panel JSON over average (LCA) traces")
    p.add_argument("--marginal", required=True, help="panel JSON over marginal (MOER) traces")
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

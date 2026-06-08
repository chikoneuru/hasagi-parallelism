"""Propagation-of-error and zone-clustered confidence intervals for the
break-even energy ratios and the carbon-signal decomposition.

Two re-analyses of already-stored aggregates, for reviewer-grade rigor:

1. Activation-recompute energy ratio (recompute-free FSDP over a
   recompute-forced DDP replica). The ratio is of two means from two
   *separate* runs, so the conservative interval previously reported took the
   independent extremes (num_lo/den_hi, num_hi/den_lo). That is wider than the
   sampling interval of the ratio. A first-order propagation-of-error (delta)
   interval is the defensible alternative when no per-seed pairing is stored:
   with independent numerator X and denominator Y,
       Var(R)/R^2 ~= (SE_X/X)^2 + (SE_Y/Y)^2 ,  R = X/Y .
   SE is recovered from each stored 95% interval as half-width / 1.96.

2. The deployable carbon signal (carbon-aware online throttle minus the
   carbon-blind same-budget throttle), differenced per zone, then a
   zone-clustered bootstrap interval over the 16 zones (one value per cluster).
   The oracle within-zone signal is re-verified the same way.

Also emits the throttle-vs-pause makespan ratio over the real traces.
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tare.stats.bootstrap import clustered_bootstrap_ci  # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _load(name):
    with open(os.path.join(ART, name)) as fh:
        return json.load(fh)


def _se_from_ci(lo, hi):
    return (hi - lo) / (2.0 * 1.96)


def ratio_propagation_ci(num, den):
    """First-order delta-method 95% interval for num/den, treating the two
    stored means as independent (separate runs, no shared seed structure)."""
    x, xlo, xhi = num["cluster_energy_per_iter_j"], num["cluster_energy_ci_lo"], num["cluster_energy_ci_hi"]
    y, ylo, yhi = den["cluster_energy_per_iter_j"], den["cluster_energy_ci_lo"], den["cluster_energy_ci_hi"]
    r = x / y
    se_x, se_y = _se_from_ci(xlo, xhi), _se_from_ci(ylo, yhi)
    rel_var = (se_x / x) ** 2 + (se_y / y) ** 2
    se_r = r * math.sqrt(rel_var)
    return {
        "point": r,
        "delta_ci_lo": r - 1.96 * se_r,
        "delta_ci_hi": r + 1.96 * se_r,
        "se": se_r,
        "independent_extremes_ci_lo": xlo / yhi,
        "independent_extremes_ci_hi": xhi / ylo,
        "num_rel_se": se_x / x,
        "den_rel_se": se_y / y,
    }


def zone_clustered_ci(values_by_zone, seed):
    point, lo, hi = clustered_bootstrap_ci(
        [[v] for v in values_by_zone], n_boot=20000, rng=random.Random(seed)
    )
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0.0 or hi < 0.0)}


def run(_args):
    out = {}

    nk = _load("recompute_nockpt.json")["layouts"]["fsdp"]
    ck = _load("recompute_ckpt.json")["layouts"]["ddp"]
    rc = ratio_propagation_ci(nk, ck)
    rc["break_even_estar"] = 0.75
    rc["straddles_break_even"] = rc["delta_ci_lo"] < 0.75 < rc["delta_ci_hi"]
    out["recompute_energy_ratio"] = rc

    par = _load("realtrace_pareto.json")["zones"]
    zones = list(par)
    online = [par[z]["throttle_online_save_pct_mean"] for z in zones]
    blind = [par[z]["throttle_blind_same_budget_save_pct_mean"] for z in zones]
    oracle = [par[z]["carbon_signal_value_pp_mean"] for z in zones]
    deployable_diff = [o - b for o, b in zip(online, blind)]
    out["deployable_signal_pp"] = zone_clustered_ci(deployable_diff, seed=0)
    out["oracle_signal_pp"] = zone_clustered_ci(oracle, seed=1)
    out["n_zones"] = len(zones)

    thr_mk = [par[z]["throttle_online_makespan_h_mean"] for z in zones]
    pause_mk = [par[z]["green_online_makespan_h_mean"] for z in zones]
    mt, mp = sum(thr_mk) / len(thr_mk), sum(pause_mk) / len(pause_mk)
    out["makespan"] = {
        "throttle_online_h_mean": mt,
        "online_pause_h_mean": mp,
        "ratio": mp / mt,
    }

    dest = os.path.join(ART, "energy_ratio_propagation_ci.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run(argparse.ArgumentParser().parse_args()))

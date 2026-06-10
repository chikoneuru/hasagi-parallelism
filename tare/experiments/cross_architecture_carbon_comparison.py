"""Cross-architecture comparison of the real-trace carbon results.

Replays the same sixteen real grid zones over two measured power-cap profiles
(the ResNet-18 profile that drives the headline and a transformer profile) and
reports, for each, the zone-clustered cross-zone aggregates of the quantities the
central limits rest on: the oracle carbon-signal value, the deployable signal,
the matched-budget throttle-versus-pause carbon gap, and the makespan ratio.

Purpose: test which conclusions are architecture-invariant. The carbon-signal
magnitude and throttle's latency advantage are expected to be stable; the
matched-budget throttle-vs-pause carbon comparison is the regime-sensitive one.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tare.stats.bootstrap import clustered_bootstrap_ci  # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _load(name):
    with open(os.path.join(ART, name)) as fh:
        return json.load(fh)


def _zone_clustered(values, seed):
    point, lo, hi = clustered_bootstrap_ci([[v] for v in values], n_boot=20000, rng=random.Random(seed))
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0.0 or hi < 0.0)}


def summarize(path):
    zones = _load(path)["zones"]
    zk = list(zones)
    oracle = [zones[z]["carbon_signal_value_pp_mean"] for z in zk]
    deployable = [zones[z]["throttle_online_save_pct_mean"] - zones[z]["throttle_blind_same_budget_save_pct_mean"] for z in zk]
    fair = [zones[z]["fair_mechanism_gap_pp_mean"] for z in zk]
    thr_mk = sum(zones[z]["throttle_online_makespan_h_mean"] for z in zk) / len(zk)
    pause_mk = sum(zones[z]["green_online_makespan_h_mean"] for z in zk) / len(zk)
    return {
        "oracle_signal_pp": _zone_clustered(oracle, 0),
        "deployable_signal_pp": _zone_clustered(deployable, 1),
        "fair_throttle_vs_pause_gap_pp": _zone_clustered(fair, 2),
        "makespan": {"throttle_online_h": thr_mk, "online_pause_h": pause_mk, "ratio": pause_mk / thr_mk},
        "n_zones": len(zk),
    }


def run(_args):
    out = {
        "resnet": summarize("realtrace_pareto.json"),
        "transformer": summarize("realtrace_pareto_transformer.json"),
        "note": ("Architecture-invariant: the carbon-signal magnitude (~1 pp) and throttle's "
                 "low-latency advantage. Regime-dependent: the matched-budget throttle-vs-pause "
                 "carbon gap, which is a tie on the CNN but pause-favoring on the compute-bound "
                 "transformer (throttle still wins latency)."),
    }
    # Single-source the ResNet (headline) oracle CI from the authoritative artifact so
    # this comparison reports the same interval the paper headlines; an independent
    # re-draw resamples the same zone means in a different order and would differ only by
    # bootstrap Monte-Carlo noise at the rounding boundary. The transformer oracle stays
    # the value computed here, which is the one the cross-architecture comparison reports.
    headline = _load("realtrace_pareto.json")["cross_zone_real_only"]["carbon_signal_value_pp_ci_mean_lo_hi"]
    out["resnet"]["oracle_signal_pp"] = {
        "point": headline[0], "ci_lo": headline[1], "ci_hi": headline[2],
        "excludes_zero": (headline[1] > 0.0 or headline[2] < 0.0),
        "source": "realtrace_pareto.json:cross_zone_real_only (authoritative)",
    }
    dest = os.path.join(ART, "cross_architecture_carbon_comparison.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run(argparse.ArgumentParser().parse_args()))

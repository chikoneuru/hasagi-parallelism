#!/usr/bin/env python
"""Certified real-reshard transition matrix on CPU process groups.

Runs a matrix of elastic-reconfiguration transitions (world-size changes and
DDP/FSDP/plain rewraps, resharded by torch.distributed.checkpoint) and, for
each transition, a clean pass plus the full fault suite emulating documented
reshard-bug classes. Reports per-transition detection results and the cost of
the certificate relative to the reshard it gates.

Decision rule, declared before each run prints:
  every clean transition must COMMIT (no false aborts), and
  every injected fault must ABORT (no silent corruption passes the gate).

Usage:
  python exp_attest_real_reshard.py                 # tiny preset, full matrix
  python exp_attest_real_reshard.py --preset small  # bigger model
  python exp_attest_real_reshard.py --transitions fsdp:2-plain:1 --faults none
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch.multiprocessing as mp

from attest.faults import FAULTS
from attest.harness import TransitionSpec, run_transition

# the checkpoint-layout changes a serverless elastic trainer actually makes
DEFAULT_TRANSITIONS = [
    ("shard", 2, "full", 1),    # scale-in: consolidate to a single worker
    ("full", 1, "shard", 2),    # scale-out from a single worker
    ("shard", 4, "shard", 2),   # world-size reshard, sharded on both sides
    ("shard", 2, "shard", 4),   # world-size reshard, growing
    ("full", 2, "shard", 2),    # rewrap: replicated -> sharded at equal world
]


def parse_transitions(arg: str):
    out = []
    for item in arg.split(","):
        pre, post = item.split("-")
        layout_pre, world_pre = pre.split(":")
        layout_post, world_post = post.split(":")
        out.append((layout_pre, int(world_pre), layout_post, int(world_post)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=["tiny", "small", "gpt2_125m"])
    ap.add_argument("--train-steps", type=int, default=2)
    ap.add_argument(
        "--transitions",
        default=None,
        help="comma list like shard:2-full:1,shard:4-shard:2 (default: built-in matrix)",
    )
    ap.add_argument(
        "--faults",
        default=None,
        help="comma list of fault names, 'none' for clean-only (default: all)",
    )
    ap.add_argument("--out", default=os.path.join("artifacts", "attest_real_reshard.json"))
    args = ap.parse_args()

    transitions = parse_transitions(args.transitions) if args.transitions else DEFAULT_TRANSITIONS
    if args.faults == "none":
        faults = []
    elif args.faults:
        faults = args.faults.split(",")
    else:
        faults = list(FAULTS)

    results = []
    n_clean_ok = n_caught = n_missed = n_false_abort = 0
    t_start = time.time()

    for layout_pre, world_pre, layout_post, world_post in transitions:
        base = dict(
            preset=args.preset,
            world_pre=world_pre,
            world_post=world_post,
            layout_pre=layout_pre,
            layout_post=layout_post,
            train_steps=args.train_steps,
        )

        spec = TransitionSpec(**base)
        r = run_transition(spec)
        ok = r.decision.committed
        n_clean_ok += int(ok)
        n_false_abort += int(not ok)
        print(f"[clean] {spec.describe():<46} -> {r.decision.summary()}")
        if not ok:
            for v in r.decision.violations[:4]:
                print(f"        {v}")
        results.append(
            {
                "transition": spec.describe(),
                "fault": None,
                "committed": r.decision.committed,
                "violations": [str(v) for v in r.decision.violations],
                "timings_s": r.timings,
            }
        )

        for fault in faults:
            spec = TransitionSpec(**base, fault=fault)
            r = run_transition(spec)
            caught = r.decision.aborted
            n_caught += int(caught)
            n_missed += int(not caught)
            via = ",".join(sorted({v.invariant for v in r.decision.violations})) or "-"
            print(f"[fault] {spec.describe():<46} -> caught={caught} via={via}")
            results.append(
                {
                    "transition": spec.describe(),
                    "fault": fault,
                    "committed": r.decision.committed,
                    "violations": [str(v) for v in r.decision.violations],
                    "timings_s": r.timings,
                }
            )

    n_faults_total = n_caught + n_missed
    detection = n_caught / n_faults_total if n_faults_total else 1.0
    cert_s = [r["timings_s"]["certificate_total_s"] for r in results]
    reshard_s = [r["timings_s"]["reshard_total_s"] for r in results]
    summary = {
        "preset": args.preset,
        "n_transitions": len(transitions),
        "n_clean_pass": n_clean_ok,
        "n_false_aborts": n_false_abort,
        "n_faults_injected": n_faults_total,
        "n_faults_caught": n_caught,
        "detection_rate": detection,
        "certificate_mean_s": sum(cert_s) / len(cert_s),
        "reshard_mean_s": sum(reshard_s) / len(reshard_s),
        "certificate_over_reshard": (sum(cert_s) / len(cert_s)) / max(sum(reshard_s) / len(reshard_s), 1e-9),
        "wall_s": time.time() - t_start,
        "gate": "PASS" if (n_false_abort == 0 and n_missed == 0) else "FAIL",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=2)

    print(
        f"\nclean {n_clean_ok}/{len(transitions)} committed, "
        f"faults {n_caught}/{n_faults_total} caught "
        f"({detection*100:.0f}%), false aborts {n_false_abort}"
    )
    print(
        f"certificate {summary['certificate_mean_s']*1e3:.0f} ms vs reshard "
        f"{summary['reshard_mean_s']*1e3:.0f} ms "
        f"({summary['certificate_over_reshard']*100:.0f}% of reshard cost)"
    )
    print(f"GATE={summary['gate']}  -> {args.out}")
    return 0 if summary["gate"] == "PASS" else 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

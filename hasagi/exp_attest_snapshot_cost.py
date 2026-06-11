"""Measure what the certificate itself costs as model bytes grow.

The snapshot is the expensive half of the certificate: every parameter and
optimizer tensor is brought to CPU, made contiguous, and sha256-hashed, plus
the independent checksum pass (count, fp64 L2, probe). The invariant
comparison afterwards touches only digests and is microseconds. This
experiment times both halves on the real model presets, on CPU tensors and
(when available) on CUDA tensors that must cross the device boundary first,
and reports throughput so the cost extrapolates linearly in state bytes to
sizes this machine cannot hold.

Run::

    python exp_attest_snapshot_cost.py --out artifacts/attest_snapshot_cost.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from attest.gate import certify_transition
from attest.model import GPT, PRESETS
from attest.snapshot import snapshot_from_state_dicts, snapshot_total_bytes

REPEATS = 3


def _fqn_optim_state(model: torch.nn.Module, optim: torch.optim.Optimizer) -> dict:
    fqns = [n for n, _ in model.named_parameters()]
    raw = optim.state_dict()["state"]
    return {"state": {fqns[i]: raw[i] for i in raw}}


def _measure(preset: str, device: str) -> dict:
    cfg = PRESETS[preset]
    model = GPT(cfg, seed=0).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq), device=device)
    for _ in range(2):  # populate Adam moments with real steps
        loss = model(x).float().pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    model_sd = model.state_dict()
    optim_sd = _fqn_optim_state(model, optim)
    param_bytes = snapshot_total_bytes(model_sd)
    state_bytes = param_bytes + sum(
        v.element_size() * v.numel()
        for slots in optim_sd["state"].values()
        for v in slots.values() if isinstance(v, torch.Tensor)
    )

    snap_times = []
    for _ in range(REPEATS):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        snap = snapshot_from_state_dicts(model_sd, optim_sd,
                                         progress={"global_step": 2})
        snap_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    decision = certify_transition(snap, snap)
    check_s = time.perf_counter() - t0
    assert decision.committed

    best = min(snap_times)
    return {
        "preset": preset,
        "device": device,
        "n_params": sum(p.numel() for p in model.parameters()),
        "state_bytes": state_bytes,
        "snapshot_seconds": best,
        "snapshot_seconds_all": snap_times,
        "snapshot_gb_per_s": state_bytes / best / 1e9,
        "check_seconds": check_s,
    }


def run(args: argparse.Namespace) -> int:
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = [_measure(p, d) for d in devices for p in args.presets]

    # linear-in-bytes projection from the largest measured row per device
    projections = {}
    for d in devices:
        biggest = max((r for r in rows if r["device"] == d), key=lambda r: r["state_bytes"])
        rate = biggest["state_bytes"] / biggest["snapshot_seconds"]
        projections[d] = {
            "basis_preset": biggest["preset"],
            "gb_per_s": rate / 1e9,
            "projected_seconds_7b_params_adamw_fp32": 7e9 * 12 / rate,
        }

    report = {
        "exp": "attest-snapshot-cost",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repeats": REPEATS,
        "note": "state_bytes covers parameters plus Adam moments; the 7B "
                "projection assumes fp32 params + two fp32 moments (12 bytes "
                "per parameter) and linear scaling in bytes",
        "rows": rows,
        "projections": projections,
    }
    print(f"{'preset':12s} {'device':6s} {'params':>12s} {'state MB':>9s} "
          f"{'snapshot s':>10s} {'GB/s':>6s} {'check s':>9s}")
    for r in rows:
        print(f"{r['preset']:12s} {r['device']:6s} {r['n_params']:12,d} "
              f"{r['state_bytes'] / 1e6:9.1f} {r['snapshot_seconds']:10.3f} "
              f"{r['snapshot_gb_per_s']:6.2f} {r['check_seconds']:9.6f}")
    for d, pr in projections.items():
        print(f"projection[{d}]: {pr['gb_per_s']:.2f} GB/s -> 7B+AdamW fp32 in "
              f"{pr['projected_seconds_7b_params_adamw_fp32']:.0f}s")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--presets", nargs="+", default=["tiny", "small", "gpt2_125m"],
                   choices=list(PRESETS))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

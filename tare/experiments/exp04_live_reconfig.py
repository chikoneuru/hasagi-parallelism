"""exp04 — live mid-job layout reconfiguration: DDP -> reshard -> FSDP (scaffold).

Trains a model for one phase, performs a real state-transporting reshard with
``tare.state.reshard.ReshardController`` (data-parallel full replica -> fully
sharded), then continues training for a second phase, recording per-iteration
loss, the reconfiguration latency, and (on GPU) the reconfiguration energy. A
no-reshard control run with the same seed and data stream gives the trajectory
the reshard run must track: a correct reshard is loss-transparent, so the gate is

    max_i | loss_reshard[i] - loss_control[i] | <= --tol-ce  (default 0.3)

across the matched steps. The reshard is verify-before-commit: if the reassembled
state does not match the pre-reshard state the controller aborts and the model is
left untouched, which the harness reports as a failed reconfiguration.

The training loop and the reshard transport run single-process here (CPU or one
GPU), which is exactly what the CPU test exercises (see
``tests/test_reshard_controller.py``). The real multi-GPU run wraps the phases in
DDP then FSDP and meters per-phase energy with NVML; that wrapping is gated behind
torchrun so this file stays runnable on CPU as a logic check.

Real run (2-GPU node)::

    torchrun --nproc_per_node=2 -m experiments.exp04_live_reconfig \\
        --d-model 1024 --layers 8 --phase-iters 50 --to-world 2 \\
        --out artifacts/exp04_live_reconfig.json

Logic smoke-test (CPU)::

    python -m experiments.exp04_live_reconfig --smoke \\
        --d-model 64 --layers 2 --phase-iters 20 --to-world 2
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn

from tare.state.reshard import ReshardController

try:
    import pynvml
    _NVML = True
except Exception:  # pragma: no cover
    _NVML = False


def _build(d_model: int, layers: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    model = nn.Sequential(*[nn.Linear(d_model, d_model) for _ in range(layers)],
                          nn.Linear(d_model, 1)).to(device)
    return model


def _fixed_batches(n: int, batch: int, d_model: int, seed: int, device: torch.device):
    """A fixed sequence of (x, y) batches so reshard and control see identical data."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(batch, d_model, generator=g).to(device),
             torch.randn(batch, 1, generator=g).to(device)) for _ in range(n)]


def _train_segment(model, opt, batches, losses: list[float]) -> None:
    loss_fn = nn.MSELoss()
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))


def _nvml_energy_j(local: int):
    if not _NVML or not torch.cuda.is_available():
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(local)
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
    except Exception:
        return None


def run(args: argparse.Namespace) -> int:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    n = args.phase_iters
    seed = args.seed

    # ---- reshard run: phase 1 -> reshard -> phase 2 -----------------------
    model = _build(args.d_model, args.layers, seed, device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    batches = _fixed_batches(2 * n, args.batch, args.d_model, seed + 1, device)
    losses_reshard: list[float] = []
    _train_segment(model, opt, batches[:n], losses_reshard)

    rc = ReshardController(atol=args.atol)
    rc.capture(model, from_world=1)
    e0 = _nvml_energy_j(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    cert = rc.reshard_and_commit(model, to_world=args.to_world)
    if device.type == "cuda":
        torch.cuda.synchronize()
    reconfig_latency_s = time.perf_counter() - t0
    e1 = _nvml_energy_j(0)
    reconfig_energy_j = (e1 - e0) if (e0 is not None and e1 is not None) else None

    _train_segment(model, opt, batches[n:], losses_reshard)

    # ---- control run: same seed + data, no reshard ------------------------
    model_c = _build(args.d_model, args.layers, seed, device)
    opt_c = torch.optim.SGD(model_c.parameters(), lr=args.lr)
    losses_control: list[float] = []
    _train_segment(model_c, opt_c, batches, losses_control)

    max_dev = max(abs(a - b) for a, b in zip(losses_reshard, losses_control, strict=True))
    continuity_ok = max_dev <= args.tol_ce

    report = {
        "exp": "exp04-live-reconfig", "device": device.type,
        "d_model": args.d_model, "layers": args.layers, "phase_iters": n,
        "to_world": args.to_world,
        "reshard_certificate": {
            "ok": cert.ok, "max_abs_diff": cert.max_abs_diff, "n_params": cert.n_params,
            "from_world": cert.from_world, "to_world": cert.to_world, "note": cert.note,
        },
        "reconfig_latency_s": reconfig_latency_s,
        "reconfig_energy_j": reconfig_energy_j,
        "loss_continuity": {
            "max_abs_deviation_vs_control": max_dev, "tol_ce": args.tol_ce, "ok": continuity_ok,
        },
        "loss_at_reshard_boundary": {
            "last_phase1": losses_reshard[n - 1], "first_phase2": losses_reshard[n],
        },
        "smoke": bool(args.smoke),
    }

    print("=" * 64)
    print(f"exp04 live reconfig (device={device.type}, layers={args.layers}, "
          f"phase_iters={n}, to_world={args.to_world})")
    if args.smoke:
        print("  [SMOKE TEST — single-process logic; real DDP/FSDP + energy need torchrun + GPUs]")
    print(f"  reshard certificate: ok={cert.ok} max_abs_diff={cert.max_abs_diff:.2e} "
          f"({cert.n_params} params) {cert.note}")
    print(f"  reconfig latency: {reconfig_latency_s * 1000:.2f} ms; "
          f"energy: {reconfig_energy_j if reconfig_energy_j is not None else 'n/a (CPU)'}")
    print(f"  loss continuity vs control: max dev {max_dev:.3e} "
          f"(tol {args.tol_ce}) -> {'OK' if continuity_ok else 'FAIL'}")
    if args.out:
        from pathlib import Path
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
        print(f"wrote {p}")
    return 0 if (cert.ok and continuity_ok) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--phase-iters", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--to-world", type=int, default=2, help="target shard count for the FSDP layout")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-6, help="reshard verification tolerance")
    p.add_argument("--tol-ce", type=float, default=0.3, help="max allowed loss deviation vs control")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

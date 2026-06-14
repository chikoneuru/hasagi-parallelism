"""Closed-loop demo: a carbon-aware controller actuates real reshards over a trace.

The characterization study evaluates each lever as a disjoint, manually-triggered
arm; the planner-to-actuation loop is never run end to end. This harness closes
that loop on the in-hand hardware: it replays a real grid-intensity trace, and on
each simulated hour a threshold policy decides the layout the controller should
run; when the decision differs from the current layout the controller fires a
real, verify-before-commit reshard (no manual trigger), training continues, and
the reconfiguration is metered. It demonstrates that the planner decision
actuates the already-measured reshard mechanism end to end and stays
loss-transparent; it does not change the paper's verdict on whether carbon
*should* drive layout (it should not), only that the control loop closes.

Two backends share one policy/trace/trigger loop:

  * ``--mode dry`` (CPU, no GPU): the actuation is the ``ReshardController``
    flat-shard transport (capture -> plan -> reassemble -> verify) between a
    simulated 1<->2 world size. Exercises the loop logic and the state-transport
    correctness with no FSDP/CUDA, so it runs in CI and as a pre-rental dry run.
  * ``--mode live`` (CUDA, >=2 GPUs under torchrun): the actuation is
    ``tare.state.reshard.live_rewrap`` (a real DDP<->FSDP flip, optimizer state
    carried, NVML energy/latency metered), the same path ``exp04_live_reconfig``
    measures, now fired by the policy rather than at a fixed phase boundary.

Usage::

    # CPU dry run (logic + transport correctness; not a measurement):
    python -m experiments.exp_closed_loop_demo --mode dry \
        --trace data_cache/real_traces/de_2024-07-01_2024-07-15_hourly.csv \
        --ticks 24 --iters-per-tick 5 --out artifacts/closed_loop_demo_dry.json

    # Live metered run on the rented node:
    torchrun --nproc_per_node=4 -m experiments.exp_closed_loop_demo --mode live \
        --trace data_cache/real_traces/de_2024-07-01_2024-07-15_hourly.csv \
        --ticks 48 --iters-per-tick 20 --out artifacts/closed_loop_demo.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from tare.energy.trace_schedule import load_electricitymaps_csv, quantile_threshold, trace_to_hourly
from tare.state.reshard import ReshardController

try:
    import pynvml
    _NVML = True
except Exception:  # pragma: no cover - depends on the node
    _NVML = False


def _build_module(d_model: int, layers: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        return nn.Sequential(*[nn.Linear(d_model, d_model) for _ in range(layers)],
                             nn.Linear(d_model, 1))


def _nvml_handle(local: int):
    if not _NVML:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(local)
    except Exception:
        return None


def _energy_j(handle) -> float | None:
    if handle is None:
        return None
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
    except Exception:
        return None


def carbon_aware_layout(intensity: float, threshold: float) -> str:
    """Threshold policy: run the lower-power sharded layout in dirty windows, the
    faster replicated layout in clean ones. (The reshard mechanism is what we
    exercise; the paper's verdict is that carbon should not actually drive this.)"""
    return "fsdp" if intensity > threshold else "ddp"


def _train_steps(model, opt, n: int, d_model: int, device: torch.device) -> float:
    last = 0.0
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, d_model, device=device)
        loss = model(x).float().pow(2).mean()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def run(args: argparse.Namespace) -> int:
    hourly = trace_to_hourly(load_electricitymaps_csv(Path(args.trace)))
    threshold = quantile_threshold(hourly, args.threshold_quantile)
    ticks = min(args.ticks, len(hourly))

    live = args.mode == "live"
    if live and not torch.cuda.is_available():
        print("--mode live requires CUDA; use --mode dry on CPU")
        return 2

    rank, world, local = 0, 1, 0
    device = torch.device("cpu")
    if live:
        import torch.distributed as dist

        from tare.state.reshard import live_rewrap, wrap_layout
        if "RANK" in os.environ:
            rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    handle = _nvml_handle(local) if live else None

    torch.manual_seed(args.seed)
    d, layers = args.d_model, args.layers
    base = _build_module(d, layers).to(device)
    make_opt = lambda p: torch.optim.SGD(p, lr=1e-3, momentum=0.9)  # noqa: E731

    layout = "ddp"
    if live:
        model = wrap_layout(base, layout, device)
    else:
        model = base
    opt = make_opt(model.parameters())

    rc = ReshardController(atol=args.atol) if not live else None
    sim_world = 1  # dry-mode simulated world size toggled by the policy
    ticklog, reshards = [], []

    for t in range(ticks):
        intensity = hourly[t]
        dirty = intensity > threshold
        loss = _train_steps(model, opt, args.iters_per_tick, d, device)
        target = carbon_aware_layout(intensity, threshold)

        fired = None
        if target != layout:
            if live:
                torch.cuda.synchronize()
                e0 = _energy_j(handle)
                t0 = time.perf_counter()
                model, opt, cert = live_rewrap(
                    model, opt, layout_from=layout, layout_to=target,
                    to_world=world, device=device, optim_factory=make_opt,
                    module_factory=lambda: _build_module(d, layers),
                    atol=args.atol, from_world=world,
                )
                torch.cuda.synchronize()
                lat = time.perf_counter() - t0
                e1 = _energy_j(handle)
                fired = {"from": layout, "to": target, "latency_s": lat,
                         "energy_j": (e1 - e0) if (e0 is not None and e1 is not None) else None,
                         "verified": bool(cert.ok), "max_abs_diff": cert.max_abs_diff}
            else:
                # dry: exercise the verify-before-commit transport over a simulated
                # world-size change, the CPU-safe core of the reshard the live
                # path actuates on GPUs.
                to_world = 2 if sim_world == 1 else 1
                rc.capture(model, from_world=sim_world)
                t0 = time.perf_counter()
                cert = rc.reshard_and_commit(model, to_world)
                lat = time.perf_counter() - t0
                sim_world = to_world
                fired = {"from": layout, "to": target, "latency_s": lat,
                         "energy_j": None, "verified": bool(cert.ok),
                         "max_abs_diff": cert.max_abs_diff,
                         "sim_world": f"{cert.from_world}->{cert.to_world}"}
            if not fired["verified"]:
                raise RuntimeError(f"reshard verification failed at tick {t}: {fired}")
            reshards.append(fired)
            layout = target

        ticklog.append({"tick": t, "intensity": intensity, "dirty": dirty,
                        "layout": layout, "loss": loss, "reshard": fired is not None})

    if rank == 0:
        report = {
            "demo": "closed-loop-controller", "mode": args.mode, "world_size": world,
            "trace": os.path.basename(args.trace), "ticks": ticks,
            "threshold_quantile": args.threshold_quantile, "threshold_g_per_kwh": threshold,
            "iters_per_tick": args.iters_per_tick,
            "n_reshards_fired_by_policy": len(reshards),
            "all_reshards_verified": all(r["verified"] for r in reshards),
            "reshard_latency_s_mean": (sum(r["latency_s"] for r in reshards) / len(reshards))
            if reshards else None,
            "reshard_energy_j": [r["energy_j"] for r in reshards],
            "reshards": reshards, "ticklog": ticklog,
        }
        print(f"closed-loop demo ({args.mode}, world={world}): {ticks} ticks, "
              f"{len(reshards)} reshards fired by policy, "
              f"all verified={report['all_reshards_verified']}")
        if args.out:
            p = Path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=2))
            print(f"wrote {p}")
    if live:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["dry", "live"], default="dry")
    p.add_argument("--trace", required=True, help="ElectricityMaps zone CSV")
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--iters-per-tick", type=int, default=5)
    p.add_argument("--threshold-quantile", type=float, default=0.6)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

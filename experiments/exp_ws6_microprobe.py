"""WS6 micro-probe (real multi-GPU): is the structural energy lever alive?

The break-even kill-switch. Measures the UN-CAPPED energy-per-iteration of distinct
hybrid-parallel layouts of the same model on real GPUs and reports the ratio
``E_eco = e_eco / e_fast``. The pre-registered decision (declare BEFORE running):

  * E_eco <= 0.75  -> the structural lever clears free DVFS-throttle's ~15% by the
                     margin the break-even needs: the layout lever is ALIVE -> WS6 is worth building.
  * 0.75 < E_eco <= 0.85 -> marginal: lever does not clear the throttle margin on its own.
  * E_eco > 0.85   -> the layout lever is DEAD at this workload: STOP, ship the
                     characterization, do not build the distributed/reshard stack.

CRITICAL: this measures energy at the DEFAULT power cap (no nvidia-smi -pl). A saving
that appears only after capping the eco ranks is DVFS-in-disguise and does not count;
we want a saving from genuine layout/arithmetic-intensity differences. Energy is read
from NVML (per-GPU total-energy counter, Zeus-style; falls back to integrated power).

Run on the rented node (2 GPUs)::

    torchrun --nproc_per_node=2 -m experiments.exp_ws6_microprobe \
        --steps 200 --warmup 30 --d-model 2048 --layers 24 --out artifacts/ws6_microprobe.json

Logic smoke-test (1 process, tiny, CPU or 1 GPU, NOT a real measurement)::

    python -m experiments.exp_ws6_microprobe --steps 5 --warmup 1 --d-model 256 --layers 2

Layouts probed (same param budget, different power profile): ``ddp`` (no shard,
compute-bound, high power = the FAST layout), ``fsdp`` (full shard, more comm),
``fsdp_offload`` (full shard + CPU offload, comm/transfer-bound, low power = an ECO
candidate). E_eco is reported for each non-fast layout vs ddp.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    import pynvml
    _NVML = True
except Exception:  # pragma: no cover - depends on the node
    _NVML = False


def _is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _setup() -> tuple[int, int, int, torch.device]:
    if _is_distributed():
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    else:
        rank, world, local = 0, 1, 0
        if torch.cuda.is_available():
            dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29555",
                                    rank=0, world_size=1)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    else:
        device = torch.device("cpu")
    return rank, world, local, device


def _build_model(d_model: int, layers: int) -> nn.Module:
    """A matmul-heavy transformer encoder stack (realistic GPU power draw)."""
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=max(1, d_model // 64),
                                       dim_feedforward=4 * d_model, batch_first=True)
    return nn.Sequential(nn.TransformerEncoder(layer, num_layers=layers),
                         nn.Linear(d_model, d_model))


def _wrap(model: nn.Module, layout: str, device: torch.device, local: int):
    """Apply the layout. Falls back to plain module when not distributed (smoke)."""
    if not _is_distributed() or dist.get_world_size() == 1:
        return model.to(device)
    from torch.distributed.fsdp import CPUOffload, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
    if layout == "ddp":
        return FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD, device_id=local)
    if layout == "fsdp":
        return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=local)
    if layout == "fsdp_offload":
        return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                    cpu_offload=CPUOffload(offload_params=True), device_id=local)
    raise ValueError(f"unknown layout {layout!r}")


def _nvml_handle(local: int):
    if not _NVML:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(local)
    except Exception:
        return None


def _energy_j(handle) -> float | None:
    """Cumulative device energy in joules via the NVML total-energy counter, or None."""
    if handle is None:
        return None
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0  # mJ -> J
    except Exception:
        return None


def _power_w(handle) -> float:
    if handle is None:
        return 0.0
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
    except Exception:
        return 0.0


def _measure(model: nn.Module, device: torch.device, handle, *, steps: int, warmup: int,
             batch: int, seq: int, d_model: int) -> dict:
    """Run warmup+steps; return per-iter energy (J) and mean power (W) for this rank's GPU."""
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    x = torch.randn(batch, seq, d_model, device=device)
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        out.float().pow(2).mean().backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    e0 = _energy_j(handle)
    p_samples: list[float] = []
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        out.float().pow(2).mean().backward()
        opt.step()
        p_samples.append(_power_w(handle))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    e1 = _energy_j(handle)
    mean_p = sum(p_samples) / max(len(p_samples), 1)
    energy_j = (e1 - e0) if (e0 is not None and e1 is not None) else mean_p * elapsed
    return {"energy_per_iter_j": energy_j / steps, "mean_power_w": mean_p,
            "iters_per_s": steps / elapsed, "elapsed_s": elapsed,
            "energy_source": "nvml-total-energy" if (e0 is not None and e1 is not None) else "nvml-power-integral"}


def _allreduce_sum(value: float) -> float:
    if _is_distributed() and dist.get_world_size() > 1:
        t = torch.tensor([value], dtype=torch.float64,
                         device=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
                         if torch.cuda.is_available() else torch.device("cpu"))
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())
    return value


def run(args: argparse.Namespace) -> int:
    rank, world, local, device = _setup()
    handle = _nvml_handle(local)
    layouts = [s.strip() for s in args.layouts.split(",") if s.strip()]
    results: dict[str, dict] = {}
    for layout in layouts:
        model = _wrap(_build_model(args.d_model, args.layers), layout, device, local)
        m = _measure(model, device, handle, steps=args.steps, warmup=args.warmup,
                     batch=args.batch, seq=args.seq, d_model=args.d_model)
        # cluster energy = sum across all ranks' GPUs (the job's real draw)
        cluster_e = _allreduce_sum(m["energy_per_iter_j"])
        cluster_p = _allreduce_sum(m["mean_power_w"])
        results[layout] = {**m, "cluster_energy_per_iter_j": cluster_e, "cluster_mean_power_w": cluster_p}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if _is_distributed() and dist.get_world_size() > 1:
            dist.barrier()

    if rank == 0:
        fast = results.get("ddp")
        report = {"world_size": world, "d_model": args.d_model, "layers": args.layers,
                  "batch": args.batch, "seq": args.seq, "steps": args.steps,
                  "energy_source": next(iter(results.values()))["energy_source"],
                  "layouts": results, "E_eco": {}, "verdict": {}}
        if fast:
            fe = fast["cluster_energy_per_iter_j"]
            for layout, r in results.items():
                if layout == "ddp" or fe <= 0:
                    continue
                ratio = r["cluster_energy_per_iter_j"] / fe
                report["E_eco"][layout] = ratio
                report["verdict"][layout] = (
                    "ALIVE (<=0.75)" if ratio <= 0.75 else
                    "MARGINAL (0.75-0.85)" if ratio <= 0.85 else "DEAD (>0.85)")
        print("=" * 60)
        print(f"WS6 micro-probe  (world_size={world}, source={report['energy_source']})")
        for layout, r in results.items():
            print(f"  {layout:14s} cluster {r['cluster_energy_per_iter_j']:.3f} J/iter "
                  f"@ {r['cluster_mean_power_w']:.0f} W, {r['iters_per_s']:.2f} it/s")
        for layout, ratio in report["E_eco"].items():
            print(f"  E_eco[{layout}] = {ratio:.3f}  -> {report['verdict'][layout]}")
        if not report["E_eco"]:
            print("  (no 'ddp' fast baseline measured or single-process smoke: ratios need >=2 GPUs)")
        if args.out:
            from pathlib import Path
            p = Path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=2))
            print(f"wrote {p}")
    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layouts", default="ddp,fsdp,fsdp_offload")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

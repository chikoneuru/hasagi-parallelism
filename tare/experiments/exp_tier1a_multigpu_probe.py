"""Multi-GPU structural-energy probe: does the eco-layout lever come alive under
memory pressure?

This extends the two-GPU micro-probe (``exp_ws6_microprobe.py``) into the regime
the characterization paper identifies as the only place the structural lever
could flip: a model large enough that a full per-device replica is near or over
device capacity, and/or a stateful optimizer (Adam) whose state no longer fits
per device, so that sharding supplies a genuine memory relief rather than pure
communication overhead. It closes the three scope gaps of the two-GPU probe:

  1. TRUE data-parallel baseline. The fast layout is real
     ``torch.nn.parallel.DistributedDataParallel``, NOT ``FSDP(NO_SHARD)``, so the
     E_eco ratio is taken against the standard reference, not a sharding-internal
     proxy.
  2. Stateful optimizer. ``--optimizer adam`` makes the optimizer state ~2x the
     parameter bytes, the canonical reason FSDP/ZeRO is built; ``sgd`` reproduces
     the near-stateless two-GPU configuration for comparison.
  3. Memory-pressure instrumentation. Per layout we record peak allocated memory
     and the fraction of device capacity it uses, so the run can DEMONSTRATE that
     the fast layout is memory-pressured (the precondition under which sharding is
     a relief, not overhead). If the true-DDP baseline OOMs, that is itself the
     finding: sharding is mandatory, not optional, and the lever is alive by
     feasibility.

Pre-registered decision on E_eco = e_eco / e_ddp (declare BEFORE running),
mirroring the break-even surface (e* ~ 0.75):

  * E_eco <= 0.75        -> the eco layout clears the free-throttle margin: the
                            structural lever is ALIVE at scale -> revisit the kill-gate.
  * 0.75 < E_eco <= 0.85 -> MARGINAL: does not clear the throttle margin alone.
  * E_eco > 0.85         -> DEAD at this workload: the two-GPU verdict holds at scale.
  * true-DDP OOM         -> FEASIBILITY regime: sharding mandatory; E_eco vs DDP is
                            undefined, report the smallest-feasible layout instead.

Energy is the per-GPU NVML total-energy counter (Zeus-style), summed across ranks
to the job's real cluster draw, measured at the DEFAULT power cap (no nvidia-smi
-pl): a saving that appears only after capping an eco rank is DVFS-in-disguise and
does not count. Each layout is measured over ``--seeds`` independent repeats so a
mean and a confidence interval are reported, not a single point.

Real run (>=4 GPUs on the rented node)::

    torchrun --nproc_per_node=4 -m experiments.exp_tier1a_multigpu_probe \
        --optimizer adam --d-model 2048 --layers 24 --batch 8 --seq 1024 \
        --steps 200 --warmup 40 --seeds 5 \
        --out artifacts/tier1a_multigpu_probe.json

Pick ``--d-model/--layers/--batch`` so the reported ``capacity_frac`` for the
``ddp`` layout is high (>= ~0.7); raise them until DDP is near capacity (or OOMs)
to put the run in the memory-pressured regime the verdict is gated on.

Logic smoke-test (1 process, tiny, CPU or 1 GPU, NOT a real measurement)::

    python -m experiments.exp_tier1a_multigpu_probe --steps 4 --warmup 1 \
        --d-model 128 --layers 2 --seeds 2 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
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
            dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29556",
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


def _apply_act_ckpt(model: nn.Module) -> None:
    """Wrap each transformer layer in non-reentrant activation checkpointing.

    Recompute trades ~33% extra forward FLOPs for a large cut in saved-activation
    memory; we use it to model the regime where the fast layout must recompute to
    fit while a sharded layout, having freed parameter/optimizer-state memory, need
    not. Applied to the raw module before any DDP/FSDP wrapping.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
            m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, nn.TransformerEncoderLayer),
    )


def _wrap(model: nn.Module, layout: str, device: torch.device, local: int,
          act_ckpt: bool = False):
    """Apply the layout. Falls back to plain module when not distributed (smoke)."""
    if act_ckpt:
        _apply_act_ckpt(model)
    if not _is_distributed() or dist.get_world_size() == 1:
        return model.to(device)
    from torch.distributed.fsdp import CPUOffload, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
    if layout == "ddp":
        # TRUE DistributedDataParallel: a full replica + optimizer state per device.
        from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
        return DDP(model.to(device), device_ids=[local] if device.type == "cuda" else None)
    if layout == "fsdp":
        return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=local)
    if layout == "fsdp_offload":
        return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                    cpu_offload=CPUOffload(offload_params=True), device_id=local)
    if layout == "hsdp":
        # Hybrid shard (shard within a group, replicate across). Recent FSDP requires
        # an explicit 2D device mesh; opt-in only, not in the default layout set.
        from torch.distributed.device_mesh import init_device_mesh
        w = dist.get_world_size()
        shard = 2 if w % 2 == 0 else w
        mesh = init_device_mesh("cuda", (w // shard, shard))
        return FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    device_mesh=mesh, device_id=local)
    raise ValueError(f"unknown layout {layout!r}")


def _make_optimizer(name: str, params):
    if name == "adam":
        return torch.optim.AdamW(params, lr=1e-4)
    if name == "sgd":
        return torch.optim.SGD(params, lr=1e-4)
    raise ValueError(f"unknown optimizer {name!r}")


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


def _device_capacity_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / 1e9


def _measure_once(model: nn.Module, device: torch.device, handle, *, steps: int, warmup: int,
                  batch: int, seq: int, d_model: int, optimizer: str) -> dict:
    """One repeat: warmup+steps; return per-iter energy (J), power (W), peak mem (GB)."""
    opt = _make_optimizer(optimizer, model.parameters())
    x = torch.randn(batch, seq, d_model, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0
    return {"energy_per_iter_j": energy_j / steps, "mean_power_w": mean_p,
            "iters_per_s": steps / elapsed, "peak_mem_gb": peak_gb,
            "energy_source": "nvml-total-energy" if (e0 is not None and e1 is not None)
            else "nvml-power-integral"}


def _ci95(xs: list[float]) -> tuple[float, float, float]:
    """mean and a small-sample 95% CI (t-interval) across repeats."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, m, m
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    tcrit = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 2.0)
    half = tcrit * sd / math.sqrt(n)
    return m, m - half, m + half


def _allreduce_sum(value: float, device: torch.device) -> float:
    if _is_distributed() and dist.get_world_size() > 1:
        t = torch.tensor([value], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())
    return value


def _allreduce_max(value: float, device: torch.device) -> float:
    if _is_distributed() and dist.get_world_size() > 1:
        t = torch.tensor([value], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())
    return value


def run(args: argparse.Namespace) -> int:
    rank, world, local, device = _setup()
    handle = _nvml_handle(local)
    capacity_gb = _device_capacity_gb(device)
    layouts = [s.strip() for s in args.layouts.split(",") if s.strip()]
    results: dict[str, dict] = {}

    for layout in layouts:
        per_seed: list[dict] = []
        oom = False
        for seed in range(args.seeds):
            torch.manual_seed(1000 + seed)
            try:
                model = _wrap(_build_model(args.d_model, args.layers), layout, device,
                              local, act_ckpt=(args.act_ckpt == "all"))
                m = _measure_once(model, device, handle, steps=args.steps, warmup=args.warmup,
                                  batch=args.batch, seq=args.seq, d_model=args.d_model,
                                  optimizer=args.optimizer)
                per_seed.append(m)
            except torch.cuda.OutOfMemoryError:  # type: ignore[attr-defined]
                oom = True
                break
            finally:
                model = None  # noqa: F841
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if _is_distributed() and dist.get_world_size() > 1:
                dist.barrier()

        # has ANY rank OOMed? if so this layout is infeasible at this scale.
        oom_any = _allreduce_max(1.0 if oom else 0.0, device) > 0.5
        if oom_any or not per_seed:
            results[layout] = {"feasible": False, "note": "OOM at this scale (memory-pressured)"}
            continue

        e_seeds = [s["energy_per_iter_j"] for s in per_seed]
        # per-seed cluster energy = sum across ranks; then mean/CI across seeds
        e_cluster_seeds = [_allreduce_sum(e, device) for e in e_seeds]
        p_cluster = _allreduce_sum(sum(s["mean_power_w"] for s in per_seed) / len(per_seed), device)
        peak_gb_max = _allreduce_max(max(s["peak_mem_gb"] for s in per_seed), device)
        em, elo, ehi = _ci95(e_cluster_seeds)
        results[layout] = {
            "feasible": True, "n_seeds": len(per_seed),
            "cluster_energy_per_iter_j": em, "cluster_energy_ci_lo": elo, "cluster_energy_ci_hi": ehi,
            "cluster_mean_power_w": p_cluster,
            "iters_per_s": sum(s["iters_per_s"] for s in per_seed) / len(per_seed),
            "peak_mem_gb_per_device": peak_gb_max,
            "capacity_frac": (peak_gb_max / capacity_gb) if capacity_gb else None,
            "energy_source": per_seed[0]["energy_source"],
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if _is_distributed() and dist.get_world_size() > 1:
            dist.barrier()

    if rank == 0:
        fast = results.get("ddp")
        ddp_feasible = bool(fast and fast.get("feasible"))
        report = {
            "probe": "tier1a-multigpu", "world_size": world, "optimizer": args.optimizer,
            "act_ckpt": args.act_ckpt,
            "d_model": args.d_model, "layers": args.layers, "batch": args.batch, "seq": args.seq,
            "steps": args.steps, "warmup": args.warmup, "seeds": args.seeds,
            "device_capacity_gb": capacity_gb,
            "break_even_estar": 0.75,
            "ddp_baseline": "true torch.nn.parallel.DistributedDataParallel",
            "ddp_feasible": ddp_feasible,
            "layouts": results, "E_eco": {}, "verdict": {},
            "smoke": bool(args.smoke),
        }
        if ddp_feasible:
            fe = fast["cluster_energy_per_iter_j"]
            for layout, r in results.items():
                if layout == "ddp" or not r.get("feasible") or fe <= 0:
                    continue
                ratio = r["cluster_energy_per_iter_j"] / fe
                # propagate the CI endpoints conservatively
                ratio_lo = r["cluster_energy_ci_lo"] / fe
                ratio_hi = r["cluster_energy_ci_hi"] / fe
                report["E_eco"][layout] = {"ratio": ratio, "ci_lo": ratio_lo, "ci_hi": ratio_hi}
                report["verdict"][layout] = (
                    "ALIVE (<=0.75)" if ratio <= 0.75 else
                    "MARGINAL (0.75-0.85)" if ratio <= 0.85 else "DEAD (>0.85)")
        else:
            report["verdict"]["_global"] = (
                "FEASIBILITY regime: true DDP infeasible at this scale; sharding mandatory, "
                "E_eco vs DDP undefined. Report smallest-feasible layout.")

        print("=" * 64)
        print(f"Tier-1a multi-GPU probe  (world_size={world}, optimizer={args.optimizer}, "
              f"source={next((r['energy_source'] for r in results.values() if r.get('feasible')), 'n/a')})")
        if args.smoke:
            print("  [SMOKE TEST — logic only, NOT a real measurement]")
        for layout, r in results.items():
            if not r.get("feasible"):
                print(f"  {layout:14s} INFEASIBLE ({r.get('note','')})")
                continue
            cf = r["capacity_frac"]
            cf_s = f"{cf*100:.0f}% cap" if cf else "n/a"
            print(f"  {layout:14s} cluster {r['cluster_energy_per_iter_j']:.3f} J/iter "
                  f"[{r['cluster_energy_ci_lo']:.3f},{r['cluster_energy_ci_hi']:.3f}] "
                  f"@ {r['cluster_mean_power_w']:.0f} W, {r['iters_per_s']:.2f} it/s, "
                  f"peak {r['peak_mem_gb_per_device']:.1f} GB ({cf_s})")
        for layout, e in report["E_eco"].items():
            print(f"  E_eco[{layout}] = {e['ratio']:.3f} [{e['ci_lo']:.3f},{e['ci_hi']:.3f}] "
                  f"-> {report['verdict'][layout]}")
        if not report["E_eco"] and ddp_feasible:
            print("  (single-process smoke or no eco layout measured: ratios need >=2 GPUs)")
        if not ddp_feasible:
            print(f"  {report['verdict']['_global']}")
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layouts", default="ddp,fsdp,fsdp_offload",
                   help="comma list; 'ddp' is the true-DDP fast baseline. 'hsdp' is opt-in (needs even world size).")
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"],
                   help="adam = stateful (memory-relief route); sgd = near-stateless (2-GPU config)")
    p.add_argument("--act-ckpt", default="none", choices=["none", "all"],
                   help="all = wrap every transformer layer in activation checkpointing (recompute)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--seeds", type=int, default=5, help="independent repeats for the CI")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=2048,
                   help="raise until ddp capacity_frac >= ~0.7 (memory-pressured regime)")
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--smoke", action="store_true", help="mark output as a logic smoke test")
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

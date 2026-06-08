"""MoE / expert-parallel structural-energy probe (scaffold).

This extends the dense multi-GPU probe (``exp_tier1a_multigpu_probe.py``) to the
one structural regime the dense study could not reach: a mixture-of-experts (MoE)
layer whose experts are *expert-parallel* (sharded across ranks, tokens routed by
all-to-all) rather than replicated. The open question the characterization paper
leaves for an MoE is whether expert parallelism makes the structural layout lever
come alive at fixed work, i.e. whether the eco (expert-parallel) layout's cluster
energy clears the free-throttle margin against a dense data-parallel baseline.

Routing uses the GShard padded-capacity dispatch/combine (Lepikhin et al. 2020):
a fixed ``[experts, capacity, d_model]`` dispatch tensor makes the all-to-all a
fixed-size collective and makes the routing *exactly* checkable on CPU (at world
size 1 the all-to-all is the identity, so the local dispatch/combine must
reproduce a per-token reference application of the selected expert — this is the
"unit-test scatter/gather on CPU" gate). See ``tests/test_moe_expert_parallel.py``.

Pre-registered decision on E_eco = e_expert_parallel / e_dense (declare BEFORE the
real run), mirroring the dense probe and the break-even surface (e* ~ 0.75):

  * E_eco <= 0.75        -> eco layout clears the free-throttle margin: ALIVE.
  * 0.75 < E_eco <= 0.85 -> MARGINAL.
  * E_eco > 0.85         -> DEAD at this workload.
  * dense OOM            -> FEASIBILITY regime (expert parallelism mandatory).

Stated hypothesis (so the result is falsifiable): ~0.95-1.05 -> DEAD, because
all-to-all communication and capacity padding add overhead while fixed-work
expert FLOPs are unchanged. The probe exists to be able to refute that.

Reported alongside energy: expert load balance (fraction of experts active, the
load coefficient of variation, and the capacity-overflow drop fraction). A seed
whose router collapses (< ``--min-active-frac`` of experts active, default 0.70)
is DEGENERATE and excluded from the energy aggregate, with the exclusion logged --
a saving that only appears because the router stopped using most experts is a
routing artifact, not a layout property.

Energy is the per-GPU NVML total-energy counter summed across ranks at the DEFAULT
power cap (a saving that needs nvidia-smi -pl is DVFS-in-disguise and does not
count), measured over ``--seeds`` repeats for a mean + CI.

Real run (>= 4 GPUs on the rented node)::

    torchrun --nproc_per_node=4 -m experiments.exp_moe_expert_parallel_probe \\
        --d-model 1024 --n-experts 8 --tokens 4096 --ffn-mult 4 \\
        --steps 200 --warmup 40 --seeds 5 --capacity-factor 1.25 \\
        --out artifacts/moe_expert_parallel_probe.json

Logic smoke-test (1 process, CPU; NOT a real measurement)::

    python -m experiments.exp_moe_expert_parallel_probe --smoke \\
        --d-model 64 --n-experts 4 --tokens 128 --steps 4 --warmup 1 --seeds 2
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
import torch.nn.functional as F  # noqa: N812

try:
    import pynvml
    _NVML = True
except Exception:  # pragma: no cover - depends on the node
    _NVML = False


# --------------------------------------------------------------------------- #
# MoE building blocks: a top-1 router + N expert FFNs, with GShard padded
# capacity dispatch/combine. The routing is layout-agnostic; the energy probe
# below wires it into a dense (replicated) vs expert-parallel (sharded) layout.
# --------------------------------------------------------------------------- #
class Expert(nn.Module):
    """A single position-wise FFN expert (the standard transformer MLP)."""

    def __init__(self, d_model: int, ffn_mult: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_mult * d_model)
        self.fc2 = nn.Linear(ffn_mult * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def top1_route(logits: torch.Tensor, n_experts: int, capacity: int):
    """GShard top-1 routing masks.

    Args:
        logits: [T, E] router logits.
        n_experts, capacity: E and the per-expert token budget C.

    Returns:
        combine_weights [T, E, C] (gate prob in the assigned slot, else 0),
        dispatch_mask   [T, E, C] (1.0 in the assigned slot, else 0),
        and routing stats (expert load, drop fraction).
    """
    gates = F.softmax(logits, dim=-1)            # [T, E]
    expert_idx = torch.argmax(gates, dim=-1)     # [T]
    gate_val = gates.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)  # [T]

    # position of each token within its expert's capacity buffer (0-based).
    onehot = F.one_hot(expert_idx, num_classes=n_experts).to(logits.dtype)  # [T, E]
    position_in_expert = (torch.cumsum(onehot, dim=0) - 1)                   # [T, E]
    position_in_expert = (position_in_expert * onehot).sum(dim=-1).long()    # [T]
    kept = position_in_expert < capacity                                     # [T] bool

    load = onehot.sum(dim=0)                                                 # [E] tokens/expert
    drop_frac = float((~kept).float().mean().item())

    n_tok = logits.shape[0]
    combine = torch.zeros(n_tok, n_experts, capacity, dtype=logits.dtype, device=logits.device)
    slot = position_in_expert.clamp(max=capacity - 1)
    idx_t = torch.arange(n_tok, device=logits.device)
    mask_keep = kept.to(logits.dtype)
    combine[idx_t, expert_idx, slot] = gate_val * mask_keep
    dispatch = (combine > 0).to(logits.dtype)
    return combine, dispatch, {"load": load, "drop_frac": drop_frac,
                               "active_experts": int((load > 0).sum().item())}


def moe_reference(x: torch.Tensor, router: nn.Linear, experts: nn.ModuleList) -> torch.Tensor:
    """Per-token reference: apply the argmax expert, scale by its gate prob.

    No capacity drop -> this is the exact target the dispatch/combine path must
    reproduce when capacity >= the busiest expert's load.
    """
    gates = F.softmax(router(x), dim=-1)
    idx = torch.argmax(gates, dim=-1)
    gate_val = gates.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(x)
    for e, expert in enumerate(experts):
        sel = idx == e
        if sel.any():
            out[sel] = expert(x[sel])
    return out * gate_val.unsqueeze(-1)


def moe_dispatch_combine(
    x: torch.Tensor, router: nn.Linear, experts: nn.ModuleList, capacity: int,
    *, expert_parallel: bool = False,
) -> tuple[torch.Tensor, dict]:
    """GShard dispatch/combine forward.

    ``expert_parallel`` shards the experts across ranks and routes tokens with two
    all-to-all collectives; at world size 1 (or non-distributed) the all-to-all is
    the identity and every rank holds every expert, so this reduces to the local
    routed forward used by the CPU correctness test.
    """
    n_experts = len(experts)
    combine, dispatch, stats = top1_route(router(x), n_experts, capacity)
    # dispatched[e, c, m] = sum_t dispatch[t,e,c] * x[t,m]
    dispatched = torch.einsum("tec,tm->ecm", dispatch, x)            # [E, C, M]

    world = dist.get_world_size() if (expert_parallel and _is_distributed()) else 1
    if world > 1:
        # Shard experts across ranks: all-to-all so each rank receives the global
        # tokens destined for its local experts, processes them, sends them back.
        assert n_experts % world == 0, "n_experts must be divisible by world size"
        local_e = n_experts // world
        recv = torch.empty_like(dispatched)
        dist.all_to_all_single(recv, dispatched.contiguous())
        rank = dist.get_rank()
        my = experts[rank * local_e:(rank + 1) * local_e]
        # recv is [E, C, M] laid out so this rank owns a contiguous expert block.
        processed = recv.clone()
        for j, expert in enumerate(my):
            e = rank * local_e + j
            processed[e] = expert(recv[e])
        out_disp = torch.empty_like(processed)
        dist.all_to_all_single(out_disp, processed.contiguous())
    else:
        out_disp = torch.stack([experts[e](dispatched[e]) for e in range(n_experts)], dim=0)

    combined = torch.einsum("tec,ecm->tm", combine, out_disp)        # [T, M]
    return combined, stats


# --------------------------------------------------------------------------- #
# Distributed plumbing (shared shape with exp_tier1a_multigpu_probe).
# --------------------------------------------------------------------------- #
def _is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _setup() -> tuple[int, int, int, torch.device]:
    if _is_distributed():
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        rank, world, local = 0, 1, 0
    device = torch.device(f"cuda:{local}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local, device


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


def _power_w(handle) -> float:
    if handle is None:
        return 0.0
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        return 0.0


def _ci95(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return math.nan, math.nan, math.nan
    m = sum(xs) / n
    if n < 2:
        return m, m, m
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    tcrit = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 2.0)
    half = tcrit * sd / math.sqrt(n)
    return m, m - half, m + half


def _allreduce(value: float, device: torch.device, op) -> float:
    if _is_distributed() and dist.get_world_size() > 1:
        t = torch.tensor([value], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=op)
        return float(t.item())
    return value


# --------------------------------------------------------------------------- #
# Energy measurement.
# --------------------------------------------------------------------------- #
def _measure_layout(
    layout: str, device: torch.device, handle, args, seed: int,
) -> dict | None:
    """One repeat of one layout; returns per-iter energy/power + routing stats.

    Returns None if the router is degenerate (< args.min_active_frac experts used).
    """
    torch.manual_seed(1000 + seed)
    d, e, mult = args.d_model, args.n_experts, args.ffn_mult
    router = nn.Linear(d, e).to(device)
    experts = nn.ModuleList([Expert(d, mult) for _ in range(e)]).to(device)
    params = list(router.parameters()) + list(experts.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    capacity = math.ceil(args.capacity_factor * args.tokens / e)
    x = torch.randn(args.tokens, d, device=device)
    expert_parallel = layout == "expert_parallel"

    last_stats: dict = {}
    for _ in range(args.warmup):
        opt.zero_grad(set_to_none=True)
        out, last_stats = moe_dispatch_combine(x, router, experts, capacity,
                                               expert_parallel=expert_parallel)
        out.float().pow(2).mean().backward()
        opt.step()
    active_frac = last_stats.get("active_experts", e) / e
    if active_frac < args.min_active_frac:
        return None  # degenerate router -> excluded

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    e0 = _energy_j(handle)
    p_samples: list[float] = []
    t0 = time.perf_counter()
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out, last_stats = moe_dispatch_combine(x, router, experts, capacity,
                                               expert_parallel=expert_parallel)
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
    load = last_stats["load"]
    load_cv = float((load.std() / load.mean()).item()) if load.mean() > 0 else math.nan
    return {
        "energy_per_iter_j": energy_j / args.steps, "mean_power_w": mean_p,
        "iters_per_s": args.steps / elapsed, "peak_mem_gb": peak_gb,
        "active_frac": active_frac, "load_cv": load_cv,
        "drop_frac": last_stats["drop_frac"],
        "energy_source": "nvml-total-energy" if (e0 is not None and e1 is not None)
        else "nvml-power-integral",
    }


def run(args: argparse.Namespace) -> int:
    rank, world, local, device = _setup()
    # NVML metering only makes sense for the CUDA device we actually compute on;
    # on CPU (smoke) leave it None so we never read an idle GPU's ambient counter.
    handle = _nvml_handle(local) if device.type == "cuda" else None
    layouts = ["dense", "expert_parallel"]
    results: dict[str, dict] = {}

    for layout in layouts:
        kept, excluded = [], 0
        for seed in range(args.seeds):
            m = _measure_layout(layout, device, handle, args, seed)
            if m is None:
                excluded += 1
                continue
            kept.append(m)
            if _is_distributed() and dist.get_world_size() > 1:
                dist.barrier()
        if not kept:
            results[layout] = {"feasible": False, "note": "all seeds degenerate", "excluded": excluded}
            continue
        e_seeds = [s["energy_per_iter_j"] for s in kept]
        e_cluster = [_allreduce(v, device, dist.ReduceOp.SUM) for v in e_seeds]
        em, elo, ehi = _ci95(e_cluster)
        results[layout] = {
            "feasible": True, "n_seeds": len(kept), "excluded_degenerate": excluded,
            "cluster_energy_per_iter_j": em, "cluster_energy_ci_lo": elo, "cluster_energy_ci_hi": ehi,
            "mean_power_w": _allreduce(sum(s["mean_power_w"] for s in kept) / len(kept), device, dist.ReduceOp.SUM),
            "iters_per_s": sum(s["iters_per_s"] for s in kept) / len(kept),
            "peak_mem_gb": _allreduce(max(s["peak_mem_gb"] for s in kept), device, dist.ReduceOp.MAX),
            "active_frac": sum(s["active_frac"] for s in kept) / len(kept),
            "load_cv": sum(s["load_cv"] for s in kept) / len(kept),
            "drop_frac": sum(s["drop_frac"] for s in kept) / len(kept),
            "energy_source": kept[0]["energy_source"],
        }

    if rank == 0:
        dense = results.get("dense")
        report = {
            "probe": "moe-expert-parallel", "world_size": world,
            "d_model": args.d_model, "n_experts": args.n_experts, "tokens": args.tokens,
            "ffn_mult": args.ffn_mult, "capacity_factor": args.capacity_factor,
            "steps": args.steps, "seeds": args.seeds, "min_active_frac": args.min_active_frac,
            "break_even_estar": 0.75, "hypothesis": "DEAD (~0.95-1.05)",
            "layouts": results, "E_eco": {}, "verdict": {}, "smoke": bool(args.smoke),
        }
        if dense and dense.get("feasible") and dense["cluster_energy_per_iter_j"] > 0:
            fe = dense["cluster_energy_per_iter_j"]
            ep = results.get("expert_parallel")
            if ep and ep.get("feasible"):
                ratio = ep["cluster_energy_per_iter_j"] / fe
                report["E_eco"]["expert_parallel"] = {
                    "ratio": ratio, "ci_lo": ep["cluster_energy_ci_lo"] / fe,
                    "ci_hi": ep["cluster_energy_ci_hi"] / fe,
                }
                report["verdict"]["expert_parallel"] = (
                    "ALIVE (<=0.75)" if ratio <= 0.75 else
                    "MARGINAL (0.75-0.85)" if ratio <= 0.85 else "DEAD (>0.85)")
        print("=" * 64)
        print(f"MoE expert-parallel probe (world_size={world}, "
              f"E={args.n_experts}, tokens={args.tokens}, cap_factor={args.capacity_factor})")
        if args.smoke:
            print("  [SMOKE TEST — logic only, NOT a real measurement; world_size 1 => all-to-all is identity]")
        for layout, r in results.items():
            if not r.get("feasible"):
                print(f"  {layout:16s} INFEASIBLE ({r.get('note', '')})")
                continue
            print(f"  {layout:16s} cluster {r['cluster_energy_per_iter_j']:.4f} J/iter "
                  f"[{r['cluster_energy_ci_lo']:.4f},{r['cluster_energy_ci_hi']:.4f}] "
                  f"active={r['active_frac']*100:.0f}% load_cv={r['load_cv']:.2f} "
                  f"drop={r['drop_frac']*100:.1f}% (excluded {r['excluded_degenerate']} degenerate)")
        for layout, ee in report["E_eco"].items():
            print(f"  E_eco[{layout}] = {ee['ratio']:.3f} [{ee['ci_lo']:.3f},{ee['ci_hi']:.3f}] "
                  f"-> {report['verdict'][layout]}")
        if not report["E_eco"]:
            print("  (E_eco needs >=2 GPUs + both layouts feasible; single-process smoke validates routing only)")
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
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-experts", type=int, default=8, help="must be divisible by world size for EP")
    p.add_argument("--tokens", type=int, default=4096, help="tokens per step (global batch x seq)")
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--capacity-factor", type=float, default=1.25,
                   help="C = ceil(factor * tokens / experts); >1 tolerates load imbalance")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--min-active-frac", type=float, default=0.70,
                   help="seed excluded as degenerate if fewer than this fraction of experts are used")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

"""Fault injectors that emulate documented reshard-bug classes on real state.

Each injector mutates the post-transition state dicts (model, optimizer,
progress) the way a buggy reshard mechanism would, BEFORE the certificate
snapshot is taken. Injection happens at the state-dict level so the same fault
applies whether the post side materialized full tensors or DTensor shards;
for a DTensor entry the injector mutates the rank-local shard, which is
exactly how a buggy reshard corrupts sharded state.

All injectors are silent by construction: none raises, none changes a tensor
shape in a way the framework would reject, so a loss-curve-only validation
would have to notice a small numeric deviation to catch any of them. The
documented instances backing each class are annotated in the corpus notes as
they are verified.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.distributed.tensor import DTensor


def _local(t: torch.Tensor) -> torch.Tensor:
    """The mutable local view: the rank-local shard for a DTensor, the tensor
    itself otherwise."""
    return t.to_local() if isinstance(t, DTensor) else t


def _tensor_items(model_sd: Dict[str, Any]) -> List[Tuple[str, torch.Tensor]]:
    return [(k, v) for k, v in sorted(model_sd.items()) if isinstance(v, torch.Tensor)]


def _pick(model_sd: Dict[str, Any], index: int) -> Tuple[str, torch.Tensor]:
    items = _tensor_items(model_sd)
    fqn, t = items[index % len(items)]
    return fqn, _local(t)


def _opt_state(optim_sd: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return optim_sd.get("state", {})


# --------------------------------------------------------------------- faults
def fault_silent_unloaded_param(model_sd, optim_sd, progress, snapshot_kwargs):
    """A parameter the load planner never materialized: it silently keeps its
    fresh-init value (the strict=False checkpoint-load failure class)."""
    fqn, loc = _pick(model_sd, 1)
    with torch.no_grad():
        g = torch.Generator().manual_seed(0xBADC0DE)
        loc.copy_(torch.randn(loc.shape, generator=g, dtype=torch.float32).to(loc.dtype) * 0.02)
    return f"re-initialized {fqn} (simulates unloaded key)"


def fault_value_corruption(model_sd, optim_sd, progress, snapshot_kwargs):
    """Transport corruption: a small additive perturbation on one tensor (a
    mis-stitched shard boundary or partial write)."""
    fqn, loc = _pick(model_sd, 2)
    with torch.no_grad():
        loc.add_(torch.full_like(loc, 1e-3))
    return f"perturbed {fqn} by +1e-3"


def fault_permuted_values(model_sd, optim_sd, progress, snapshot_kwargs):
    """Mis-mapped shard geometry: the values of a 2-D weight are transposed
    and reshaped back, preserving shape and norm but scrambling positions
    (the tensor-parallel transpose / wrong-stride class)."""
    for fqn, t in _tensor_items(model_sd):
        loc = _local(t)
        if loc.ndim == 2 and loc.shape[0] != loc.shape[1]:
            with torch.no_grad():
                loc.copy_(loc.t().reshape(loc.shape))
            return f"transposed-and-reshaped {fqn}"
    fqn, loc = _pick(model_sd, 0)
    with torch.no_grad():
        loc.copy_(loc.flatten().flip(0).reshape(loc.shape))
    return f"reversed {fqn}"


def fault_cross_param_overwrite(model_sd, optim_sd, progress, snapshot_kwargs):
    """FQN mis-mapping: one parameter receives another parameter's bytes
    (the renamed/misordered key-mapping class in checkpoint converters)."""
    items = _tensor_items(model_sd)
    for (na, ta), (nb, tb) in zip(items, items[1:]):
        la, lb = _local(ta), _local(tb)
        if la.shape == lb.shape and na != nb:
            with torch.no_grad():
                lb.copy_(la)
            return f"overwrote {nb} with {na}"
    na, la = _pick(model_sd, 0)
    nb, lb = _pick(model_sd, 3)
    n = min(la.numel(), lb.numel())
    with torch.no_grad():
        lb.flatten()[:n].copy_(la.flatten()[:n])
    return f"partially overwrote {nb} with {na}"


def fault_stale_optimizer_moment(model_sd, optim_sd, progress, snapshot_kwargs):
    """Optimizer state dropped and re-initialized for one parameter while the
    weights load fine (the optimizer-state merge/reshard class)."""
    for fqn, slots in sorted(_opt_state(optim_sd).items()):
        if "exp_avg" in slots:
            with torch.no_grad():
                _local(slots["exp_avg"]).zero_()
                if "exp_avg_sq" in slots:
                    _local(slots["exp_avg_sq"]).zero_()
            return f"zeroed exp_avg/exp_avg_sq of {fqn}"
    return "no optimizer state present (fault not applicable)"


def fault_step_counter_reset(model_sd, optim_sd, progress, snapshot_kwargs):
    """The per-param step counter resets on load: bias correction restarts
    and the next updates are silently mis-scaled."""
    for fqn, slots in sorted(_opt_state(optim_sd).items()):
        if "step" in slots:
            s = slots["step"]
            if isinstance(s, torch.Tensor):
                with torch.no_grad():
                    _local(s).zero_()
            else:
                slots["step"] = 0
            return f"reset step counter of {fqn}"
    return "no step counter present (fault not applicable)"


def fault_precision_cast(model_sd, optim_sd, progress, snapshot_kwargs):
    """A round-trip through a narrower dtype during transport: values are
    quantized but shape and dtype look right afterwards."""
    fqn, loc = _pick(model_sd, 4)
    with torch.no_grad():
        loc.copy_(loc.to(torch.float16).to(loc.dtype))
    return f"fp16 round-trip on {fqn}"


def fault_progress_reset(model_sd, optim_sd, progress, snapshot_kwargs):
    """Trainer-level progress (global step / samples seen) lost in the
    transition metadata."""
    if progress:
        key = sorted(progress)[0]
        progress[key] = 0
        return f"reset progress counter '{key}'"
    return "no progress counters (fault not applicable)"


def fault_dropped_fqn(model_sd, optim_sd, progress, snapshot_kwargs):
    """A tensor the reshard never produced at all: the post-side state dict
    is missing one FQN (a placement/materialization failure)."""
    snapshot_kwargs["drop_fqns"] = snapshot_kwargs.get("drop_fqns", []) + ["__first__"]
    return "post-side state dict will omit its first FQN"


def fault_reduction_order_swap(model_sd, optim_sd, progress, snapshot_kwargs):
    """The declared gradient-reduction order changes across the transition
    without a declared bound change (unbounded summation-order drift)."""
    snapshot_kwargs["swap_reduction_order"] = True
    return "declared reduction order will be permuted"


FaultFn = Callable[..., str]

FAULTS: Dict[str, FaultFn] = {
    "silent_unloaded_param": fault_silent_unloaded_param,
    "value_corruption": fault_value_corruption,
    "permuted_values": fault_permuted_values,
    "cross_param_overwrite": fault_cross_param_overwrite,
    "stale_optimizer_moment": fault_stale_optimizer_moment,
    "step_counter_reset": fault_step_counter_reset,
    "precision_cast": fault_precision_cast,
    "progress_reset": fault_progress_reset,
    "dropped_fqn": fault_dropped_fqn,
    "reduction_order_swap": fault_reduction_order_swap,
}


def get_fault(name: Optional[str]) -> Optional[FaultFn]:
    if name is None:
        return None
    return FAULTS[name]

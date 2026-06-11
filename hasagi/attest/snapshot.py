"""Canonical, layout-independent snapshots of training state.

A snapshot maps every fully-qualified parameter name (FQN) to a content
fingerprint, and every optimizer slot (exp_avg, exp_avg_sq, step, ...) to its
own fingerprint, together with cheap independent checksums (element count,
L2 norm) computed through a separate code path from the byte hash. Two layouts
of the same logical state must produce identical snapshots; any transport bug
that drops, corrupts, duplicates, retypes, or mis-maps a tensor changes one.

Fingerprints cover dtype and shape explicitly: a reshard that silently casts
bf16 to fp32, or transposes a shard, changes the fingerprint even when the
values survive.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch


def fingerprint_tensor(t: torch.Tensor) -> str:
    """A deterministic content fingerprint: sha256 over (dtype, shape, bytes).

    The tensor is brought to CPU and made contiguous; the dtype and shape are
    hashed as a header so that casts and reshapes are distinguishable from
    value corruption.
    """
    t = t.detach()
    if t.is_sparse:
        t = t.to_dense()
    t = t.cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    # reinterpret the contiguous buffer as raw bytes (works for every dtype,
    # including those numpy lacks, e.g. bf16) and hash at C speed
    if t.numel() == 0:
        return h.hexdigest()
    raw = t.reshape(-1).view(torch.uint8)
    h.update(raw.numpy().tobytes())
    return h.hexdigest()


def _independent_checksum(t: torch.Tensor) -> Dict[str, float]:
    """Cheap checksums computed WITHOUT the byte-hash path, so the certificate
    does not stand on a single serialization routine: element count, fp64 L2
    norm, and the value at a fixed probe index."""
    t = t.detach().cpu()
    flat = t.reshape(-1)
    n = flat.numel()
    norm = float(torch.linalg.vector_norm(flat.to(torch.float64))) if n else 0.0
    probe = float(flat[n // 2].to(torch.float64)) if n else 0.0
    return {"numel": float(n), "l2": norm, "probe_mid": probe}


@dataclass
class StateSnapshot:
    """The logical training state at one side of a transition."""

    # FQN -> content fingerprint of the parameter tensor
    params: Dict[str, str]
    # FQN -> slot name -> fingerprint (Adam exp_avg / exp_avg_sq / step, ...)
    opt_slots: Dict[str, Dict[str, str]]
    # FQN -> independent checksums (numel / l2 / probe), separate code path
    checksums: Dict[str, Dict[str, float]]
    # training-progress counters that the transition must preserve
    progress: Dict[str, int] = field(default_factory=dict)
    # declared gradient reduction order (bucket order) and its drift bound
    reduction_order: Optional[list] = None
    reduction_drift_bound: float = 1e-6
    # free-form provenance (world size, wrapper, step) for diagnostics only
    meta: Dict[str, Any] = field(default_factory=dict)


def _is_tensor(v: Any) -> bool:
    return isinstance(v, torch.Tensor)


def snapshot_from_state_dicts(
    model_state: Mapping[str, Any],
    optim_state: Optional[Mapping[str, Any]] = None,
    *,
    progress: Optional[Dict[str, int]] = None,
    reduction_order: Optional[list] = None,
    reduction_drift_bound: float = 1e-6,
    meta: Optional[Dict[str, Any]] = None,
) -> StateSnapshot:
    """Build a snapshot from (full, unsharded) model and optimizer state dicts.

    ``optim_state`` accepts either the flattened FQN-keyed form produced by
    ``torch.distributed.checkpoint.state_dict.get_optimizer_state_dict``
    (``{"state": {fqn: {slot: tensor}}, "param_groups": [...]}``) or the
    classic index-keyed ``torch.optim.Optimizer.state_dict()`` form, in which
    case indices are kept as keys (callers should prefer the FQN form so the
    mapping is layout-independent).
    """
    params: Dict[str, str] = {}
    checks: Dict[str, Dict[str, float]] = {}
    for fqn, tensor in model_state.items():
        if not _is_tensor(tensor):
            continue
        params[fqn] = fingerprint_tensor(tensor)
        checks[fqn] = _independent_checksum(tensor)

    slots: Dict[str, Dict[str, str]] = {}
    if optim_state:
        state = optim_state.get("state", optim_state)
        for key, per_param in state.items():
            if not isinstance(per_param, Mapping):
                continue
            entry: Dict[str, str] = {}
            for slot, val in per_param.items():
                if _is_tensor(val):
                    # 0-dim step tensors compare by value: their device/layout is
                    # incidental, but the count itself must be preserved.
                    if val.ndim == 0:
                        entry[slot] = f"scalar:{val.item():.17g}"
                    else:
                        entry[slot] = fingerprint_tensor(val)
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    # scalar step counters participate in the certificate too
                    entry[slot] = f"scalar:{float(val):.17g}"
            slots[str(key)] = entry

    return StateSnapshot(
        params=params,
        opt_slots=slots,
        checksums=checks,
        progress=dict(progress or {}),
        reduction_order=list(reduction_order) if reduction_order is not None else None,
        reduction_drift_bound=reduction_drift_bound,
        meta=dict(meta or {}),
    )


def snapshot_total_bytes(model_state: Mapping[str, Any]) -> int:
    """Total parameter bytes covered by a snapshot (for overhead accounting)."""
    return sum(
        t.element_size() * t.numel() for t in model_state.values() if _is_tensor(t)
    )


def l2_close(a: Dict[str, float], b: Dict[str, float], rtol: float = 1e-7) -> bool:
    """Compare two independent-checksum records: exact on counts, relative
    tolerance on the norm and on the fixed probe element. The probe gives the
    independent path a position-sensitive witness — a value permutation
    preserves numel and L2 exactly but moves the mid element with high
    probability."""
    if a["numel"] != b["numel"]:
        return False
    for key in ("l2", "probe_mid"):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:  # records from older snapshots lack the probe
            continue
        if not (math.isfinite(va) and math.isfinite(vb)):
            return False
        ref = max(abs(va), abs(vb), 1e-30)
        if abs(va - vb) / ref > rtol:
            return False
    return True

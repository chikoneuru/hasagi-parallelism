"""Reshard controller for live mid-job layout reconfiguration (Uplift 3 scaffold).

The characterization paper treats the structural layout lever's reconfiguration
cost analytically (the resume model). This module is the executable counterpart:
it moves real training state across a parallelism layout change -- the canonical
case being data-parallel (a full per-rank replica, DDP) to fully-sharded (a flat
contiguous shard per rank, FSDP FULL_SHARD) and back -- so a harness can measure
the reconfiguration latency and energy and show the loss trajectory is continuous
across the flip.

The state model matches FSDP's FlatParameter: all parameters are flattened in a
fixed manifest order into one 1-D buffer, padded to a multiple of the world size,
and split into equal contiguous shards. Resharding from W to W' is therefore
"reassemble the flat buffer from the old shards, re-split into W' shards" -- and
``unflatten(flatten(s)) == s`` and ``unshard(shard(f, W)) == f`` are exact, which
is what makes the round-trip checkable on CPU without any GPU (see
``tests/test_reshard_controller.py``).

The controller is transactional: it reassembles the post-reshard state, verifies
it against the state captured before the move, and only then commits it into the
model; on a verification mismatch it aborts and leaves the model untouched (the
last verified state). When ``torch.distributed.checkpoint`` is available and the
job is distributed, the same flat-shard plan can be persisted/loaded through DCP;
the in-memory path here is the layout-transport core the harness exercises.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParamManifest:
    """Ordered (name, shape) records — the deterministic flatten/unflatten order."""

    entries: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def total_numel(self) -> int:
        return sum(_numel(shape) for _, shape in self.entries)


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def flatten_state(state: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ParamManifest]:
    """Flatten a state dict into (1-D buffer, manifest) in sorted-key order."""
    entries: list[tuple[str, tuple[int, ...]]] = []
    chunks: list[torch.Tensor] = []
    for name in sorted(state):
        t = state[name]
        entries.append((name, tuple(t.shape)))
        chunks.append(t.detach().reshape(-1).to(torch.float32))
    flat = torch.cat(chunks) if chunks else torch.zeros(0)
    return flat, ParamManifest(tuple(entries))


def unflatten_state(flat: torch.Tensor, manifest: ParamManifest) -> dict[str, torch.Tensor]:
    """Inverse of :func:`flatten_state`."""
    out: dict[str, torch.Tensor] = {}
    off = 0
    for name, shape in manifest.entries:
        n = _numel(shape)
        out[name] = flat[off:off + n].reshape(shape).clone()
        off += n
    return out


def shard_flat(flat: torch.Tensor, world: int) -> list[torch.Tensor]:
    """Pad to a multiple of ``world`` and split into ``world`` equal contiguous shards."""
    if world < 1:
        raise ValueError("world must be >= 1")
    total = flat.numel()
    pad = (-total) % world
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    return list(flat.reshape(world, -1).contiguous())


def unshard_flat(shards: list[torch.Tensor], total_numel: int) -> torch.Tensor:
    """Concatenate shards and drop the padding back to ``total_numel``."""
    return torch.cat([s.reshape(-1) for s in shards])[:total_numel]


@dataclass
class ReshardCertificate:
    """Result of the verify-before-commit check over a reshard transition."""

    ok: bool
    max_abs_diff: float
    n_params: int
    from_world: int
    to_world: int
    note: str = ""


class ReshardController:
    """Capture training state, reshard it to a new world size, verify, then commit.

    Usage::

        rc = ReshardController()
        rc.capture(model)                      # before the layout flip
        cert = rc.reshard_and_commit(model, to_world=4)
        assert cert.ok                         # else the model is untouched
    """

    def __init__(self, atol: float = 1e-6) -> None:
        self.atol = atol
        self._captured: dict[str, torch.Tensor] | None = None
        self._flat: torch.Tensor | None = None
        self._manifest: ParamManifest | None = None
        self.from_world = 1

    def capture(self, model: torch.nn.Module, from_world: int = 1) -> ParamManifest:
        """Snapshot the model's parameters as the last verified state."""
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self._captured = state
        self._flat, self._manifest = flatten_state(state)
        self.from_world = from_world
        return self._manifest

    def plan_shards(self, to_world: int) -> list[torch.Tensor]:
        """The per-rank flat shards the new (FSDP) layout would hold."""
        if self._flat is None:
            raise RuntimeError("capture() must be called before planning a reshard")
        return shard_flat(self._flat, to_world)

    def verify(self, reassembled: dict[str, torch.Tensor], to_world: int) -> ReshardCertificate:
        """Check the reassembled post-reshard state equals the captured state."""
        assert self._captured is not None and self._manifest is not None
        max_diff = 0.0
        for name, _ in self._manifest.entries:
            a = self._captured[name].reshape(-1).to(torch.float32)
            b = reassembled[name].reshape(-1).to(torch.float32)
            if a.numel() != b.numel():
                return ReshardCertificate(False, float("inf"), len(self._manifest.entries),
                                          self.from_world, to_world, f"numel mismatch at {name}")
            max_diff = max(max_diff, float((a - b).abs().max().item()) if a.numel() else 0.0)
        return ReshardCertificate(max_diff <= self.atol, max_diff,
                                  len(self._manifest.entries), self.from_world, to_world)

    def reshard_and_commit(self, model: torch.nn.Module, to_world: int) -> ReshardCertificate:
        """Reshard to ``to_world``, verify, and commit into ``model`` iff verification passes.

        On failure the model is left at its last verified state (abort-to-last-verified).
        """
        assert self._manifest is not None
        shards = self.plan_shards(to_world)
        flat = unshard_flat(shards, self._manifest.total_numel)
        reassembled = unflatten_state(flat, self._manifest)
        cert = self.verify(reassembled, to_world)
        if cert.ok:
            model.load_state_dict(reassembled, strict=True)
        return cert

"""Multiprocess transition harness: a real reshard, certified.

The harness runs one elastic-reconfiguration transition end to end on CPU
process groups (gloo backend), with torch.distributed.checkpoint (DCP) as the
reshard mechanism under test. A transition moves the checkpointed training
state between layouts:

  full(N)    every rank holds the complete, replicated state
  shard(N)   every tensor is a DTensor split Shard(0) across an N-rank mesh

  stage A (pre)   spawn ``world_pre`` ranks, train the model for a few
                  deterministic steps (DDP when world_pre > 1) so weights and
                  Adam moments are non-trivial, snapshot the full logical
                  state, convert to the pre layout, and DCP-save.

  stage B (post)  spawn ``world_post`` ranks, build a fresh differently-seeded
                  template in the post layout, DCP-load the checkpoint into it
                  (this resharding load is the transition under test),
                  optionally inject a fault emulating a documented reshard-bug
                  class, gather, and snapshot.

  parent          certify pre vs post and decide commit/abort; report wall
                  times so the certificate cost is comparable against the
                  reshard cost it gates.

Stages exchange snapshots as JSON files and run as separate OS process
groups, exactly like an elastic restart. Sharded training itself (FSDP)
requires an accelerator in this torch build; the checkpoint-level shard
layouts here exercise the same DCP resharding path that elastic systems use,
and the harness accepts an FSDP training stage as a drop-in once a GPU node
is available.
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from .faults import get_fault
from .gate import CommitDecision, certify_transition
from .model import GPT, PRESETS, synthetic_batch
from .snapshot import StateSnapshot, snapshot_from_state_dicts

LAYOUTS = ("full", "shard")


@dataclass
class TransitionSpec:
    preset: str = "tiny"
    world_pre: int = 2
    world_post: int = 1
    layout_pre: str = "shard"
    layout_post: str = "full"
    train_steps: int = 3
    batch: int = 4
    fault: Optional[str] = None
    seed: int = 1234

    def describe(self) -> str:
        f = f" fault={self.fault}" if self.fault else ""
        return (
            f"{self.layout_pre}({self.world_pre}) -> {self.layout_post}({self.world_post})"
            f" [{self.preset}]{f}"
        )


@dataclass
class TransitionResult:
    spec: TransitionSpec
    decision: CommitDecision
    timings: Dict[str, float]

    def caught(self) -> bool:
        return self.decision.aborted


# --------------------------------------------------------------------------- helpers
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_pg(rank: int, world: int, port: int) -> None:
    # the harness is a CPU process-group testbed by design: hide any host GPU
    # so every stage process computes on CPU regardless of the machine
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)


def _shardable(t: torch.Tensor, world: int) -> bool:
    return t.ndim >= 1 and t.shape[0] >= world


def _to_layout_sd(
    model_sd: Dict[str, Any],
    optim_sd: Dict[str, Any],
    layout: str,
    world: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert full state dicts to the requested checkpoint layout."""
    if layout == "full" or world == 1:
        return model_sd, optim_sd
    mesh = init_device_mesh("cpu", (world,))

    def conv(t: Any) -> Any:
        if isinstance(t, torch.Tensor) and _shardable(t, world):
            return distribute_tensor(t, mesh, [Shard(0)])
        return t

    model_out = {k: conv(v) for k, v in model_sd.items()}
    optim_out = {
        "state": {
            fqn: {slot: conv(v) for slot, v in slots.items()}
            for fqn, slots in optim_sd.get("state", {}).items()
        },
        "param_groups": optim_sd.get("param_groups", []),
    }
    return model_out, optim_out


def _materialize_full(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Gather every DTensor entry to its full logical tensor (collective)."""

    def gather(v: Any) -> Any:
        if isinstance(v, DTensor):
            return v.full_tensor()
        if isinstance(v, dict):
            return {k: gather(x) for k, x in v.items()}
        return v

    return {k: gather(v) for k, v in sd.items()}


def _snapshot(
    model_sd: Dict[str, Any],
    optim_sd: Dict[str, Any],
    progress: Dict[str, int],
    *,
    meta: Dict[str, Any],
    drop_fqns: Optional[list] = None,
    swap_reduction_order: bool = False,
) -> StateSnapshot:
    model_full = _materialize_full(model_sd)
    optim_full = {
        "state": _materialize_full(optim_sd.get("state", {})),
        "param_groups": optim_sd.get("param_groups", []),
    }
    if drop_fqns:
        keys = sorted(model_full)
        for marker in drop_fqns:
            victim = keys[0] if marker == "__first__" else marker
            model_full.pop(victim, None)
    order = sorted(model_full)
    if swap_reduction_order and len(order) >= 2:
        order[0], order[1] = order[1], order[0]
    return snapshot_from_state_dicts(
        model_full,
        optim_full,
        progress=progress,
        reduction_order=order,
        meta=meta,
    )


def _snapshot_to_json(snap: StateSnapshot, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(dataclasses.asdict(snap), fh)


def _snapshot_from_json(path: str) -> StateSnapshot:
    with open(path) as fh:
        return StateSnapshot(**json.load(fh))


def _build_trained_state(
    spec: TransitionSpec, rank: int
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, int]]:
    """Train the model deterministically and return canonical full state."""
    cfg = PRESETS[spec.preset]
    torch.manual_seed(spec.seed)
    model: torch.nn.Module = GPT(cfg, seed=spec.seed)
    if spec.world_pre > 1:
        model = DDP(model)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(spec.train_steps):
        batch = synthetic_batch(cfg, spec.batch, seed=spec.seed + step)
        shard = batch.chunk(spec.world_pre)[rank] if spec.world_pre > 1 else batch
        logits = model(shard[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), shard[:, 1:].reshape(-1)
        )
        optim.zero_grad()
        loss.backward()
        optim.step()

    # canonical FQN-keyed full state (strips the DDP "module." prefix)
    model_sd, optim_sd = get_state_dict(model, optim)
    progress = {"global_step": spec.train_steps, "samples_seen": spec.train_steps * spec.batch}
    return model_sd, optim_sd, progress


def _build_template_state(
    spec: TransitionSpec,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A fresh, differently-seeded template in the post layout for DCP to
    reshard the checkpoint into. The different seed matters: a load that
    silently skips a tensor leaves template values that cannot collide with
    the trained state."""
    cfg = PRESETS[spec.preset]
    model = GPT(cfg, seed=spec.seed + 999)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # materialize optimizer slots (zeros) so DCP has targets to load into
    optim.zero_grad()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optim.step()
    optim.zero_grad(set_to_none=True)
    model_sd, optim_sd = get_state_dict(model, optim)
    return _to_layout_sd(model_sd, optim_sd, spec.layout_post, spec.world_post)


# --------------------------------------------------------------------------- stage A
def _stage_pre(rank: int, spec: TransitionSpec, workdir: str, port: int) -> None:
    _init_pg(rank, spec.world_pre, port)
    try:
        model_sd, optim_sd, progress = _build_trained_state(spec, rank)

        t0 = time.perf_counter()
        snap = _snapshot(
            model_sd, optim_sd, progress,
            meta={"side": "pre", "world": spec.world_pre, "layout": spec.layout_pre},
        )
        snap_s = time.perf_counter() - t0
        if rank == 0:
            _snapshot_to_json(snap, os.path.join(workdir, "pre.json"))
            with open(os.path.join(workdir, "progress.json"), "w") as fh:
                json.dump(progress, fh)

        # the checkpoint, in the pre layout, that the transition reshards from
        t0 = time.perf_counter()
        m_sd, o_sd = _to_layout_sd(model_sd, optim_sd, spec.layout_pre, spec.world_pre)
        dcp.save({"model": m_sd, "optim": o_sd}, checkpoint_id=os.path.join(workdir, "ckpt"))
        save_s = time.perf_counter() - t0
        if rank == 0:
            with open(os.path.join(workdir, "timings_pre.json"), "w") as fh:
                json.dump({"snapshot_pre_s": snap_s, "dcp_save_s": save_s}, fh)
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- stage B
def _stage_post(rank: int, spec: TransitionSpec, workdir: str, port: int) -> None:
    _init_pg(rank, spec.world_post, port)
    try:
        t0 = time.perf_counter()
        model_sd, optim_sd = _build_template_state(spec)
        dcp.load({"model": model_sd, "optim": optim_sd}, checkpoint_id=os.path.join(workdir, "ckpt"))
        load_s = time.perf_counter() - t0

        with open(os.path.join(workdir, "progress.json")) as fh:
            progress = {k: int(v) for k, v in json.load(fh).items()}

        snapshot_kwargs: Dict[str, Any] = {}
        fault_note = ""
        fn = get_fault(spec.fault)
        if fn is not None:
            fault_note = fn(model_sd, optim_sd, progress, snapshot_kwargs)

        t0 = time.perf_counter()
        snap = _snapshot(
            model_sd, optim_sd, progress,
            meta={
                "side": "post", "world": spec.world_post, "layout": spec.layout_post,
                "fault": spec.fault or "", "fault_note": fault_note,
            },
            drop_fqns=snapshot_kwargs.get("drop_fqns"),
            swap_reduction_order=snapshot_kwargs.get("swap_reduction_order", False),
        )
        snap_s = time.perf_counter() - t0
        if rank == 0:
            _snapshot_to_json(snap, os.path.join(workdir, "post.json"))
            with open(os.path.join(workdir, "timings_post.json"), "w") as fh:
                json.dump({"snapshot_post_s": snap_s, "dcp_load_s": load_s}, fh)
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- driver
def run_transition(spec: TransitionSpec, workdir: Optional[str] = None) -> TransitionResult:
    """Run one certified transition; returns the gate decision and timings."""
    own_tmp = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="attest_")
    try:
        mp.spawn(_stage_pre, args=(spec, workdir, _free_port()), nprocs=spec.world_pre, join=True)
        mp.spawn(_stage_post, args=(spec, workdir, _free_port()), nprocs=spec.world_post, join=True)

        pre = _snapshot_from_json(os.path.join(workdir, "pre.json"))
        post = _snapshot_from_json(os.path.join(workdir, "post.json"))
        decision = certify_transition(pre, post)

        timings: Dict[str, float] = {}
        for name in ("timings_pre.json", "timings_post.json"):
            with open(os.path.join(workdir, name)) as fh:
                timings.update(json.load(fh))
        timings["certificate_check_s"] = decision.check_seconds
        timings["certificate_total_s"] = (
            timings.get("snapshot_pre_s", 0.0)
            + timings.get("snapshot_post_s", 0.0)
            + decision.check_seconds
        )
        timings["reshard_total_s"] = timings.get("dcp_save_s", 0.0) + timings.get("dcp_load_s", 0.0)
        return TransitionResult(spec=spec, decision=decision, timings=timings)
    finally:
        if own_tmp:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)

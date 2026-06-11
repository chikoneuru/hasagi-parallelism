"""Reshard controller for live mid-job layout reconfiguration.

The characterization paper treats the structural layout lever's reconfiguration
cost analytically (the resume model). This module is the executable counterpart:
it moves real training state across a parallelism layout change -- the canonical
case being data-parallel (a full per-rank replica, DDP) to fully-sharded (a flat
contiguous shard per rank, FSDP FULL_SHARD) and back -- so a harness can measure
the reconfiguration latency and energy and show the loss trajectory is continuous
across the flip. :func:`live_rewrap` performs that flip on a live model: it
tears down one wrapper (plain/DDP/FSDP), stands the next one up over the
verified state, and carries the optimizer state across.

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

import time
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
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    return list(flat.reshape(world, -1).contiguous())


def unshard_flat(shards: list[torch.Tensor], total_numel: int) -> torch.Tensor:
    """Concatenate shards and drop the padding back to ``total_numel``."""
    return torch.cat([s.reshape(-1) for s in shards])[:total_numel]


@dataclass
class ReshardCertificate:
    """Result of the verify-before-commit check over a reshard transition.

    ``timings`` (filled by :func:`live_rewrap`) breaks the rewrap into stages —
    wall-clock seconds without per-stage device sync, so the harness's bracketed
    window remains the authoritative stall figure.
    """

    ok: bool
    max_abs_diff: float
    n_params: int
    from_world: int
    to_world: int
    note: str = ""
    timings: dict | None = None


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
        return self.capture_state(model.state_dict(), from_world=from_world)

    def capture_state(self, state: Mapping[str, torch.Tensor],
                      from_world: int = 1) -> ParamManifest:
        """Like :meth:`capture`, from an already-extracted state dict (e.g. one
        gathered through FSDP's full-state-dict context, whose keys a plain
        ``model.state_dict()`` on the wrapper would not match)."""
        self._captured = {k: v.detach().cpu().clone() for k, v in state.items()}
        self._flat, self._manifest = flatten_state(self._captured)
        self.from_world = from_world
        return self._manifest

    def last_verified_state(self) -> dict[str, torch.Tensor] | None:
        """The captured baseline (CPU clones) — what abort-to-last-verified restores."""
        return self._captured

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


# --------------------------------------------------------------------------- #
# Live layout rewrap: tear down one parallelism wrapper, stand up the next one
# over the verified state, and carry the optimizer state across. torch.distributed
# imports stay inside the functions so this module keeps importing without a
# process group (and the wrappers are only touched when a layout asks for them).
# --------------------------------------------------------------------------- #

LAYOUTS = ("plain", "ddp", "fsdp")


def unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    """The inner plain module regardless of wrapper (plain / DDP / FSDP)."""
    return getattr(model, "module", model)


def wrap_layout(module: torch.nn.Module, layout: str, device: torch.device) -> torch.nn.Module:
    """Wrap a plain module in the given parallelism layout.

    ``plain`` returns the module unchanged. ``ddp``/``fsdp`` require an
    initialized process group; both are exercised for real even at world=1
    (DDP simply has no peer to bucket with, and FSDP degrades FULL_SHARD to
    NO_SHARD — the wrapper, flat-parameter handling, and state-dict machinery
    are the same code that runs multi-rank). FSDP needs CUDA.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    if layout == "plain":
        return module
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError(f"layout {layout!r} requires an initialized process group")
    if layout == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
        ids = None
        if device.type == "cuda":
            ids = [device.index if device.index is not None else torch.cuda.current_device()]
        return DDP(module, device_ids=ids)
    if layout == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
        from torch.distributed.fsdp import ShardingStrategy
        kwargs: dict = {"sharding_strategy": ShardingStrategy.FULL_SHARD,
                        "use_orig_params": True}
        if device.type == "cuda":
            kwargs["device_id"] = device
        return FSDP(module, **kwargs)
    raise AssertionError(f"unhandled layout {layout!r}")  # unreachable: validated above


def extract_full_state(model: torch.nn.Module, layout: str) -> dict[str, torch.Tensor]:
    """A clean-keyed full state dict on CPU, whatever the wrapper.

    For FSDP the state is gathered (unsharded) through the FULL_STATE_DICT
    context so the keys match the plain module's; for DDP/plain the inner
    module's state dict already has clean keys.
    """
    if layout == "fsdp":
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            sd = model.state_dict()
    else:
        sd = unwrap_module(model).state_dict()
    return {k: v.detach().cpu() for k, v in sd.items()}


def named_optim_state(model: torch.nn.Module, layout: str, optim) -> dict:
    """The optimizer state keyed by parameter FQN (the layout-portable form).

    A plain optimizer ``state_dict()`` keys entries by integer parameter index,
    which is meaningless across a rewrap; FSDP's optim APIs speak FQNs. For an
    FSDP model the state is additionally gathered (unsharded) here.
    """
    if layout == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
        return FSDP.optim_state_dict(model, optim)
    names = [n for n, _ in unwrap_module(model).named_parameters()]
    osd = optim.state_dict()
    return {
        "state": {names[i]: v for i, v in osd["state"].items()},
        "param_groups": [{**g, "params": [names[i] for i in g["params"]]}
                         for g in osd["param_groups"]],
    }


def load_named_optim_state(model: torch.nn.Module, layout: str, optim, named: dict) -> None:
    """Load FQN-keyed optimizer state into an optimizer over ``model``'s params.

    The optimizer must have been constructed over the post-rewrap parameters
    with the same param-group structure as the source.
    """
    if not named.get("state"):
        return  # stateless source (e.g. momentum-free SGD): nothing to carry
    if layout == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
        optim.load_state_dict(FSDP.optim_state_dict_to_load(model, optim, named))
        return
    names = [n for n, _ in unwrap_module(model).named_parameters()]
    index = {n: i for i, n in enumerate(names)}
    optim.load_state_dict({
        "state": {index[n]: v for n, v in named["state"].items()},
        "param_groups": [{**g, "params": [index[n] for n in g["params"]]}
                         for g in named["param_groups"]],
    })


def _agree_certificate(cert: ReshardCertificate, device: torch.device) -> ReshardCertificate:
    """Make the commit/abort decision identical on every rank.

    A certificate computed from rank-local state can differ across ranks; if
    one rank commits while another aborts, the two post different collective
    sequences and the job deadlocks (NCCL hangs until the watchdog fires).
    All-reducing the failure bit (and the worst observed diff) before branching
    makes the whole group take the same path.
    """
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return cert
    t = torch.tensor([0.0 if cert.ok else 1.0, cert.max_abs_diff],
                     device="cuda" if dist.get_backend() == "nccl" else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if t[0].item() > 0 and cert.ok:
        cert.ok = False
        cert.note = (cert.note + " [peer-rank failed]").strip()
    cert.max_abs_diff = max(cert.max_abs_diff, float(t[1].item()))
    return cert


def live_rewrap(
    model: torch.nn.Module,
    optim,
    *,
    layout_from: str,
    layout_to: str,
    to_world: int,
    device: torch.device,
    optim_factory,
    module_factory=None,
    atol: float = 1e-6,
    from_world: int = 1,
):
    """Flip a live model from one parallelism layout to another, verify-before-commit.

    Mechanics: capture the full state out of the current wrapper; run the
    controller's flat-shard transport plan to ``to_world`` and commit the
    reassembled state into a target module; stand the new wrapper up over it;
    gather the post-rewrap state back out and verify it against the captured
    baseline (this second check is what catches the wrapper itself mangling
    state). The optimizer is rebuilt over the new wrapper's parameters via
    ``optim_factory`` (which must mirror the original construction) and its
    state is carried across through the FQN-keyed form.

    ``module_factory`` (callable returning a fresh module on ``device``) is
    required when ``layout_from == 'fsdp'`` — FSDP owns its inner module's
    storage, so the state must be stood up on fresh storage — and enables true
    abort-to-last-verified: on a failed certificate the baseline state is
    reloaded into a fresh module rewrapped in the ORIGINAL layout. Without it,
    a pre-teardown failure leaves the input model untouched and a post-wrap
    failure raises, since there is nothing safe to hand back.

    Multi-rank semantics (torchrun): every rank runs this same function. The
    flat-plan transport is rank-local; the cross-rank state movement is done by
    the wrappers themselves — FSDP shards the committed full state across the
    process group (a full-replica gather/scatter pipe, not flat-shard
    transport), and the DDP constructor broadcasts rank 0's parameters. The
    ``[transport]`` certificate therefore checks a rank-local round-trip, and
    the ``[post-rewrap]`` certificate is the one that exercises the cross-rank
    machinery. Both decisions are all-reduced across the group so every rank
    commits or aborts together. Note the DDP broadcast means a plain->ddp flip
    at world>1 is state-preserving only if all ranks start rank-identical
    (anything trained under DDP/FSDP already is; a freshly built unseeded
    module is not, and the certificate will catch exactly that).

    Returns ``(model, optim, certificate)``. ``certificate.timings`` carries a
    per-stage wall-clock breakdown (no per-stage device sync — the caller's
    bracketed window stays authoritative).
    """
    if layout_from not in LAYOUTS or layout_to not in LAYOUTS:
        raise ValueError(f"layouts must be in {LAYOUTS}: {layout_from!r} -> {layout_to!r}")
    if layout_from == "fsdp" and module_factory is None:
        raise ValueError("module_factory is required when rewrapping from fsdp "
                         "(FSDP owns the inner module's storage)")
    import torch.distributed as dist
    if (layout_to in ("ddp", "fsdp") and dist.is_available() and dist.is_initialized()
            and to_world != dist.get_world_size()):
        raise ValueError(
            f"to_world={to_world} != process-group world={dist.get_world_size()}: the "
            f"{layout_to} wrapper shards across the existing group; changing the group "
            f"size is a restart/checkpoint operation, not a live rewrap")

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    rc = ReshardController(atol=atol)
    rc.capture_state(extract_full_state(model, layout_from), from_world=from_world)
    carried = named_optim_state(model, layout_from, optim) if optim is not None else None
    timings["snapshot_s"] = time.perf_counter() - t0

    def _abort(cert: ReshardCertificate):
        if module_factory is None:
            cert.timings = timings
            return model, optim, cert
        restored = module_factory()
        restored.load_state_dict(rc.last_verified_state(), strict=True)
        rewrapped = wrap_layout(restored.to(device), layout_from, device)
        new_optim = optim_factory(rewrapped.parameters())
        if carried is not None:
            load_named_optim_state(rewrapped, layout_from, new_optim, carried)
        cert.timings = timings
        return rewrapped, new_optim, cert

    # 1) flat-shard transport plan to the target world, verified (rank-local)
    t0 = time.perf_counter()
    flat = unshard_flat(rc.plan_shards(to_world), rc._manifest.total_numel)
    reassembled = unflatten_state(flat, rc._manifest)
    cert = rc.verify(reassembled, to_world)
    timings["transport_verify_s"] = time.perf_counter() - t0
    if not cert.ok:
        cert.note = (cert.note + " [transport]").strip()
    cert = _agree_certificate(cert, device)
    if not cert.ok:
        return _abort(cert)

    # 2) commit into the target module and stand the new wrapper up over it
    t0 = time.perf_counter()
    target = module_factory() if module_factory is not None else unwrap_module(model)
    target.load_state_dict(reassembled, strict=True)
    new_model = wrap_layout(target.to(device), layout_to, device)
    timings["commit_wrap_s"] = time.perf_counter() - t0

    # 3) gather back out through the NEW wrapper and re-verify (catches the
    #    wrapper mangling/renaming state, not just the transport)
    t0 = time.perf_counter()
    cert = rc.verify(extract_full_state(new_model, layout_to), to_world)
    timings["post_verify_s"] = time.perf_counter() - t0
    if not cert.ok:
        cert.note = (cert.note + " [post-rewrap]").strip()
    cert = _agree_certificate(cert, device)
    if not cert.ok:
        if module_factory is None:
            raise RuntimeError(f"post-rewrap verification failed with no module_factory "
                               f"to abort to: {cert}")
        return _abort(cert)

    # 4) optimizer over the new wrapper's params, state carried across
    t0 = time.perf_counter()
    new_optim = optim_factory(new_model.parameters())
    if carried is not None:
        load_named_optim_state(new_model, layout_to, new_optim, carried)
    timings["optim_rebuild_s"] = time.perf_counter() - t0
    cert.timings = timings
    return new_model, new_optim, cert

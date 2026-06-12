"""Certificate-as-oracle sweep over distributed-checkpoint transition configs.

Every cell in this sweep performs a transition that is SUPPOSED to be clean:
save real state through ``torch.distributed.checkpoint`` (DCP) under one
configuration, load it back, and certify the loaded state against a snapshot
taken at save-call time. The certificate is the oracle: a cell that commits
behaved; a cell that aborts WITHOUT raising is a silent-divergence candidate,
the class that ships to users. Loud failures on documented-supported paths
are recorded separately.

The sweep is seeded by the neighborhoods of two documented DCP issues and
fans out over their variants:

  * pytorch/pytorch#126881 (open): in-place load resolves what to read from
    the TEMPLATE state dict, so entries absent from the template are silently
    not restored. Family A varies what the template is missing or mis-typing.
  * pytorch/pytorch#144657 (fixed by #145408): a mutation landing between
    ``async_save`` and the completion of its future used to be written to
    the checkpoint. Family B re-opens the mutation window over object kinds
    the fix may not stage (nested non-tensors, aliases, optimizer-style
    nests).
  * Family C probes save/load fidelity corners: aliased storages, strided
    views, dtype preservation, empty and zero-dim tensors.

Run the same script under different torch versions; the artifact records the
version so calibration arms (a known-buggy torch must produce ABORTs where
the documented bug lives) validate the oracle before any new finding is
trusted on a current torch::

    .venv/sweep/bin/python exp_attest_oracle_sweep.py \\
        --out artifacts/attest_oracle_sweep_$(python -c 'import torch; \\
        print(torch.__version__.split("+")[0])').json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
import traceback

import torch
import torch.distributed.checkpoint as dcp

from attest.gate import certify_transition
from attest.snapshot import snapshot_from_state_dicts


# --------------------------------------------------------------------------- #
# State builders: every cell starts from a freshly built, painted state.
# --------------------------------------------------------------------------- #
def _painted(shape, base):
    t = torch.full(shape, float(base))
    t += torch.arange(t.numel(), dtype=torch.float32).reshape(shape) * 1e-3
    return t


def build_state(kind: str = "plain") -> dict:
    """A state dict with the shapes real trainers persist: tensors, nested
    optimizer-style state, and non-tensor leaves."""
    state = {
        "model.weight": _painted((4, 3), 100),
        "model.bias": _painted((4,), 200),
        "optim": {
            "state": {
                "0": {"exp_avg": _painted((4, 3), 300),
                      "exp_avg_sq": _painted((4, 3), 400),
                      "step": torch.tensor(7.0)},
            },
            "param_groups": [{"lr": 1e-3}],
        },
        "meta": {"epoch": 3, "samples": 96},
    }
    if kind == "aliased":
        base = _painted((8, 3), 500)
        state["alias.full"] = base
        state["alias.view"] = base[2:6]
    if kind == "strided":
        state["strided.t"] = _painted((6, 4), 600).t()  # non-contiguous
    if kind == "dtypes":
        state["bf16.w"] = _painted((4, 3), 700).to(torch.bfloat16)
        state["int64.w"] = torch.arange(12).reshape(4, 3)
        state["empty.w"] = torch.empty(0)
        state["zerodim.w"] = torch.tensor(3.5)
    return state


def _flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}.{i}", v, out)
    else:
        out[prefix] = obj  # non-tensor leaf -> auxiliary stream


def snapshot(state: dict, progress=None):
    flat: dict = {}
    _flatten("", state, flat)
    tensors = {k: v for k, v in flat.items() if isinstance(v, torch.Tensor)}
    aux = {k: v for k, v in flat.items() if not isinstance(v, torch.Tensor)}
    return snapshot_from_state_dicts(tensors, progress=progress, aux_streams=aux)


def deep_clone(state):
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, dict):
        return {k: deep_clone(v) for k, v in state.items()}
    if isinstance(state, list):
        return [deep_clone(v) for v in state]
    if isinstance(state, tuple):
        return tuple(deep_clone(v) for v in state)
    return copy.deepcopy(state)


# --------------------------------------------------------------------------- #
# Cells. Each returns (post_state, notes) given a workdir; the harness
# snapshots before/after and certifies. A cell may also raise (recorded as
# loud, not silent).
# --------------------------------------------------------------------------- #
def cell_roundtrip(work, kind="plain"):
    state = build_state(kind)
    truth = deep_clone(state)
    dcp.save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"))
    template = deep_clone(state)
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, "plain save/load round-trip"


def cell_template_missing_subtree(work):
    # pytorch/pytorch#126881's shape: the lazily-initialized entry is absent
    # from the load template
    state = build_state()
    truth = deep_clone(state)
    dcp.save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"))
    template = deep_clone(state)
    template.pop("meta")
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, "template missing a non-tensor subtree (#126881)"


def cell_template_missing_tensor(work):
    state = build_state()
    truth = deep_clone(state)
    dcp.save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"))
    template = deep_clone(state)
    template.pop("model.bias")
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, "template missing a tensor key"


def cell_template_wrong_dtype(work, to=torch.float64):
    state = build_state()
    truth = deep_clone(state)
    dcp.save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"))
    template = deep_clone(state)
    template["model.weight"] = template["model.weight"].to(to)
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, f"template tensor declared {to} for a float32 ckpt"


def cell_template_wrong_shape(work):
    state = build_state()
    truth = deep_clone(state)
    dcp.save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"))
    template = deep_clone(state)
    template["model.weight"] = torch.zeros(3, 4)
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, "template tensor has transposed shape"


def cell_template_stale_extra_key(work):
    # the inverse of missing: the template carries a key the checkpoint never
    # saved; whatever survives in it is stale memory presented as restored
    state = build_state()
    truth = deep_clone(state)
    dcp.save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"))
    template = deep_clone(state)
    template["model.stale"] = _painted((2, 2), 900)
    truth["model.stale"] = None  # the checkpoint has nothing for it
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, "template carries a key the checkpoint lacks"


def cell_async_mutation_window(work, kind="plain", mutate="tensor"):
    # pytorch/pytorch#144657's shape: mutate state between async_save and
    # future completion, then load what the checkpoint actually stored
    state = build_state(kind)
    truth = deep_clone(state)
    fut = dcp.async_save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"))
    with torch.no_grad():
        if mutate == "tensor":
            state["model.weight"].add_(1000.0)
        elif mutate == "optim":
            state["optim"]["state"]["0"]["exp_avg"].add_(1000.0)
            state["optim"]["state"]["0"]["step"].add_(1.0)
        elif mutate == "nontensor":
            state["meta"]["epoch"] = 9999
        elif mutate == "alias":
            state["alias.view"].add_(1000.0)  # writes through shared storage
    fut.result()
    template = deep_clone(truth)
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, f"async_save with {mutate} mutation in the window (#144657)"



# --------------------------------------------------------------------------- #
# Multi-process resharding cells: save DTensor state at world=2, load it back
# under a different layout. Workers are top-level for mp.spawn.
# --------------------------------------------------------------------------- #
def _painted_global(rows: int = 4):
    return {
        "model.weight": _painted((rows, 3), 100),
        "optim.exp_avg": _painted((rows, 3), 300),
    }


def _dt_save_worker(rank, world, work, port, rows):
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(world),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    state = {k: distribute_tensor(v, mesh, [Shard(0)])
             for k, v in _painted_global(rows).items()}
    dcp.save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"))
    dist.destroy_process_group()


def _dt_load_shard1_worker(rank, world, work, port, rows):
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(world),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    template = {k: distribute_tensor(torch.zeros_like(v), mesh, [Shard(1)])
                for k, v in _painted_global(rows).items()}
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    if rank == 0:
        full = {k: v.full_tensor() for k, v in template.items()}
    else:
        full = {k: v.full_tensor() for k, v in template.items()}  # collective
    if rank == 0:
        torch.save(full, os.path.join(work, "loaded_full.pt"))
    dist.destroy_process_group()


def _spawn(fn, work, port, rows, world=2):
    import torch.multiprocessing as mp

    mp.spawn(fn, args=(world, work, port, rows), nprocs=world, join=True)


def cell_dtensor_reshard_to_full(work, rows=4, port=29590):
    truth = _painted_global(rows)
    _spawn(_dt_save_worker, work, port, rows)
    template = {k: torch.zeros_like(v) for k, v in truth.items()}
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, f"DTensor Shard(0) world=2 -> plain full load (rows={rows})"


def cell_dtensor_reshard_to_shard1(work, port=29592):
    truth = _painted_global(4)
    _spawn(_dt_save_worker, work, port, 4)
    _spawn(_dt_load_shard1_worker, work, port + 1, 4)
    loaded = torch.load(os.path.join(work, "loaded_full.pt"), weights_only=True)
    return truth, loaded, "DTensor Shard(0) world=2 -> Shard(1) world=2 reload"


def cell_async_process_checkpointer(work):
    # the process-based checkpointer requires an initialized process group
    # even single-rank, unlike every other path in this sweep
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType

    os.environ.update({"RANK": "0", "WORLD_SIZE": "1",
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29594"})
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        state = build_state()
        truth = deep_clone(state)
        fut = dcp.async_save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"),
                             async_checkpointer_type=AsyncCheckpointerType.PROCESS)
        with torch.no_grad():
            state["model.weight"].add_(1000.0)
        fut.result()
        template = deep_clone(truth)
        dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    finally:
        dist.destroy_process_group()
    return truth, template, "process-based async_save with mutation in the window"


def cell_async_explicit_stager(work):
    from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

    state = build_state()
    truth = deep_clone(state)
    stager = DefaultStager(StagingOptions(use_pinned_memory=False,
                                          use_shared_memory=True,
                                          use_async_staging=False,
                                          use_non_blocking_copy=False))
    fut = dcp.async_save(state_dict=state, checkpoint_id=os.path.join(work, "ckpt"),
                         async_stager=stager)
    with torch.no_grad():
        state["model.weight"].add_(1000.0)
        state["optim"]["state"]["0"]["exp_avg"].add_(1000.0)
    fut.result()
    try:
        stager.close()
    except Exception:
        pass
    template = deep_clone(truth)
    dcp.load(state_dict=template, checkpoint_id=os.path.join(work, "ckpt"))
    return truth, template, "async_save via explicit shared-memory stager, mutation in window"


CELLS = [
    ("roundtrip_plain", lambda w: cell_roundtrip(w, "plain")),
    ("roundtrip_aliased", lambda w: cell_roundtrip(w, "aliased")),
    ("roundtrip_strided", lambda w: cell_roundtrip(w, "strided")),
    ("roundtrip_dtypes", lambda w: cell_roundtrip(w, "dtypes")),
    ("inplace_missing_subtree", cell_template_missing_subtree),
    ("inplace_missing_tensor", cell_template_missing_tensor),
    ("inplace_dtype_upcast_f64", cell_template_wrong_dtype),
    ("inplace_dtype_downcast_bf16",
     lambda w: cell_template_wrong_dtype(w, torch.bfloat16)),
    ("inplace_dtype_downcast_f16",
     lambda w: cell_template_wrong_dtype(w, torch.float16)),
    ("inplace_wrong_shape", cell_template_wrong_shape),
    ("inplace_stale_extra_key", cell_template_stale_extra_key),
    ("async_mutate_tensor", lambda w: cell_async_mutation_window(w, mutate="tensor")),
    ("async_mutate_optim", lambda w: cell_async_mutation_window(w, mutate="optim")),
    ("async_mutate_nontensor", lambda w: cell_async_mutation_window(w, mutate="nontensor")),
    ("async_mutate_alias", lambda w: cell_async_mutation_window(w, "aliased", "alias")),
    ("reshard_dt_to_full", lambda w: cell_dtensor_reshard_to_full(w, 4, 29590)),
    ("reshard_dt_uneven_to_full", lambda w: cell_dtensor_reshard_to_full(w, 5, 29591)),
    ("reshard_dt_shard0_to_shard1", cell_dtensor_reshard_to_shard1),
    ("async_process_checkpointer", cell_async_process_checkpointer),
    ("async_explicit_stager", cell_async_explicit_stager),
]


def run(args: argparse.Namespace) -> int:
    results = []
    for name, fn in CELLS:
        work = tempfile.mkdtemp(prefix=f"oracle_{name}_")
        rec = {"cell": name}
        try:
            truth, loaded, note = fn(work)
            rec["note"] = note
            pre = snapshot(truth)
            post = snapshot(loaded)
            decision = certify_transition(pre, post)
            tflat_pre, tflat_post = {}, {}
            _flatten("", truth, tflat_pre)
            _flatten("", loaded, tflat_post)
            devs = [float((tflat_pre[k].double() - tflat_post[k].double()).abs().max())
                    for k in tflat_pre
                    if isinstance(tflat_pre.get(k), torch.Tensor)
                    and isinstance(tflat_post.get(k), torch.Tensor)
                    and tflat_pre[k].shape == tflat_post[k].shape
                    and tflat_pre[k].numel() > 0]
            rec.update({
                "max_value_dev_dtype_blind": max(devs) if devs else None,
                "raised": None,
                "committed": decision.committed,
                "n_violations": len(decision.violations),
                "violations_by_invariant": sorted({v.invariant for v in decision.violations}),
                "violating_fqns": sorted({v.fqn for v in decision.violations})[:8],
                "sample_violation": str(decision.violations[0]) if decision.violations else None,
                "silent_divergence": not decision.committed,
            })
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # DCP's CheckpointException subclasses
            # BaseException, so a plain `except Exception` misses exactly the
            # failures this sweep is probing
            rec.update({
                "raised": f"{type(e).__name__}: {e}",
                "traceback_tail": traceback.format_exc()[-600:],
                "silent_divergence": False,
            })
        finally:
            shutil.rmtree(work, ignore_errors=True)
        results.append(rec)

    findings = [r for r in results if r.get("silent_divergence")]
    report = {
        "exp": "attest-oracle-sweep",
        "torch": torch.__version__,
        "oracle": "transition certificate over snapshot(truth at save call) vs "
                  "snapshot(state after load); silent_divergence = certificate "
                  "aborts on a path that raised nothing",
        "cells": results,
        "n_cells": len(results),
        "n_silent_divergence": len(findings),
        "silent_cells": [r["cell"] for r in findings],
    }
    print(f"torch {torch.__version__}: {len(findings)}/{len(results)} cells "
          f"silently diverged")
    for r in results:
        if r.get("raised"):
            status = f"LOUD   {r['raised'][:70]}"
        elif r["committed"]:
            status = "COMMIT"
        else:
            status = (f"SILENT-ABORT {r['violations_by_invariant']} "
                      f"{r['violating_fqns'][:3]}")
        print(f"  {r['cell']:26s} {status}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

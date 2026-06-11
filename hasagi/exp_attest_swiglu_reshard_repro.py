"""Natural-bug reproduction: Megatron-LM's pre-fix SwiGLU sharding caught by the certificate.

Before NVIDIA/Megatron-LM PR#520 (merged 2023-10-04), the fused gate/up
projection weight of a gated-MLP (SwiGLU) layer was declared to distributed
checkpointing as ONE contiguous tensor-parallel-sharded tensor
(``megatron/core/transformer/transformer_layer.py`` at the PR's base commit
``ab0336a``). The fused local weight is the row-concatenation
``[gate_r; up_r]`` of the rank's gate and up chunks, so the declared global
tensor interleaves gate and up blocks by rank: saving at TP=2 stores
``[gate0; up0; gate1; up1]``. Loading that checkpoint at a different TP degree
re-splits the rows at wrong boundaries: a TP=1 load reconstructs
``fc1.weight`` as ``[gate0; up0; gate1; up1]`` where the model expects
``[gate0; gate1; up0; up1]`` -- a silent permutation of perfectly preserved
values. Keys, shapes, dtype, element count, and even the L2 norm all pass;
only byte-level content comparison can catch it. Restoring at the SAME TP
degree is bit-exact, so the bug fires exclusively on the resharding
transition. The shipped fix (today ``apply_swiglu_sharded_factory``) chunks
the fused weight into separately-sharded gate and up tensors before saving.

This experiment replays both declaration styles through the REAL
``megatron.core.dist_checkpointing`` pipeline (megatron-core 0.17.1, CPU,
gloo) and shows the transition certificate catches the pre-fix arm blind,
with zero injected faults:

  1. save    -- one gated MLP, built by megatron-core at TP=2 with
                recognizable per-row values, saved twice: once declaring fc1
                the pre-fix way (vendored verbatim from the PR#520 base
                commit), once through the current factory (the shipped fix).
  2. load    -- each checkpoint loaded back through the same declaration
                style at TP=1 (the resharding transition) and, for the
                pre-fix arm, also at TP=2 (the same-degree control).
  3. certify -- the certificate compares each loaded state against the
                logical full weights assembled from the live TP=2 module at
                save time. Expected: pre-fix TP-change ABORTs on
                content_equivalence; pre-fix same-TP and fixed TP-change
                COMMIT bit-exact.

The torch_dist checkpoint strategy in megatron-core 0.17.1 calls
``torch.cuda`` unconditionally in a few places (preload sync, failure-flag
device); on a CPU-only torch build those calls raise, so the stage processes
no-op them. They never touch checkpoint content.

Setup (one-time, ~2 min; the venv is local and gitignored)::

    python3 -m venv .venv/mcore
    .venv/mcore/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    .venv/mcore/bin/pip install megatron-core==0.17.1 psutil

Run::

    python exp_attest_swiglu_reshard_repro.py --out artifacts/attest_swiglu_reshard_repro.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MCORE_VENV = os.path.join(HERE, ".venv", "mcore")
HIDDEN = 6
FFN = 8  # fused fc1 weight is [2*FFN, HIDDEN] globally
TP_SAVE = 2
PORTS = {("save", "buggy"): 29565, ("save", "fixed"): 29566,
         ("load-tp1", "buggy"): 29567, ("load-tp1", "fixed"): 29568,
         ("load-tp2", "buggy"): 29569}


# --------------------------------------------------------------------------- #
# Stage helpers (these run inside the megatron-core venv).
# --------------------------------------------------------------------------- #
def _cpu_cuda_shim() -> None:
    """No-op the unconditional torch.cuda calls in mcore's torch_dist
    checkpoint strategy (stream sync + failure-flag device placement) that
    raise on a CPU-only torch build; checkpoint content is unaffected."""
    import torch

    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.current_device = lambda: "cpu"


def _init_parallel(rank: int, world: int, port: int, tp: int) -> None:
    os.environ.update({
        "RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
    })
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    _cpu_cuda_shim()


def _build_mlp(tp: int):
    import torch
    import torch.nn.functional as torch_fn
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(7)
    cfg = TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=2, ffn_hidden_size=FFN,
        gated_linear_unit=True, activation_func=torch_fn.silu, add_bias_linear=False,
        use_cpu_initialization=True, tensor_model_parallel_size=tp,
    )
    return MLP(cfg, MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                  linear_fc2=RowParallelLinear))


def _paint(mlp, rank: int, tp: int) -> None:
    """Give every row/column a value encoding its GLOBAL index (gate row g ->
    1000+g, up row u -> 2000+u, fc2 column c -> 3000+c, plus a small ramp
    over the other axis), so a mis-split is readable in the artifact."""
    import torch

    half = FFN // tp
    with torch.no_grad():
        w1 = mlp.linear_fc1.weight  # local fused [2*half, HIDDEN] = [gate_r; up_r]
        ramp = torch.arange(HIDDEN, dtype=w1.dtype) * 1e-3
        for i in range(half):
            w1[i] = 1000.0 + rank * half + i + ramp
            w1[half + i] = 2000.0 + rank * half + i + ramp
        w2 = mlp.linear_fc2.weight  # local [HIDDEN, half]
        ramp = torch.arange(HIDDEN, dtype=w2.dtype) * 1e-3
        for j in range(half):
            w2[:, j] = 3000.0 + rank * half + j + ramp


def _logical_full(per_rank: list) -> dict:
    """The layout-independent logical state: all gates, then all ups."""
    import torch

    fused = [s["fc1"] for s in per_rank]
    gates = torch.cat([f[: f.shape[0] // 2] for f in fused])
    ups = torch.cat([f[f.shape[0] // 2:] for f in fused])
    return {
        "mlp.linear_fc1.weight": torch.cat([gates, ups]),
        "mlp.linear_fc2.weight": torch.cat([s["fc2"] for s in per_rank], dim=1),
    }


# --------------------------------------------------------------------------- #
# The pre-fix declaration, vendored from NVIDIA/Megatron-LM commit
# ab0336a5c8eab77aa74ae604ba1e73decbf6d560 (the base of fix PR#520),
# megatron/core/transformer/transformer_layer.py::TransformerLayer
# .sharded_state_dict, with mechanical adaptations only: it takes the module
# as an argument instead of `self`; the single-layer values layer_number-1=0
# and _get_layer_offset()=0 are inlined (num_layers=1); the state dict is
# taken with prefix='mlp.' so the names match the original axis-map rows on a
# bare MLP; imports use megatron-core 0.17.1's paths. The sharding geometry
# -- ONE contiguous TP shard for the fused fc1 weight -- is untouched; that
# geometry is the bug.
# --------------------------------------------------------------------------- #
def _sharded_state_dict_pre_pr520(module, prefix: str = "") -> dict:
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing import ShardedTensor
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    state_dict = module.state_dict(prefix="mlp.", keep_vars=True)

    tensor_parallel_layers_axis_map = {
        'mlp.linear_fc1.weight': 0,
        'mlp.linear_fc1.bias': 0,
        'mlp.linear_fc2.weight': 1,
    }

    num_layers = 1
    global_layer_offset = 0
    sharded_state_dict = {}

    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{prefix}{global_layer_offset}.{layer_name}'
        sharded_offsets = [(0, global_layer_offset, num_layers)]  # PP sharding

        if layer_name in tensor_parallel_layers_axis_map:
            tp_axis = tensor_parallel_layers_axis_map[layer_name]
            # TP sharding
            sharded_offsets.append(
                [
                    tp_axis + 1,  # +1 for PP dimension
                    parallel_state.get_tensor_model_parallel_rank(),
                    parallel_state.get_tensor_model_parallel_world_size(),
                ]
            )
            replica_id = parallel_state.get_data_parallel_rank()
        else:
            replica_id = (
                parallel_state.get_data_parallel_rank()
                * parallel_state.get_data_parallel_world_size()
                + parallel_state.get_tensor_model_parallel_rank()
            )

        if layer_name.endswith('._extra_state'):
            sharded_state_dict[layer_key] = ShardedObject(
                f'{prefix}{layer_name}',
                tensor,
                (num_layers,),
                (global_layer_offset,),
                replica_id,
            )
        else:
            sharded_state_dict[layer_key] = ShardedTensor.from_rank_offsets(
                f'{prefix}{layer_name}',
                tensor,
                *sharded_offsets,
                replica_id=replica_id,
                prepend_axis_num=1,  # for PP sharding
            )

    return sharded_state_dict


def _sharded_state_dict_fixed(module) -> dict:
    """The shipped declaration: MLP.sharded_state_dict routes the fused fc1
    weight through apply_swiglu_sharded_factory (chunk into gate and up, shard
    each separately)."""
    from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

    return module.sharded_state_dict(prefix="mlp.",
                                     metadata=ensure_metadata_has_dp_cp_group(None))


def _declare(module, arm: str) -> dict:
    return (_sharded_state_dict_pre_pr520(module) if arm == "buggy"
            else _sharded_state_dict_fixed(module))


# --------------------------------------------------------------------------- #
# Stages (re-executed under the megatron-core venv).
# --------------------------------------------------------------------------- #
def _save_worker(rank: int, world: int, workdir: str, arm: str, port: int) -> None:
    import torch
    import torch.distributed as dist

    _init_parallel(rank, world, port, tp=world)
    from megatron.core import dist_checkpointing

    mlp = _build_mlp(tp=world)
    _paint(mlp, rank, world)
    local = {"fc1": mlp.linear_fc1.weight.detach().clone(),
             "fc2": mlp.linear_fc2.weight.detach().clone()}
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        # ground truth assembled from the live module tensors, not re-derived
        torch.save(_logical_full(gathered), os.path.join(workdir, f"truth_{arm}.pt"))
        os.makedirs(os.path.join(workdir, f"ckpt_{arm}"), exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(_declare(mlp, arm), os.path.join(workdir, f"ckpt_{arm}"))


def _stage_load_tp1(workdir: str, arm: str, port: int) -> None:
    import torch

    _init_parallel(0, 1, port, tp=1)
    from megatron.core import dist_checkpointing

    mlp = _build_mlp(tp=1)
    with torch.no_grad():  # placeholders: a silent no-op load must not pass
        mlp.linear_fc1.weight.fill_(-7.5)
        mlp.linear_fc2.weight.fill_(-7.5)
    loaded = dist_checkpointing.load(_declare(mlp, arm), os.path.join(workdir, f"ckpt_{arm}"))
    key_prefix = "0." if arm == "buggy" else ""  # the vendored declaration keys by layer index
    out = {k: loaded[f"{key_prefix}{k}"].detach().clone()
           for k in ("mlp.linear_fc1.weight", "mlp.linear_fc2.weight")}
    torch.save(out, os.path.join(workdir, f"loaded_{arm}_tp1.pt"))


def _load_tp2_worker(rank: int, world: int, workdir: str, port: int) -> None:
    import torch
    import torch.distributed as dist

    _init_parallel(rank, world, port, tp=world)
    from megatron.core import dist_checkpointing

    mlp = _build_mlp(tp=world)
    with torch.no_grad():
        mlp.linear_fc1.weight.fill_(-7.5)
        mlp.linear_fc2.weight.fill_(-7.5)
    loaded = dist_checkpointing.load(_declare(mlp, "buggy"),
                                     os.path.join(workdir, "ckpt_buggy"))
    local = {"fc1": loaded["0.mlp.linear_fc1.weight"].detach().clone(),
             "fc2": loaded["0.mlp.linear_fc2.weight"].detach().clone()}
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        torch.save(_logical_full(gathered),
                   os.path.join(workdir, "loaded_buggy_roundtrip_tp2.pt"))


# --------------------------------------------------------------------------- #
# Orchestration (runs under the project env; stages use the mcore venv).
# --------------------------------------------------------------------------- #
def _run(cmd: list, what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {r.returncode}):\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def _mcore_version(venv: str) -> str:
    r = subprocess.run([f"{venv}/bin/python", "-c",
                        "import megatron.core; print(megatron.core.__version__)"],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "unknown"


def run(args: argparse.Namespace) -> int:
    import torch

    from attest.gate import certify_transition
    from attest.snapshot import snapshot_from_state_dicts

    work = os.path.abspath(args.workdir)
    if args.fresh and os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    py = f"{args.mcore_venv}/bin/python"

    for stage, arm in (("save", "buggy"), ("save", "fixed"), ("load-tp1", "buggy"),
                       ("load-tp1", "fixed"), ("load-tp2", "buggy")):
        _run([py, os.path.abspath(__file__), "--stage", stage, "--arm", arm,
              "--workdir", work], f"stage {stage}/{arm}")

    truth = torch.load(os.path.join(work, "truth_buggy.pt"), weights_only=True)
    truth_fixed = torch.load(os.path.join(work, "truth_fixed.pt"), weights_only=True)
    for k in truth:  # both arms painted the same module the same way
        assert torch.equal(truth[k], truth_fixed[k]), f"truth diverged on {k}"

    # the mis-split the bug mechanism predicts: per-rank fused blocks in order
    fc1 = truth["mlp.linear_fc1.weight"]
    gates, ups = fc1[:FFN], fc1[FFN:]
    half = FFN // TP_SAVE
    predicted_missplit = torch.cat([gates[:half], ups[:half], gates[half:], ups[half:]])

    pre = snapshot_from_state_dicts(truth, meta={"side": "live TP=2 module at save time"})
    report: dict = {
        "exp": "attest-swiglu-reshard-repro",
        "bug": {"fix_pr": "NVIDIA/Megatron-LM#520", "base_commit": "ab0336a5",
                "mechanism": "fused SwiGLU gate/up weight declared as one contiguous "
                             "TP-sharded tensor; a TP-degree-changing load re-splits "
                             "rows at wrong boundaries, interleaving gate and up blocks",
                "injected_faults": 0},
        "megatron_core_version": _mcore_version(args.mcore_venv),
        "geometry": {"hidden": HIDDEN, "ffn": FFN, "tp_save": TP_SAVE},
        "arms": {},
    }
    arms = {
        "buggy_tp2_to_tp1": ("loaded_buggy_tp1.pt", False),
        "buggy_tp2_roundtrip": ("loaded_buggy_roundtrip_tp2.pt", True),
        "fixed_tp2_to_tp1": ("loaded_fixed_tp1.pt", True),
    }
    ok = True
    for name, (fname, expect_commit) in arms.items():
        post_state = torch.load(os.path.join(work, fname), weights_only=True)
        post = snapshot_from_state_dicts(post_state, meta={"side": name})
        decision = certify_transition(pre, post)
        devs = [float((truth[k] - post_state[k]).abs().max()) for k in truth]
        multiset = all(
            torch.equal(truth[k].reshape(-1).sort().values,
                        post_state[k].reshape(-1).sort().values)
            for k in truth
        )
        report["arms"][name] = {
            "committed": decision.committed,
            "expected_committed": expect_commit,
            "as_expected": decision.committed == expect_commit,
            "n_violations": len(decision.violations),
            "violations_by_invariant": sorted({v.invariant for v in decision.violations}),
            "sample_violation": str(decision.violations[0]) if decision.violations else None,
            "check_seconds": decision.check_seconds,
            "max_abs_dev": max(devs),
            "values_preserved_as_multiset": multiset,
            "fc1_matches_predicted_missplit": bool(
                torch.equal(post_state["mlp.linear_fc1.weight"], predicted_missplit)
            ),
        }
        ok = ok and (decision.committed == expect_commit)

    report["natural_bug_caught_blind"] = ok
    print("=" * 64)
    print(f"SwiGLU reshard natural-bug repro (megatron-core "
          f"{report['megatron_core_version']})")
    for name, arm in report["arms"].items():
        verdict = "COMMIT" if arm["committed"] else "ABORT"
        print(f"  {name:20s} -> {verdict:6s} ({arm['n_violations']} violations, "
              f"{arm['violations_by_invariant']}; max|dev|={arm['max_abs_dev']:.3g}, "
              f"multiset-intact={arm['values_preserved_as_multiset']}, "
              f"predicted-missplit={arm['fc1_matches_predicted_missplit']}) "
              f"expected_commit={arm['expected_committed']}")
    print(f"  natural bug caught blind, zero injected faults: {ok}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcore-venv", default=MCORE_VENV,
                   help="venv with megatron-core==0.17.1 (CPU torch)")
    p.add_argument("--workdir", default=os.path.join(HERE, "_ckpt", "swiglu_reshard_repro"))
    p.add_argument("--out", default=None)
    p.add_argument("--fresh", action="store_true", help="discard previous stage outputs")
    p.add_argument("--stage", default="all",
                   choices=["all", "save", "load-tp1", "load-tp2"],
                   help="internal: stages run inside the mcore venv")
    p.add_argument("--arm", default=None, choices=["buggy", "fixed"])
    args = p.parse_args()
    if args.stage != "all":
        import torch.multiprocessing as mp

        work = os.path.abspath(args.workdir)
        port = PORTS[(args.stage, args.arm)]
        if args.stage == "save":
            mp.spawn(_save_worker, args=(TP_SAVE, work, args.arm, port),
                     nprocs=TP_SAVE, join=True)
        elif args.stage == "load-tp1":
            _stage_load_tp1(work, args.arm, port)
        else:
            mp.spawn(_load_tp2_worker, args=(TP_SAVE, work, port),
                     nprocs=TP_SAVE, join=True)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

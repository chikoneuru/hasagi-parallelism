"""Natural-bug reproduction: Megatron-LM's pre-fix RNG sharding caught by the certificate.

Before NVIDIA/Megatron-LM PR#2658 (merged 2025-12-17), checkpointing declared
the per-rank RNG state to distributed checkpointing as one object sharded
over (pipeline, tensor) coordinates with the data-parallel rank as a replica
id (``megatron/training/checkpointing.py::get_rng_state`` at the PR's base
commit ``4bdd7b10e``). Under expert parallelism the ranks that share a
(pipeline, tensor) cell hold DIFFERENT generator streams, deliberately seeded
apart so that expert routing and dropout decorrelate, yet the declaration
calls them replicas of one object: the save keeps only the primary replica's
streams, and on load every rank in the cell restores that one stream. Each
restored stream is internally valid, every parameter and optimizer slot is
intact, training-progress counters survive, and the resumed job trains on
silently re-correlated randomness. The shipped fix adds the data-parallel
coordinate to the sharding so each rank's streams own their slot.

This experiment replays both declarations through the real
``megatron.core.dist_checkpointing`` pipeline (megatron-core 0.17.1, CPU,
gloo, two ranks, expert parallelism 2) and shows the auxiliary-stream
invariant catches the wrong restore blind while the other five invariants
correctly stay silent:

  1. seed     -- each expert rank seeds python/numpy/torch streams apart,
                 mirroring how megatron decorrelates expert ranks.
  2. save     -- the rank streams plus a small painted model are saved twice:
                 once declaring the RNG object the pre-fix way (vendored
                 verbatim from the PR#2658 base commit), once the fixed way.
  3. load     -- each checkpoint is loaded back at the same topology through
                 the same declaration style.
  4. certify  -- per arm, the pre snapshot holds each rank's true streams
                 keyed by logical expert coordinate; the post snapshot holds
                 what each rank actually restored. Expected: the pre-fix arm
                 ABORTs on aux_stream_residency only (the secondary expert
                 rank restored the primary's streams); the fixed arm COMMITs
                 bit-exact. Zero injected faults.

A functional epilogue draws one random tensor per rank from the restored
torch stream: under the pre-fix arm both expert ranks draw identical values,
the re-correlation the seeding existed to prevent; under the fix they stay
distinct.

Setup: the same pinned venv as the SwiGLU reproduction
(``exp_attest_swiglu_reshard_repro.py``).

Run::

    python exp_attest_rng_restore_repro.py --out artifacts/attest_rng_restore_repro.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MCORE_VENV = os.path.join(HERE, ".venv", "mcore")
EP = 2
PORTS = {"buggy": 29585, "fixed": 29586}
STREAM_KEYS = ("random_rng_state", "np_rng_state", "torch_rng_state")


def _cpu_cuda_shim() -> None:
    import torch

    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.current_device = lambda: "cpu"


# --------------------------------------------------------------------------- #
# The pre-fix declaration, vendored from NVIDIA/Megatron-LM commit
# 4bdd7b10e (the base of fix PR#2658),
# megatron/training/checkpointing.py::get_rng_state, with mechanical
# adaptations only: the surrounding training-args plumbing is dropped (the
# data_parallel_random_init branch is not taken, matching its default), the
# ``mpu`` alias is megatron.core.parallel_state, CUDA-only stream entries are
# guarded for a CPU host, and the torch_dist branch is kept verbatim. The
# sharding geometry -- one (pp, tp) cell with the data-parallel rank as a
# REPLICA id, so expert ranks' distinct streams collide into one object --
# is untouched; that geometry is the bug. The fixed declaration is the same
# function's ep_size > 1 branch after the fix: the data-parallel coordinate
# joins the sharding and every rank owns its slot.
# --------------------------------------------------------------------------- #
def _collect_rng_state() -> dict:
    import numpy as np
    import torch

    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state['cuda_rng_state'] = torch.cuda.get_rng_state()
    return rng_state


def _declare_rng(rng_state_list: list, arm: str):
    from megatron.core import parallel_state as mpu
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if arm == "buggy":
        return ShardedObject('rng_state', rng_state_list, (pp_size, tp_size),
                             (pp_rank, tp_rank),
                             replica_id=mpu.get_data_parallel_rank(with_context_parallel=True))
    dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
    return ShardedObject(
        'rng_state',
        rng_state_list,
        (pp_size, tp_size, dp_size),
        (pp_rank, tp_rank, dp_rank),
        replica_id=0,
    )


# --------------------------------------------------------------------------- #
# Stage (re-executed under the megatron-core venv, two ranks, EP=2).
# --------------------------------------------------------------------------- #
def _roundtrip_worker(rank: int, world: int, workdir: str, arm: str, port: int) -> None:
    os.environ.update({
        "RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
    })
    import numpy as np
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core import dist_checkpointing, parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=EP)
    _cpu_cuda_shim()
    ep_rank = parallel_state.get_expert_model_parallel_rank()

    # seed the rank's streams apart, as megatron seeds expert ranks apart
    seed = 1234 + 100 * ep_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rng_state = _collect_rng_state()
    truth = {f"ep{ep_rank}.{k}": rng_state[k] for k in STREAM_KEYS}
    torch.save(truth, os.path.join(workdir, f"truth_{arm}_rank{rank}.pt"))

    # a small painted model so the content invariants have real work to do
    model_local = torch.full((4, 3), 500.0 + ep_rank) + torch.arange(3) * 1e-3
    from megatron.core.dist_checkpointing import ShardedTensor

    sharded = {
        "model.weight": ShardedTensor.from_rank_offsets(
            "model.weight", model_local, (0, rank, world)),
        "rng_state": _declare_rng([rng_state], arm),
    }
    ckpt = os.path.join(workdir, f"ckpt_{arm}")
    if rank == 0:
        os.makedirs(ckpt, exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(sharded, ckpt)
    dist.barrier()

    # reload at the same topology through the same declaration style
    placeholder = {
        "model.weight": ShardedTensor.from_rank_offsets(
            "model.weight", torch.zeros_like(model_local), (0, rank, world)),
        "rng_state": _declare_rng([_collect_rng_state()], arm),
    }
    loaded = dist_checkpointing.load(placeholder, ckpt)
    restored_list = loaded["rng_state"]
    restored = {f"ep{ep_rank}.{k}": restored_list[0][k] for k in STREAM_KEYS}
    torch.save(restored, os.path.join(workdir, f"restored_{arm}_rank{rank}.pt"))
    torch.save({"model.weight": loaded["model.weight"]},
               os.path.join(workdir, f"model_{arm}_rank{rank}.pt"))

    # functional epilogue: what the next dropout mask would be built from
    torch.set_rng_state(restored_list[0]["torch_rng_state"])
    torch.save(torch.randn(4), os.path.join(workdir, f"draw_{arm}_rank{rank}.pt"))


def _stage(workdir: str, arm: str) -> None:
    import torch.multiprocessing as mp

    mp.spawn(_roundtrip_worker, args=(EP, workdir, arm, PORTS[arm]),
             nprocs=EP, join=True)


# --------------------------------------------------------------------------- #
# Orchestration (runs under the project env; stages use the mcore venv).
# --------------------------------------------------------------------------- #
def _run(cmd: list, what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {r.returncode}):\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def run(args: argparse.Namespace) -> int:
    import torch

    from attest.gate import certify_transition
    from attest.snapshot import snapshot_from_state_dicts

    work = os.path.abspath(args.workdir)
    if args.fresh and os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    py = f"{args.mcore_venv}/bin/python"

    report: dict = {
        "exp": "attest-rng-restore-repro",
        "bug": {"fix_pr": "NVIDIA/Megatron-LM#2658", "base_commit": "4bdd7b10e",
                "mechanism": "RNG state declared as one (pp, tp) object with the "
                             "data-parallel rank as replica id; expert ranks' "
                             "distinct streams collide, the save keeps the primary "
                             "replica, every rank restores it",
                "injected_faults": 0},
        "topology": {"world": EP, "tp": 1, "pp": 1, "ep": EP},
        "arms": {},
    }
    expected = {"buggy": False, "fixed": True}
    ok = True
    for arm in ("buggy", "fixed"):
        _run([py, os.path.abspath(__file__), "--stage", "roundtrip",
              "--arm", arm, "--workdir", work], f"stage roundtrip/{arm}")
        truth, restored, model_pre, model_post = {}, {}, {}, {}
        for rank in range(EP):
            truth.update(torch.load(os.path.join(work, f"truth_{arm}_rank{rank}.pt"),
                                    weights_only=False))
            restored.update(torch.load(os.path.join(work, f"restored_{arm}_rank{rank}.pt"),
                                       weights_only=False))
            m = torch.load(os.path.join(work, f"model_{arm}_rank{rank}.pt"),
                           weights_only=False)
            model_pre[f"rank{rank}.model.weight"] = (
                torch.full((4, 3), 500.0 + rank) + torch.arange(3) * 1e-3)
            model_post[f"rank{rank}.model.weight"] = m["model.weight"]

        pre = snapshot_from_state_dicts(model_pre, progress={"global_step": 1},
                                        aux_streams=truth)
        post = snapshot_from_state_dicts(model_post, progress={"global_step": 1},
                                         aux_streams=restored)
        decision = certify_transition(pre, post)
        draws = [torch.load(os.path.join(work, f"draw_{arm}_rank{r}.pt"),
                            weights_only=True) for r in range(EP)]
        report["arms"][arm] = {
            "committed": decision.committed,
            "expected_committed": expected[arm],
            "as_expected": decision.committed == expected[arm],
            "n_violations": len(decision.violations),
            "violations_by_invariant": sorted({v.invariant for v in decision.violations}),
            "violating_streams": sorted({v.fqn for v in decision.violations
                                         if v.invariant == "aux_stream_residency"}),
            "sample_violation": str(decision.violations[0]) if decision.violations else None,
            "other_invariants_silent": all(v.invariant == "aux_stream_residency"
                                           for v in decision.violations),
            "expert_ranks_draw_identical_randomness": bool(
                torch.equal(draws[0], draws[1])),
        }
        ok = ok and (decision.committed == expected[arm])

    b = report["arms"]["buggy"]
    ok = ok and b["other_invariants_silent"] and b["expert_ranks_draw_identical_randomness"]
    ok = ok and not report["arms"]["fixed"]["expert_ranks_draw_identical_randomness"]
    report["natural_bug_caught_blind"] = ok

    print("=" * 64)
    print("RNG-restore natural-bug repro (megatron-core dist_checkpointing)")
    for arm, a in report["arms"].items():
        verdict = "COMMIT" if a["committed"] else "ABORT"
        print(f"  {arm:6s} -> {verdict:6s} ({a['n_violations']} violations, "
              f"{a['violations_by_invariant']}; streams={a['violating_streams']}; "
              f"identical-draws={a['expert_ranks_draw_identical_randomness']}) "
              f"expected_commit={a['expected_committed']}")
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
    p.add_argument("--mcore-venv", default=MCORE_VENV)
    p.add_argument("--workdir", default=os.path.join(HERE, "_ckpt", "rng_restore_repro"))
    p.add_argument("--out", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--stage", default="all", choices=["all", "roundtrip"])
    p.add_argument("--arm", default=None, choices=["buggy", "fixed"])
    args = p.parse_args()
    if args.stage == "roundtrip":
        _stage(os.path.abspath(args.workdir), args.arm)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Natural-bug reproduction: DeepSpeed's zero_to_fp32.py caught by the transition certificate.

DeepSpeed v0.16.0 shipped a consolidation bug
(https://github.com/deepspeedai/DeepSpeed/issues/6791, fixed in PR#6792): in
``zero_to_fp32.py``, ``to_torch_tensor(state_dict, return_empty_tensor=True)``
mutates its input, replacing every (lazy) tensor with ``torch.empty`` while
computing the shard layout, so the writer then saves uninitialized memory.
The output is well-formed (index json + shards), the script exits 0, and the
CLI's DEFAULT flags take the same path (``--max_shard_size`` defaults to 5GB),
so every standard conversion silently corrupts. Loss-curve validation cannot
see it: the corruption happens at export, after training.

This experiment reproduces the bug through the REAL pipeline and shows the
certificate catches it blind, with no injected fault anywhere:

  1. train   -- real DeepSpeed ZeRO-2 training on CPU (gloo, 2 ranks, client
                torch Adam) under a version-pinned venv; ``save_checkpoint``
                writes a real ZeRO checkpoint (and DeepSpeed itself copies
                ``zero_to_fp32.py`` into it -- the exact artifact users run).
  2. convert -- three conversions of that one checkpoint: the checkpoint's own
                v0.16.0 script with a small ``--max_shard_size`` (multi-shard),
                the same script with DEFAULT flags, and the v0.16.1 (fixed)
                script from a second pinned venv.
  3. certify -- the transition certificate compares each converted state dict
                against a snapshot of the live module taken at save time
                (ground truth independent of any converter). Expected:
                both v0.16.0 arms ABORT on content_equivalence; v0.16.1 COMMITs.

Setup (one-time, ~2 min; the venvs are local and gitignored)::

    python3 -m venv .venv/ds0160 && python3 -m venv .venv/ds0161
    .venv/ds0160/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    .venv/ds0160/bin/pip install deepspeed==0.16.0 huggingface_hub safetensors ninja
    .venv/ds0161/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    .venv/ds0161/bin/pip install deepspeed==0.16.1 huggingface_hub safetensors ninja

Run::

    python exp_attest_zero_to_fp32_repro.py --out artifacts/attest_zero_to_fp32_repro.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUGGY_VENV = os.path.join(HERE, ".venv", "ds0160")
FIXED_VENV = os.path.join(HERE, ".venv", "ds0161")
TAG = "step3"


# --------------------------------------------------------------------------- #
# Stage: train (re-executed under the version-pinned DeepSpeed venv).
# --------------------------------------------------------------------------- #
def _build_model(seed: int = 0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64), nn.Linear(64, 8))


def _train_worker(rank: int, world: int, ckpt_dir: str) -> None:
    os.environ.update({
        "RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29563",
    })
    import deepspeed
    import torch

    deepspeed.init_distributed(dist_backend="gloo")
    model = _build_model()
    config = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 4,
        "zero_optimization": {"stage": 2},
    }
    # a client torch optimizer avoids DeepSpeed JIT-compiling its fused CPUAdam
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=config)
    torch.manual_seed(100 + rank)
    for _ in range(3):
        x = torch.randn(4, 64)
        loss = engine(x).pow(2).mean()
        engine.backward(loss)
        engine.step()
    if rank == 0:
        truth = {k: v.detach().clone().float() for k, v in engine.module.state_dict().items()}
        torch.save(truth, os.path.join(ckpt_dir, "live_truth.pt"))
    engine.save_checkpoint(ckpt_dir, tag=TAG)


def _stage_train(ckpt_dir: str) -> None:
    import torch.multiprocessing as mp

    os.makedirs(ckpt_dir, exist_ok=True)
    mp.spawn(_train_worker, args=(2, ckpt_dir), nprocs=2, join=True)


# --------------------------------------------------------------------------- #
# Orchestration (runs under the project env; subprocesses use the pinned venvs).
# --------------------------------------------------------------------------- #
def _venv_env(venv: str) -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "")  # ninja for the shm comm op
    env["DS_ACCELERATOR"] = "cpu"
    return env


def _run(cmd: list, env: dict, what: str) -> None:
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def _ds_version(venv: str) -> str:
    r = subprocess.run([f"{venv}/bin/python", "-c",
                        "import deepspeed; print(deepspeed.__version__)"],
                       env=_venv_env(venv), capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "unknown"


def _load_converted(out_dir: str) -> dict:
    import torch

    out: dict = {}
    files = sorted(glob.glob(os.path.join(out_dir, "pytorch_model*.bin")))
    for f in files:
        if f.endswith("index.json"):
            continue
        out.update(torch.load(f, map_location="cpu", weights_only=False))
    return out


def _diagnostics(truth: dict, converted: dict) -> dict:
    missing = [k for k in truth if k not in converted]
    devs, zero_fracs = [], []
    for k in truth:
        if k not in converted:
            continue
        devs.append(float((truth[k].float() - converted[k].float()).abs().max()))
        zero_fracs.append(float((converted[k] == 0).float().mean()))
    return {
        "missing_keys": missing,
        "max_abs_dev": max(devs) if devs else None,
        "mean_zero_frac": sum(zero_fracs) / len(zero_fracs) if zero_fracs else None,
    }


def run(args: argparse.Namespace) -> int:
    from attest.gate import certify_transition
    from attest.snapshot import snapshot_from_state_dicts

    import torch

    work = os.path.abspath(args.workdir)
    ckpt = os.path.join(work, "ckpt")
    if args.fresh and os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)

    versions = {"buggy": _ds_version(args.ds_venv), "fixed": _ds_version(args.fixed_venv)}

    # 1) real ZeRO-2 training + checkpoint under the pinned (buggy) DeepSpeed
    if not os.path.exists(os.path.join(ckpt, "live_truth.pt")):
        t0 = time.perf_counter()
        _run([f"{args.ds_venv}/bin/python", os.path.abspath(__file__),
              "--stage", "train", "--workdir", work],
             _venv_env(args.ds_venv), "ZeRO-2 CPU training")
        print(f"trained + checkpointed in {time.perf_counter() - t0:.1f}s")
    ckpt_script = os.path.join(ckpt, "zero_to_fp32.py")
    assert os.path.exists(ckpt_script), "DeepSpeed should have copied zero_to_fp32.py into the checkpoint"

    # 2) three conversions of the SAME checkpoint
    fixed_script = os.path.join(work, "zero_to_fp32_fixed.py")
    shutil.copy(os.path.join(args.fixed_venv, "lib/python3.12/site-packages/deepspeed/utils/zero_to_fp32.py"),
                fixed_script)
    arms = {
        "buggy_sharded": ([f"{args.ds_venv}/bin/python", ckpt_script, ckpt,
                           os.path.join(work, "out_buggy_sharded"), "-t", TAG,
                           "--max_shard_size", "10KB"], args.ds_venv),
        "buggy_default": ([f"{args.ds_venv}/bin/python", ckpt_script, ckpt,
                           os.path.join(work, "out_buggy_default"), "-t", TAG], args.ds_venv),
        "fixed": ([f"{args.fixed_venv}/bin/python", fixed_script, ckpt,
                   os.path.join(work, "out_fixed"), "-t", TAG,
                   "--max_shard_size", "10KB"], args.fixed_venv),
    }
    for name, (cmd, venv) in arms.items():
        _run(cmd, _venv_env(venv), f"conversion {name}")

    # 3) certify each conversion against the live-module ground truth
    truth = torch.load(os.path.join(ckpt, "live_truth.pt"), map_location="cpu",
                       weights_only=False)
    pre = snapshot_from_state_dicts(truth, meta={"side": "live module at save time",
                                                 "world": 2, "zero_stage": 2})
    report: dict = {
        "exp": "attest-zero-to-fp32-repro",
        "bug": {"issue": "deepspeedai/DeepSpeed#6791", "fix_pr": "#6792",
                "mechanism": "to_torch_tensor(return_empty_tensor=True) mutates the "
                             "consolidated state dict; shards are written from the "
                             "emptied tensors (uninitialized memory)",
                "injected_faults": 0},
        "deepspeed_versions": versions,
        "n_params": len(truth),
        "arms": {},
    }
    expected = {"buggy_sharded": False, "buggy_default": False, "fixed": True}
    ok = True
    for name in arms:
        out_dir = os.path.join(work, f"out_{name}")
        converted = _load_converted(out_dir)
        post = snapshot_from_state_dicts(converted, meta={"side": f"zero_to_fp32 {name}"})
        decision = certify_transition(pre, post)
        diag = _diagnostics(truth, converted)
        report["arms"][name] = {
            "committed": decision.committed,
            "expected_committed": expected[name],
            "as_expected": decision.committed == expected[name],
            "n_violations": len(decision.violations),
            "violations_by_invariant": sorted({v.invariant for v in decision.violations}),
            "sample_violation": str(decision.violations[0]) if decision.violations else None,
            "check_seconds": decision.check_seconds,
            **diag,
        }
        ok = ok and (decision.committed == expected[name])

    report["natural_bug_caught_blind"] = ok
    print("=" * 64)
    print(f"zero_to_fp32 natural-bug repro (deepspeed {versions['buggy']} vs {versions['fixed']})")
    for name, arm in report["arms"].items():
        verdict = "COMMIT" if arm["committed"] else "ABORT"
        print(f"  {name:14s} -> {verdict:6s} ({arm['n_violations']} violations, "
              f"{arm['violations_by_invariant']}; max|dev|={arm['max_abs_dev']:.3g}, "
              f"zero-frac={arm['mean_zero_frac']:.2f}) expected_commit={arm['expected_committed']}")
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
    p.add_argument("--ds-venv", default=BUGGY_VENV, help="venv with deepspeed==0.16.0")
    p.add_argument("--fixed-venv", default=FIXED_VENV, help="venv with deepspeed==0.16.1")
    p.add_argument("--workdir", default=os.path.join(HERE, "_ckpt", "zero_to_fp32_repro"))
    p.add_argument("--out", default=None)
    p.add_argument("--fresh", action="store_true", help="discard any previous checkpoint")
    p.add_argument("--stage", default="all", choices=["all", "train"],
                   help="internal: 'train' runs inside the pinned venv")
    args = p.parse_args()
    if args.stage == "train":
        _stage_train(os.path.join(os.path.abspath(args.workdir), "ckpt"))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Live reproducibility break in HF Trainer mid-epoch resume, caught as a stream misuse.

huggingface/transformers#39215 reported that the ``Trainer`` resume path
fetches the first batch before restoring the checkpoint's RNG state, so any
randomness the data pipeline consumes for that batch draws from the wrong
stream and bit-reproducible resume silently breaks. The issue was closed by
the stale bot in August 2025 without a fix, and the ordering is still
present in transformers 5.11.0: on a mid-epoch resume the trainer skips
consumed batches, calls ``get_batch_samples`` for the first resumed update
step, and only then runs ``_load_rng_state`` (the ``rng_to_sync`` block).
The epoch-boundary resume path restores the RNG state before any fetch, so
it serves as the in-framework control.

This experiment measures the break end to end with a stochastic dataset
(every ``__getitem__`` draws from the global torch stream, the in-batch
augmentation case from the report):

  full          one uninterrupted run; every random draw is recorded in
                order, and a checkpoint is saved mid-epoch at step K.
  resume        a fresh process resumes from that checkpoint; the draws it
                produces for the resumed steps should be bit-identical to
                the full run's draws for the same steps.
  resume_epoch  the control: the checkpoint falls exactly on an epoch
                boundary, taking the early-restore path.

The certificate sees the cause, not just the symptom: the auxiliary-stream
snapshot pairs the torch RNG state the checkpoint stored (what
``_load_rng_state`` will restore) against the live global stream at the
moment the first resumed batch is materialized. On the mid-epoch path they
differ (the stream was consumed by batch skipping and the fetch happens
before the restore); on the epoch-boundary path they match.

Run::

    python exp_attest_hf_resume_repro.py --out artifacts/attest_hf_resume_repro.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_VENV = os.path.join(HERE, ".venv", "sweep")
BS = 2
DATASET_LEN = 64  # 32 steps per epoch at batch size 2


# --------------------------------------------------------------------------- #
# Stage: one Trainer run (executed under the sweep venv).
# --------------------------------------------------------------------------- #
def _stage_train(workdir: str, mode: str, save_step: int, max_steps: int,
                 capture_at: int) -> None:
    import torch
    from torch import nn
    from transformers import Trainer, TrainingArguments, set_seed

    records = []          # every random draw, in production order
    capture = {}          # global stream state at selected production indices

    class StochasticDataset(torch.utils.data.Dataset):
        """The in-batch augmentation case: each item consumes global RNG."""

        def __len__(self):
            return DATASET_LEN

        def __getitem__(self, idx):
            if len(records) == capture_at:
                # the stream state from which the first post-checkpoint item
                # will draw; reproducible resume requires the resumed run to
                # reach this exact state before producing that item
                capture["stream_at_first_resumed_item"] = torch.get_rng_state()
            draw = torch.rand(4)
            records.append(draw)
            return {"x": torch.full((4,), float(idx)) + draw}

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(4))

        def forward(self, x, **kwargs):
            loss = (x * self.w).sum() ** 2 * 1e-4
            return {"loss": loss}

    set_seed(7)
    run_dir = os.path.join(workdir, f"trainer_{mode}")
    ckpt_dir = os.path.join(workdir, "trainer_full")  # checkpoints live in the full run
    args = TrainingArguments(
        output_dir=run_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BS,
        save_strategy="steps",
        save_steps=save_step,
        seed=7,
        report_to=[],
        use_cpu=True,
        logging_strategy="no",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=TinyModel(), args=args, train_dataset=StochasticDataset())
    if mode == "full":
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=os.path.join(ckpt_dir, f"checkpoint-{save_step}"))

    torch.save({"records": records, "capture": capture},
               os.path.join(workdir, f"draws_{mode}.pt"))


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def _run_stage(py: str, workdir: str, mode: str, save_step: int, max_steps: int,
               capture_at: int) -> None:
    r = subprocess.run(
        [py, os.path.abspath(__file__), "--stage", "train", "--mode", mode,
         "--workdir", workdir, "--save-step", str(save_step),
         "--max-steps", str(max_steps), "--capture-at", str(capture_at)],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": HERE},
    )
    if r.returncode != 0:
        raise RuntimeError(f"stage {mode} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def _compare(work: str, scenario: str, save_step: int, resume_offset: int,
             py_label: str) -> dict:
    import torch

    from attest.gate import certify_transition
    from attest.snapshot import snapshot_from_state_dicts

    full = torch.load(os.path.join(work, "draws_full.pt"), weights_only=False)
    res = torch.load(os.path.join(work, f"draws_{scenario}.pt"), weights_only=False)
    skip_items = save_step * BS

    # item-level truth: resume-run draw i corresponds to full-run draw
    # resume_offset + i (a mid-epoch resume re-materializes the skipped
    # items, an epoch-boundary resume starts at the boundary directly)
    diverged_at = None
    compared = 0
    for i in range(len(res["records"])):
        j = resume_offset + i
        if j >= len(full["records"]):
            break
        compared += 1
        if not torch.equal(full["records"][j], res["records"][i]):
            diverged_at = j
            break

    # certificate view: the stream that produced the full run's first
    # post-checkpoint item, against the stream the resumed run had when it
    # produced the same logical item
    pre = snapshot_from_state_dicts(
        {}, aux_streams={"dataloader.torch_stream":
                         full["capture"].get("stream_at_first_resumed_item")})
    post = snapshot_from_state_dicts(
        {}, aux_streams={"dataloader.torch_stream":
                         res["capture"].get("stream_at_first_resumed_item")})
    decision = certify_transition(pre, post)

    return {
        "scenario": scenario,
        "first_divergent_item": diverged_at,
        "first_resumed_item_index": skip_items,
        "items_compared": compared,
        "reproducible": diverged_at is None and compared > 0,
        "certificate_committed": decision.committed,
        "violations_by_invariant": sorted({v.invariant for v in decision.violations}),
        "python": py_label,
    }


def run(args: argparse.Namespace) -> int:
    work = os.path.abspath(args.workdir)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    py = f"{args.venv}/bin/python"

    tf_version = subprocess.run(
        [py, "-c", "import transformers; print(transformers.__version__)"],
        capture_output=True, text=True).stdout.strip()

    results = []
    # mid-epoch checkpoint at step 4 of 32 per epoch: the rng_to_sync path;
    # the resumed run re-materializes the skipped items, so draws align 1:1
    _run_stage(py, work, "full", save_step=4, max_steps=8, capture_at=8)
    _run_stage(py, work, "resume", save_step=4, max_steps=8, capture_at=8)
    results.append(_compare(work, "resume", 4, resume_offset=0, py_label=py))

    # control: checkpoint on the epoch boundary takes the early-restore path;
    # the resumed run starts producing at the boundary (offset 64)
    work2 = os.path.join(work, "epoch_boundary")
    os.makedirs(work2, exist_ok=True)
    _run_stage(py, work2, "full", save_step=32, max_steps=36, capture_at=64)
    _run_stage(py, work2, "resume", save_step=32, max_steps=36, capture_at=0)
    epoch_cmp = _compare(work2, "resume", 32, resume_offset=64, py_label=py)
    epoch_cmp["scenario"] = "resume_epoch_boundary_control"
    results.append(epoch_cmp)

    midepoch, control = results[0], results[1]
    ok = ((not midepoch["reproducible"]) and (not midepoch["certificate_committed"])
          and control["reproducible"] and control["certificate_committed"])
    report = {
        "exp": "attest-hf-resume-repro",
        "bug": {"issue": "huggingface/transformers#39215",
                "status": "closed by stale bot 2025-08-25, unfixed",
                "mechanism": "mid-epoch resume calls get_batch_samples before "
                             "_load_rng_state (the rng_to_sync block), so the "
                             "first resumed batch draws data-pipeline randomness "
                             "from the wrong stream",
                "injected_faults": 0},
        "transformers": tf_version,
        "scenarios": results,
        "live_bug_demonstrated": ok,
    }
    print("=" * 64)
    print(f"HF Trainer mid-epoch resume repro (transformers {tf_version})")
    for r in results:
        print(f"  {r['scenario']:32s} reproducible={r['reproducible']} "
              f"first_divergent_item={r['first_divergent_item']} "
              f"certificate={'COMMIT' if r['certificate_committed'] else 'ABORT'}")
    print(f"  live bug demonstrated with control: {ok}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--venv", default=SWEEP_VENV)
    p.add_argument("--workdir", default=os.path.join(HERE, "_ckpt", "hf_resume_repro"))
    p.add_argument("--out", default=None)
    p.add_argument("--stage", default="all", choices=["all", "train"])
    p.add_argument("--mode", default=None, choices=["full", "resume"])
    p.add_argument("--save-step", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--capture-at", type=int, default=8)
    args = p.parse_args()
    if args.stage == "train":
        _stage_train(os.path.abspath(args.workdir), args.mode, args.save_step,
                     args.max_steps, args.capture_at)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

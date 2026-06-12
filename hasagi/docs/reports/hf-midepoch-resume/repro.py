# huggingface/transformers#39215: a mid-epoch resume fetches the first batch
# (get_batch_samples) before restoring the checkpoint's RNG state
# (_load_rng_state, the rng_to_sync block), so randomness the data pipeline
# consumes for that batch draws from the wrong stream and bit-reproducible
# resume silently breaks. An epoch-boundary resume restores the RNG state
# before any fetch, so it serves as the in-framework control.
# CPU-only, ~30 s. Run: python repro.py
import os
import tempfile

import torch
import transformers
from torch import nn
from transformers import Trainer, TrainingArguments, set_seed

BS = 2
N = 64  # 32 steps per epoch at batch size 2


class StochasticDataset(torch.utils.data.Dataset):
    """In-batch augmentation: every item consumes the global torch stream."""

    def __init__(self):
        self.draws = []

    def __len__(self):
        return N

    def __getitem__(self, idx):
        draw = torch.rand(4)
        self.draws.append(draw)
        return {"x": torch.full((4,), float(idx)) + draw}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x, **kwargs):
        return {"loss": (x * self.w).sum() ** 2 * 1e-4}


def run(output_dir, save_step, max_steps, resume_from=None):
    set_seed(7)
    ds = StochasticDataset()
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BS,
        save_strategy="steps",
        save_steps=save_step,
        seed=7,
        report_to=[],
        use_cpu=True,
        logging_strategy="no",
        disable_tqdm=True,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=TinyModel(), args=args, train_dataset=ds)
    trainer.train(resume_from_checkpoint=resume_from)
    return ds.draws


def first_divergence(full, resumed, first_resumed_item, resume_offset=0):
    """Index of the first post-checkpoint item whose draw differs."""
    for i in range(first_resumed_item, len(resumed)):
        j = resume_offset + i
        if j >= len(full):
            return None
        if not torch.equal(full[j], resumed[i]):
            return i
    return None


def main():
    transformers.logging.set_verbosity_error()
    work = tempfile.mkdtemp()
    print(f"transformers {transformers.__version__} | torch {torch.__version__}")

    # mid-epoch checkpoint at step 4 of 32 per epoch: the rng_to_sync path.
    # The resumed run re-materializes the 8 skipped items (the late restore is
    # meant to neutralize that), then must reproduce the full run's items 8+.
    full = run(os.path.join(work, "full"), save_step=4, max_steps=8)
    resumed = run(os.path.join(work, "resume"), save_step=4, max_steps=8,
                  resume_from=os.path.join(work, "full", "checkpoint-4"))
    div = first_divergence(full, resumed, first_resumed_item=8)
    shifted = all(
        torch.equal(full[i - BS], resumed[i]) for i in range(10, len(resumed))
        if i - BS < len(full)
    )
    print(f"mid-epoch resume    : first divergent post-checkpoint item = {div} "
          f"(first resumed batch starts at item 8) -> "
          f"{'NOT reproducible' if div is not None else 'reproducible'}")
    if div is not None and shifted:
        print("                      items 10+ equal the full run's items shifted"
              " back by one batch: the RNG restore landed one fetch too late")

    # control: a checkpoint exactly on the epoch boundary takes the
    # early-restore path (line `self._load_rng_state(...)` before any fetch)
    full_c = run(os.path.join(work, "full_c"), save_step=32, max_steps=36)
    resumed_c = run(os.path.join(work, "resume_c"), save_step=32, max_steps=36,
                    resume_from=os.path.join(work, "full_c", "checkpoint-32"))
    div_c = first_divergence(full_c, resumed_c, first_resumed_item=0,
                             resume_offset=64)
    print(f"epoch-boundary ctrl : first divergent post-checkpoint item = {div_c}"
          f" -> {'NOT reproducible' if div_c is not None else 'reproducible'}")


if __name__ == "__main__":
    main()

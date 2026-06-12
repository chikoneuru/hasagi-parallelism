# Comment draft for huggingface/transformers#39215

This is still present on transformers 5.11.0 (torch 2.12.0) and still breaks
bit-reproducible mid-epoch resume. The issue was closed by the stale bot
without a fix, and the original report didn't include a runnable
reproduction, so here is one, with a control that pins the cause to the
fetch/restore ordering. Could this be reopened?

The ordering in `Trainer._inner_training_loop` is unchanged in 5.11.0: an
epoch-boundary resume restores the RNG state before any fetch, but a
mid-epoch resume only restores it after `get_batch_samples` has already
pulled the first resumed batch:

```python
# trainer.py (5.11.0)
if steps_trained_in_current_epoch > 0 and not self.args.ignore_data_skip:
    train_dataloader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
    ...
    rng_to_sync = True
elif steps_trained_in_current_epoch == 0:
    self._load_rng_state(resume_from_checkpoint)   # epoch boundary: restore BEFORE any fetch
...
batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, self.args.device)

# need to sync after if we skipped the batches in `get_batch_samples` for shuffle order reason
if rng_to_sync:
    self._load_rng_state(resume_from_checkpoint)   # mid-epoch: restore AFTER the fetch
```

So any randomness the data pipeline consumes for the first resumed batch
(in-batch augmentation, anything drawing from the global stream in
`__getitem__`) comes from the wrong RNG state.

**Reproduction** (CPU-only, ~30 s): a dataset whose `__getitem__` draws from
the global torch stream, an uninterrupted 8-step run with a checkpoint at
step 4 (mid-epoch, 32 steps per epoch), and a resume from that checkpoint.
Bit-reproducible resume requires the resumed run's draws to match the full
run's draws for the same items.

<details><summary>repro.py</summary>

```python
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
```

</details>

Output:

```
transformers 5.11.0 | torch 2.12.0+cpu
{'train_runtime': '0.0092', 'train_samples_per_second': '1736', 'train_steps_per_second': '867.9', 'train_loss': '9.423', 'epoch': '0.25'}
{'train_runtime': '0.0049', 'train_samples_per_second': '3233', 'train_steps_per_second': '1616', 'train_loss': '7.006', 'epoch': '0.25'}
mid-epoch resume    : first divergent post-checkpoint item = 8 (first resumed batch starts at item 8) -> NOT reproducible
                      items 10+ equal the full run's items shifted back by one batch: the RNG restore landed one fetch too late
{'train_runtime': '0.0204', 'train_samples_per_second': '3533', 'train_steps_per_second': '1766', 'train_loss': '7.647', 'epoch': '1.125'}
{'train_runtime': '0.0046', 'train_samples_per_second': '1.558e+04', 'train_steps_per_second': '7788', 'train_loss': '0.5896', 'epoch': '1.125'}
epoch-boundary ctrl : first divergent post-checkpoint item = None -> reproducible
```

Three things the output shows:

1. The first divergent post-checkpoint item is exactly item 8, the very
   first resumed batch.
2. From the second resumed batch on, the resumed run's draws equal the full
   run's draws shifted back by exactly one batch: the restored RNG state is
   correct, it just landed one fetch too late, and the divergence never
   self-heals.
3. The epoch-boundary control (checkpoint at step 32, the early-restore
   path) is bit-identical across all post-checkpoint items under the same
   harness, so the divergence is the mid-epoch ordering, not the harness.

The fix proposed above (restore before the first fetch) matches what the
epoch-boundary path already does, and the shift-by-one-batch signature
confirms it would close the gap. Happy to help validate a fix.

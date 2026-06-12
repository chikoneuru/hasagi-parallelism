"""Healthy single-process DeepSpeed ZeRO-2 training run, TrainCheck-instrumentable.

Mirrors the model and config of ``exp_attest_zero_to_fp32_repro.py`` at world
size one so that TrainCheck's collector can run it directly (no launcher) and
its inference sees a healthy reference pipeline: init, training steps, and a
checkpoint save. Paths come from environment variables because the collector
runs the script without arguments:

  TC_H2H_CKPT   checkpoint directory (default ./h2h_ckpt)
  TC_H2H_STEPS  training steps (default 30)
  TC_H2H_SEED   data seed (default 0)
"""
# ruff: noqa: E402  (environment setup must precede framework imports)

import os

os.environ.setdefault("DS_ACCELERATOR", "cpu")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29575")

import deepspeed  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

try:  # stage annotations sharpen TrainCheck's inference; optional otherwise
    from traincheck import annotate_stage
    from traincheck.instrumentor import META_VARS
except ImportError:  # running uninstrumented (e.g. to produce the checkpoint)
    META_VARS = {}

    def annotate_stage(name):
        return None


META_VARS["step"] = -1


def build_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64),
                         nn.Linear(64, 8))


def main() -> None:
    ckpt_dir = os.environ.get("TC_H2H_CKPT", "./h2h_ckpt")
    steps = int(os.environ.get("TC_H2H_STEPS", "30"))
    seed = int(os.environ.get("TC_H2H_SEED", "0"))

    annotate_stage("init")
    deepspeed.init_distributed(dist_backend="gloo")
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 4,
        "zero_optimization": {"stage": 2},
    }
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,
                                           config=config)

    annotate_stage("training")
    torch.manual_seed(100 + seed)
    for step in range(steps):
        META_VARS["step"] = step
        x = torch.randn(4, 64)
        loss = engine(x).pow(2).mean()
        engine.backward(loss)
        engine.step()
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.6f}")

    annotate_stage("checkpointing")
    os.makedirs(ckpt_dir, exist_ok=True)
    engine.save_checkpoint(ckpt_dir, tag="final")
    print(f"saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()

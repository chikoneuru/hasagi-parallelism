"""Consumer run: load converted fp32 weights and train a few steps.

This is the charitable arm shape for a checker that watches live training: if
loading corrupted weights violates any invariant TrainCheck inferred from the
healthy reference pipeline, it should fire within the first resumed steps.
The same script doubles as the healthy reference consumer and as the clean
control by pointing it at differently produced weights:

  TC_H2H_WEIGHTS  directory holding pytorch_model*.bin (required)
  TC_H2H_STEPS    training steps after the load (default 5)
  TC_H2H_SEED     data seed (default 0)
"""
# ruff: noqa: E402  (environment setup must precede framework imports)

import glob
import os

import torch
import torch.nn as nn


try:
    from traincheck import annotate_stage
    from traincheck.instrumentor import META_VARS
except ImportError:
    META_VARS = {}

    def annotate_stage(name):
        return None


META_VARS["step"] = -1


def build_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64),
                         nn.Linear(64, 8))


def load_converted(out_dir: str) -> dict:
    state: dict = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "pytorch_model*.bin"))):
        if f.endswith("index.json"):
            continue
        state.update(torch.load(f, map_location="cpu", weights_only=False))
    if not state:
        raise FileNotFoundError(f"no pytorch_model*.bin under {out_dir}")
    return state


def main() -> None:
    weights = os.environ["TC_H2H_WEIGHTS"]
    steps = int(os.environ.get("TC_H2H_STEPS", "5"))
    seed = int(os.environ.get("TC_H2H_SEED", "0"))

    annotate_stage("init")
    model = build_model()
    model.load_state_dict(load_converted(weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    annotate_stage("training")
    torch.manual_seed(200 + seed)
    for step in range(steps):
        META_VARS["step"] = step
        x = torch.randn(4, 64)
        loss = model(x).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step} loss {loss.item():.6f}")


if __name__ == "__main__":
    main()

"""Load a TP=2 SwiGLU checkpoint at TP=1 and train a few steps.

Single process; this is the resharding consumer whose first steps are the
charitable detection window for a checker watching live training.

  TC_H2H_ARM    buggy | fixed (declaration style, matching the save side)
  TC_H2H_CKPT   checkpoint directory to load
  TC_H2H_STEPS  training steps after the load (default 5)
"""
# ruff: noqa: E402  (environment setup must precede framework imports)

import os

import torch

import swiglu_common as sw

try:
    from traincheck import annotate_stage
    from traincheck.instrumentor import META_VARS
except ImportError:
    META_VARS = {}

    def annotate_stage(name):
        return None


META_VARS["step"] = -1


def main() -> None:
    arm = os.environ["TC_H2H_ARM"]
    ckpt = os.environ["TC_H2H_CKPT"]
    steps = int(os.environ.get("TC_H2H_STEPS", "5"))
    tp = int(os.environ.get("TC_H2H_TP", "1"))

    annotate_stage("init")
    sw.init_parallel(tp=tp)
    from megatron.core import dist_checkpointing

    mlp = sw.build_mlp(tp=tp)
    with torch.no_grad():  # placeholders so a no-op load cannot pass silently
        mlp.linear_fc1.weight.fill_(-7.5)
        mlp.linear_fc2.weight.fill_(-7.5)
    loaded = dist_checkpointing.load(sw.declare(mlp, arm), ckpt)
    key_prefix = "0." if arm == "buggy" else ""
    with torch.no_grad():
        mlp.linear_fc1.weight.copy_(loaded[f"{key_prefix}mlp.linear_fc1.weight"])
        mlp.linear_fc2.weight.copy_(loaded[f"{key_prefix}mlp.linear_fc2.weight"])
    print(f"loaded {arm} checkpoint; fc1[0,:3] = "
          f"{mlp.linear_fc1.weight[0, :3].tolist()}")

    annotate_stage("training")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    torch.manual_seed(int(os.environ.get("TC_H2H_DATA_SEED", "300")))
    for step in range(steps):
        META_VARS["step"] = step
        x = torch.randn(4, 1, sw.HIDDEN)
        out, _ = mlp(x)
        loss = out.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step} loss {loss.item():.6f}")


if __name__ == "__main__":
    main()

"""Save a painted megatron-core gated MLP at TP=2 through dist_checkpointing.

Launched by torchrun with two ranks (see run_save_tp2.sh). The declaration
style comes from the environment so the same script produces the pre-fix and
the fixed-factory checkpoints:

  TC_H2H_ARM   buggy | fixed
  TC_H2H_CKPT  checkpoint output directory
"""
# ruff: noqa: E402  (environment setup must precede framework imports)

import os

import torch  # noqa: F401  (instrumented surface; dist below is what we call)
import torch.distributed as dist

import swiglu_common as sw

try:
    from traincheck import annotate_stage
except ImportError:
    def annotate_stage(name):
        return None


def main() -> None:
    arm = os.environ["TC_H2H_ARM"]
    ckpt = os.environ["TC_H2H_CKPT"]

    annotate_stage("init")
    sw.init_parallel(tp=sw.TP_SAVE)
    rank = dist.get_rank()
    from megatron.core import dist_checkpointing

    mlp = sw.build_mlp(tp=sw.TP_SAVE)
    sw.paint(mlp, rank, sw.TP_SAVE)

    annotate_stage("checkpointing")
    if rank == 0:
        os.makedirs(ckpt, exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(sw.declare(mlp, arm), ckpt)
    if rank == 0:
        print(f"saved {arm} checkpoint to {ckpt}")
    dist.barrier()


if __name__ == "__main__":
    main()

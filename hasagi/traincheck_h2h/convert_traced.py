"""Run a checkpoint's own zero_to_fp32.py under TrainCheck instrumentation.

The conversion is a standalone post-training script with no model or
optimizer object to track, which is exactly the structural point this arm
documents: the collector can instrument the torch API surface it calls, but
there is nothing for invariants over training state to attach to.

  TC_H2H_Z2F   path to the zero_to_fp32.py to execute (the checkpoint's copy)
  TC_H2H_CKPT  checkpoint directory (input)
  TC_H2H_OUT   output directory for the converted weights
  TC_H2H_TAG   checkpoint tag (default: final)

The script runs with its DEFAULT command-line flags, the configuration that
silently corrupts under deepspeed 0.16.0.
"""
# ruff: noqa: E402  (environment setup must precede framework imports)

import os
import runpy
import sys

# imported solely so the collector instruments the torch API surface the
# conversion uses; the wrapper itself never touches it
import torch  # noqa: F401


def main() -> None:
    script = os.environ["TC_H2H_Z2F"]
    ckpt = os.environ["TC_H2H_CKPT"]
    out = os.environ["TC_H2H_OUT"]
    tag = os.environ.get("TC_H2H_TAG", "final")
    sys.argv = [script, ckpt, out, "-t", tag]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

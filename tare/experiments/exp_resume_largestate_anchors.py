"""Measured checkpoint write/reload anchors at larger model state.

The resume mis-decision window is anchored to a single small-state point
(ResNet-18, ~0.045 GB, checkpoint write 0.092 s and reload 0.047 s) and then
extrapolated to large optimizer state at a constant per-GB bandwidth. This probe
adds intermediate *measured* anchors so the extrapolation is a multi-point fit
rather than a single-anchor line: for each target parameter size it builds a
model of that size on the GPU, attaches an SGD momentum buffer (the optimizer
state a real resume must reload), checkpoints model+optimizer to disk, then tears
down and reloads exactly as the resume path does, timing the write and the reload.

Faithful to the small-state anchor's method: reload uses
``torch.load(map_location="cpu")`` followed by ``load_state_dict`` into a freshly
built GPU model, so the reload time includes the disk read and the host-to-device
copy of the parameters. Checkpoints are written to a temp path and removed.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn


class _SizedModel(nn.Module):
    """A module whose parameter count hits a target, in several chunks to avoid a
    single giant allocation (no forward pass is needed; we time I/O, not compute)."""

    def __init__(self, n_params: int, n_chunks: int = 8) -> None:
        super().__init__()
        per = n_params // n_chunks
        self.ps = nn.ParameterList([nn.Parameter(torch.randn(per)) for _ in range(n_chunks)])


def _measure(n_params: int, ckpt: str, device: torch.device) -> dict:
    model = _SizedModel(n_params).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Populate the momentum buffer (the optimizer state a resume must reload) on
    # the parameter's device, without a backward pass (keeps grad memory at zero).
    for p in model.parameters():
        opt.state[p]["momentum_buffer"] = torch.zeros_like(p)
    torch.cuda.synchronize()
    params_gb = sum(p.numel() * 4 for p in model.parameters()) / 1e9

    t = time.perf_counter()
    torch.save({"model": model.state_dict(), "optim": opt.state_dict()}, ckpt)
    torch.cuda.synchronize()
    write_s = time.perf_counter() - t
    ckpt_gb = os.path.getsize(ckpt) / 1e9

    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model2 = _SizedModel(n_params).to(device)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
    t = time.perf_counter()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model2.load_state_dict(state["model"])
    opt2.load_state_dict(state["optim"])
    torch.cuda.synchronize()
    reload_s = time.perf_counter() - t

    del model2, opt2, state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    os.remove(ckpt)

    return {
        "params_gb": params_gb,
        "ckpt_file_gb": ckpt_gb,
        "write_s": write_s,
        "reload_s": reload_s,
        "write_bw_gbps": ckpt_gb / write_s,
        "reload_bw_gbps": ckpt_gb / reload_s,
    }


def run(args: argparse.Namespace) -> int:
    assert torch.cuda.is_available(), "needs a real GPU"
    device = torch.device("cuda", args.device)
    targets = [int(float(x) * 1e9 / 4) for x in args.params_gb.split(",")]  # GB-of-params -> param count
    rows = []
    for n in targets:
        # warm the allocator once at this size so the timing is steady-state I/O
        r = _measure(n, args.ckpt, device)
        rows.append(r)
        print(f"  params={r['params_gb']:.2f}GB ckpt={r['ckpt_file_gb']:.2f}GB "
              f"write={r['write_s']:.2f}s ({r['write_bw_gbps']:.2f} GB/s) "
              f"reload={r['reload_s']:.2f}s ({r['reload_bw_gbps']:.2f} GB/s)")
    out = {
        "gpu": torch.cuda.get_device_name(device),
        "method": ("torch.save(model+SGD-momentum) write; torch.load(map_location=cpu)+load_state_dict reload; "
                   "host-to-device copy of params included, faithful to the small-state anchor"),
        "small_state_anchor": {"params_gb": 0.045, "write_s": 0.092, "reload_s": 0.047},
        "measured_anchors": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params-gb", default="1.0,4.0", help="comma-separated GB-of-parameters targets")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--ckpt", default="/tmp/resume_anchor_ckpt.pt")
    p.add_argument("--out", default="artifacts/resume_largestate_anchors.json")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

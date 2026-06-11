#!/usr/bin/env python
"""IdleLedger kill-test bench (Project B / Green Serverless).

Measures the three primitives that decide whether "consolidate-then-actually-power-down"
beats "keep warm" for elastic AI-training pools:
  P_active        - mean GPU power during a small training step loop (W)
  P_alloc_idle    - mean GPU power while a process holds the CUDA context but does no compute (W)
  P_powered_down  - power of a truly deallocated / deep-sleep GPU (W); NOT measurable from inside a
                    live process (it needs node sleep / MIG-off / persistence-off), so it is a
                    parameter (--powered-down-w, default 15W = a typical deep-idle floor).
  E_reload(size)  - wall energy to save+reload weights+optimizer state, per model size (J)

Then the BREAK-EVEN idle duration:
  t*  =  E_reload / (P_alloc_idle - P_powered_down)
i.e. if a freed GPU will sit idle longer than t*, powering it down saves net energy vs keeping it
warm; if the idle gap is shorter than t*, the reload energy dominates and you should keep it warm.

*** PRE-REGISTERED GATE (decide the verdict mapping BEFORE running) ***
Compare t* against the MEDIAN idle gap of a real training-queue trace (e.g. Philly/Alibaba, ~tens of
seconds to minutes between jobs / within gang-wait). Report honestly regardless of outcome:
  t* << median gap        -> ALIVE     (powering down pays often; IdleLedger has headroom)
  t* ~  median gap         -> MARGINAL  (knife-edge; the break-even MODEL is the contribution, win is small)
  t* >> median gap         -> DEAD      (reload dominates; IdleLedger collapses to PWR+FGD's published result)

The honest risk this bench exists to surface (from the research plan): idle is only ~6% of *training*
energy and the recoverable pool lives at cluster level, so a large per-GPU t* would foreclose the lead
direction before any system is built. Run on the real RTX 3080 Ti (and A100 if available).

Usage:
  python idle_breakeven_bench.py --gpu 0 --sizes 125M,350M,1.3B --median-gap-s 30 --out artifacts/idle_breakeven.json
"""
import argparse
import json
import os
import threading
import time

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ---------------------------------------------------------------- power metering
class PowerMonitor:
    """Background sampler of GPU power via NVML; returns mean power (W) and energy (J)."""

    def __init__(self, gpu_index, hz=20):
        if not _HAS_NVML:
            raise RuntimeError("pynvml not available; install nvidia-ml-py to meter power")
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.hz = hz
        self._stop = threading.Event()
        self._samples = []
        self._t = None

    def _loop(self):
        dt = 1.0 / self.hz
        while not self._stop.is_set():
            self._samples.append(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0)  # mW -> W
            time.sleep(dt)

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t0 = time.time()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self._dur = time.time() - self._t0

    @property
    def mean_w(self):
        return sum(self._samples) / len(self._samples) if self._samples else float("nan")

    @property
    def duration_s(self):
        return self._dur

    @property
    def energy_j(self):
        return self.mean_w * self._dur


# ---------------------------------------------------------------- toy workload
_SIZES = {  # (d_model, n_layers) toy GPT-ish blocks; tune to fit the GPU under test
    "125M": (768, 12),
    "350M": (1024, 24),
    "1.3B": (2048, 24),
}


def _build(size, device):
    d, L = _SIZES[size]
    # a deliberately simple stack so the bench is portable; the POINT is power/energy, not the model
    layers = []
    for _ in range(L):
        layers += [nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.LayerNorm(d)]
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, opt, d


def measure_active(size, gpu, device, steps=40, batch=8, seq=128):
    model, opt, d = _build(size, device)
    x = torch.randn(batch, seq, d, device=device)
    for _ in range(5):  # warmup
        opt.zero_grad()
        model(x).mean().backward()
        opt.step()
    torch.cuda.synchronize()
    with PowerMonitor(gpu) as pm:
        for _ in range(steps):
            opt.zero_grad()
            model(x).mean().backward()
            opt.step()
        torch.cuda.synchronize()
    del model, opt, x
    torch.cuda.empty_cache()
    return pm.mean_w


def measure_alloc_idle(gpu, device, seconds=20):
    # hold a context + a small resident tensor, do NO compute (the "freed-but-warm" state)
    _resident = torch.zeros(1024, 1024, device=device)
    torch.cuda.synchronize()
    with PowerMonitor(gpu) as pm:
        time.sleep(seconds)
    del _resident
    return pm.mean_w


def measure_reload_energy(size, gpu, device, ckpt_dir):
    model, opt, _ = _build(size, device)
    path = os.path.join(ckpt_dir, f"ckpt_{size}.pt")
    torch.cuda.synchronize()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, path)
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # measure the cold reload: rebuild + load weights+optimizer onto the device
    with PowerMonitor(gpu) as pm:
        m2, o2, _ = _build(size, device)
        sd = torch.load(path, map_location=device)
        m2.load_state_dict(sd["model"])
        o2.load_state_dict(sd["opt"])
        torch.cuda.synchronize()
    e = pm.energy_j
    ckpt_bytes = os.path.getsize(path)
    del m2, o2
    torch.cuda.empty_cache()
    return {"reload_energy_j": e, "reload_s": pm.duration_s, "ckpt_bytes": ckpt_bytes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sizes", default="125M,350M")
    ap.add_argument("--powered-down-w", type=float, default=15.0,
                    help="power of a truly deallocated/deep-sleep GPU (the scale-to-zero target floor)")
    ap.add_argument("--median-gap-s", type=float, default=30.0,
                    help="median idle gap from a real training-queue trace, for the pre-registered gate")
    ap.add_argument("--idle-seconds", type=int, default=20)
    ap.add_argument("--out", default="artifacts/idle_breakeven.json")
    args = ap.parse_args()

    if not (_HAS_NVML and _HAS_TORCH):
        raise SystemExit("Need pynvml + torch + a CUDA GPU to run this bench.")
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt_dir = os.path.join(os.path.dirname(args.out) or ".", "_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    p_alloc_idle = measure_alloc_idle(args.gpu, device, seconds=args.idle_seconds)
    delta = p_alloc_idle - args.powered_down_w

    rows = []
    for s in sizes:
        p_active = measure_active(s, args.gpu, device)
        r = measure_reload_energy(s, args.gpu, device, ckpt_dir)
        t_star = r["reload_energy_j"] / delta if delta > 0 else float("inf")
        if t_star < 0.5 * args.median_gap_s:
            verdict = "ALIVE"
        elif t_star <= 2.0 * args.median_gap_s:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        rows.append({"size": s, "p_active_w": round(p_active, 1), **{k: round(v, 3) for k, v in r.items()},
                     "breakeven_t_star_s": round(t_star, 2), "verdict_vs_gate": verdict})
        print(f"[{s}] active={p_active:.0f}W  reload={r['reload_energy_j']:.0f}J/{r['reload_s']:.1f}s  "
              f"t*={t_star:.1f}s  vs median-gap={args.median_gap_s}s  -> {verdict}")

    out = {
        "gpu_index": args.gpu,
        "p_alloc_idle_w": round(p_alloc_idle, 1),
        "p_powered_down_w_assumed": args.powered_down_w,
        "idle_minus_powerdown_delta_w": round(delta, 1),
        "median_idle_gap_s": args.median_gap_s,
        "per_size": rows,
        "gate": "t* << gap ALIVE | ~gap MARGINAL | >> gap DEAD (pre-registered)",
        "note": "P_powered_down is assumed (true power-down needs node sleep/MIG-off, not measurable in-process). "
                "Sweep --powered-down-w to bound sensitivity. Cluster-level headroom needs a trace replay on top of t*.",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)
    # TODO(next): replay a real Philly/Alibaba idle-gap distribution and integrate fraction(gap > t*)
    #             * mean idle power to get the cluster-level Joules-saved-from-idle headline.


if __name__ == "__main__":
    main()

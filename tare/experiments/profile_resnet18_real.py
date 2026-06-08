"""Profile ResNet-18 / CIFAR-10 on RTX 3080 Ti at multiple power caps.

Outputs a JSON with per-cap (throughput, mean power, energy-per-iter) measured
under real training (not synthetic). The downstream
``exp_scheduler_head_to_head_real.py`` consumes this profile so the allocator
comparison runs on measured workload characteristics instead of the synthetic
``linear_profile`` curve.

We treat the power-cap level as a proxy for "GPU count" along the
energy/throughput Pareto: caps {350, 300, 250, 200, 150} W give 5 operating
points spanning the U-shape Zeus shows on every datacenter GPU. The
``EnergyProfile`` consumed downstream interprets index i as the i-th
operating point on this curve.

Usage::

    python -m experiments.profile_resnet18_real --iters 200 \\
        --power-caps 150 200 250 300 350 \\
        --out artifacts/resnet18_real_profile.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfilePoint:
    power_cap_w: int
    iters: int
    wall_seconds: float
    energy_joules: float
    throughput_iter_per_s: float
    mean_power_w: float
    energy_per_iter_j: float


class NvmlMeter:
    """Background NVML polling for power × time integration."""

    def __init__(self, sample_seconds: float = 0.1) -> None:
        self.sample_seconds = sample_seconds
        self._joules = 0.0
        self._sum_power = 0.0
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        import pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._pynvml = pynvml

    def _loop(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            try:
                power_mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            except Exception:   # noqa: BLE001
                power_mw = 0
            now = time.monotonic()
            dt = now - last
            last = now
            p_w = power_mw / 1000.0
            self._joules += p_w * dt
            self._sum_power += p_w
            self._samples += 1
            self._stop.wait(self.sample_seconds)

    def start(self) -> None:
        self._joules = 0.0
        self._sum_power = 0.0
        self._samples = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def set_power_cap(watts: int) -> None:
    try:
        subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-pl", str(int(watts))],
            check=False, capture_output=True, text=True, timeout=10.0,
        )
    except Exception:   # noqa: BLE001
        pass


def _build_resnet18(num_classes: int = 10):
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _build_loader(data_root: Path, batch_size: int):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    data_root.mkdir(parents=True, exist_ok=True)
    ds = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                      pin_memory=True, drop_last=True)


def profile_at_cap(
    power_cap_w: int,
    iters: int,
    batch_size: int,
    data_root: Path,
    warmup_iters: int = 20,
) -> ProfilePoint:
    import torch
    device = "cuda"
    set_power_cap(power_cap_w)
    time.sleep(2.0)   # let GPU settle on the new cap

    model = _build_resnet18().to(device)
    loader = _build_loader(data_root, batch_size)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    it = iter(loader)
    # Warmup so cuDNN auto-tune settles before we measure.
    model.train()
    for _ in range(warmup_iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimiser.step()
    torch.cuda.synchronize()

    meter = NvmlMeter()
    meter.start()
    start = time.monotonic()
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimiser.step()
    torch.cuda.synchronize()
    wall = time.monotonic() - start
    meter.stop()

    throughput = iters / wall
    mean_power = meter._sum_power / max(1, meter._samples)
    return ProfilePoint(
        power_cap_w=power_cap_w,
        iters=iters,
        wall_seconds=wall,
        energy_joules=meter._joules,
        throughput_iter_per_s=throughput,
        mean_power_w=mean_power,
        energy_per_iter_j=meter._joules / iters,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=200,
                        help="Iters per power cap (after warmup).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--power-caps", nargs="+", type=int,
                        default=[150, 200, 250, 300, 350])
    parser.add_argument("--data-root", default="data_cache/cifar10")
    parser.add_argument("--out", default="artifacts/resnet18_real_profile.json")
    args = parser.parse_args()

    points: list[ProfilePoint] = []
    data_root = Path(args.data_root)
    for cap in args.power_caps:
        print(f"\n[cap={cap}W] profiling {args.iters} iters batch={args.batch_size}")
        p = profile_at_cap(cap, args.iters, args.batch_size, data_root)
        points.append(p)
        print(f"  throughput={p.throughput_iter_per_s:.2f} it/s, "
              f"mean_power={p.mean_power_w:.1f}W, "
              f"E/iter={p.energy_per_iter_j:.3f}J, wall={p.wall_seconds:.1f}s")
    # Restore default cap
    set_power_cap(300)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "workload": "resnet18_cifar10",
        "batch_size": args.batch_size,
        "iters_per_cap": args.iters,
        "points": [asdict(p) for p in points],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

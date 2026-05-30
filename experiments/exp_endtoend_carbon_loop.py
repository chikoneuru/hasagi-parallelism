"""End-to-end carbon-aware training — real ResNet-18 driven by a carbon trace.

Closes Extension 3 of the pre-paper review: prior H5-C experiments measure
the policy in pure simulation (no actual training), and the H2 end-to-end
training run tests DVFS + preempt proxies but is driven by a fixed schedule,
not the carbon trace. This harness wires the *real* training loop to a real
carbon trace and the HASAGI-threshold policy that decides per simulated-hour
whether to train.

Setup:
  - 30 epochs ResNet-18 / CIFAR-10, single seed
  - Simulated 7-day carbon trace (parametric DE)
  - Each "simulated hour" the policy queries intensity at that hour:
      - Carbon-aware: train one epoch if intensity ≤ threshold; else defer
      - Continuous: always train (control)
  - Energy is measured via NVML per epoch; carbon footprint is energy ×
    intensity_at_simulated_hour
  - We cap total simulated hours at the trace length (168 for 7 days); if
    the carbon-aware run can't fit 30 epochs into low-intensity windows,
    we report what fraction completed.

The headline metric is **carbon footprint reduction at matched final
accuracy**. Pass criterion is: carbon-aware achieves the *same or better*
accuracy as continuous baseline while using less carbon (gCO2).

Usage::

    python -m experiments.exp_endtoend_carbon_loop --epochs 30 \\
        --out artifacts/endtoend_carbon_loop.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hasagi.energy.carbon_trace import published_grid_trace

ACTIVE_POWER_W = 210.0   # Zeus reference; only used when NVML unavailable


@dataclass
class EpochRecord:
    epoch_idx: int
    simulated_hour: int
    intensity_g_per_kwh: float
    wall_seconds: float
    energy_joules: float
    test_top1: float
    deferred: bool = False


@dataclass
class RunResult:
    condition: str
    seed: int
    epochs_target: int
    threshold_g_per_kwh: float
    final_top1: float
    total_energy_joules: float
    total_carbon_grams: float
    total_simulated_hours: int
    epochs_completed: int
    epochs_deferred_hours: int
    epoch_records: list[EpochRecord] = field(default_factory=list)


class NvmlMeter:
    def __init__(self, sample_seconds: float = 0.1) -> None:
        self.sample_seconds = sample_seconds
        self._joules = 0.0
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
            self._joules += (power_mw / 1000.0) * dt
            self._stop.wait(self.sample_seconds)

    def start(self) -> None:
        self._joules = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def read_joules(self) -> float:
        return self._joules


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


def _build_loaders(data_root: Path, batch_size: int = 128):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    data_root.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


def _seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(model, loader, optimiser, criterion, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimiser.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def _eval_top1(model, loader, device) -> float:
    import torch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(1, total)


def run_training(
    *,
    seed: int,
    condition: str,
    epochs_target: int,
    zone: str,
    days: int,
    threshold_multiplier: float,
    data_root: Path,
) -> RunResult:
    """Drive one (seed, condition) training run.

    condition:
      - ``continuous``: ignore carbon trace; train one epoch per simulated hour
        until ``epochs_target`` epochs are done
      - ``carbon_aware``: per simulated hour, train if intensity ≤ threshold,
        else skip (defer)
    """
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(seed)

    trace = published_grid_trace(zone, days=days, sample_minutes=60, seed=seed)
    intensities = list(trace.intensities)
    median = statistics.median(intensities)
    threshold = median * threshold_multiplier
    n_hours = len(intensities)

    model = _build_resnet18().to(device)
    train_loader, test_loader = _build_loaders(data_root)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs_target)
    criterion = nn.CrossEntropyLoss()
    set_power_cap(300)

    epoch_records: list[EpochRecord] = []
    deferred = 0
    epoch_idx = 0
    hour = 0

    while epoch_idx < epochs_target and hour < n_hours:
        intensity = intensities[hour]
        if condition == "carbon_aware" and intensity > threshold:
            epoch_records.append(EpochRecord(
                epoch_idx=-1, simulated_hour=hour,
                intensity_g_per_kwh=intensity,
                wall_seconds=0.0, energy_joules=0.0, test_top1=0.0,
                deferred=True,
            ))
            deferred += 1
            hour += 1
            continue

        meter = NvmlMeter()
        meter.start()
        epoch_start = time.monotonic()
        _train_one_epoch(model, train_loader, optimiser, criterion, device)
        epoch_wall = time.monotonic() - epoch_start
        meter.stop()
        scheduler.step()
        top1 = _eval_top1(model, test_loader, device)
        epoch_records.append(EpochRecord(
            epoch_idx=epoch_idx,
            simulated_hour=hour,
            intensity_g_per_kwh=intensity,
            wall_seconds=epoch_wall,
            energy_joules=meter.read_joules(),
            test_top1=top1,
            deferred=False,
        ))
        epoch_idx += 1
        hour += 1

    set_power_cap(300)
    total_energy = sum(r.energy_joules for r in epoch_records)
    # Carbon footprint = sum(energy × intensity at that hour) ; energy in kWh × g/kWh.
    total_carbon_g = sum(
        (r.energy_joules / 3_600_000.0) * r.intensity_g_per_kwh
        for r in epoch_records if not r.deferred
    )
    final_top1 = next(
        (r.test_top1 for r in reversed(epoch_records) if not r.deferred), 0.0,
    )

    return RunResult(
        condition=condition,
        seed=seed,
        epochs_target=epochs_target,
        threshold_g_per_kwh=threshold,
        final_top1=final_top1,
        total_energy_joules=total_energy,
        total_carbon_grams=total_carbon_g,
        total_simulated_hours=hour,
        epochs_completed=epoch_idx,
        epochs_deferred_hours=deferred,
        epoch_records=epoch_records,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--zone", default="DE")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--threshold-multiplier", type=float, default=1.10)
    parser.add_argument("--data-root", default="data_cache/cifar10")
    parser.add_argument("--out", default="artifacts/endtoend_carbon_loop.json")
    args = parser.parse_args()

    print(f"\n=== seed {args.seed} / continuous ===")
    cont = run_training(
        seed=args.seed, condition="continuous", epochs_target=args.epochs,
        zone=args.zone, days=args.days, threshold_multiplier=args.threshold_multiplier,
        data_root=Path(args.data_root),
    )
    print(f"  top-1={cont.final_top1:.2f}%, energy={cont.total_energy_joules/3.6e6:.4f} kWh, "
          f"carbon={cont.total_carbon_grams:.1f} g, epochs={cont.epochs_completed}/{args.epochs}, "
          f"sim hours={cont.total_simulated_hours}, deferred={cont.epochs_deferred_hours}")

    print(f"\n=== seed {args.seed} / carbon_aware (threshold {cont.threshold_g_per_kwh:.0f} g) ===")
    aware = run_training(
        seed=args.seed, condition="carbon_aware", epochs_target=args.epochs,
        zone=args.zone, days=args.days, threshold_multiplier=args.threshold_multiplier,
        data_root=Path(args.data_root),
    )
    print(f"  top-1={aware.final_top1:.2f}%, energy={aware.total_energy_joules/3.6e6:.4f} kWh, "
          f"carbon={aware.total_carbon_grams:.1f} g, epochs={aware.epochs_completed}/{args.epochs}, "
          f"sim hours={aware.total_simulated_hours}, deferred={aware.epochs_deferred_hours}")

    # Headline comparison.
    print("\n--- HEADLINE ---")
    print(f"  accuracy delta: {aware.final_top1 - cont.final_top1:+.2f} pp")
    if cont.total_carbon_grams > 0:
        pct = 100.0 * (aware.total_carbon_grams - cont.total_carbon_grams) / cont.total_carbon_grams
        print(f"  carbon delta:   {pct:+.2f}% ({aware.total_carbon_grams - cont.total_carbon_grams:+.1f} g)")
    print(f"  sim hours: continuous={cont.total_simulated_hours} vs carbon_aware={aware.total_simulated_hours}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "continuous": asdict(cont),
        "carbon_aware": asdict(aware),
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

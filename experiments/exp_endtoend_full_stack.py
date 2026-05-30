"""End-to-end full-stack carbon-aware training — joint throttle + defer + deadline.

Closes the gaps the prior ``exp_endtoend_carbon_loop.py`` left open
(per the pre-paper critical review):

  - **multi-seed × multi-zone × threshold sweep** for statistical signal +
    cross-zone robustness (Ext 4)
  - **joint policy** — defer when very high carbon, *throttle* when
    moderately high, run at max-throughput when low. Uses the measured
    ResNet-18 Pareto curve from profile_resnet18_real.py so the throttle
    decisions land on real (cap, throughput, energy) trade-offs (Ext 5)
  - **deadline-aware** defer — only defer if the deadline slack permits;
    bounds the JCT penalty that the naive carbon-aware policy paid (Ext 6)

The harness drives real ResNet-18 / CIFAR-10 training. Per simulated hour
the policy decides among ``{train_at_max, train_at_optimal, defer}``; each
training hour records measured energy (NVML), accuracy (test top-1), and
the active power-cap. ``defer`` simulates zero compute / zero energy at
that hour while the trace advances.

Policy modes:
  - ``static_max``: train every hour at the max-throughput cap (350W).
    Baseline; ignores carbon trace.
  - ``static_optimal``: train every hour at the energy-optimal cap (200W).
    Lower energy, longer JCT.
  - ``carbon_defer``: train at max-throughput; defer if intensity > threshold.
    Matches the prior end-to-end harness.
  - ``carbon_throttle``: never defer; train at energy-optimal cap if
    intensity > threshold else at max cap. JCT penalty 0 (no defer); energy
    benefit from staying on the energy-optimal U-shape point during dirty
    grid hours.
  - ``carbon_joint``: defer if intensity > defer_threshold; throttle to
    energy-optimal if intensity > throttle_threshold; max otherwise.
    Full-stack policy — the closest approximation to HASAGI's
    joint(defer, throttle) decision on a single-stage workload.
  - ``carbon_deadline_aware``: identical to ``carbon_defer`` but only
    defers when ``deadline_slack_hours >= remaining_epochs``. Bounds the
    JCT penalty by the deadline budget.

Usage::

    # Sensitivity (Ext 4): multi-seed × multi-zone × threshold
    python -m experiments.exp_endtoend_full_stack \\
        --policy carbon_defer --seeds 0 1 2 --zones NO DE ZA \\
        --threshold-multipliers 1.05 1.10 1.20 \\
        --out artifacts/endtoend_sensitivity

    # Full-stack (Ext 5): 5 policy modes × DE × seed 0
    python -m experiments.exp_endtoend_full_stack \\
        --policy static_max carbon_defer carbon_throttle carbon_joint \\
        --seeds 0 --zones DE --out artifacts/endtoend_fullstack

    # Deadline-aware (Ext 6): deadline-multiplier sweep
    python -m experiments.exp_endtoend_full_stack \\
        --policy carbon_deadline_aware --seeds 0 --zones DE \\
        --deadline-multipliers 1.0 1.5 2.0 3.0 \\
        --out artifacts/endtoend_deadline
"""
from __future__ import annotations

import argparse
import itertools
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

# ResNet-18 Pareto curve — measured on RTX 3080 Ti via profile_resnet18_real.py
# at 150 iters/cap (Ext 1). Cap (W) → energy-per-iter (J).
MAX_CAP_W = 350
OPTIMAL_CAP_W = 200


@dataclass
class HourRecord:
    hour: int
    intensity_g_per_kwh: float
    action: str   # "train_at_max", "train_at_optimal", "defer"
    epoch_idx: int   # -1 if deferred
    wall_seconds: float
    energy_joules: float
    test_top1: float
    power_cap_w: int


@dataclass
class RunResult:
    policy: str
    seed: int
    zone: str
    epochs_target: int
    threshold_multiplier: float
    throttle_threshold_multiplier: float
    deadline_multiplier: float
    final_top1: float
    total_energy_joules: float
    total_carbon_grams: float
    total_simulated_hours: int
    epochs_completed: int
    deferred_hours: int
    throttle_hours: int
    max_cap_hours: int
    jct_penalty_pct: float
    hour_records: list[HourRecord] = field(default_factory=list)


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


def _train_one_epoch(model, loader, optimiser, criterion, device) -> None:
    model.train()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()


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


def _decide(
    policy: str,
    intensity: float,
    median_intensity: float,
    threshold_mult: float,
    throttle_threshold_mult: float,
    deadline_slack_hours: int,
    epochs_remaining: int,
) -> str:
    """Return one of ``{"train_at_max", "train_at_optimal", "defer"}``."""
    defer_thr = median_intensity * threshold_mult
    throttle_thr = median_intensity * throttle_threshold_mult

    if policy == "static_max":
        return "train_at_max"
    if policy == "static_optimal":
        return "train_at_optimal"
    if policy == "carbon_defer":
        return "defer" if intensity > defer_thr else "train_at_max"
    if policy == "carbon_throttle":
        # Never defer; throttle to energy-optimal when carbon is high.
        return "train_at_optimal" if intensity > defer_thr else "train_at_max"
    if policy == "carbon_joint":
        # Defer at the highest tier; throttle at the middle tier; max otherwise.
        if intensity > defer_thr:
            return "defer"
        if intensity > throttle_thr:
            return "train_at_optimal"
        return "train_at_max"
    if policy == "carbon_deadline_aware":
        # Defer only if the deadline slack permits.
        if intensity > defer_thr and deadline_slack_hours >= epochs_remaining:
            return "defer"
        return "train_at_max"
    raise ValueError(f"unknown policy {policy!r}")


def run_training(
    *,
    policy: str,
    seed: int,
    zone: str,
    epochs_target: int,
    days: int,
    threshold_multiplier: float,
    throttle_threshold_multiplier: float,
    deadline_multiplier: float,
    data_root: Path,
) -> RunResult:
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(seed)

    trace = published_grid_trace(zone, days=days, sample_minutes=60, seed=seed)
    intensities = list(trace.intensities)
    median = statistics.median(intensities)
    n_hours = len(intensities)
    # Deadline: epochs_target × deadline_multiplier hours of slack from start.
    deadline_hour_budget = int(round(epochs_target * deadline_multiplier))

    model = _build_resnet18().to(device)
    train_loader, test_loader = _build_loaders(data_root)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs_target)
    criterion = nn.CrossEntropyLoss()

    records: list[HourRecord] = []
    epoch_idx = 0
    hour = 0
    deferred = 0
    throttle_hrs = 0
    max_hrs = 0
    last_top1 = 0.0

    while epoch_idx < epochs_target and hour < min(n_hours, deadline_hour_budget):
        intensity = intensities[hour]
        deadline_slack = deadline_hour_budget - hour
        epochs_remaining = epochs_target - epoch_idx
        action = _decide(
            policy, intensity, median,
            threshold_multiplier, throttle_threshold_multiplier,
            deadline_slack, epochs_remaining,
        )

        if action == "defer":
            records.append(HourRecord(
                hour=hour, intensity_g_per_kwh=intensity,
                action=action, epoch_idx=-1,
                wall_seconds=0.0, energy_joules=0.0,
                test_top1=0.0, power_cap_w=0,
            ))
            deferred += 1
            hour += 1
            continue

        cap = MAX_CAP_W if action == "train_at_max" else OPTIMAL_CAP_W
        set_power_cap(cap)
        if action == "train_at_optimal":
            throttle_hrs += 1
        else:
            max_hrs += 1

        meter = NvmlMeter()
        meter.start()
        epoch_start = time.monotonic()
        _train_one_epoch(model, train_loader, optimiser, criterion, device)
        epoch_wall = time.monotonic() - epoch_start
        meter.stop()
        scheduler.step()
        last_top1 = _eval_top1(model, test_loader, device)
        records.append(HourRecord(
            hour=hour, intensity_g_per_kwh=intensity,
            action=action, epoch_idx=epoch_idx,
            wall_seconds=epoch_wall, energy_joules=meter.read_joules(),
            test_top1=last_top1, power_cap_w=cap,
        ))
        epoch_idx += 1
        hour += 1

    set_power_cap(MAX_CAP_W)
    total_energy = sum(r.energy_joules for r in records)
    total_carbon_g = sum(
        (r.energy_joules / 3_600_000.0) * r.intensity_g_per_kwh
        for r in records if r.action != "defer"
    )
    # JCT penalty: extra simulated hours vs the ``static_max`` baseline (= epochs_target hours).
    jct_penalty_pct = 100.0 * (hour - epochs_target) / epochs_target

    return RunResult(
        policy=policy, seed=seed, zone=zone,
        epochs_target=epochs_target,
        threshold_multiplier=threshold_multiplier,
        throttle_threshold_multiplier=throttle_threshold_multiplier,
        deadline_multiplier=deadline_multiplier,
        final_top1=last_top1,
        total_energy_joules=total_energy,
        total_carbon_grams=total_carbon_g,
        total_simulated_hours=hour,
        epochs_completed=epoch_idx,
        deferred_hours=deferred,
        throttle_hours=throttle_hrs,
        max_cap_hours=max_hrs,
        jct_penalty_pct=jct_penalty_pct,
        hour_records=records,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", default=["carbon_joint"],
                        choices=["static_max", "static_optimal", "carbon_defer",
                                 "carbon_throttle", "carbon_joint", "carbon_deadline_aware"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--zones", nargs="+", default=["DE"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--threshold-multipliers", nargs="+", type=float, default=[1.10])
    parser.add_argument("--throttle-threshold-multipliers", nargs="+", type=float,
                        default=[0.95],
                        help="Below median × this multiplier ⇒ throttle (in carbon_joint).")
    parser.add_argument("--deadline-multipliers", nargs="+", type=float, default=[10.0],
                        help="Deadline budget = epochs × this. Default 10 = generous (7 days).")
    parser.add_argument("--data-root", default="data_cache/cifar10")
    parser.add_argument("--out", default="artifacts/endtoend_fullstack")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(
        args.policies, args.seeds, args.zones,
        args.threshold_multipliers, args.throttle_threshold_multipliers,
        args.deadline_multipliers,
    ))
    print(f"\nRunning {len(grid)} cells\n")
    all_results: list[RunResult] = []
    for i, (policy, seed, zone, thr_mult, throttle_mult, ddl_mult) in enumerate(grid):
        cell_name = (f"{policy}_seed{seed}_{zone}_thr{thr_mult:.2f}"
                     f"_throt{throttle_mult:.2f}_ddl{ddl_mult:.1f}")
        print(f"[{i + 1}/{len(grid)}] {cell_name}")
        result = run_training(
            policy=policy, seed=seed, zone=zone, epochs_target=args.epochs,
            days=args.days, threshold_multiplier=thr_mult,
            throttle_threshold_multiplier=throttle_mult,
            deadline_multiplier=ddl_mult,
            data_root=Path(args.data_root),
        )
        all_results.append(result)
        path = out_dir / f"{cell_name}.json"
        path.write_text(json.dumps(asdict(result), indent=2))
        print(f"  top-1={result.final_top1:.2f}%, carbon={result.total_carbon_grams:.1f} g, "
              f"energy={result.total_energy_joules / 3.6e6:.4f} kWh, "
              f"sim_h={result.total_simulated_hours}, "
              f"defer/throt/max={result.deferred_hours}/{result.throttle_hours}/{result.max_cap_hours}, "
              f"jct_penalty={result.jct_penalty_pct:+.1f}%")

    # Roll-up summary.
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

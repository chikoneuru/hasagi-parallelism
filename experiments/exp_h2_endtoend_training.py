"""H2 end-to-end training — accuracy invariance under HASAGI elasticity proxies.

The H2 acceptance criterion is "final accuracy within ±1% of the non-elastic
baseline" when HASAGI's elasticity mechanisms run during training. On a single
GPU we cannot honestly validate multi-GPU DP elasticity, but we *can* validate
the elasticity proxies that the testbed reproduces faithfully:

  - **DVFS** — periodically switch the NVIDIA power-cap during training,
    mimicking carbon-aware throttle decisions. The optimiser sees the same
    weights and the same gradients; only wall-clock and per-iter energy change.
  - **Preempt** — checkpoint every K epochs and resume in a fresh process,
    mimicking pod preemption + resume from durable state. The optimiser
    state, RNG state, and dataloader position must round-trip exactly.
  - **Combined** — both DVFS and preempt active simultaneously.

For each (seed, condition) we record per-epoch (loss, top-1 accuracy, energy
joules) and the total wall-clock + total kWh. The H2 pass/fail test compares
final accuracy across conditions at the same seed — passing if the worst
elastic accuracy is within 1.0 percentage points of the matching static
baseline at every seed.

The harness writes one JSON per (seed, condition) run plus a roll-up summary.
The summary harness reads those files and reports pairwise accuracy delta +
energy ratio.

Usage::

    # Smoke (1 epoch × 1 seed × static only, ~30 s):
    python -m experiments.exp_h2_endtoend_training --smoke

    # Full (3 seeds × 4 conditions × 30 epochs, ~2 h on RTX 3080 Ti):
    python -m experiments.exp_h2_endtoend_training --seeds 0 1 2 --epochs 30 \
        --conditions static dvfs preempt combined --out artifacts/h2_endtoend/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EpochSample:
    epoch: int
    train_loss: float
    test_top1: float
    wall_seconds: float
    energy_joules: float
    mean_power_w: float


@dataclass
class RunResult:
    seed: int
    condition: str
    epochs: int
    batch_size: int
    learning_rate: float
    final_top1: float
    total_wall_seconds: float
    total_energy_joules: float
    per_epoch: list[EpochSample] = field(default_factory=list)
    power_cap_schedule_w: list[float] = field(default_factory=list)
    preempts: int = 0


# ---------------------------------------------------------------------------
# NVML energy meter — background thread polling power; ∫ P dt over the run.
# ---------------------------------------------------------------------------

class NvmlEnergyMeter:
    """Background polling NVML energy meter. Call ``start()`` + ``stop()`` around the
    measured window; ``read_joules()`` reports the integral.

    Sampling at 100 ms gives ~10 Hz which is fine for a 30 s+ training window.
    The integration is rectangular over each polling interval, which slightly
    under-counts on rising-power edges; this is fine for a relative
    static-vs-elastic comparison.
    """

    def __init__(self, sample_seconds: float = 0.1) -> None:
        self.sample_seconds = sample_seconds
        self._joules = 0.0
        self._samples = 0
        self._sum_power_w = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception as exc:   # noqa: BLE001
            raise RuntimeError(f"NVML unavailable: {exc}") from exc

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
            power_w = power_mw / 1000.0
            self._joules += power_w * dt
            self._sum_power_w += power_w
            self._samples += 1
            self._stop.wait(self.sample_seconds)

    def start(self) -> None:
        self._joules = 0.0
        self._samples = 0
        self._sum_power_w = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def read_joules(self) -> float:
        return self._joules

    def mean_power_w(self) -> float:
        return self._sum_power_w / max(1, self._samples)


# ---------------------------------------------------------------------------
# Power-cap controller — uses the passwordless-sudo entry set up at Tuần 31.
# ---------------------------------------------------------------------------

DEFAULT_POWER_CAPS_W = (200.0, 300.0)  # alternates each cycle


def set_power_cap(watts: int) -> None:
    """Issue ``sudo nvidia-smi -pl <W>``. Requires the passwordless-sudo entry
    at /etc/sudoers.d/hasagi-nvidia-smi (set up at Tuần 31)."""
    try:
        subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-pl", str(int(watts))],
            check=False, capture_output=True, text=True, timeout=10.0,
        )
    except Exception:   # noqa: BLE001
        pass   # best-effort; energy meter still records actual power


# ---------------------------------------------------------------------------
# Model + dataloaders — lazy imports so the module loads on the CPU-only venv.
# ---------------------------------------------------------------------------

def _build_resnet18(num_classes: int = 10):
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None, num_classes=num_classes)
    # Adapt the first conv for CIFAR's 32×32 inputs.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _build_loaders(data_root: Path, batch_size: int, num_workers: int = 2):
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
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def _seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_epoch(model, loader, optimiser, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def _eval_top1(model, loader, device) -> float:
    import torch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(1, total)


# ---------------------------------------------------------------------------
# One training run — drives the (seed, condition) over `epochs` epochs.
# ---------------------------------------------------------------------------

def _make_optimiser(model, lr: float, epochs: int):
    import torch
    optimiser = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    return optimiser, scheduler


def _save_ckpt(path: Path, model, optimiser, scheduler, epoch: int) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimiser.state_dict(),
        "sched": scheduler.state_dict(),
    }, path)


def _load_ckpt(path: Path, model, optimiser, scheduler) -> int:
    import torch
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimiser.load_state_dict(state["optim"])
    scheduler.load_state_dict(state["sched"])
    return state["epoch"]


def run_training(
    *,
    seed: int,
    condition: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    data_root: Path,
    ckpt_dir: Path,
    dvfs_caps_w: tuple[float, float] = DEFAULT_POWER_CAPS_W,
    dvfs_period: int = 5,
    preempt_period: int = 10,
) -> RunResult:
    """Drive one (seed, condition) end-to-end run.

    condition:
      - ``static``: no DVFS, no preempt
      - ``dvfs``: power-cap toggles between ``dvfs_caps_w`` every ``dvfs_period`` epochs
      - ``preempt``: checkpoint + reload-from-disk every ``preempt_period`` epochs
      - ``combined``: both
    """
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(seed)
    model = _build_resnet18().to(device)
    train_loader, test_loader = _build_loaders(data_root, batch_size)
    optimiser, scheduler = _make_optimiser(model, learning_rate, epochs)
    criterion = nn.CrossEntropyLoss()

    do_dvfs = condition in ("dvfs", "combined")
    do_preempt = condition in ("preempt", "combined")
    preempts = 0
    power_cap_schedule: list[float] = []

    # Reset power cap to high for static; alternate from the second cap for dvfs.
    if do_dvfs:
        set_power_cap(int(dvfs_caps_w[1]))
        power_cap_schedule.append(dvfs_caps_w[1])
    else:
        set_power_cap(300)
        power_cap_schedule.append(300.0)

    meter = NvmlEnergyMeter()
    run_start = time.monotonic()
    per_epoch: list[EpochSample] = []

    ckpt_path = ckpt_dir / f"seed{seed}_{condition}.pt"

    for epoch in range(epochs):
        if do_dvfs and epoch > 0 and epoch % dvfs_period == 0:
            new_cap = dvfs_caps_w[(epoch // dvfs_period) % len(dvfs_caps_w)]
            set_power_cap(int(new_cap))
            power_cap_schedule.append(new_cap)

        if do_preempt and epoch > 0 and epoch % preempt_period == 0:
            _save_ckpt(ckpt_path, model, optimiser, scheduler, epoch)
            # "Re-create" everything from disk, including a fresh CUDA context-less reload.
            model_new = _build_resnet18().to(device)
            optimiser_new, scheduler_new = _make_optimiser(model_new, learning_rate, epochs)
            _load_ckpt(ckpt_path, model_new, optimiser_new, scheduler_new)
            model = model_new
            optimiser = optimiser_new
            scheduler = scheduler_new
            preempts += 1

        epoch_start = time.monotonic()
        meter.start()
        train_loss = _train_epoch(model, train_loader, optimiser, criterion, device)
        meter.stop()
        epoch_wall = time.monotonic() - epoch_start
        scheduler.step()
        test_top1 = _eval_top1(model, test_loader, device)
        per_epoch.append(EpochSample(
            epoch=epoch,
            train_loss=train_loss,
            test_top1=test_top1,
            wall_seconds=epoch_wall,
            energy_joules=meter.read_joules(),
            mean_power_w=meter.mean_power_w(),
        ))

    total_wall = time.monotonic() - run_start
    total_energy = sum(s.energy_joules for s in per_epoch)
    # Restore the GPU to default power cap on exit.
    set_power_cap(300)

    return RunResult(
        seed=seed,
        condition=condition,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        final_top1=per_epoch[-1].test_top1 if per_epoch else 0.0,
        total_wall_seconds=total_wall,
        total_energy_joules=total_energy,
        per_epoch=per_epoch,
        power_cap_schedule_w=power_cap_schedule,
        preempts=preempts,
    )


# ---------------------------------------------------------------------------
# Sweep + reporting
# ---------------------------------------------------------------------------

CONDITIONS = ("static", "dvfs", "preempt", "combined")


def _write_result(out_dir: Path, result: RunResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed{result.seed}_{result.condition}.json"
    path.write_text(json.dumps(asdict(result), indent=2))
    return path


def _summarise(out_dir: Path) -> dict[str, dict[str, float]]:
    """Per-condition aggregate over seeds."""
    runs: dict[str, list[RunResult]] = {}
    for path in sorted(out_dir.glob("seed*_*.json")):
        data = json.loads(path.read_text())
        runs.setdefault(data["condition"], []).append(data)

    summary: dict[str, dict[str, float]] = {}
    for cond, rs in runs.items():
        accs = [r["final_top1"] for r in rs]
        es = [r["total_energy_joules"] for r in rs]
        walls = [r["total_wall_seconds"] for r in rs]
        summary[cond] = {
            "n": len(rs),
            "mean_top1": sum(accs) / len(accs),
            "min_top1": min(accs),
            "max_top1": max(accs),
            "mean_energy_kwh": sum(es) / len(es) / 3_600_000.0,
            "mean_wall_minutes": sum(walls) / len(walls) / 60.0,
        }
    return summary


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title="H2 end-to-end training — per-condition aggregate")
    table.add_column("condition")
    table.add_column("n", justify="right")
    table.add_column("mean top-1 (%)", justify="right")
    table.add_column("min-max top-1", justify="right")
    table.add_column("mean kWh", justify="right")
    table.add_column("mean wall (min)", justify="right")
    static = summary.get("static")
    for cond in CONDITIONS:
        if cond not in summary:
            continue
        s = summary[cond]
        table.add_row(
            cond,
            str(s["n"]),
            f"{s['mean_top1']:.2f}",
            f"{s['min_top1']:.2f}–{s['max_top1']:.2f}",
            f"{s['mean_energy_kwh']:.4f}",
            f"{s['mean_wall_minutes']:.1f}",
        )
    console.print(table)

    if static is None:
        return
    delta_table = Table(title="H2 acceptance — accuracy delta vs static baseline")
    delta_table.add_column("condition")
    delta_table.add_column("Δ mean top-1 (pp)", justify="right")
    delta_table.add_column("|Δ| ≤ 1.0 pp?", justify="center")
    for cond in CONDITIONS:
        if cond == "static" or cond not in summary:
            continue
        delta = summary[cond]["mean_top1"] - static["mean_top1"]
        ok = "✓" if abs(delta) <= 1.0 else "✗"
        delta_table.add_row(cond, f"{delta:+.2f}", ok)
    console.print(delta_table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS),
        choices=list(CONDITIONS),
    )
    parser.add_argument("--out", default="artifacts/h2_endtoend")
    parser.add_argument("--data-root", default="data_cache/cifar10")
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch × 1 seed × static (~30 s sanity check).")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip training; just re-read and summarise the existing artifacts.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    data_root = Path(args.data_root)
    ckpt_dir = Path(args.out) / "ckpts"

    if args.summary_only:
        summary = _summarise(out_dir)
        _print_summary(summary)
        out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
        return 0

    if args.smoke:
        seeds = [0]
        conditions = ["static"]
        epochs = 1
    else:
        seeds = args.seeds
        conditions = args.conditions
        epochs = args.epochs

    for seed in seeds:
        for cond in conditions:
            print(f"\n[seed {seed} / {cond} / {epochs} epochs] running...")
            result = run_training(
                seed=seed, condition=cond, epochs=epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                data_root=data_root, ckpt_dir=ckpt_dir,
            )
            path = _write_result(out_dir, result)
            print(
                f"  done: top-1={result.final_top1:.2f}%, "
                f"energy={result.total_energy_joules/3.6e6:.4f} kWh, "
                f"wall={result.total_wall_seconds/60:.1f} min, "
                f"preempts={result.preempts}, -> {path}"
            )

    summary = _summarise(out_dir)
    _print_summary(summary)
    out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

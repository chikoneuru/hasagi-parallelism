"""End-to-end carbon-aware fine-tuning of BERT-base on SST-2.

Generalises the ResNet-18 / CIFAR-10 end-to-end harness in
``exp_endtoend_full_stack.py`` to a transformer workload, so the
``carbon_throttle`` policy and the Pareto cap (200 W energy-optimal)
can be checked for transferability across model architectures.

Per simulated hour the policy decides among
``{train_at_max, train_at_optimal, defer}``. Each training "epoch" is
one full pass over SST-2 train (~67k examples at batch 32 → ~2100
iterations) on BERT-base (~110M params). NVML samples power at 100 ms
during the active span; ``nvidia-smi -pl`` (passwordless sudo) sets the
cap before each epoch starts.

Policy modes mirror exactly the ResNet harness so the two studies can
be cross-referenced:
    static_max | static_optimal | carbon_defer | carbon_throttle |
    carbon_joint | carbon_deadline_aware

Usage::

    python -m experiments.exp_c2_bert_sst2 \\
        --policies static_max carbon_throttle carbon_defer \\
        --seeds 0 1 2 --zones DE --epochs 5 \\
        --out artifacts/c2_bert_sst2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.exp_endtoend_full_stack import (
    MAX_CAP_W,
    OPTIMAL_CAP_W,
    NvmlMeter,
    _decide,
    set_power_cap,
)
from tare.energy.carbon_trace import published_grid_trace

MAX_SEQ_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


@dataclass
class HourRecord:
    hour: int
    intensity_g_per_kwh: float
    action: str
    epoch_idx: int
    wall_seconds: float
    energy_joules: float
    test_top1: float
    power_cap_w: int


@dataclass
class RunResult:
    policy: str
    seed: int
    zone: str
    workload: str
    epochs_target: int
    threshold_multiplier: float
    throttle_threshold_multiplier: float
    deadline_multiplier: float
    throttle_cap_w: int
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


def _seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_bert_sst2(num_labels: int = 2):
    """Return a fresh BERT-base + SST-2 head + matching tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels,
    )
    return model, tokenizer


def _build_sst2_loaders(tokenizer, data_cache_dir: Path):
    """Return ``(train_loader, eval_loader)`` over tokenised SST-2.

    Uses the GLUE/SST-2 split from HuggingFace ``datasets``. Tokenisation
    is cached on disk via the ``cache_dir`` argument so re-runs are fast.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    data_cache_dir.mkdir(parents=True, exist_ok=True)
    # datasets >= 4.0 requires a namespaced repo id; "glue" alone fails with
    # HfUriError. nyu-mll/glue is the canonical HF mirror.
    raw = load_dataset("nyu-mll/glue", "sst2", cache_dir=str(data_cache_dir))

    def encode(batch):
        out = tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        out["labels"] = batch["label"]
        return out

    encoded = raw.map(encode, batched=True, load_from_cache_file=True)
    columns = ["input_ids", "attention_mask", "labels"]
    encoded.set_format("torch", columns=columns)

    train_loader = DataLoader(
        encoded["train"], batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        encoded["validation"], batch_size=128, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, eval_loader


def _train_one_epoch(model, loader, optimiser, scheduler, device) -> None:
    import torch.nn.utils as nn_utils
    model.train()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()


def _eval_top1(model, loader, device) -> float:
    import torch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(1, total)


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
    data_cache_dir: Path,
    throttle_cap_w: int = OPTIMAL_CAP_W,
) -> RunResult:
    import torch
    from transformers import get_linear_schedule_with_warmup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(seed)

    trace = published_grid_trace(zone, days=days, sample_minutes=60, seed=seed)
    intensities = list(trace.intensities)
    median = statistics.median(intensities)
    n_hours = len(intensities)
    deadline_hour_budget = int(round(epochs_target * deadline_multiplier))
    # Spread seeds across the 24h-periodic trace so each samples a different
    # daily-cycle phase (~8h apart). Without this, all seeds start at hour 0
    # and only ever see the first deadline_hour_budget hours of the trace,
    # which on synthetic DE is below the throttle threshold.
    start_hour = (seed * 8 + 5) % max(1, n_hours - deadline_hour_budget)

    model, tokenizer = _build_bert_sst2()
    model = model.to(device)
    train_loader, eval_loader = _build_sst2_loaders(tokenizer, data_cache_dir)

    n_steps_per_epoch = len(train_loader)
    total_steps = n_steps_per_epoch * epochs_target
    warmup_steps = int(0.1 * total_steps)

    no_decay = {"bias", "LayerNorm.weight"}
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimiser = torch.optim.AdamW(params, lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    records: list[HourRecord] = []
    epoch_idx = 0
    hour = start_hour
    deferred = 0
    throttle_hrs = 0
    max_hrs = 0
    last_top1 = 0.0
    deadline_hour_limit = start_hour + deadline_hour_budget

    while epoch_idx < epochs_target and hour < min(n_hours, deadline_hour_limit):
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

        cap = MAX_CAP_W if action == "train_at_max" else throttle_cap_w
        set_power_cap(cap)
        if action == "train_at_optimal":
            throttle_hrs += 1
        else:
            max_hrs += 1

        meter = NvmlMeter()
        meter.start()
        epoch_start = time.monotonic()
        _train_one_epoch(model, train_loader, optimiser, scheduler, device)
        epoch_wall = time.monotonic() - epoch_start
        meter.stop()
        last_top1 = _eval_top1(model, eval_loader, device)
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
    jct_penalty_pct = 100.0 * ((hour - start_hour) - epochs_target) / epochs_target

    return RunResult(
        policy=policy, seed=seed, zone=zone, workload="bert-base-sst2",
        epochs_target=epochs_target,
        threshold_multiplier=threshold_multiplier,
        throttle_threshold_multiplier=throttle_threshold_multiplier,
        deadline_multiplier=deadline_multiplier,
        throttle_cap_w=throttle_cap_w,
        final_top1=last_top1,
        total_energy_joules=total_energy,
        total_carbon_grams=total_carbon_g,
        total_simulated_hours=hour - start_hour,
        epochs_completed=epoch_idx,
        deferred_hours=deferred,
        throttle_hours=throttle_hrs,
        max_cap_hours=max_hrs,
        jct_penalty_pct=jct_penalty_pct,
        hour_records=records,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies", nargs="+",
        default=["static_max", "carbon_throttle", "carbon_defer"],
        choices=["static_max", "static_optimal", "carbon_defer",
                 "carbon_throttle", "carbon_joint", "carbon_deadline_aware"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--zones", nargs="+", default=["DE"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--threshold-multipliers", nargs="+", type=float, default=[1.10])
    parser.add_argument("--throttle-threshold-multipliers", nargs="+", type=float,
                        default=[0.95])
    parser.add_argument("--deadline-multipliers", nargs="+", type=float, default=[10.0])
    parser.add_argument("--data-cache-dir", default="data_cache/glue_sst2")
    parser.add_argument(
        "--throttle-cap", type=int, default=OPTIMAL_CAP_W,
        help="Power cap (W) used when the policy throttles. The energy-optimal "
             "cap is workload-dependent; for BERT-base SST-2 it is ~250 W, "
             "whereas the CV harness uses 200 W.",
    )
    parser.add_argument("--out", default="artifacts/c2_bert_sst2")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(
        args.policies, args.seeds, args.zones,
        args.threshold_multipliers, args.throttle_threshold_multipliers,
        args.deadline_multipliers,
    ))
    print(f"\nRunning {len(grid)} cells (BERT-base / SST-2)\n")
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
            data_cache_dir=Path(args.data_cache_dir),
            throttle_cap_w=args.throttle_cap,
        )
        all_results.append(result)
        path = out_dir / f"{cell_name}.json"
        path.write_text(json.dumps(asdict(result), indent=2))
        print(f"  top-1={result.final_top1:.2f}%, carbon={result.total_carbon_grams:.1f} g, "
              f"energy={result.total_energy_joules / 3.6e6:.4f} kWh, "
              f"sim_h={result.total_simulated_hours}, "
              f"defer/throt/max={result.deferred_hours}/{result.throttle_hours}/{result.max_cap_hours}, "
              f"jct_penalty={result.jct_penalty_pct:+.1f}%")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

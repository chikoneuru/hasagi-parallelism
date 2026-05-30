"""Pareto power-cap sweep for BERT-base SST-2 on the local GPU.

Runs one fine-tuning epoch at each candidate power cap and records
(wall_seconds, total_energy_joules, top-1). The output identifies the
energy-minimising cap (the throttle target for a compute-bound
transformer workload), separately from the wall-time-minimising cap
(typically the device TDP). Used to pick the ``OPTIMAL_CAP_W`` override
that the carbon_throttle policy should adopt for transformer workloads,
mirroring how the ResNet-CIFAR study picked 200 W as the Pareto cap
for a CV workload.

Usage::

    python -m experiments.exp_c2_bert_pareto_cap \\
        --caps 350 320 300 280 250 220 200 \\
        --seed 0 --out artifacts/c2_bert_pareto_cap
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.exp_c2_bert_sst2 import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    _build_bert_sst2,
    _build_sst2_loaders,
    _eval_top1,
    _seed_everything,
    _train_one_epoch,
)
from experiments.exp_endtoend_full_stack import NvmlMeter, set_power_cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps", nargs="+", type=int,
                        default=[350, 320, 300, 280, 250, 220, 200])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-cache", type=Path,
                        default=Path("artifacts/hf_cache"))
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/c2_bert_pareto_cap"))
    args = parser.parse_args()

    import torch
    from transformers import get_linear_schedule_with_warmup

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for i, cap in enumerate(args.caps):
        print(f"[{i+1}/{len(args.caps)}] cap={cap} W")
        _seed_everything(args.seed)
        model, tokenizer = _build_bert_sst2()
        model = model.to(device)
        train_loader, eval_loader = _build_sst2_loaders(tokenizer, args.data_cache)
        n_steps = len(train_loader)
        warmup_steps = int(0.1 * n_steps)

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
            optimiser, num_warmup_steps=warmup_steps, num_training_steps=n_steps,
        )

        set_power_cap(cap)
        meter = NvmlMeter()
        meter.start()
        t0 = time.monotonic()
        _train_one_epoch(model, train_loader, optimiser, scheduler, device)
        wall_s = time.monotonic() - t0
        meter.stop()
        energy_j = meter.read_joules()
        top1 = _eval_top1(model, eval_loader, device)
        avg_power_w = energy_j / wall_s if wall_s > 0 else 0.0

        row = {
            "cap_w": cap,
            "wall_seconds": wall_s,
            "total_energy_joules": energy_j,
            "avg_power_w": avg_power_w,
            "top1_after_one_epoch": top1,
            "energy_kwh": energy_j / 3_600_000.0,
            "samples_per_second": (n_steps * 32) / wall_s if wall_s > 0 else 0.0,
        }
        rows.append(row)
        print(
            f"  wall={wall_s:.1f}s  energy={energy_j/1000:.2f}kJ "
            f"avg_power={avg_power_w:.1f}W  top1={top1:.2f}%"
        )
        del model, optimiser, scheduler, train_loader, eval_loader
        torch.cuda.empty_cache()

    set_power_cap(350)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    energy_opt = min(rows, key=lambda r: r["total_energy_joules"])
    time_opt = min(rows, key=lambda r: r["wall_seconds"])
    print("\n=== Pareto summary ===")
    print(f"Energy-optimal cap: {energy_opt['cap_w']} W "
          f"(energy={energy_opt['total_energy_joules']/1000:.2f} kJ, "
          f"wall={energy_opt['wall_seconds']:.1f} s)")
    print(f"Wall-time-optimal cap: {time_opt['cap_w']} W "
          f"(wall={time_opt['wall_seconds']:.1f} s, "
          f"energy={time_opt['total_energy_joules']/1000:.2f} kJ)")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

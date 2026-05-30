"""Analytical carbon model for BERT-base SST-2 at a workload-tuned throttle cap.

============================ RESULT WITHDRAWN ============================
The headline this module produced (carbon_throttle BERT carbon saving at a
250 W cap) is WITHDRAWN and must NOT be cited. Three confirmed defects: (a) the
250 W per-epoch energy is from a single-seed, single-epoch sweep and was never
validated end-to-end -- the only clean end-to-end run was at 200 W, where the
policy is carbon-POSITIVE; (b) the 16-zone x 3-seed design is pseudo-replicated
(seeds share one synthetic diurnal phase, effective N is ~3); (c) the carbon
traces are synthetic, not real. Kept only as an unvalidated record. A real
end-to-end run at the reported throttle cap, zone-clustered statistics, and
real carbon traces are required before any BERT carbon number can be reported.
=========================================================================

A carbon-aware policy's per-hour decision (``train_at_max`` /
``train_at_optimal`` / ``defer``) depends only on the carbon-intensity trace
and the policy thresholds, never on the power cap. The cap only sets how much
energy each active hour draws, and the per-epoch energy-vs-cap curve is a
property of the model and the GPU, not of the carbon trace or the region.
Two facts follow, and together they let us compute region-by-region carbon
*without re-measuring on the GPU* (hence immune to shared-GPU co-tenancy):

  1. The per-(zone, seed) schedule is reproducible by replaying the policy
     decision over ``published_grid_trace(zone, seed)`` -- pure arithmetic.
  2. The per-epoch energy at any cap is the controlled Pareto-sweep
     measurement (``c2_bert_pareto_cap``), taken on an exclusive GPU.

Energy per active hour:
  - ``train_at_max``     -> Pareto energy at 350 W (the device max cap).
  - ``train_at_optimal`` -> Pareto energy at ``--throttle-cap`` (250 W is the
                            BERT energy-optimal point; 200 W, the CV harness
                            cap, sits past it and *raises* energy).
  - ``defer``            -> 0 (job paused that hour).

Accuracy is unchanged by the cap: a throttled or deferred run executes the
exact same gradient steps in the same order for a given seed, only later or
slower, so the top-1 delta versus ``static_max`` is identically zero (the
clean cap-200 run confirms this: 91.74 == 91.74 per seed). We therefore carry
the measured per-seed top-1 (seeds 0/1/2 = 91.74/92.89/93.00) as a fixed
seed property, identical across zones and policies.

This mirrors the 16-zone x 3-seed design of the ResNet/CIFAR global power-cap
throttle sweep so the two studies aggregate through the same statistical
pipeline.

Usage::

    python -m experiments.exp_c2_analytical_carbon \\
        --throttle-cap 250 --seeds 0 1 2 \\
        --zones DE US-CA FR PL VN JP GB SG KR BR NO ZA AU IN CN AE \\
        --pareto artifacts/c2_bert_pareto_cap/summary.json \\
        --out artifacts/c2_bert_sst2_analytical_cap250
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.exp_endtoend_full_stack import _decide
from hasagi.energy.carbon_trace import published_grid_trace

MAX_CAP_W = 350

# Measured per-seed top-1 from the clean cap-200 run. The cap does not change
# which gradient steps run, so the top-1 delta vs static_max is exactly zero;
# these absolute values are a fixed seed property, identical across zones.
SEED_TOP1 = {0: 91.74311926605505, 1: 92.88990825688073, 2: 93.0045871559633}

ALL_ZONES = [
    "DE", "US-CA", "FR", "PL", "VN", "JP", "GB", "SG",
    "KR", "BR", "NO", "ZA", "AU", "IN", "CN", "AE",
]


def replay_schedule(
    *, policy: str, seed: int, zone: str, epochs_target: int, days: int,
    threshold_multiplier: float, throttle_threshold_multiplier: float,
    deadline_multiplier: float,
) -> list[dict]:
    """Replay the policy decision loop over a trace; return hour records.

    Mirrors ``run_training`` in ``exp_c2_bert_sst2`` exactly, minus the GPU
    training: same seed-offset start hour, same deadline budget, same advance
    rules (defer advances the clock but not the epoch counter).
    """
    trace = published_grid_trace(zone, days=days, sample_minutes=60, seed=seed)
    intensities = list(trace.intensities)
    median = statistics.median(intensities)
    n_hours = len(intensities)
    deadline_hour_budget = int(round(epochs_target * deadline_multiplier))
    start_hour = (seed * 8 + 5) % max(1, n_hours - deadline_hour_budget)

    records: list[dict] = []
    epoch_idx = 0
    hour = start_hour
    deadline_hour_limit = start_hour + deadline_hour_budget
    while epoch_idx < epochs_target and hour < min(n_hours, deadline_hour_limit):
        intensity = intensities[hour]
        deadline_slack = deadline_hour_limit - hour
        epochs_remaining = epochs_target - epoch_idx
        action = _decide(
            policy, intensity, median,
            threshold_multiplier, throttle_threshold_multiplier,
            deadline_slack, epochs_remaining,
        )
        if action == "defer":
            records.append({"hour": hour, "intensity_g_per_kwh": intensity,
                            "action": action, "epoch_idx": -1})
            hour += 1
            continue
        records.append({"hour": hour, "intensity_g_per_kwh": intensity,
                        "action": action, "epoch_idx": epoch_idx})
        epoch_idx += 1
        hour += 1
    return records, start_hour, epochs_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throttle-cap", type=int, default=250)
    parser.add_argument("--policies", nargs="+",
                        default=["static_max", "carbon_throttle", "carbon_defer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--zones", nargs="+", default=ALL_ZONES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--threshold-multiplier", type=float, default=1.10)
    parser.add_argument("--throttle-threshold-multiplier", type=float, default=0.95)
    parser.add_argument("--deadline-multiplier", type=float, default=10.0)
    parser.add_argument("--pareto", type=Path,
                        default=Path("artifacts/c2_bert_pareto_cap/summary.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/c2_bert_sst2_analytical_cap250"))
    args = parser.parse_args()

    pareto = json.loads(args.pareto.read_text())
    energy_at = {r["cap_w"]: r["total_energy_joules"] for r in pareto}
    if MAX_CAP_W not in energy_at or args.throttle_cap not in energy_at:
        raise SystemExit(
            f"Pareto sweep needs both {MAX_CAP_W} W and {args.throttle_cap} W; "
            f"have {sorted(energy_at)}"
        )
    e_max = energy_at[MAX_CAP_W]
    e_throttle = energy_at[args.throttle_cap]

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for zone in args.zones:
        for seed in args.seeds:
            for policy in args.policies:
                recs, start_hour, epochs_target = replay_schedule(
                    policy=policy, seed=seed, zone=zone,
                    epochs_target=args.epochs, days=args.days,
                    threshold_multiplier=args.threshold_multiplier,
                    throttle_threshold_multiplier=args.throttle_threshold_multiplier,
                    deadline_multiplier=args.deadline_multiplier,
                )
                total_energy = 0.0
                total_carbon = 0.0
                throttle_hrs = max_hrs = deferred = 0
                hour_records = []
                for r in recs:
                    action = r["action"]
                    if action == "defer":
                        energy, cap = 0.0, 0
                        deferred += 1
                    elif action == "train_at_optimal":
                        energy, cap = e_throttle, args.throttle_cap
                        throttle_hrs += 1
                    else:
                        energy, cap = e_max, MAX_CAP_W
                        max_hrs += 1
                    total_energy += energy
                    if action != "defer":
                        total_carbon += (energy / 3_600_000.0) * r["intensity_g_per_kwh"]
                    hour_records.append({**r, "wall_seconds": 0.0,
                                         "energy_joules": energy,
                                         "test_top1": 0.0, "power_cap_w": cap})
                sim_hours = (hour_records[-1]["hour"] - start_hour + 1) if hour_records else 0
                jct = 100.0 * (sim_hours - epochs_target) / epochs_target
                cell = {
                    "policy": policy, "seed": seed, "zone": zone,
                    "workload": "bert-base-sst2", "epochs_target": epochs_target,
                    "threshold_multiplier": args.threshold_multiplier,
                    "throttle_threshold_multiplier": args.throttle_threshold_multiplier,
                    "deadline_multiplier": args.deadline_multiplier,
                    "throttle_cap_w": args.throttle_cap,
                    "final_top1": SEED_TOP1[seed],
                    "total_energy_joules": total_energy,
                    "total_carbon_grams": total_carbon,
                    "total_simulated_hours": sim_hours,
                    "epochs_completed": epochs_target,
                    "deferred_hours": deferred, "throttle_hours": throttle_hrs,
                    "max_cap_hours": max_hrs, "jct_penalty_pct": jct,
                    "energy_model": (f"analytical: max=Pareto@350W({e_max/1000:.2f}kJ), "
                                     f"throttle=Pareto@{args.throttle_cap}W"
                                     f"({e_throttle/1000:.2f}kJ), defer=0"),
                    "hour_records": hour_records,
                }
                results.append(cell)
                fname = (f"{policy}_seed{seed}_{zone}_thr"
                         f"{args.threshold_multiplier:.2f}"
                         f"_throt{args.throttle_threshold_multiplier:.2f}"
                         f"_ddl{args.deadline_multiplier:.1f}.json")
                (args.out / fname).write_text(json.dumps(cell, indent=2))

    (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"Analytical BERT-SST2 carbon: {len(args.zones)} zones x "
          f"{len(args.seeds)} seeds x {len(args.policies)} policies "
          f"= {len(results)} cells")
    print(f"throttle-cap {args.throttle_cap} W = {e_throttle/1000:.2f} kJ/epoch; "
          f"max 350 W = {e_max/1000:.2f} kJ/epoch")
    print(f"wrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

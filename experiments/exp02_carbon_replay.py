"""exp02 — replay an energy + carbon-proxy trace against scaling policies.

Reports both **primary energy** (kWh measured) and **derived carbon** (kWh × grid
intensity proxy). The primary claim is the energy delta; carbon delta is shown for
context and depends entirely on which grid trace you load.

Compared across:
    - Static-max (no scaling, max GPUs always)
    - Rule-based HISE
    - MPC HISE

Usage:
    python experiments/exp02_carbon_replay.py --trace traces/synthetic_solar.csv --hours 24
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from hise.energy.carbon_trace import load_csv_trace
from hise.energy.policy import MPCPolicy, RuleBasedPolicy


@dataclass
class SimResult:
    name: str
    total_energy_kwh: float       # primary (would be measured by NVML in real run)
    total_emissions_g: float      # secondary (derived via grid intensity proxy)
    average_gpus: float
    final_iter_done: int


def simulate(name: str, decide_fn, trace, *, target_iters: int, max_gpus: int,
             power_per_gpu_w: float, base_throughput: float, tick_s: int) -> SimResult:
    """Walk ``trace`` in ``tick_s`` steps. ``decide_fn`` returns target_gpus for each tick."""
    sim_t = 0.0
    iter_done = 0
    total_energy_kwh = 0.0
    total_emissions = 0.0
    gpus_history: list[int] = []
    current_gpus = 1

    while iter_done < target_iters and sim_t < trace.duration_seconds:
        intensity = trace.intensity_at(sim_t)
        forecast = [(t, trace.intensity_at(sim_t + t)) for t in range(0, tick_s * 6, tick_s)]
        current_gpus = decide_fn(current_gpus, intensity, forecast)
        current_gpus = max(1, min(max_gpus, current_gpus))
        gpus_history.append(current_gpus)

        # Concave scaling: throughput = base * gpus^0.85.
        throughput = base_throughput * (current_gpus ** 0.85)
        iter_done += int(throughput * tick_s)

        # Primary: energy this tick (kWh). In a real run this comes from NVML.
        kwh = (power_per_gpu_w * current_gpus * tick_s) / 3_600_000.0
        total_energy_kwh += kwh
        # Derived: carbon proxy = energy × grid intensity.
        total_emissions += kwh * intensity

        sim_t += tick_s

    return SimResult(
        name=name,
        total_energy_kwh=total_energy_kwh,
        total_emissions_g=total_emissions,
        average_gpus=sum(gpus_history) / max(1, len(gpus_history)),
        final_iter_done=iter_done,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="traces/synthetic_solar.csv")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--target-iters", type=int, default=200_000)
    parser.add_argument("--max-gpus", type=int, default=8)
    args = parser.parse_args()

    trace = load_csv_trace(args.trace)
    tick_s = 300  # 5-minute decision cadence

    rb = RuleBasedPolicy(min_gpus=1, max_gpus=args.max_gpus)
    mpc = MPCPolicy(
        min_gpus=1,
        max_gpus=args.max_gpus,
        horizon_steps=6,
        step_seconds=tick_s,
        power_per_gpu_w=300.0,
        throughput_per_gpu=lambda g: 5.0 * (g ** 0.85),
        iterations_remaining=args.target_iters,
        deadline_seconds_remaining=args.hours * 3600.0,
    )

    results = [
        simulate("static-max", lambda c, i, f: args.max_gpus, trace,
                 target_iters=args.target_iters, max_gpus=args.max_gpus,
                 power_per_gpu_w=300.0, base_throughput=5.0, tick_s=tick_s),
        simulate("rule-based", lambda c, i, f: rb.decide(c, i).target_gpus, trace,
                 target_iters=args.target_iters, max_gpus=args.max_gpus,
                 power_per_gpu_w=300.0, base_throughput=5.0, tick_s=tick_s),
        simulate("mpc", lambda c, i, f: mpc.decide(c, f).target_gpus, trace,
                 target_iters=args.target_iters, max_gpus=args.max_gpus,
                 power_per_gpu_w=300.0, base_throughput=5.0, tick_s=tick_s),
    ]

    console = Console()
    table = Table(title=f"HISE exp02 — energy + carbon-proxy replay ({args.trace})")
    table.add_column("policy")
    table.add_column("energy (kWh)\n[primary]", justify="right")
    table.add_column("Δ vs static", justify="right")
    table.add_column("carbon proxy (kg CO2)\n[derived]", justify="right")
    table.add_column("Δ vs static", justify="right")
    table.add_column("avg gpus", justify="right")
    table.add_column("iters done", justify="right")
    baseline_kwh = results[0].total_energy_kwh
    baseline_g = results[0].total_emissions_g
    for r in results:
        kwh_ratio = r.total_energy_kwh / max(baseline_kwh, 1e-9)
        g_ratio = r.total_emissions_g / max(baseline_g, 1e-9)
        table.add_row(
            r.name,
            f"{r.total_energy_kwh:.3f}",
            f"{kwh_ratio:.0%}",
            f"{r.total_emissions_g / 1000:.2f}",
            f"{g_ratio:.0%}",
            f"{r.average_gpus:.2f}",
            f"{r.final_iter_done:,}",
        )
    console.print(table)
    console.print(
        "\n[dim]Note: energy is the primary, defensible metric (in a real run this is "
        "measured via NVML). Carbon column is derived via grid intensity proxy and varies "
        "with the trace.[/dim]"
    )


if __name__ == "__main__":
    main()

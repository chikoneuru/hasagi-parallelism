"""exp01 — smoke test: one job through the control loop over a synthetic 24h carbon trace.

Runs entirely in-process (no Docker, no GPU). Prints a table of per-tick decisions so you
can eyeball whether the rule-based policy scales down during the solar-peak inversion.

Usage:
    python experiments/exp01_smoke_test.py
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from hasagi.admission.mss import ScalingCurve
from hasagi.energy.carbon_trace import synthetic_solar_trace
from hasagi.energy.policy import RuleBasedPolicy
from hasagi.orchestrator.control_loop import ControlLoop, admit_or_drop
from hasagi.orchestrator.job import Job, JobState, JobStore
from hasagi.parallel.planner import SimpleRuntimeModel
from hasagi.pool.local_pool import SimulatedPool


def main() -> None:
    console = Console()
    trace = synthetic_solar_trace(hours=24)
    store = JobStore()
    pool = SimulatedPool()

    job = Job.new(
        model_name="resnet18",
        dataset="cifar10",
        deadline_s=8 * 3600.0,
        iterations_target=20_000,
        carbon_budget_g=2_000.0,
    )

    curve = ScalingCurve(throughput_per_gpu_count=[(x ** 0.85) for x in range(1, 17)])
    if not admit_or_drop(job, curve):
        console.print(f"[red]Job dropped: {job.last_decision_reason}")
        return
    job.state = JobState.RUNNING
    store.add(job)

    sim_time_s = 0.0
    intensity_holder = {"t": 0.0}

    def intensity_now() -> float:
        return trace.intensity_at(intensity_holder["t"])

    runtime_model = SimpleRuntimeModel(
        per_sample_flops=2e9,
        model_bytes=12_000_000,
        device_throughput_flops=1e12,
        network_bandwidth_bps=10e9,
    )

    loop = ControlLoop(
        job_store=store,
        energy_policy=RuleBasedPolicy(min_gpus=1, max_gpus=8),
        intensity_at_now=intensity_now,
        runtime_model=runtime_model,
        pool_scale_fn=lambda jid, target: pool.scale(jid, target),
    )

    table = Table(title="HASAGI smoke test — 24h synthetic solar trace")
    table.add_column("hour")
    table.add_column("intensity\n(gCO2/kWh)", justify="right")
    table.add_column("gpus", justify="right")
    table.add_column("(dp, mp)")
    table.add_column("state")
    table.add_column("reason")

    tick_s = 30 * 60.0  # 30 minutes
    for tick in range(48):
        intensity_holder["t"] = sim_time_s
        loop.tick(now_seconds=sim_time_s)
        sim_time_s += tick_s
        j = store.get(job.job_id)
        table.add_row(
            f"{tick * 0.5:4.1f}",
            f"{intensity_now():.0f}",
            str(j.allocated_gpus),
            str(j.parallelism),
            j.state.value,
            j.last_decision_reason,
        )

    console.print(table)
    console.print(f"\n[green]Pool scale events:[/green] {len(pool.events)}")


if __name__ == "__main__":
    main()

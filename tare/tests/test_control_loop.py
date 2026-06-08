"""Unit test the orchestrator control loop end-to-end (no FastAPI)."""
from __future__ import annotations

from tare.admission.mss import ScalingCurve
from tare.energy.carbon_trace import synthetic_solar_trace
from tare.energy.policy import RuleBasedPolicy
from tare.orchestrator.control_loop import ControlLoop, admit_or_drop
from tare.orchestrator.job import Job, JobState, JobStore
from tare.parallel.planner import SimpleRuntimeModel
from tare.pool.local_pool import SimulatedPool


def test_smoke_run_records_scale_events() -> None:
    trace = synthetic_solar_trace(hours=24)
    store = JobStore()
    pool = SimulatedPool()
    curve = ScalingCurve(throughput_per_gpu_count=[x ** 0.85 for x in range(1, 17)])

    job = Job.new(
        model_name="resnet18", dataset="cifar10",
        deadline_s=8 * 3600.0, iterations_target=20_000,
    )
    assert admit_or_drop(job, curve)
    job.state = JobState.RUNNING
    store.add(job)

    holder = {"t": 0.0}

    loop = ControlLoop(
        job_store=store,
        energy_policy=RuleBasedPolicy(min_gpus=1, max_gpus=8),
        intensity_at_now=lambda: trace.intensity_at(holder["t"]),
        runtime_model=SimpleRuntimeModel(
            per_sample_flops=2e9, model_bytes=12_000_000,
            device_throughput_flops=1e12, network_bandwidth_bps=10e9,
        ),
        pool_scale_fn=lambda jid, target: pool.scale(jid, target),
    )

    for tick in range(12):
        holder["t"] = tick * 30 * 60.0
        loop.tick(now_seconds=holder["t"])

    assert len(pool.events) == 12
    # State should still be RUNNING or PAUSED, not COMPLETED/FAILED.
    j = store.get(job.job_id)
    assert j.state in {JobState.RUNNING, JobState.PAUSED}

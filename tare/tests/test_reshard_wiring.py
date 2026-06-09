"""CPU tests for the live-reconfiguration wiring (Uplift 3 control-loop side).

Covers the three integration points that let a layout decision flow to a state
move without a GPU: the control loop emits a ``ReshardEvent`` when a running
job's world size or parallelism shape changes (and not when it holds or pauses),
the host trainer applies the move through ``ReshardController``, and the Knative
worker records the reconfiguration as a lifecycle event.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tare.energy.policy import PowerAwareRulePolicy
from tare.energy.telemetry import WorkerTelemetry
from tare.orchestrator.energy_aware_control_loop import (
    EnergyAwareControlLoop,
    ReshardEvent,
)
from tare.orchestrator.job import Job, JobState, JobStore
from tare.parallel.planner import SimpleRuntimeModel
from tare.worker import knative_main as km
from tare.worker.host_trainer import HostTrainer


def _runtime() -> SimpleRuntimeModel:
    return SimpleRuntimeModel(
        per_sample_flops=2e9, model_bytes=12_000_000,
        device_throughput_flops=1e12, network_bandwidth_bps=10e9,
    )


def _policy() -> PowerAwareRulePolicy:
    return PowerAwareRulePolicy(
        min_gpus=1, max_gpus=8,
        scale_down_above_j_per_iter=3.0, scale_up_below_j_per_iter=1.5,
        hysteresis_ticks=1,
    )


def _job(allocated: int) -> Job:
    job = Job.new(model_name="resnet18", dataset="cifar10",
                  deadline_s=8 * 3600.0, iterations_target=20_000)
    job.state = JobState.RUNNING
    job.allocated_gpus = allocated
    return job


def _telem(power_w: float, throughput: float) -> WorkerTelemetry:
    return WorkerTelemetry(
        worker_id="w1", stage_id=0, gpu_type="A100",
        power_draw_w=power_w, throughput_iters_per_s=throughput,
        energy_cumulative_kwh=0.0, power_cap_w=400.0,
        memory_used_bytes=8 << 30, temperature_c=60.0, timestamp_s=0.0,
    )


# --- control loop emission ------------------------------------------------- #
def test_control_loop_emits_reshard_event_on_world_change() -> None:
    store = JobStore()
    job = _job(allocated=4)
    store.add(job)
    tel = {"w1": _telem(power_w=400.0, throughput=100.0)}  # 4 J/iter -> scale down
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: tel, runtime_model=_runtime(),
    )
    result = loop.tick(now_seconds=0.0)

    assert result.decisions[job.job_id].target_gpus == 3
    ev = result.reshard_events.get(job.job_id)
    assert isinstance(ev, ReshardEvent)
    assert ev.from_world == 4 and ev.to_world == 3
    assert "->" in ev.reason


def test_no_reshard_event_when_layout_held() -> None:
    """A single-GPU job with no telemetry holds at (1,(1,1)) -> no transition."""
    store = JobStore()
    job = _job(allocated=1)
    store.add(job)
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: {}, runtime_model=_runtime(),
    )
    result = loop.tick(now_seconds=0.0)

    assert result.decisions[job.job_id].target_gpus == 1
    assert job.job_id not in result.reshard_events


def test_second_identical_tick_emits_no_event() -> None:
    """Once the layout is established, an unchanged tick emits nothing."""
    store = JobStore()
    job = _job(allocated=4)
    store.add(job)
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: {}, runtime_model=_runtime(),  # no telemetry -> hold
    )
    loop.tick(now_seconds=0.0)                       # establishes layout for 4 GPUs
    result2 = loop.tick(now_seconds=1.0)             # same world + same shape
    assert job.job_id not in result2.reshard_events


def test_no_reshard_event_when_multigpu_held() -> None:
    """A 4-GPU job merely held emits nothing, even though its default
    parallelism (1,1) differs from the planner shape for 4 GPUs (regression:
    the gate must key off world size, not the shape tuple)."""
    store = JobStore()
    job = _job(allocated=4)
    store.add(job)
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: {}, runtime_model=_runtime(),  # no telemetry -> hold at 4
    )
    result = loop.tick(now_seconds=0.0)
    assert result.decisions[job.job_id].target_gpus == 4
    assert job.job_id not in result.reshard_events


def test_no_reshard_event_on_paused_resume() -> None:
    """A PAUSED job resumed with a changed world reconfigures via the
    pause/resume checkpoint path, so no reshard event is emitted."""
    store = JobStore()
    job = _job(allocated=2)
    job.state = JobState.PAUSED
    job.parallelism = (1, 2)
    store.add(job)
    tel = {"w1": _telem(power_w=100.0, throughput=100.0)}  # 1 J/iter -> scale up
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: tel, runtime_model=_runtime(),
    )
    result = loop.tick(now_seconds=0.0)
    assert result.decisions[job.job_id].target_gpus == 3   # world changed 2 -> 3
    assert job.job_id not in result.reshard_events          # but was PAUSED -> no event


# --- host trainer hook ----------------------------------------------------- #
def test_host_trainer_reshard_preserves_params() -> None:
    ht = HostTrainer()
    ht._model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    before = {k: v.clone() for k, v in ht._model.state_dict().items()}
    cert = ht.reshard(to_world=4)
    assert cert.ok and cert.max_abs_diff == 0.0
    after = ht._model.state_dict()
    for k in before:
        assert torch.allclose(after[k], before[k]), k


def test_host_trainer_reshard_requires_built_model() -> None:
    with pytest.raises(RuntimeError, match="model must be built"):
        HostTrainer().reshard(to_world=2)


# --- knative lifecycle event ----------------------------------------------- #
def test_knative_reshard_records_lifecycle() -> None:
    km.RESHARD_S = 0.0                                   # no busy-sleep in the test
    km.TIMESTAMPS.cuda_init_complete_offset_s = 0.0      # skip the mock CUDA init
    before = km.TIMESTAMPS.reshard_count
    out = km.reshard(km.ReshardRequest(job_id="j1", from_world=4, to_world=2))
    assert out["status"] == "ok"
    assert out["request"]["from_world"] == 4 and out["request"]["to_world"] == 2
    assert out["lifecycle"]["reshard_count"] == before + 1
    assert out["lifecycle"]["first_reshard_completed_offset_s"] is not None


def test_knative_reshard_triggers_cuda_init_and_idempotent_stamps() -> None:
    """First /reshard drives the mock CUDA init (like /work); the first-stamps
    are set once and not overwritten on a later call."""
    saved_ts, saved_r, saved_c = km.TIMESTAMPS, km.RESHARD_S, km.CUDA_INIT_S
    try:
        km.RESHARD_S = 0.0
        km.CUDA_INIT_S = 0.0
        km.TIMESTAMPS = km.LifecycleTimestamps(container_start_unix=0.0)
        assert km.TIMESTAMPS.cuda_init_complete_offset_s is None
        km.reshard(km.ReshardRequest(job_id="j", from_world=2, to_world=1))
        assert km.TIMESTAMPS.cuda_init_complete_offset_s is not None   # first call drove CUDA init
        first_recv = km.TIMESTAMPS.first_reshard_received_offset_s
        assert first_recv is not None
        km.reshard(km.ReshardRequest(job_id="j", from_world=1, to_world=2))
        assert km.TIMESTAMPS.first_reshard_received_offset_s == first_recv   # idempotent stamp
        assert km.TIMESTAMPS.reshard_count == 2
    finally:
        km.TIMESTAMPS, km.RESHARD_S, km.CUDA_INIT_S = saved_ts, saved_r, saved_c

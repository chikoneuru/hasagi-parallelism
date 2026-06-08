"""Mid-run deadline-floor selector + control-loop override tests."""
from __future__ import annotations

import time

from tare.admission.mss import ScalingCurve
from tare.energy.policy import PowerAwareRulePolicy
from tare.energy.telemetry import WorkerTelemetry
from tare.orchestrator.deadline_selector import DeadlineFloor, DeadlineFloorSelector
from tare.orchestrator.energy_aware_control_loop import EnergyAwareControlLoop
from tare.orchestrator.job import Job, JobState, JobStore
from tare.parallel.planner import SimpleRuntimeModel

# --- Helpers ---


def _runtime() -> SimpleRuntimeModel:
    return SimpleRuntimeModel(
        per_sample_flops=2e9, model_bytes=12_000_000,
        device_throughput_flops=1e12, network_bandwidth_bps=10e9,
    )


def _curve() -> ScalingCurve:
    # Concave, monotone-increasing curve in iter/s for 1..8 GPUs.
    return ScalingCurve(
        throughput_per_gpu_count=(1.0, 1.9, 2.7, 3.4, 4.0, 4.5, 4.9, 5.2),
    )


def _telem(power_w: float, throughput: float) -> WorkerTelemetry:
    return WorkerTelemetry(
        worker_id="w1", stage_id=0, gpu_type="A100",
        power_draw_w=power_w, throughput_iters_per_s=throughput,
        energy_cumulative_kwh=0.0, power_cap_w=400.0,
        memory_used_bytes=8 << 30, temperature_c=60.0, timestamp_s=0.0,
    )


def _policy() -> PowerAwareRulePolicy:
    return PowerAwareRulePolicy(
        min_gpus=1, max_gpus=8,
        scale_down_above_j_per_iter=3.0,
        scale_up_below_j_per_iter=1.5,
        hysteresis_ticks=1,
    )


# --- Job helpers ---


def test_job_iters_remaining_basic() -> None:
    job = Job.new(model_name="m", dataset="d", deadline_s=100.0, iterations_target=1000)
    assert job.iters_remaining() == 1000
    job.iterations_done = 400
    assert job.iters_remaining() == 600


def test_job_iters_remaining_clamped_at_zero() -> None:
    """iterations_done overshoot must not produce a negative remainder."""
    job = Job.new(model_name="m", dataset="d", deadline_s=100.0, iterations_target=500)
    job.iterations_done = 600
    assert job.iters_remaining() == 0


def test_job_deadline_seconds_remaining_in_window() -> None:
    job = Job.new(model_name="m", dataset="d", deadline_s=1000.0, iterations_target=10)
    # submitted_at defaults to time.time(); evaluate just after.
    now = job.submitted_at + 200.0
    assert abs(job.deadline_seconds_remaining(now) - 800.0) < 1e-6


def test_job_deadline_seconds_remaining_after_expiry() -> None:
    job = Job.new(model_name="m", dataset="d", deadline_s=100.0, iterations_target=10)
    now = job.submitted_at + 150.0
    assert job.deadline_seconds_remaining(now) == 0.0


def test_job_deadline_seconds_remaining_default_now() -> None:
    """Default ``now=None`` reads ``time.time()`` — sanity-check it's positive."""
    job = Job.new(model_name="m", dataset="d", deadline_s=3600.0, iterations_target=10)
    remaining = job.deadline_seconds_remaining()
    assert 0.0 < remaining <= 3600.0


# --- DeadlineFloorSelector ---


def test_floor_zero_iters_returns_min() -> None:
    sel = DeadlineFloorSelector(curve=_curve(), min_gpus=2)
    floor = sel.evaluate(iters_remaining=0, deadline_seconds_remaining=100.0)
    assert floor.gpus == 2
    assert floor.feasible is True
    assert "no iters remaining" in floor.reason


def test_floor_expired_deadline_escalates() -> None:
    sel = DeadlineFloorSelector(curve=_curve())
    floor = sel.evaluate(iters_remaining=100, deadline_seconds_remaining=0.0)
    assert floor.gpus == _curve().max_gpus
    assert floor.feasible is False
    assert "deadline already passed" in floor.reason


def test_floor_infeasible_max_throughput_insufficient() -> None:
    """Required rate > max throughput → infeasible, escalate to ceiling."""
    sel = DeadlineFloorSelector(curve=_curve())
    # 1000 iters in 100s → 10 iter/s; max curve throughput is 5.2 iter/s.
    floor = sel.evaluate(iters_remaining=1000, deadline_seconds_remaining=100.0)
    assert floor.feasible is False
    assert floor.gpus == _curve().max_gpus
    assert "infeasible" in floor.reason


def test_floor_feasible_matches_mss() -> None:
    """100 iters in 50s → 2 iter/s required → curve hits ≥2 at 3 GPUs (2.7)."""
    sel = DeadlineFloorSelector(curve=_curve())
    floor = sel.evaluate(iters_remaining=100, deadline_seconds_remaining=50.0)
    assert floor.feasible is True
    assert floor.gpus == 3
    assert "MSS=3" in floor.reason


def test_floor_respects_min_gpus_clamp() -> None:
    """When MSS=1 but min_gpus=3, floor lifts to 3."""
    sel = DeadlineFloorSelector(curve=_curve(), min_gpus=3)
    floor = sel.evaluate(iters_remaining=10, deadline_seconds_remaining=100.0)
    assert floor.gpus == 3
    assert floor.feasible is True


def test_floor_infeasible_floor_override() -> None:
    """Custom infeasible_floor caps escalation."""
    sel = DeadlineFloorSelector(curve=_curve(), infeasible_floor=4)
    floor = sel.evaluate(iters_remaining=1000, deadline_seconds_remaining=100.0)
    assert floor.gpus == 4
    assert floor.feasible is False


# --- Control loop integration ---


def _job_running(allocated: int) -> Job:
    job = Job.new(
        model_name="resnet18", dataset="cifar10",
        deadline_s=10_000.0, iterations_target=20_000,
    )
    job.state = JobState.RUNNING
    job.allocated_gpus = allocated
    return job


def test_control_loop_no_selector_no_override() -> None:
    """Backwards compat: jobs without a selector follow policy verbatim."""
    store = JobStore()
    job = _job_running(allocated=4)
    store.add(job)
    tel = {"w1": _telem(power_w=400.0, throughput=100.0)}   # 4 J/iter → scale down
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: tel, runtime_model=_runtime(),
    )
    result = loop.tick(now_seconds=time.time())
    assert result.decisions[job.job_id].target_gpus == 3
    assert result.deadline_floors == {}
    assert result.deadline_overrides == {}
    assert store.get(job.job_id).allocated_gpus == 3


def test_control_loop_floor_lifts_policy_target() -> None:
    """Policy says 3, but deadline floor demands more → loop overrides upward."""
    store = JobStore()
    # 18_000 iters left in 5_000s → required rate 3.6 iter/s → MSS hits at 5 GPUs (curve 4.0).
    job = _job_running(allocated=4)
    job.iterations_target = 20_000
    job.iterations_done = 2_000
    job.submitted_at = time.time() - 5_000.0   # deadline_s=10_000 → 5_000s remain
    store.add(job)

    tel = {"w1": _telem(power_w=400.0, throughput=100.0)}   # 4 J/iter → policy says 3
    sel = DeadlineFloorSelector(curve=_curve())
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: tel, runtime_model=_runtime(),
        deadline_floor_selectors={job.job_id: sel},
    )
    result = loop.tick(now_seconds=time.time())

    assert result.decisions[job.job_id].target_gpus == 3   # raw policy
    assert job.job_id in result.deadline_floors
    floor = result.deadline_floors[job.job_id]
    assert floor.feasible is True
    assert floor.gpus >= 5
    assert job.job_id in result.deadline_overrides
    assert "deadline floor lifted" in result.deadline_overrides[job.job_id]
    # Store reflects the lifted allocation.
    assert store.get(job.job_id).allocated_gpus == floor.gpus


def test_control_loop_floor_passthrough_when_policy_higher() -> None:
    """Policy already >= floor → no override emitted (but floor still recorded)."""
    store = JobStore()
    # Generous deadline: 1000 iters in 10_000s → required 0.1 iter/s → MSS=1.
    job = _job_running(allocated=4)
    job.iterations_target = 2_000
    job.iterations_done = 1_000
    store.add(job)

    tel = {"w1": _telem(power_w=100.0, throughput=100.0)}   # 1 J/iter → policy scale up to 5
    sel = DeadlineFloorSelector(curve=_curve())
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: tel, runtime_model=_runtime(),
        deadline_floor_selectors={job.job_id: sel},
    )
    result = loop.tick(now_seconds=time.time())

    assert result.decisions[job.job_id].target_gpus == 5
    assert result.deadline_floors[job.job_id].gpus == 1
    assert result.deadline_overrides == {}
    assert store.get(job.job_id).allocated_gpus == 5


def test_control_loop_infeasible_deadline_escalates_to_ceiling() -> None:
    """Expired deadline → selector returns ceiling → loop runs at ceiling."""
    store = JobStore()
    job = _job_running(allocated=2)
    job.iterations_target = 20_000
    job.iterations_done = 5_000
    job.deadline_s = 100.0
    job.submitted_at = time.time() - 200.0   # already expired
    store.add(job)

    tel = {"w1": _telem(power_w=400.0, throughput=100.0)}   # policy wants scale-down
    sel = DeadlineFloorSelector(curve=_curve())
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: tel, runtime_model=_runtime(),
        deadline_floor_selectors={job.job_id: sel},
    )
    result = loop.tick(now_seconds=time.time())

    assert result.deadline_floors[job.job_id].feasible is False
    assert result.deadline_floors[job.job_id].gpus == _curve().max_gpus
    assert store.get(job.job_id).allocated_gpus == _curve().max_gpus
    assert "deadline already passed" in result.deadline_overrides[job.job_id]


def test_control_loop_selector_only_one_job() -> None:
    """Selector is per-job: jobs without a selector still skip the floor."""
    store = JobStore()
    job_with = _job_running(allocated=4)
    job_with.iterations_target = 20_000
    job_with.iterations_done = 2_000
    job_with.submitted_at = time.time() - 5_000.0   # 5_000s remain → needs MSS≥5
    store.add(job_with)

    job_without = _job_running(allocated=4)
    store.add(job_without)

    tel = {"w1": _telem(power_w=400.0, throughput=100.0)}   # both → scale down
    sel = DeadlineFloorSelector(curve=_curve())
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=_policy(),
        telemetry_source=lambda: tel, runtime_model=_runtime(),
        deadline_floor_selectors={job_with.job_id: sel},
    )
    result = loop.tick(now_seconds=time.time())

    assert job_with.job_id in result.deadline_overrides
    assert job_without.job_id not in result.deadline_overrides
    assert store.get(job_with.job_id).allocated_gpus >= 5
    assert store.get(job_without.job_id).allocated_gpus == 3


def test_deadline_floor_dataclass_repr() -> None:
    """DeadlineFloor is informational — its fields must be stable for logging."""
    floor = DeadlineFloor(
        gpus=3, feasible=True, iters_remaining=100,
        deadline_seconds_remaining=50.0, reason="MSS=3",
    )
    assert floor.gpus == 3
    assert floor.feasible is True
    assert floor.iters_remaining == 100
    assert floor.deadline_seconds_remaining == 50.0

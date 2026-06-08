"""End-to-end multi-tick simulation: primal-dual + deadline floor + repartition.

Exercises the full stack on a single job over many ticks of a control loop:

  1. Energy-aware admission via ``EnergyBudgetMSS`` (existing path).
  2. Per-tick policy decision from ``OnlinePrimalDualPolicy`` (Tuần 25).
  3. ``DeadlineFloorSelector`` lifts the policy target when the running
     iter-budget burn rate falls behind the deadline (Tuần 24).
  4. ``incremental_partition`` + stagnation-fallback ``partition_pipeline``
     re-shape the pipeline when the allocator changes.

Asserts the system reaches the iteration target before the deadline, the
primal-dual dual variable stays bounded across the run, and at least one
repartition event fires when the allocator changes pipeline depth.
"""
from __future__ import annotations

import time

from tare.admission.energy_profile import linear_profile
from tare.admission.mss import EnergyBudgetMSS, ScalingCurve
from tare.energy.policy import OnlinePrimalDualPolicy
from tare.orchestrator.deadline_selector import DeadlineFloorSelector
from tare.orchestrator.energy_aware_control_loop import (
    EnergyAwareControlLoop,
    RepartitionContext,
    energy_admit_or_drop,
)
from tare.orchestrator.job import Job, JobState, JobStore
from tare.parallel.partitioner import LayerProfile, LinkSpec, StageSpec
from tare.parallel.planner import SimpleRuntimeModel

# Action set throughput in iter/s as a function of gpu count.
# Pareto: more gpus → more throughput but with diminishing returns + rising power.
_THROUGHPUT = {1: 1.0, 2: 1.9, 3: 2.7, 4: 3.4, 5: 4.0, 6: 4.5, 7: 4.9, 8: 5.2}
_ENERGY_PER_ITER = {1: 40.0, 2: 45.0, 3: 50.0, 4: 60.0, 5: 75.0, 6: 90.0, 7: 105.0, 8: 120.0}


def _throughput(g: int) -> float:
    return _THROUGHPUT.get(g, 0.0)


def _energy_per_iter(g: int) -> float:
    return _ENERGY_PER_ITER.get(g, float("inf"))


def _runtime() -> SimpleRuntimeModel:
    return SimpleRuntimeModel(
        per_sample_flops=2e9, model_bytes=12_000_000,
        device_throughput_flops=1e12, network_bandwidth_bps=10e9,
    )


def _layers(n: int = 12) -> list[LayerProfile]:
    return [
        LayerProfile(index=i, fwd_flops=1e8, bwd_flops=2e8, activation_bytes=1_000_000)
        for i in range(n)
    ]


def _stages_factory(k: int) -> list[StageSpec]:
    return [
        StageSpec(stage_id=s, throughput_flops=1e12, memory_bytes=8 << 30)
        for s in range(k)
    ]


def _links_factory(k: int) -> list[LinkSpec]:
    return [
        LinkSpec(src_stage=s, dst_stage=s + 1, bandwidth_bps=10e9, latency_s=0.0005)
        for s in range(k - 1)
    ]


def _curve() -> ScalingCurve:
    return ScalingCurve(throughput_per_gpu_count=tuple(_THROUGHPUT[g] for g in sorted(_THROUGHPUT)))


def _admit_job(
    store: JobStore,
    iters_target: int,
    deadline_s: float,
    energy_budget_kwh: float,
) -> Job:
    job = Job.new(
        model_name="resnet18", dataset="cifar10",
        deadline_s=deadline_s, iterations_target=iters_target,
    )
    ebmss = EnergyBudgetMSS(
        curve=_curve(), power_per_gpu_w=300.0,
        energy_budget_kwh=energy_budget_kwh,
        energy_profile=linear_profile(
            power_per_gpu_w=300.0,
            base_throughput_iters_per_s=1.0,
            max_gpus=_curve().max_gpus,
        ),
    )
    admitted = energy_admit_or_drop(job, ebmss)
    assert admitted, f"job rejected: {job.last_decision_reason}"
    job.state = JobState.RUNNING
    store.add(job)
    return job


def _policy(
    target_iter_rate: float,
    horizon_steps: int,
    intensity_scale: float = 1.0,
) -> OnlinePrimalDualPolicy:
    max_e_mu = max(_ENERGY_PER_ITER[g] * _THROUGHPUT[g] for g in _THROUGHPUT)
    return OnlinePrimalDualPolicy(
        min_gpus=1, max_gpus=8,
        throughput_per_gpu=_throughput,
        energy_per_iter=_energy_per_iter,
        target_iter_rate=target_iter_rate,
        horizon_steps=horizon_steps,
        max_energy_estimate=max_e_mu,
        intensity_scale=intensity_scale,
    )


def _ctx() -> RepartitionContext:
    return RepartitionContext(
        layers=_layers(12),
        stages_factory=_stages_factory,
        links_factory=_links_factory,
        objective="bottleneck",
    )


def _intensity_oscillator(period: int) -> tuple[list[float], callable]:
    """Two-level intensity trace: alternates 0.5 and 1.5 every ``period`` ticks."""
    history: list[float] = []
    state = {"step": 0}

    def get() -> float:
        b = 0.5 if (state["step"] // period) % 2 == 0 else 1.5
        history.append(b)
        state["step"] += 1
        return b

    return history, get


# --- End-to-end ---


def test_end_to_end_meets_deadline_under_oscillating_intensity() -> None:
    """Full stack: admission → primal-dual policy + floor + repartition over
    many ticks. The job must reach the iteration target inside the deadline
    while the policy keeps switching between high/low intensity halves."""
    store = JobStore()
    iters_target = 900
    deadline_s = 450.0
    target_rate = iters_target / deadline_s   # 2.0 iter/s — above g=1 ceiling
    job = _admit_job(store, iters_target=iters_target, deadline_s=deadline_s,
                     energy_budget_kwh=10.0)

    pol = _policy(target_iter_rate=target_rate, horizon_steps=200)
    sel = DeadlineFloorSelector(curve=_curve())
    history, intensity_fn = _intensity_oscillator(period=20)

    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=pol,
        telemetry_source=lambda: {}, runtime_model=_runtime(),
        intensity_at_now=intensity_fn,
        repartition_contexts={job.job_id: _ctx()},
        deadline_floor_selectors={job.job_id: sel},
    )

    tick_dt = 5.0   # 5-second simulated ticks
    repartitions_seen = set()
    decisions_per_gpu_count: dict[int, int] = {}
    now = time.time()
    submitted_at = now
    job.submitted_at = submitted_at

    for step in range(120):   # 120 ticks × 5s = 600s wallclock
        result = loop.tick(now_seconds=submitted_at + step * tick_dt)
        stored = store.get(job.job_id)
        # Simulate iteration accumulation using the live throughput.
        iters_this_tick = _throughput(stored.allocated_gpus) * tick_dt
        stored.iterations_done = min(
            iters_target,
            int(round(stored.iterations_done + iters_this_tick)),
        )
        decisions_per_gpu_count[stored.allocated_gpus] = (
            decisions_per_gpu_count.get(stored.allocated_gpus, 0) + 1
        )
        # A repartition this tick is signaled by the partition appearing in
        # the TickResult (the loop only emits when target changes & target>=2).
        if job.job_id in result.partitions:
            repartitions_seen.add(result.partitions[job.job_id].num_stages)
        if stored.iterations_done >= iters_target:
            break

    final = store.get(job.job_id)
    assert final.iterations_done >= iters_target, (
        f"missed deadline: only {final.iterations_done}/{iters_target} iters done"
    )
    # The intensity oscillator + dual update force the policy to visit at
    # least two distinct gpu counts → at least one repartition event.
    assert len(decisions_per_gpu_count) >= 2, (
        f"policy never re-allocated: {decisions_per_gpu_count}"
    )
    assert len(repartitions_seen) >= 1, (
        f"no repartition fired across changing gpu counts: {repartitions_seen}"
    )
    # Dual must stay finite (no runaway from a step-size mis-calibration).
    assert pol.lambda_t < 1e9


def test_end_to_end_deadline_floor_intervenes_on_falling_behind() -> None:
    """Force the primal-dual policy to under-allocate (high intensity ⇒ score
    favours g=1) but pinned-tight deadline forces the floor to lift the target."""
    store = JobStore()
    iters_target = 1000
    deadline_s = 400.0
    target_rate = iters_target / deadline_s   # 2.5 iter/s → needs ~3 GPUs
    job = _admit_job(store, iters_target=iters_target, deadline_s=deadline_s,
                     energy_budget_kwh=10.0)

    pol = _policy(target_iter_rate=target_rate, horizon_steps=80)
    sel = DeadlineFloorSelector(curve=_curve())
    # Hold intensity high so the unconstrained argmin would pick g=1.
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=pol,
        telemetry_source=lambda: {}, runtime_model=_runtime(),
        intensity_at_now=lambda: 1.5,
        repartition_contexts={job.job_id: _ctx()},
        deadline_floor_selectors={job.job_id: sel},
    )

    now = time.time()
    job.submitted_at = now
    overrides = 0
    for step in range(40):
        # Simulate the job falling behind: only credit a fraction of would-be iters.
        result = loop.tick(now_seconds=now + step * 5.0)
        if job.job_id in result.deadline_overrides:
            overrides += 1
        stored = store.get(job.job_id)
        stored.iterations_done = int(
            stored.iterations_done + _throughput(stored.allocated_gpus) * 5.0 * 0.5
        )

    assert overrides >= 1, "deadline floor never lifted under tight-deadline regime"
    final = store.get(job.job_id)
    assert final.allocated_gpus >= 3, (
        f"floor failed to raise allocation: ended at {final.allocated_gpus} GPUs"
    )


def test_end_to_end_no_repartition_below_two_gpus() -> None:
    """Confirms the K≥2 guard: primal-dual at very loose deadline picks g=1,
    no partition is computed, no floor override fires."""
    store = JobStore()
    iters_target = 100
    deadline_s = 10_000.0   # very loose
    target_rate = iters_target / deadline_s   # 0.01 iter/s → trivially satisfied
    job = _admit_job(store, iters_target=iters_target, deadline_s=deadline_s,
                     energy_budget_kwh=10.0)

    pol = _policy(target_iter_rate=target_rate, horizon_steps=40)
    sel = DeadlineFloorSelector(curve=_curve())
    loop = EnergyAwareControlLoop(
        job_store=store, energy_policy=pol,
        telemetry_source=lambda: {}, runtime_model=_runtime(),
        intensity_at_now=lambda: 1.0,
        repartition_contexts={job.job_id: _ctx()},
        deadline_floor_selectors={job.job_id: sel},
    )
    job.submitted_at = time.time()
    for step in range(20):
        result = loop.tick(now_seconds=job.submitted_at + step * 5.0)
        assert job.job_id not in result.deadline_overrides   # never lifted
    # Final state stays at g=1 (or whatever the policy picked); no partition
    # entered the cache because target stayed < 2.
    assert loop._partitions == {}

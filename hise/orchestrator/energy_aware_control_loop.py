"""Energy-first closed-loop orchestrator wiring Phase 2 Tuần 1-2 algorithms.

This is the **HISE primary control path**. The original `ControlLoop` keeps the
Phase 1 carbon-driven `RuleBasedPolicy` for ablation; this class swaps in the
energy-first stack:

* **A3 policy**: `PowerAwareRulePolicy` (J/iter threshold + hysteresis) OR
  `MPCPolicy` (receding-horizon with reconfig penalty).
* **A1 partitioner**: optional `incremental_partition` with `StagnationTracker`
  fallback to full `partition_pipeline` when the local window stagnates.
* **A2 admission** is consumed via `EnergyBudgetMSS` (called externally, like
  the original `admit_or_drop`) — kept as a separate helper since admission
  runs once per job submission, not per tick.
* **Telemetry source**: zero-arg callable returning the current per-worker
  telemetry map; production wires this to the NVML sidecar, tests wire it to
  a stub dict.

Each tick:
    1. Snapshot telemetry + (optional) carbon intensity.
    2. For each RUNNING / PAUSED job:
       a. Run energy policy → ``EnergyDecision``.
       b. If `target_gpus` changed AND repartition context is set: run
          `incremental_partition`; if `StagnationTracker.observe` triggers,
          run full `partition_pipeline` and reset the tracker.
       c. Update job allocation + parallelism strategy; trigger pool scale.
    3. Return `TickResult` with per-job decisions + strategies + partitions.

The control loop is a plain dataclass so unit tests can drive ticks
deterministically without FastAPI.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from hise.admission.mss import EnergyBudgetMSS
from hise.energy.policy import EnergyDecision, MPCPolicy, PowerAwareRulePolicy
from hise.energy.telemetry import WorkerTelemetry
from hise.orchestrator.job import Job, JobState, JobStore
from hise.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    Partition,
    StageSpec,
    StagnationTracker,
    incremental_partition,
    partition_pipeline,
)
from hise.parallel.planner import HybridStrategy, SimpleRuntimeModel, select_hybrid_strategy

logger = logging.getLogger(__name__)


@dataclass
class RepartitionContext:
    """Inputs needed to re-partition a job when its GPU count changes.

    The orchestrator stores one ``RepartitionContext`` per job. When the energy
    policy decides to scale, the loop calls ``incremental_partition`` with these
    inputs; if the ``StagnationTracker`` triggers, it falls back to
    ``partition_pipeline`` (full DP).

    Args:
        layers: per-layer profile of the model (fwd/bwd FLOPS + activation
            bytes). Stable across ticks for a single job.
        stages_factory: callable ``(num_stages) -> list[StageSpec]`` that builds
            the per-stage spec list at a given pipeline depth. Closes over
            current telemetry-driven ``power_draw_w`` and ``memory_bytes``.
        links_factory: callable ``(num_stages) -> list[LinkSpec]`` for the
            K-1 inter-stage links. Closes over current network bandwidth.
        objective: ``"bottleneck"`` (default) or ``"energy"`` for the DP scoring.
        boundary_window: incremental sliding window radius (default 3).
        num_microbatches: M for the pipeline bottleneck cost; default 1.
    """

    layers: Sequence[LayerProfile]
    stages_factory: Callable[[int], list[StageSpec]]
    links_factory: Callable[[int], list[LinkSpec]]
    objective: str = "bottleneck"
    boundary_window: int = 3
    num_microbatches: int = 1


@dataclass
class TickResult:
    """What the energy-aware control loop produced this tick."""

    tick_seconds: float
    intensity_g_per_kwh: float | None = None
    decisions: dict[str, EnergyDecision] = field(default_factory=dict)
    strategies: dict[str, HybridStrategy] = field(default_factory=dict)
    partitions: dict[str, Partition] = field(default_factory=dict)
    fallbacks: dict[str, str] = field(default_factory=dict)   # job_id → reason


@dataclass
class EnergyAwareControlLoop:
    """Energy-first closed-loop orchestrator. Replaces RuleBasedPolicy with
    PowerAwareRulePolicy (default) or MPCPolicy; adds telemetry feedback,
    optional pipeline repartitioning with stagnation-driven full-DP fallback.

    Args:
        job_store: shared job store.
        energy_policy: ``PowerAwareRulePolicy`` or ``MPCPolicy``. Both are
            accepted; the loop dispatches on type when invoking ``decide``.
        telemetry_source: zero-arg callable returning the latest per-worker
            telemetry map ``{worker_id: WorkerTelemetry}``. Empty dict is OK
            (PowerAwareRule falls back to "no telemetry — hold"; MPC does not
            require telemetry but the loop reads ``intensity_at_now`` instead).
        runtime_model: ``SimpleRuntimeModel`` for the hybrid strategy planner.
        intensity_at_now: optional zero-arg callable returning current grid
            intensity gCO2/kWh. Required for MPC; optional for PowerAwareRule
            (only used as carbon spatial-shift gate).
        intensity_forecast: optional zero-arg callable returning the forecast
            list ``[(t_offset, intensity), ...]`` for MPC's horizon. Ignored
            when policy is PowerAwareRule.
        repartition_contexts: optional ``{job_id: RepartitionContext}`` map.
            Jobs without a context skip repartitioning (allocation update only).
        stagnation_patience: ticks of no-improvement before fallback to full DP.
        cluster_size: total GPU budget for the cluster (used by hybrid planner).
        pool_scale_fn: optional ``(job_id, target_gpus) -> None`` side effect.
    """

    job_store: JobStore
    energy_policy: PowerAwareRulePolicy | MPCPolicy
    telemetry_source: Callable[[], Mapping[str, WorkerTelemetry]]
    runtime_model: SimpleRuntimeModel
    intensity_at_now: Callable[[], float] | None = None
    intensity_forecast: Callable[[], list[tuple[float, float]]] | None = None
    repartition_contexts: dict[str, RepartitionContext] = field(default_factory=dict)
    stagnation_patience: int = 3
    cluster_size: int = 16
    pool_scale_fn: Callable[[str, int], None] | None = None

    _partitions: dict[str, Partition] = field(default_factory=dict, init=False, repr=False)
    _trackers: dict[str, StagnationTracker] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.energy_policy, MPCPolicy) and self.intensity_forecast is None:
            raise ValueError(
                "MPCPolicy requires intensity_forecast callable for the horizon"
            )

    def _invoke_policy(self, current_gpus: int) -> EnergyDecision:
        """Dispatch to the right ``decide()`` signature based on policy type."""
        if isinstance(self.energy_policy, PowerAwareRulePolicy):
            tel = self.telemetry_source()
            intensity = self.intensity_at_now() if self.intensity_at_now else None
            return self.energy_policy.decide(current_gpus, tel, intensity)
        if isinstance(self.energy_policy, MPCPolicy):
            forecast = self.intensity_forecast() if self.intensity_forecast else []
            return self.energy_policy.decide(current_gpus, forecast)
        raise TypeError(f"Unsupported energy policy type: {type(self.energy_policy).__name__}")

    def _tracker_for(self, job_id: str, objective: str) -> StagnationTracker:
        """Lazy-create + cache per-job StagnationTracker."""
        if job_id not in self._trackers:
            self._trackers[job_id] = StagnationTracker(
                patience=self.stagnation_patience,
                objective=objective,
            )
        return self._trackers[job_id]

    def _repartition(
        self,
        job_id: str,
        target_gpus: int,
        result: TickResult,
    ) -> Partition | None:
        """Run incremental partition; fall back to full DP on stagnation.

        Returns the new ``Partition`` or ``None`` if no repartition context was
        registered for this job. Updates ``result.fallbacks[job_id]`` when the
        full DP escape is taken.
        """
        ctx = self.repartition_contexts.get(job_id)
        if ctx is None:
            return None

        stages = ctx.stages_factory(target_gpus)
        links = ctx.links_factory(target_gpus)
        prev = self._partitions.get(job_id)
        tracker = self._tracker_for(job_id, ctx.objective)

        try:
            if prev is None or len(prev.cuts) != target_gpus - 1:
                # First partition for this job, OR pipeline depth changed →
                # incremental cannot reuse the old cuts; do full DP.
                new_partition = partition_pipeline(
                    ctx.layers, stages, links,
                    num_microbatches=ctx.num_microbatches,
                    objective=ctx.objective,
                )
                tracker.reset_all()
            else:
                # Same depth, slide cuts within window.
                new_partition = incremental_partition(
                    prev, ctx.layers, stages, links,
                    boundary_window=ctx.boundary_window,
                    num_microbatches=ctx.num_microbatches,
                    objective=ctx.objective,
                )
                if tracker.observe(new_partition):
                    # Window has stagnated — escape via full DP.
                    new_partition = partition_pipeline(
                        ctx.layers, stages, links,
                        num_microbatches=ctx.num_microbatches,
                        objective=ctx.objective,
                    )
                    tracker.reset()
                    result.fallbacks[job_id] = (
                        f"stagnation after {self.stagnation_patience} ticks → full DP"
                    )
        except RuntimeError as e:
            logger.warning("repartition failed for job %s: %s", job_id, e)
            return prev   # keep old partition; caller handles allocation change

        self._partitions[job_id] = new_partition
        return new_partition

    def tick(self, now_seconds: float | None = None) -> TickResult:
        now = now_seconds if now_seconds is not None else time.time()
        intensity = self.intensity_at_now() if self.intensity_at_now else None
        result = TickResult(tick_seconds=now, intensity_g_per_kwh=intensity)

        jobs = (
            self.job_store.by_state(JobState.RUNNING)
            + self.job_store.by_state(JobState.PAUSED)
        )
        for job in jobs:
            current_gpus = max(1, job.allocated_gpus or 1)
            decision = self._invoke_policy(current_gpus)
            result.decisions[job.job_id] = decision

            target = max(1, decision.target_gpus)
            strategy = select_hybrid_strategy(target, self.runtime_model)
            result.strategies[job.job_id] = strategy

            if target != current_gpus and target >= 2:
                # Pipeline repartition makes sense for K≥2 only.
                partition = self._repartition(job.job_id, target, result)
                if partition is not None:
                    result.partitions[job.job_id] = partition
            elif target >= 2 and self.repartition_contexts.get(job.job_id) is not None:
                # Same gpu count but a partition context exists — keep the
                # current partition in the result for observability.
                cached = self._partitions.get(job.job_id)
                if cached is not None:
                    result.partitions[job.job_id] = cached

            self.job_store.update(
                job.job_id,
                allocated_gpus=target,
                parallelism=(strategy.data_parallel, strategy.model_parallel),
                state=JobState.PAUSED if decision.pause else JobState.RUNNING,
                last_decision_reason=decision.reason,
            )
            if self.pool_scale_fn:
                try:
                    self.pool_scale_fn(job.job_id, target)
                except Exception:  # pragma: no cover
                    logger.exception("pool scale failed for job %s", job.job_id)
        return result


def energy_admit_or_drop(job: Job, ebmss: EnergyBudgetMSS) -> bool:
    """Energy-budget admission helper — paralleling Phase 1 ``admit_or_drop``.

    Uses ``EnergyBudgetMSS.find()`` which respects both deadline AND energy
    budget (and optionally carbon proxy if the EB-MSS was configured with a
    grid intensity forecast + carbon budget). On admission, sets
    ``job.allocated_gpus`` to the smallest GPU count meeting both constraints.

    Returns ``True`` on admission, ``False`` on drop. Mutates ``job.state``
    and ``job.last_decision_reason`` in place.

    This is the **energy-first admission path** for HISE (contribution C3
    reframed — see [docs/powerflow-comparison.md](../../docs/powerflow-comparison.md)).
    Wire-in pattern in tests / experiments::

        ebmss = EnergyBudgetMSS(curve=..., power_per_gpu_w=300.0,
                                energy_budget_kwh=1.0, energy_profile=profile)
        if energy_admit_or_drop(job, ebmss):
            store.add(job)
    """
    decision = ebmss.find(
        iterations_remaining=job.iterations_target - job.iterations_done,
        deadline_seconds=job.deadline_s,
    )
    if not decision.admitted:
        job.state = JobState.DROPPED
        job.last_decision_reason = decision.reason
        return False
    job.state = JobState.ADMITTED
    job.allocated_gpus = decision.gpus
    job.last_decision_reason = decision.reason
    return True

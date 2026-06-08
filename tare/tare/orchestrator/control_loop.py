"""The orchestrator's main control loop — the closed-loop feedback that ties HASAGI together.

Each tick:
    1. Pull current carbon intensity from the energy scheduler (or trace replay).
    2. For each admitted job, ask the energy policy for a target GPU count.
    3. Run the planner to pick a new (dp, mp) strategy under that target.
    4. Tell the pool manager to scale workers; tell workers about the new partition.
    5. Update job state, record metrics.

Implemented as a plain callable so we can unit-test it deterministically and also run it
from FastAPI (`api.py`) as a background task.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tare.admission.mss import ScalingCurve, minimum_satisfactory_share
from tare.energy.policy import EnergyDecision, RuleBasedPolicy
from tare.orchestrator.job import Job, JobState, JobStore
from tare.parallel.planner import HybridStrategy, SimpleRuntimeModel, select_hybrid_strategy

logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    """What the control loop produced this tick. Used for testing + observability."""

    tick_seconds: float
    intensity_g_per_kwh: float
    decisions: dict[str, EnergyDecision] = field(default_factory=dict)
    strategies: dict[str, HybridStrategy] = field(default_factory=dict)


@dataclass
class ControlLoop:
    job_store: JobStore
    energy_policy: RuleBasedPolicy
    intensity_at_now: Callable[[], float]                   # current intensity
    runtime_model: SimpleRuntimeModel
    cluster_size: int = 16
    pool_scale_fn: Callable[[str, int], None] | None = None  # (job_id, target_gpus) → side effect

    def tick(self, now_seconds: float | None = None) -> TickResult:
        now = now_seconds if now_seconds is not None else time.time()
        intensity = self.intensity_at_now()
        result = TickResult(tick_seconds=now, intensity_g_per_kwh=intensity)

        for job in self.job_store.by_state(JobState.RUNNING) + self.job_store.by_state(JobState.PAUSED):
            decision = self.energy_policy.decide(job.allocated_gpus or 1, intensity)
            result.decisions[job.job_id] = decision

            # Decide new parallelism strategy under the new gpu budget.
            target = max(1, decision.target_gpus)
            strategy = select_hybrid_strategy(target, self.runtime_model)
            result.strategies[job.job_id] = strategy

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


def admit_or_drop(job: Job, curve: ScalingCurve) -> bool:
    """Convenience wrapper — for the smoke test we run admission inline."""
    mss = minimum_satisfactory_share(
        iterations_remaining=job.iterations_target - job.iterations_done,
        deadline_seconds=job.deadline_s,
        curve=curve,
    )
    if mss == 0:
        job.state = JobState.DROPPED
        job.last_decision_reason = "no allocation meets deadline (MSS=0)"
        return False
    job.state = JobState.ADMITTED
    job.allocated_gpus = mss
    job.last_decision_reason = f"MSS={mss}"
    return True

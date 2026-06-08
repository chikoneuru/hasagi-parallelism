"""In-memory job store + state machine for the orchestrator.

For multi-orchestrator HA, swap this for ``state.redis_store`` later.
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock


class JobState(str, enum.Enum):
    PENDING   = "PENDING"
    ADMITTED  = "ADMITTED"
    RUNNING   = "RUNNING"
    PAUSED    = "PAUSED"      # paused by energy policy (high-carbon hour)
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    DROPPED   = "DROPPED"     # admission rejected


@dataclass
class Job:
    job_id: str
    model_name: str
    dataset: str
    deadline_s: float
    iterations_target: int
    iterations_done: int = 0
    carbon_budget_g: float = 0.0
    state: JobState = JobState.PENDING
    allocated_gpus: int = 0
    parallelism: tuple[int, int] = (1, 1)   # (data_parallel, model_parallel)
    submitted_at: float = field(default_factory=time.time)
    last_decision_reason: str = ""

    @classmethod
    def new(cls, **kwargs) -> Job:
        return cls(job_id=str(uuid.uuid4()), **kwargs)

    def iters_remaining(self) -> int:
        """Iterations still owed to finish this job. Clamped at zero."""
        return max(0, self.iterations_target - self.iterations_done)

    def deadline_seconds_remaining(self, now_seconds: float | None = None) -> float:
        """Wall-clock seconds until this job's deadline.

        Deadline anchored at ``submitted_at + deadline_s``. Returns 0.0 once
        the deadline has passed so downstream allocators treat the job as
        time-critical rather than negative-budget.
        """
        now = now_seconds if now_seconds is not None else time.time()
        return max(0.0, self.submitted_at + self.deadline_s - now)


class JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = RLock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def by_state(self, state: JobState) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.state == state]

    def update(self, job_id: str, **changes) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in changes.items():
                setattr(job, k, v)
            return job

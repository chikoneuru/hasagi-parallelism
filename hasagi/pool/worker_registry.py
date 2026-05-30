"""In-memory worker registry — workers register on startup, heartbeat periodically.

Decoupled from the pool backend (Docker / Knative) so we can swap them later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock


@dataclass
class WorkerInfo:
    worker_id: str
    gpu_type: str        # e.g. "A100", "T4", "V100"
    address: str         # host:port or k8s pod IP
    last_heartbeat: float = field(default_factory=time.time)
    busy: bool = False
    current_job: str | None = None


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = RLock()

    def register(self, info: WorkerInfo) -> None:
        with self._lock:
            self._workers[info.worker_id] = info

    def heartbeat(self, worker_id: str) -> None:
        with self._lock:
            w = self._workers.get(worker_id)
            if w is not None:
                w.last_heartbeat = time.time()

    def assign(self, worker_id: str, job_id: str) -> None:
        with self._lock:
            w = self._workers.get(worker_id)
            if w is not None:
                w.busy = True
                w.current_job = job_id

    def release(self, worker_id: str) -> None:
        with self._lock:
            w = self._workers.get(worker_id)
            if w is not None:
                w.busy = False
                w.current_job = None

    def healthy(self, max_age_s: float = 60.0) -> list[WorkerInfo]:
        now = time.time()
        with self._lock:
            return [w for w in self._workers.values() if (now - w.last_heartbeat) <= max_age_s]

    def by_gpu_type(self, gpu_type: str) -> list[WorkerInfo]:
        with self._lock:
            return [w for w in self._workers.values() if w.gpu_type == gpu_type]

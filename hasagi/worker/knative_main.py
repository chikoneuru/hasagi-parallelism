"""Knative-friendly worker entrypoint.

The Knative autoscaler wakes pods on HTTP traffic and scales them back to
zero when idle. This module exposes the worker as a FastAPI app so the
autoscaler can drive its lifecycle, and records monotonic timestamps at
each lifecycle event so the test harness can measure cold-start
breakdowns:

  - ``container_start`` — process boot (``time.monotonic()`` at import)
  - ``app_ready`` — first ``/healthz`` returning 200
  - ``cuda_init_complete`` — mock CUDA initialisation finished
  - ``first_work_request_received`` — first ``/work`` call accepted
  - ``first_work_request_completed`` — first ``/work`` call returned

Pure CPU image. The "CUDA init" is a configurable busy sleep so the
real cost can be replayed without an actual GPU inside the pod. The
host's NVML stream is the source of truth for energy; this worker only
emits timestamps that the harness correlates with NVML.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("hasagi.worker.knative")
logging.basicConfig(
    level=os.environ.get("HASAGI_WORKER_LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@dataclass
class LifecycleTimestamps:
    """Monotonic wall-clock offsets in seconds from container start."""

    container_start_unix: float
    app_ready_offset_s: float | None = None
    cuda_init_complete_offset_s: float | None = None
    first_work_request_received_offset_s: float | None = None
    first_work_request_completed_offset_s: float | None = None
    work_request_count: int = 0
    work_request_history: list[dict[str, Any]] = field(default_factory=list)


CONTAINER_START_MONO = time.monotonic()
CONTAINER_START_UNIX = time.time()
TIMESTAMPS = LifecycleTimestamps(container_start_unix=CONTAINER_START_UNIX)


def _offset_now() -> float:
    return time.monotonic() - CONTAINER_START_MONO


# The mock CUDA init busy-sleeps for a configurable duration so the harness can
# replay realistic warm-up costs without needing GPU passthrough inside the pod.
# Default 3.0 s ≈ first cudaInitialize on Ampere with the loaded driver (matches
# our host's measured ~2.8 s from a torch.cuda.init() probe earlier).
CUDA_INIT_S = float(os.environ.get("HASAGI_MOCK_CUDA_INIT_S", "3.0"))
# Per-request "work" cost in seconds; mimics a single training iteration.
WORK_ITER_S = float(os.environ.get("HASAGI_MOCK_WORK_ITER_S", "0.05"))


def _mock_cuda_init_once() -> None:
    """Idempotent — does the busy sleep on first call, no-op afterwards."""
    if TIMESTAMPS.cuda_init_complete_offset_s is not None:
        return
    logger.info("mock CUDA init starting (%.3fs)", CUDA_INIT_S)
    start = time.monotonic()
    # Spin-wait so we burn the CPU cycle the way a real CUDA init would
    # (memory allocation + compiler warm-up) — sleep would give a deceptively
    # cheap signal under "kubectl describe" probes.
    while time.monotonic() - start < CUDA_INIT_S:
        pass
    TIMESTAMPS.cuda_init_complete_offset_s = _offset_now()
    logger.info(
        "mock CUDA init done at offset %.3fs",
        TIMESTAMPS.cuda_init_complete_offset_s,
    )


# --- FastAPI app ---


app = FastAPI()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Knative readiness probe; also marks the first-readiness offset."""
    if TIMESTAMPS.app_ready_offset_s is None:
        TIMESTAMPS.app_ready_offset_s = _offset_now()
        logger.info("app ready at offset %.3fs", TIMESTAMPS.app_ready_offset_s)
    return {"status": "ok", "uptime_s": _offset_now()}


class WorkRequest(BaseModel):
    iterations: int = 1
    job_id: str = "unknown"


@app.post("/work")
def work(req: WorkRequest) -> dict[str, Any]:
    """Mock training-iteration endpoint. Triggers CUDA init on first call."""
    received_offset = _offset_now()
    if TIMESTAMPS.first_work_request_received_offset_s is None:
        TIMESTAMPS.first_work_request_received_offset_s = received_offset
    _mock_cuda_init_once()
    start = time.monotonic()
    total = req.iterations * WORK_ITER_S
    while time.monotonic() - start < total:
        pass
    completed_offset = _offset_now()
    if TIMESTAMPS.first_work_request_completed_offset_s is None:
        TIMESTAMPS.first_work_request_completed_offset_s = completed_offset
    TIMESTAMPS.work_request_count += 1
    record = {
        "job_id": req.job_id,
        "iterations": req.iterations,
        "received_offset_s": received_offset,
        "completed_offset_s": completed_offset,
        "wall_s": completed_offset - received_offset,
    }
    TIMESTAMPS.work_request_history.append(record)
    return {
        "status": "ok",
        "lifecycle": _lifecycle_payload(),
        "request": record,
    }


@app.get("/lifecycle")
def lifecycle() -> dict[str, Any]:
    """Return the full lifecycle-timestamp record."""
    return _lifecycle_payload()


def _lifecycle_payload() -> dict[str, Any]:
    """Plain-dict snapshot for JSON serialisation."""
    return {
        "container_start_unix": TIMESTAMPS.container_start_unix,
        "now_offset_s": _offset_now(),
        "app_ready_offset_s": TIMESTAMPS.app_ready_offset_s,
        "cuda_init_complete_offset_s": TIMESTAMPS.cuda_init_complete_offset_s,
        "first_work_request_received_offset_s":
            TIMESTAMPS.first_work_request_received_offset_s,
        "first_work_request_completed_offset_s":
            TIMESTAMPS.first_work_request_completed_offset_s,
        "work_request_count": TIMESTAMPS.work_request_count,
        "cuda_init_s_target": CUDA_INIT_S,
        "work_iter_s_target": WORK_ITER_S,
    }


def main() -> None:
    import uvicorn
    uvicorn.run(
        "hasagi.worker.knative_main:app",
        host="0.0.0.0",   # noqa: S104 — container intentionally binds all
        port=int(os.environ.get("PORT", "8080")),
        log_level=os.environ.get("HASAGI_WORKER_LOGLEVEL", "info").lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()

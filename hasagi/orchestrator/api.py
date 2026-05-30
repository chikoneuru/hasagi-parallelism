"""FastAPI server — exposes the job submission interface and a metrics endpoint.

Run with: ``uvicorn hasagi.orchestrator.api:app --host 0.0.0.0 --port 8000``

The orchestrator runs the ``ControlLoop.tick`` in a background asyncio task on a fixed
cadence (``HASAGI_TICK_SECONDS``).
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from hasagi.admission.mss import ScalingCurve
from hasagi.config import SETTINGS
from hasagi.energy.carbon_trace import CarbonTrace, load_csv_trace, synthetic_solar_trace
from hasagi.energy.policy import RuleBasedPolicy
from hasagi.orchestrator.control_loop import ControlLoop, admit_or_drop
from hasagi.orchestrator.job import Job, JobState, JobStore
from hasagi.parallel.planner import SimpleRuntimeModel

logger = logging.getLogger("hasagi.orchestrator")
logging.basicConfig(level=SETTINGS.log_level)


JOBS_TOTAL = Counter("hasagi_jobs_submitted_total", "Number of jobs submitted")
JOBS_ADMITTED = Counter("hasagi_jobs_admitted_total", "Number of jobs admitted")
JOBS_DROPPED = Counter("hasagi_jobs_dropped_total", "Number of jobs dropped at admission")
CARBON_GAUGE = Gauge("hasagi_carbon_intensity_g_per_kwh", "Current carbon intensity")


def _load_trace() -> CarbonTrace:
    path = Path(SETTINGS.carbon_trace_path)
    if path.exists():
        logger.info("loading carbon trace from %s", path)
        return load_csv_trace(path)
    logger.warning("no trace at %s, falling back to synthetic_solar_trace()", path)
    return synthetic_solar_trace()


class SubmitJobRequest(BaseModel):
    model_name: str
    dataset: str
    deadline_s: float
    iterations_target: int
    carbon_budget_g: float = 0.0


class SubmitJobResponse(BaseModel):
    job_id: str
    state: str
    allocated_gpus: int
    reason: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = JobStore()
    app.state.trace = _load_trace()
    app.state.trace_start = datetime.utcnow()

    def intensity_now() -> float:
        elapsed = (datetime.utcnow() - app.state.trace_start).total_seconds()
        v = app.state.trace.intensity_at(elapsed % max(app.state.trace.duration_seconds, 1.0))
        CARBON_GAUGE.set(v)
        return v

    runtime_model = SimpleRuntimeModel(
        per_sample_flops=2e9,
        model_bytes=120_000_000,
        device_throughput_flops=1e12,
        network_bandwidth_bps=10e9,
    )
    app.state.loop = ControlLoop(
        job_store=app.state.store,
        energy_policy=RuleBasedPolicy(min_gpus=1, max_gpus=16),
        intensity_at_now=intensity_now,
        runtime_model=runtime_model,
    )

    async def loop_task():
        while True:
            try:
                result = app.state.loop.tick()
                logger.info("tick intensity=%.0f gCO2/kWh decisions=%d",
                            result.intensity_g_per_kwh, len(result.decisions))
            except Exception:
                logger.exception("control-loop tick crashed")
            await asyncio.sleep(SETTINGS.tick_seconds)

    task = asyncio.create_task(loop_task())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="HASAGI Orchestrator", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=SubmitJobResponse)
def submit_job(req: SubmitJobRequest) -> SubmitJobResponse:
    JOBS_TOTAL.inc()
    job = Job.new(
        model_name=req.model_name,
        dataset=req.dataset,
        deadline_s=req.deadline_s,
        iterations_target=req.iterations_target,
        carbon_budget_g=req.carbon_budget_g,
    )
    # Toy scaling curve — concave, capped at 16 GPUs. Production should profile per-model.
    curve = ScalingCurve(throughput_per_gpu_count=[(x ** 0.85) / 1.0 for x in range(1, 17)])
    if admit_or_drop(job, curve):
        job.state = JobState.RUNNING
        JOBS_ADMITTED.inc()
    else:
        JOBS_DROPPED.inc()
    app.state.store.add(job)
    return SubmitJobResponse(
        job_id=job.job_id, state=job.state.value,
        allocated_gpus=job.allocated_gpus, reason=job.last_decision_reason,
    )


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return [_job_dict(j) for j in app.state.store.all()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_dict(job)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _job_dict(job: Job) -> dict:
    return {
        "job_id": job.job_id,
        "model_name": job.model_name,
        "dataset": job.dataset,
        "deadline_s": job.deadline_s,
        "iterations_done": job.iterations_done,
        "iterations_target": job.iterations_target,
        "state": job.state.value,
        "allocated_gpus": job.allocated_gpus,
        "parallelism": list(job.parallelism),
        "last_decision_reason": job.last_decision_reason,
    }


def run() -> None:
    """Entry point for ``hasagi-orchestrator`` console script."""
    import uvicorn
    uvicorn.run("hasagi.orchestrator.api:app", host="0.0.0.0", port=8000, reload=False)

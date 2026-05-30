"""Centralised runtime configuration (env-driven, validated by pydantic)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.environ.get("HASAGI_REDIS_URL", "redis://localhost:6379/0")
    orchestrator_url: str = os.environ.get("HASAGI_ORCHESTRATOR_URL", "http://localhost:8000")
    carbon_trace_path: str = os.environ.get("HASAGI_CARBON_TRACE", "traces/synthetic_solar.csv")
    tick_seconds: int = int(os.environ.get("HASAGI_TICK_SECONDS", "30"))
    log_level: str = os.environ.get("HASAGI_LOG_LEVEL", "INFO")

    worker_id: str = os.environ.get("HASAGI_WORKER_ID", "worker-local")
    worker_gpu_type: str = os.environ.get("HASAGI_WORKER_GPU_TYPE", "A100")


SETTINGS = Settings()

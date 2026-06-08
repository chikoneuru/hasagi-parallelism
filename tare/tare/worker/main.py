"""Worker entrypoint — registers with the orchestrator, heartbeats, accepts training jobs.

The actual training loop lives in ``tare.worker.trainer``. This module is the long-running
process; it stays alive across pause/resume cycles.
"""
from __future__ import annotations

import logging
import signal
import sys
import time

import httpx
from prometheus_client import Gauge, start_http_server

from tare.config import SETTINGS

WORKER_GPU_UTIL = Gauge("tare_worker_gpu_util", "Approx GPU util (0..1)", ["worker_id"])
WORKER_HB = Gauge("tare_worker_last_heartbeat_unix", "Unix time of last heartbeat", ["worker_id"])

logger = logging.getLogger("tare.worker")


def _heartbeat() -> None:
    url = f"{SETTINGS.orchestrator_url}/workers/{SETTINGS.worker_id}/heartbeat"
    try:
        httpx.post(url, json={"gpu_type": SETTINGS.worker_gpu_type}, timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("heartbeat to %s failed: %s", url, exc)
    WORKER_HB.labels(worker_id=SETTINGS.worker_id).set(time.time())


def run() -> None:
    logging.basicConfig(level=SETTINGS.log_level)
    logger.info("worker %s starting (gpu_type=%s)", SETTINGS.worker_id, SETTINGS.worker_gpu_type)
    start_http_server(9001)  # prometheus scrape

    stop = False

    def _shutdown(signum, frame):  # noqa: ARG001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while not stop:
        _heartbeat()
        # Future: poll orchestrator for assigned (job, partition) and call trainer.run_step()
        time.sleep(10)

    logger.info("worker %s exiting", SETTINGS.worker_id)
    sys.exit(0)


if __name__ == "__main__":
    run()

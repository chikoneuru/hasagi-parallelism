"""Local pool backend — spawn workers as Docker containers on a single host.

For real cluster mode, see ``knative_pool.py``.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LocalDockerPool:
    image: str = "tare-worker:dev"
    network: str = "tare-net"
    redis_url: str = "redis://redis:6379/0"
    orchestrator_url: str = "http://orchestrator:8000"
    spawned: list[str] = field(default_factory=list)

    def _docker(self) -> str:
        binary = shutil.which("docker")
        if binary is None:
            raise RuntimeError("docker not on PATH; install Docker or use the simulator path.")
        return binary

    def scale(self, job_id: str, target: int) -> Sequence[str]:
        """Spawn ``target`` worker containers for ``job_id``; return container names."""
        # Idempotent: count existing containers tagged for this job.
        running = [n for n in self.spawned if n.startswith(f"tare-{job_id[:8]}-")]
        delta = target - len(running)
        if delta == 0:
            return tuple(running)
        if delta > 0:
            for i in range(len(running), target):
                name = f"tare-{job_id[:8]}-{i}"
                self._spawn(name, job_id)
                self.spawned.append(name)
        else:
            for name in running[target:]:
                self._kill(name)
                self.spawned.remove(name)
        return tuple(n for n in self.spawned if n.startswith(f"tare-{job_id[:8]}-"))

    def _spawn(self, name: str, job_id: str) -> None:
        cmd = [
            self._docker(), "run", "-d",
            "--rm", "--name", name,
            "--network", self.network,
            "-e", f"HASAGI_WORKER_ID={name}",
            "-e", f"HASAGI_REDIS_URL={self.redis_url}",
            "-e", f"HASAGI_ORCHESTRATOR_URL={self.orchestrator_url}",
            "-e", f"HASAGI_JOB_ID={job_id}",
            self.image,
        ]
        logger.info("spawning worker %s", name)
        subprocess.run(cmd, check=True)

    def _kill(self, name: str) -> None:
        logger.info("stopping worker %s", name)
        subprocess.run([self._docker(), "stop", name], check=False)


@dataclass
class SimulatedPool:
    """No-op pool used in unit tests + smoke runs — records scale events without launching."""

    events: list[tuple[str, int]] = field(default_factory=list)

    def scale(self, job_id: str, target: int) -> Sequence[str]:
        self.events.append((job_id, target))
        return tuple(f"sim-{job_id[:8]}-{i}" for i in range(target))

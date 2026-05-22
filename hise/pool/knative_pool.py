"""Knative pool backend — sketch for Kubernetes deployment.

Talks to the Kubernetes API to set the ``serving.knative.dev/v1`` Service's
``autoscaling.knative.dev/minScale`` / ``maxScale`` annotations, then waits for
Knative's scaler to fan workers out.

Left as a stub; flesh out with the ``kubernetes`` Python client when running on cluster.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnativePool:
    namespace: str = "hise"
    service_template: str = "hise-worker"

    def scale(self, job_id: str, target: int) -> None:  # pragma: no cover
        raise NotImplementedError(
            "KnativePool.scale not yet implemented; use LocalDockerPool or SimulatedPool."
        )

"""Knative pool backend.

Talks to the Kubernetes API to set the ``serving.knative.dev/v1`` Service's
``autoscaling.knative.dev/min-scale`` / ``max-scale`` annotations and reports
back the resulting pod-lifecycle timestamps.

This implementation shells out to ``kubectl`` rather than depending on the
``kubernetes`` Python client because (a) the cluster is local Kind in our
testbed and (b) the orchestrator's deployment shape keeps ``kubectl`` in the
container image already. Swap to the in-cluster client when running multi-
tenant; the contract (``scale()``, ``observe()``) doesn't change.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PodLifecycle:
    """A snapshot of one Knative-served pod's K8s-side timeline."""

    pod_name: str
    created_unix: float
    scheduled_unix: float | None
    container_started_unix: float | None
    ready_unix: float | None
    phase: str


@dataclass
class KnativeScaleResult:
    """Outcome of one ``scale()`` call.

    Attributes:
      target: requested replica count.
      observed_replicas: number of pods kubectl saw at the end of the wait.
      wait_seconds: wall-clock spent inside ``scale()``.
      pods: lifecycle snapshots, one per pod that appeared during the wait.
      timed_out: True when the requested replica count was not reached
                 within ``timeout_seconds`` (caller may treat as failure).
    """

    target: int
    observed_replicas: int
    wait_seconds: float
    pods: list[PodLifecycle] = field(default_factory=list)
    timed_out: bool = False


@dataclass
class KnativePool:
    """Manage one Knative Service via ``kubectl`` annotations.

    Args:
      service: Knative Service name (e.g. ``hasagi-worker-lifecycle``).
      namespace: K8s namespace the service lives in.
      kubectl: binary path; default ``kubectl`` from PATH.
      poll_interval_s: seconds between replica-count polls inside ``scale()``.
    """

    service: str = "hasagi-worker-lifecycle"
    namespace: str = "hasagi-validation"
    kubectl: str = "kubectl"
    poll_interval_s: float = 0.5

    def _patch_annotations(self, annotations: dict[str, str]) -> None:
        """Apply a batch of annotations on the Knative template spec in one
        atomic call. Knative's validation webhook rejects half-updates that
        leave min-scale > max-scale, so callers must always patch
        consistent pairs together.
        """
        patch = json.dumps({
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": annotations
                    }
                }
            }
        })
        cmd = [
            self.kubectl, "patch", "ksvc", self.service,
            "-n", self.namespace, "--type", "merge", "-p", patch,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"kubectl patch ksvc/{self.service} failed: {exc.stderr.strip()}"
            ) from exc

    def _list_pods(self) -> list[dict]:
        """Return raw pod items from kubectl for the service's selector."""
        cmd = [
            self.kubectl, "get", "pods",
            "-n", self.namespace,
            "-l", f"serving.knative.dev/service={self.service}",
            "-o", "json",
        ]
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"kubectl get pods failed: {exc.stderr.strip()}"
            ) from exc
        return json.loads(out).get("items", [])

    def _snapshot_pod(self, item: dict) -> PodLifecycle:
        """Project one kubectl pod item to a ``PodLifecycle`` row."""
        meta = item.get("metadata", {})
        status = item.get("status", {})
        created_str = meta.get("creationTimestamp")
        created_unix = _parse_iso8601(created_str) if created_str else time.time()
        scheduled_unix: float | None = None
        ready_unix: float | None = None
        for cond in status.get("conditions", []):
            t = cond.get("lastTransitionTime")
            if not t:
                continue
            if cond.get("type") == "PodScheduled":
                scheduled_unix = _parse_iso8601(t)
            elif cond.get("type") == "Ready" and cond.get("status") == "True":
                ready_unix = _parse_iso8601(t)
        container_started_unix: float | None = None
        for cstatus in status.get("containerStatuses", []):
            running = cstatus.get("state", {}).get("running", {})
            started_at = running.get("startedAt")
            if started_at:
                container_started_unix = _parse_iso8601(started_at)
                break
        return PodLifecycle(
            pod_name=meta.get("name", "?"),
            created_unix=created_unix,
            scheduled_unix=scheduled_unix,
            container_started_unix=container_started_unix,
            ready_unix=ready_unix,
            phase=status.get("phase", "?"),
        )

    def scale(
        self,
        target: int,
        timeout_seconds: float = 120.0,
        wait_for_ready: bool = True,
    ) -> KnativeScaleResult:
        """Drive the Knative service to exactly ``target`` replicas.

        Implementation strategy: set both ``min-scale`` and ``max-scale``
        equal to ``target`` so the autoscaler is forced. Knative still
        respects request-driven scaling above ``min-scale`` if requests
        arrive; for cold-start measurement we want determinism, so we
        pin both.

        ``target == 0`` requests scale-to-zero. The wait condition for
        a successful zero-replica state is "no running pods" — *not*
        "0 ready pods", which is trivially true while pods still exist.

        ``wait_for_ready`` polls until ``target`` ready pods are observed
        (or the timeout fires). For ``target == 0``, the flag means
        "wait for full drain" instead.
        """
        if target < 0:
            raise ValueError(f"target must be >= 0, got {target}")
        target_str = str(target)
        self._patch_annotations({
            "autoscaling.knative.dev/min-scale": target_str,
            "autoscaling.knative.dev/max-scale": target_str,
        })
        logger.info(
            "knative-pool: patched %s/%s min=max=%d", self.namespace, self.service, target,
        )

        start = time.monotonic()
        observed_replicas = 0
        active_replicas = 0
        pods: list[PodLifecycle] = []
        timed_out = False
        while time.monotonic() - start < timeout_seconds:
            items = self._list_pods()
            pods = [
                self._snapshot_pod(it)
                for it in items
                if it.get("status", {}).get("phase") not in ("Succeeded", "Failed")
            ]
            ready = [p for p in pods if p.ready_unix is not None]
            observed_replicas = len(ready)
            active_replicas = len(pods)
            if target == 0:
                # Drained when no active pods remain.
                if active_replicas == 0:
                    break
            elif wait_for_ready:
                if observed_replicas >= target:
                    break
            else:
                if active_replicas >= target:
                    break
            time.sleep(self.poll_interval_s)
        else:
            if target == 0:
                timed_out = active_replicas > 0
            else:
                timed_out = (
                    observed_replicas < target if wait_for_ready
                    else active_replicas < target
                )

        wait_seconds = time.monotonic() - start
        return KnativeScaleResult(
            target=target,
            observed_replicas=observed_replicas if target > 0 else active_replicas,
            wait_seconds=wait_seconds,
            pods=pods,
            timed_out=timed_out,
        )

    def observe(self, include_terminating: bool = False) -> list[PodLifecycle]:
        """Return the current pod-lifecycle snapshots without scaling.

        By default Terminating pods (those with a non-null
        ``deletionTimestamp``) are excluded — they no longer take traffic
        and including them pollutes scale-event diffs. Pass
        ``include_terminating=True`` if you specifically need the dying
        pods (e.g., to time their drain wall-clock).
        """
        out: list[PodLifecycle] = []
        for it in self._list_pods():
            if not include_terminating and it.get("metadata", {}).get("deletionTimestamp"):
                continue
            if it.get("status", {}).get("phase") in ("Succeeded", "Failed"):
                continue
            out.append(self._snapshot_pod(it))
        return out


def _parse_iso8601(s: str) -> float:
    """K8s timestamps are RFC3339 like ``2026-05-26T12:34:56Z``."""
    # ``datetime.fromisoformat`` handles the trailing 'Z' on 3.11+; replace
    # for older interpreters anyway since the cost is trivial.
    from datetime import datetime, timezone
    s = s.rstrip("Z") + "+00:00" if s.endswith("Z") else s
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()

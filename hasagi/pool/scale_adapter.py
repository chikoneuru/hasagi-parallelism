"""Adapt the orchestrator's ``pool_scale_fn`` hook to a real ``KnativePool``.

The energy-aware control loop drives scaling through a side effect with the
signature ``pool_scale_fn(job_id, target_replicas) -> None``. The in-process
``SimulatedPool`` / ``LocalDockerPool`` already match that shape via
``scale(job_id, target)``. ``KnativePool`` does not: it manages exactly one
Knative Service, so its ``scale(target, ...)`` takes no ``job_id`` (the service
identity *is* the job binding) and returns a rich ``KnativeScaleResult`` rather
than ``None``.

``make_knative_scale_fn`` closes that gap, producing a ``pool_scale_fn`` the
control loop can call directly so the carbon/energy policy drives **real**
Knative scale-to-zero end to end. The per-call ``KnativeScaleResult`` (cold-start
wall-clock, pod lifecycle timestamps) is optionally captured into a caller-owned
dict so an energy ledger can attribute resume cost to the right tick.

Single-service by default; pass ``pool_for`` to route different jobs to
different Knative Services.
"""
from __future__ import annotations

from collections.abc import Callable, MutableMapping

from hasagi.pool.knative_pool import KnativePool, KnativeScaleResult


def make_knative_scale_fn(
    pool: KnativePool,
    *,
    pool_for: Callable[[str], KnativePool] | None = None,
    results: MutableMapping[str, KnativeScaleResult] | None = None,
    timeout_seconds: float = 120.0,
    wait_for_ready: bool = True,
) -> Callable[[str, int], None]:
    """Build a ``pool_scale_fn(job_id, target)`` backed by ``KnativePool``.

    Args:
        pool: the default Knative pool driven for every job.
        pool_for: optional resolver ``job_id -> KnativePool`` for multi-service
            deployments. When ``None``, every job drives ``pool``.
        results: optional dict updated in place with the latest
            ``KnativeScaleResult`` per ``job_id`` — feed this to the pod energy
            ledger so a resume's cold-start window is attributed correctly.
        timeout_seconds: forwarded to ``KnativePool.scale``.
        wait_for_ready: forwarded to ``KnativePool.scale`` (for ``target == 0``
            it means "wait for full drain").

    Returns:
        A ``(job_id, target) -> None`` callable. A negative ``target`` is clamped
        to ``0`` (scale-to-zero) rather than raising, so a policy that emits a
        pause as a negative width still drains cleanly.
    """

    def scale_fn(job_id: str, target: int) -> None:
        chosen = pool_for(job_id) if pool_for is not None else pool
        result = chosen.scale(
            target=max(0, target),
            timeout_seconds=timeout_seconds,
            wait_for_ready=wait_for_ready,
        )
        if results is not None:
            results[job_id] = result

    return scale_fn

"""Tests for the KnativePool ``pool_scale_fn`` adapter.

These pin the signature bridge between the orchestrator's
``pool_scale_fn(job_id, target)`` hook and ``KnativePool.scale(target, ...)``,
including scale-to-zero on a paused job and per-job result capture for the
energy ledger. No cluster is required — a duck-typed fake pool records calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tare.pool.knative_pool import KnativeScaleResult
from tare.pool.scale_adapter import make_knative_scale_fn


@dataclass
class _FakePool:
    """Records ``scale`` calls; returns a plausible KnativeScaleResult."""

    calls: list[tuple[int, float, bool]] = field(default_factory=list)

    def scale(
        self, target: int, timeout_seconds: float = 120.0, wait_for_ready: bool = True,
    ) -> KnativeScaleResult:
        self.calls.append((target, timeout_seconds, wait_for_ready))
        return KnativeScaleResult(
            target=target, observed_replicas=target, wait_seconds=0.01,
        )


def test_adapter_maps_target_to_scale() -> None:
    pool = _FakePool()
    fn = make_knative_scale_fn(pool)
    fn("job-1", 2)
    assert pool.calls == [(2, 120.0, True)]


def test_adapter_scale_to_zero() -> None:
    pool = _FakePool()
    fn = make_knative_scale_fn(pool)
    fn("job-1", 0)
    assert pool.calls[0][0] == 0


def test_adapter_clamps_negative_to_zero() -> None:
    """A policy that encodes a pause as a negative width still drains cleanly."""
    pool = _FakePool()
    fn = make_knative_scale_fn(pool)
    fn("job-1", -3)
    assert pool.calls[0][0] == 0


def test_adapter_records_results_per_job() -> None:
    pool = _FakePool()
    results: dict[str, KnativeScaleResult] = {}
    fn = make_knative_scale_fn(pool, results=results)
    fn("job-A", 1)
    fn("job-B", 0)
    assert results["job-A"].target == 1
    assert results["job-B"].target == 0


def test_adapter_pool_for_routing() -> None:
    pa, pb = _FakePool(), _FakePool()
    pools = {"job-A": pa, "job-B": pb}
    fn = make_knative_scale_fn(pa, pool_for=lambda jid: pools[jid])
    fn("job-A", 1)
    fn("job-B", 2)
    assert pa.calls[0][0] == 1
    assert pb.calls[0][0] == 2


def test_adapter_forwards_timeout_and_wait() -> None:
    pool = _FakePool()
    fn = make_knative_scale_fn(pool, timeout_seconds=5.0, wait_for_ready=False)
    fn("job-1", 1)
    assert pool.calls[0] == (1, 5.0, False)

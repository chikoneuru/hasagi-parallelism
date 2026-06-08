"""Hybrid parallelism strategy selection — Hydrozoa Algorithm 1 (MLSys'22 §3.4).

Given a cluster size, exhaustively enumerate (data_parallel × model_parallel) factorisations
and pick the one with the lowest predicted runtime. Used by the orchestrator at job start
and during reconfiguration when the burst pool grows/shrinks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HybridStrategy:
    data_parallel: int
    model_parallel: int
    estimated_runtime_s: float

    @property
    def total_workers(self) -> int:
        return self.data_parallel * self.model_parallel


def select_hybrid_strategy(
    cluster_size: int,
    runtime_model,
    *,
    min_data_parallel: int = 1,
) -> HybridStrategy:
    """Iterate ``mp ∈ [1, cluster_size]`` with ``dp = floor(cluster_size / mp)`` and pick the
    strategy minimising ``runtime_model(dp, mp)``.

    ``runtime_model`` is a callable ``(dp, mp) -> seconds``. In production this is fit from
    profiling data (per-layer fwd/bwd, all-reduce time α + β/B); for the testbed we wire in a
    simple analytic stand-in (see ``SimpleRuntimeModel`` below).
    """
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    best = HybridStrategy(data_parallel=0, model_parallel=0, estimated_runtime_s=math.inf)
    for mp in range(1, cluster_size + 1):
        dp = cluster_size // mp
        if dp < min_data_parallel:
            continue
        runtime = runtime_model(dp, mp)
        if runtime < best.estimated_runtime_s:
            best = HybridStrategy(data_parallel=dp, model_parallel=mp, estimated_runtime_s=runtime)
    if best.total_workers == 0:
        raise RuntimeError("No feasible (dp, mp) strategy found.")
    return best


@dataclass
class SimpleRuntimeModel:
    """Rough analytic runtime model: compute + allreduce + pipeline-bubble.

    Useful for unit tests and smoke runs. Real evaluation should fit (alpha, beta) via linear
    regression on profiling traces (Hydrozoa §3.4).

    Two batch conventions, selected by ``global_batch_size``:

    - ``global_batch_size is None`` (default): each replica processes a FIXED
      ``microbatch_count`` per step (per-step latency). Data parallelism then adds
      only all-reduce cost with no compute benefit, so the lowest-runtime strategy
      degenerates to maximal model-parallel at every bandwidth — the planner is
      bandwidth-insensitive in this mode. This is the legacy behaviour relied on by
      the existing control-loop/admission callers and their tests.
    - ``global_batch_size`` set: a fixed GLOBAL batch is split across the ``dp``
      replicas (``samples_per_replica = global_batch_size / dp``), the standard
      data-parallel framing (Megatron, Hydrozoa). Here dp reduces per-replica
      compute at the cost of an all-reduce while mp reduces compute at the cost of a
      pipeline bubble, so the optimal split is genuinely bandwidth-dependent.
    """

    per_sample_flops: float
    model_bytes: int
    device_throughput_flops: float
    network_bandwidth_bps: float
    pipeline_alpha: float = 0.05   # bubble fraction per stage
    microbatch_count: int = 16
    global_batch_size: int | None = None

    def __call__(self, dp: int, mp: int) -> float:
        # Samples processed per replica per step: a fixed global batch split across
        # dp replicas (data-parallel framing), else a fixed per-replica microbatch.
        samples_per_replica = (self.global_batch_size / max(dp, 1)
                               if self.global_batch_size is not None else self.microbatch_count)
        # Per-device compute time — model-parallel splits the FLOPs across mp stages.
        comp = self.per_sample_flops * samples_per_replica / max(self.device_throughput_flops * mp, 1.0)
        # All-reduce over dp groups: 2 * (dp - 1) * shard_bytes / bw.
        shard_bytes = self.model_bytes / max(mp, 1)
        allreduce = 2.0 * max(dp - 1, 0) * shard_bytes * 8.0 / max(self.network_bandwidth_bps, 1.0)
        bubble = self.pipeline_alpha * (mp - 1) * comp
        return comp + allreduce + bubble

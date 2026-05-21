"""PipeDream-style k-way pipeline DNN partitioning for serverless hybrid parallelism.

Given a sequential DNN ``L = [l_1, ..., l_n]`` and K pipeline stages (serverless containers),
find k-1 cut points that split ``L`` into segments minimising the pipeline bottleneck time.

Literature foundation:
    - PipeDream [Narayanan et al., SOSP'19]: O(n² K) DP for K-stage pipelines.
    - GPipe [Huang et al., NeurIPS'19]: equal-FLOPs heuristic.
    - Hydrozoa [Guo et al., MLSys'22]: hybrid-parallel planner on serverless containers.

Objective (HISE contribution C2):
    Primary: minimize pipeline bottleneck (max stage time) → maximise steady-state throughput.
    Secondary: sigma_exec (std-dev of stage times) retained for ablation comparison.
    Phase 2 adds energy-per-iteration once NVML/RAPL telemetry is wired in.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayerProfile:
    """Per-layer profiling info; populated by running a few forward/backward passes."""

    index: int
    fwd_flops: float
    bwd_flops: float
    activation_bytes: int

    @property
    def flops(self) -> float:
        return self.fwd_flops + self.bwd_flops


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage in the serverless container pool."""

    stage_id: int
    throughput_flops: float  # aggregate FLOPS of all workers assigned to this stage
    memory_bytes: int


@dataclass(frozen=True)
class LinkSpec:
    """Bandwidth + latency between consecutive pipeline stages."""

    src_stage: int
    dst_stage: int
    bandwidth_bps: float
    latency_s: float = 0.0


@dataclass(frozen=True)
class Partition:
    """A k-way pipeline partition of layers across stages."""

    cuts: tuple[int, ...] = ()
    stage_layers: dict[int, tuple[int, ...]] = field(default_factory=dict)
    stage_exec_time: dict[int, float] = field(default_factory=dict)
    sigma_exec: float = math.inf
    pipeline_time: float = math.inf
    num_stages: int = 0

    def is_feasible(self) -> bool:
        return all(math.isfinite(t) for t in self.stage_exec_time.values())


# ---------------------------------------------------------------------------
# Helpers (stage-agnostic)
# ---------------------------------------------------------------------------

def _segment_flops(layers: Sequence[LayerProfile], indices: Iterable[int]) -> tuple[float, float]:
    fwd = bwd = 0.0
    for idx in indices:
        layer = layers[idx]
        fwd += layer.fwd_flops
        bwd += layer.bwd_flops
    return fwd, bwd


def _comp_time(stage: StageSpec, fwd: float, bwd: float) -> float:
    return (fwd + bwd) / max(stage.throughput_flops, 1.0)


def _comm_time(link: LinkSpec, payload_bytes: float) -> float:
    if link.bandwidth_bps <= 0:
        return math.inf
    return link.latency_s + (payload_bytes * 8.0) / link.bandwidth_bps


def _exec_time(comp: float, comm_out: float, comm_in: float) -> float:
    fwd_t = comp + comm_out
    bwd_t = comp + comm_in
    return 0.5 * (fwd_t + bwd_t)


# ---------------------------------------------------------------------------
# PipeDream DP partitioner — O(n² K)
# ---------------------------------------------------------------------------

def partition_pipeline(
    layers: Sequence[LayerProfile],
    stages: Sequence[StageSpec],
    links: Sequence[LinkSpec],
    num_microbatches: int = 1,
) -> Partition:
    """PipeDream-style DP: find cut points minimising the bottleneck stage time.

    Args:
        layers: n LayerProfile objects, indexed 0..n-1.
        stages: K StageSpec objects, ordered by stage_id 0..K-1.
        links: K-1 LinkSpec objects for consecutive stage pairs.
        num_microbatches: M, the number of microbatches per minibatch.

    Complexity: O(n² · K) time, O(n · K) space.
    """
    n = len(layers)
    K = len(stages)

    if K < 1:
        raise ValueError("Need at least 1 stage.")
    if n < K:
        raise ValueError(f"Need at least {K} layers for {K} stages.")

    link_map: dict[int, LinkSpec] = {}
    for lk in links:
        link_map[lk.src_stage] = lk
    for s in range(K - 1):
        if s not in link_map:
            raise ValueError(f"Missing link from stage {s} to stage {s+1}.")

    prefix_fwd = [0.0] * (n + 1)
    prefix_bwd = [0.0] * (n + 1)
    for i in range(n):
        prefix_fwd[i + 1] = prefix_fwd[i] + layers[i].fwd_flops
        prefix_bwd[i + 1] = prefix_bwd[i] + layers[i].bwd_flops

    def seg_exec(stage_id: int, start: int, end: int) -> float:
        fwd = prefix_fwd[end + 1] - prefix_fwd[start]
        bwd = prefix_bwd[end + 1] - prefix_bwd[start]
        comp = _comp_time(stages[stage_id], fwd, bwd)

        comm_in = 0.0
        if stage_id > 0 and start > 0:
            comm_in = _comm_time(link_map[stage_id - 1], layers[start - 1].activation_bytes)

        comm_out = 0.0
        if stage_id < K - 1:
            comm_out = _comm_time(link_map[stage_id], layers[end].activation_bytes)

        return _exec_time(comp, comm_out, comm_in)

    # K=1: no pipeline
    if K == 1:
        t = seg_exec(0, 0, n - 1)
        return Partition(
            cuts=(),
            stage_layers={0: tuple(range(n))},
            stage_exec_time={0: t},
            sigma_exec=0.0,
            pipeline_time=t * num_microbatches,
            num_stages=1,
        )

    # DP: dp[j][s] = (min_bottleneck, backpointer)
    INF = float("inf")
    dp = [[(INF, -1) for _ in range(K)] for _ in range(n)]

    for j in range(n):
        dp[j][0] = (seg_exec(0, 0, j), -1)

    for s in range(1, K):
        for j in range(s, n):
            best_val, best_i = INF, -1
            for i in range(s - 1, j):
                prev = dp[i][s - 1][0]
                if prev >= INF:
                    continue
                cur = seg_exec(s, i + 1, j)
                bottleneck = max(prev, cur)
                if bottleneck < best_val:
                    best_val = bottleneck
                    best_i = i
            dp[j][s] = (best_val, best_i)

    if dp[n - 1][K - 1][0] >= INF:
        raise RuntimeError("No feasible partition found.")

    # Backtrack cuts
    cuts_list: list[int] = []
    j = n - 1
    for s in range(K - 1, 0, -1):
        i = dp[j][s][1]
        cuts_list.append(i)
        j = i
    cuts_list.reverse()

    return _build_partition(layers, stages, link_map, tuple(cuts_list), K, num_microbatches)


# ---------------------------------------------------------------------------
# Incremental partition — slide k-1 cuts within ±window
# ---------------------------------------------------------------------------

def incremental_partition(
    previous: Partition,
    layers: Sequence[LayerProfile],
    stages: Sequence[StageSpec],
    links: Sequence[LinkSpec],
    boundary_window: int = 3,
    num_microbatches: int = 1,
) -> Partition:
    """Re-partition by sliding all k-1 cuts within ±boundary_window of the previous solution.

    Cost: O(window^(k-1)) — tractable for k≤5, window≤5.
    """
    n = len(layers)
    K = len(stages)
    prev_cuts = list(previous.cuts)

    if len(prev_cuts) != K - 1:
        raise ValueError(f"Previous partition has {len(prev_cuts)} cuts but {K} stages needs {K-1}.")

    link_map: dict[int, LinkSpec] = {lk.src_stage: lk for lk in links}

    ranges: list[range] = []
    for c_idx, c_val in enumerate(prev_cuts):
        lo = max(c_idx, c_val - boundary_window)
        hi = min(n - (K - 1 - c_idx), c_val + boundary_window)
        ranges.append(range(lo, hi + 1))

    best = previous
    best_bottleneck = max(previous.stage_exec_time.values()) if previous.stage_exec_time else math.inf

    for candidate_cuts in itertools.product(*ranges):
        if not all(candidate_cuts[i] < candidate_cuts[i + 1] for i in range(len(candidate_cuts) - 1)):
            continue
        try:
            p = _build_partition(layers, stages, link_map, candidate_cuts, K, num_microbatches)
        except RuntimeError:
            continue
        bottleneck = max(p.stage_exec_time.values())
        if bottleneck < best_bottleneck:
            best_bottleneck = bottleneck
            best = p

    return best


# ---------------------------------------------------------------------------
# Shared partition builder
# ---------------------------------------------------------------------------

def _build_partition(
    layers: Sequence[LayerProfile],
    stages: Sequence[StageSpec],
    link_map: dict[int, LinkSpec],
    cuts: tuple[int, ...],
    K: int,
    num_microbatches: int,
) -> Partition:
    n = len(layers)
    boundaries = [-1, *cuts, n - 1]
    stage_layers: dict[int, tuple[int, ...]] = {}
    stage_exec: dict[int, float] = {}

    for s in range(K):
        start = boundaries[s] + 1
        end = boundaries[s + 1]
        if start > end:
            raise RuntimeError("empty segment")
        stage_layers[s] = tuple(range(start, end + 1))

        fwd_f, bwd_f = _segment_flops(layers, range(start, end + 1))
        comp = _comp_time(stages[s], fwd_f, bwd_f)

        comm_in = 0.0
        if s > 0:
            comm_in = _comm_time(link_map[s - 1], layers[start - 1].activation_bytes)
        comm_out = 0.0
        if s < K - 1:
            comm_out = _comm_time(link_map[s], layers[end].activation_bytes)

        stage_exec[s] = _exec_time(comp, comm_out, comm_in)

    mean_t = sum(stage_exec.values()) / K
    sigma = math.sqrt(sum((t - mean_t) ** 2 for t in stage_exec.values()) / K)
    bottleneck = max(stage_exec.values())
    pipeline_time = sum(stage_exec.values()) + (num_microbatches - 1) * bottleneck

    return Partition(
        cuts=tuple(cuts),
        stage_layers=stage_layers,
        stage_exec_time=stage_exec,
        sigma_exec=sigma,
        pipeline_time=pipeline_time,
        num_stages=K,
    )

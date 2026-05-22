"""PipeDream-style k-way pipeline DNN partitioning for serverless hybrid parallelism.

Given a sequential DNN ``L = [l_1, ..., l_n]`` and K pipeline stages (serverless containers),
find k-1 cut points that split ``L`` into segments minimising the pipeline bottleneck time.

Literature foundation:
    - PipeDream [Narayanan et al., SOSP'19]: O(n² K) DP for K-stage pipelines.
    - GPipe [Huang et al., NeurIPS'19]: equal-FLOPs heuristic.
    - Hydrozoa [Guo et al., MLSys'22]: hybrid-parallel planner on serverless containers.

Objectives:
    Primary: minimize pipeline bottleneck (max stage time) → maximise steady-state throughput.
    Secondary: sigma_exec (std-dev of stage times) retained for ablation comparison.
    Energy: minimise per-iteration kWh via per-stage power_draw_w telemetry.
    Feasibility: hard memory + power-cap constraints prune infeasible (stage, segment)
    assignments in both the DP and incremental paths.
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
    """One pipeline stage in the serverless container pool.

    ``power_cap_w`` is the aggregate per-stage power ceiling — Σ NVML power limits
    across the workers assigned to this stage. Used by the feasibility check;
    default math.inf preserves throughput-only behaviour for callers that don't
    set it.

    ``power_draw_w`` is the **current** aggregate power draw across all workers
    in the stage (Σ WorkerTelemetry.power_draw_w). Used by ``objective="energy"``
    partitioning to compute ``E_per_iter = Σ_s P_s · T_s``. Default 0.0 means "no
    telemetry"; with the energy objective and all-zero powers, every partition
    scores 0 (degenerate, falls through to ties — caller should use the bottleneck
    objective until telemetry is wired).
    """

    stage_id: int
    throughput_flops: float  # aggregate FLOPS of all workers assigned to this stage
    memory_bytes: int
    power_cap_w: float = math.inf
    power_draw_w: float = 0.0


@dataclass(frozen=True)
class LinkSpec:
    """Bandwidth + latency between consecutive pipeline stages."""

    src_stage: int
    dst_stage: int
    bandwidth_bps: float
    latency_s: float = 0.0


@dataclass(frozen=True)
class Partition:
    """A k-way pipeline partition of layers across stages.

    Both ``pipeline_time`` (bottleneck-driven) and ``energy_per_iter`` (telemetry-
    driven) are computed for every partition regardless of the optimisation
    objective; reporting both lets downstream code ablate objectives without
    re-running the DP.
    """

    cuts: tuple[int, ...] = ()
    stage_layers: dict[int, tuple[int, ...]] = field(default_factory=dict)
    stage_exec_time: dict[int, float] = field(default_factory=dict)
    sigma_exec: float = math.inf
    pipeline_time: float = math.inf
    energy_per_iter: float = math.inf
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


def _segment_activation_bytes(layers: Sequence[LayerProfile], start: int, end: int) -> int:
    """Activation footprint proxy for layers [start..end] on a single stage.

    Sum of per-layer activation_bytes — captures the memory needed to retain
    activations for the backward pass. Parameters are not modelled separately;
    extend LayerProfile when a tighter bound is needed.
    """
    return sum(layers[i].activation_bytes for i in range(start, end + 1))


def _segment_feasible(
    layers: Sequence[LayerProfile],
    stage: StageSpec,
    start: int,
    end: int,
) -> tuple[bool, str]:
    """Feasibility check: does assigning layers[start..end] to ``stage`` fit?

    Memory constraint: segment activation footprint ≤ stage.memory_bytes.
    Power-cap constraint: stage.power_draw_w ≤ stage.power_cap_w (stage-only,
    segment-independent — an over-cap stage can hold no layers).

    Returns ``(True, "")`` when feasible; otherwise ``(False, reason)`` with a
    short diagnostic suitable for RuntimeError messages.
    """
    mem = _segment_activation_bytes(layers, start, end)
    if mem > stage.memory_bytes:
        return False, f"stage {stage.stage_id} mem {mem}B > cap {stage.memory_bytes}B"
    if stage.power_draw_w > stage.power_cap_w:
        return False, (
            f"stage {stage.stage_id} power {stage.power_draw_w}W "
            f"> cap {stage.power_cap_w}W"
        )
    return True, ""


# ---------------------------------------------------------------------------
# PipeDream DP partitioner — O(n² K)
# ---------------------------------------------------------------------------

def partition_pipeline(
    layers: Sequence[LayerProfile],
    stages: Sequence[StageSpec],
    links: Sequence[LinkSpec],
    num_microbatches: int = 1,
    objective: str = "bottleneck",
) -> Partition:
    """PipeDream-style DP partitioner with selectable objective.

    Args:
        layers: n LayerProfile objects, indexed 0..n-1.
        stages: K StageSpec objects, ordered by stage_id 0..K-1.
        links: K-1 LinkSpec objects for consecutive stage pairs.
        num_microbatches: M, the number of microbatches per minibatch.
        objective: ``"bottleneck"`` (default, throughput-optimal, PipeDream SOSP'19) or
            ``"energy"``. Energy objective minimises ``E_per_iter = Σ_s P_s · T_s``
            where ``P_s = stages[s].power_draw_w`` and ``T_s`` is the stage
            execution time; requires non-zero per-stage power on at least one
            stage to produce non-degenerate decisions.

    DP recurrences:
        bottleneck: ``dp[j][s] = min_i max(dp[i][s-1], T_s(i+1..j))``
        energy:     ``dp[j][s] = min_i (dp[i][s-1] + P_s · T_s(i+1..j))``

    The two objectives commute with ``min`` selection so backtracking is identical;
    only the scoring function changes.

    Complexity: O(n² · K) time, O(n · K) space.
    """
    if objective not in ("bottleneck", "energy"):
        raise ValueError(f"objective must be 'bottleneck' or 'energy', got {objective!r}")

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
    prefix_mem = [0] * (n + 1)
    for i in range(n):
        prefix_fwd[i + 1] = prefix_fwd[i] + layers[i].fwd_flops
        prefix_bwd[i + 1] = prefix_bwd[i] + layers[i].bwd_flops
        prefix_mem[i + 1] = prefix_mem[i] + layers[i].activation_bytes

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

    def seg_feasible(stage_id: int, start: int, end: int) -> bool:
        """Feasibility check using O(1) prefix-sum memory lookup."""
        mem = prefix_mem[end + 1] - prefix_mem[start]
        if mem > stages[stage_id].memory_bytes:
            return False
        if stages[stage_id].power_draw_w > stages[stage_id].power_cap_w:
            return False
        return True

    def seg_score(stage_id: int, start: int, end: int, prev_score: float) -> float:
        """Combine prev cumulative score with stage (stage_id) covering layers
        [start..end] under the active objective. Returns INF when the assignment
        violates a feasibility constraint — INF propagates through both max
        (bottleneck) and add (energy), pruning the transition in the DP."""
        if not seg_feasible(stage_id, start, end):
            return math.inf
        t = seg_exec(stage_id, start, end)
        if objective == "bottleneck":
            return max(prev_score, t)
        # energy: add P_s · T_s
        return prev_score + stages[stage_id].power_draw_w * t

    # K=1: no pipeline
    if K == 1:
        ok, reason = _segment_feasible(layers, stages[0], 0, n - 1)
        if not ok:
            raise RuntimeError(f"K=1 partition infeasible: {reason}")
        t = seg_exec(0, 0, n - 1)
        return Partition(
            cuts=(),
            stage_layers={0: tuple(range(n))},
            stage_exec_time={0: t},
            sigma_exec=0.0,
            pipeline_time=t * num_microbatches,
            energy_per_iter=stages[0].power_draw_w * t,
            num_stages=1,
        )

    # DP: dp[j][s] = (min_score, backpointer). Score interpretation depends on objective.
    INF = float("inf")
    dp = [[(INF, -1) for _ in range(K)] for _ in range(n)]

    # Base case: stage 0 covers layers 0..j. Initial "prev_score" is 0 for both
    # objectives (max with 0 = identity for non-negative times; sum with 0 = identity).
    for j in range(n):
        dp[j][0] = (seg_score(0, 0, j, 0.0), -1)

    for s in range(1, K):
        for j in range(s, n):
            best_val, best_i = INF, -1
            for i in range(s - 1, j):
                prev = dp[i][s - 1][0]
                if prev >= INF:
                    continue
                score = seg_score(s, i + 1, j, prev)
                if score < best_val:
                    best_val = score
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
    objective: str = "bottleneck",
) -> Partition:
    """Re-partition by sliding all k-1 cuts within ±boundary_window of the previous solution.

    Cost: O(window^(k-1)) — tractable for k≤5, window≤5.

    ``objective`` selects which Partition field drives candidate comparison
    (``"bottleneck"`` → minimise max stage_exec_time, ``"energy"`` → minimise
    energy_per_iter). Same parameter as ``partition_pipeline``.
    """
    if objective not in ("bottleneck", "energy"):
        raise ValueError(f"objective must be 'bottleneck' or 'energy', got {objective!r}")

    def _score(p: Partition) -> float:
        if objective == "bottleneck":
            return max(p.stage_exec_time.values()) if p.stage_exec_time else math.inf
        return p.energy_per_iter

    n = len(layers)
    K = len(stages)
    prev_cuts = list(previous.cuts)

    if len(prev_cuts) != K - 1:
        raise ValueError(f"Previous partition has {len(prev_cuts)} cuts but {K} stages needs {K-1}.")

    link_map: dict[int, LinkSpec] = {lk.src_stage: lk for lk in links}

    # Rebuild `previous` against the CURRENT layers + stages before using its score
    # as a baseline. Its stored stage_exec_time / energy_per_iter may be stale if the
    # layer set or stages changed since `previous` was computed; comparing against
    # stale values would let the function return `previous` unchanged with mismatched
    # stage_layers for current n.
    prev_valid = (
        all(0 <= c < n - 1 for c in prev_cuts)
        and all(prev_cuts[i] < prev_cuts[i + 1] for i in range(len(prev_cuts) - 1))
    )
    if prev_valid:
        try:
            best = _build_partition(layers, stages, link_map, tuple(prev_cuts), K, num_microbatches)
            best_score = _score(best)
        except RuntimeError:
            best, best_score = previous, math.inf
    else:
        best, best_score = previous, math.inf

    ranges: list[range] = []
    for c_idx, c_val in enumerate(prev_cuts):
        lo = max(c_idx, c_val - boundary_window)
        hi = min(n - (K - 1 - c_idx), c_val + boundary_window)
        ranges.append(range(lo, hi + 1))

    for candidate_cuts in itertools.product(*ranges):
        if not all(candidate_cuts[i] < candidate_cuts[i + 1] for i in range(len(candidate_cuts) - 1)):
            continue
        if candidate_cuts == tuple(prev_cuts):
            continue  # already evaluated as baseline above
        try:
            p = _build_partition(layers, stages, link_map, candidate_cuts, K, num_microbatches)
        except RuntimeError:
            continue
        score = _score(p)
        if score < best_score:
            best_score = score
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
        ok, reason = _segment_feasible(layers, stages[s], start, end)
        if not ok:
            raise RuntimeError(f"infeasible: {reason}")
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
    energy_per_iter = sum(stages[s].power_draw_w * stage_exec[s] for s in range(K))

    return Partition(
        cuts=tuple(cuts),
        stage_layers=stage_layers,
        stage_exec_time=stage_exec,
        sigma_exec=sigma,
        pipeline_time=pipeline_time,
        energy_per_iter=energy_per_iter,
        num_stages=K,
    )


# ---------------------------------------------------------------------------
# Stagnation tracker — escape local windows via full DP fallback
# ---------------------------------------------------------------------------

def _partition_score(partition: Partition, objective: str) -> float:
    """Extract the comparison score for a partition under the given objective."""
    if objective == "bottleneck":
        return max(partition.stage_exec_time.values()) if partition.stage_exec_time else math.inf
    if objective == "energy":
        return partition.energy_per_iter
    raise ValueError(f"objective must be 'bottleneck' or 'energy', got {objective!r}")


@dataclass
class StagnationTracker:
    """Detect when successive `incremental_partition` calls stop making progress.

    The orchestrator runs incremental repartition between training steps to keep
    cuts well-placed cheaply (O(window^(k-1))) instead of paying the full O(n² K)
    DP every time. But incremental can only see ±boundary_window around the
    previous cuts — if the true optimum drifts outside that window (e.g., after
    a large workload shift or a power-cap change), incremental gets stuck at a
    local minimum. This tracker counts consecutive non-improving calls and
    signals when to fall back to ``partition_pipeline`` (full DP).

    Typical usage::

        tracker = StagnationTracker(patience=3, objective="energy")
        for step in steps:
            incr = incremental_partition(prev, layers, stages, links, objective="energy")
            if tracker.observe(incr):
                # Window failed to improve `patience` times in a row — escape.
                incr = partition_pipeline(layers, stages, links, objective="energy")
                tracker.reset()
            prev = incr

    Attributes:
        patience: Consecutive non-improving observations that trigger fallback.
        min_delta: Absolute improvement required to count as "progress" (default 0
            means any score reduction resets the counter). Units match the
            objective (seconds for bottleneck, joules-per-iter for energy).
        objective: Which Partition field to compare on each call.
    """

    patience: int = 3
    min_delta: float = 0.0
    objective: str = "bottleneck"
    _best_score: float = field(default=math.inf, init=False, repr=False)
    _stagnant_calls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")
        if self.objective not in ("bottleneck", "energy"):
            raise ValueError(
                f"objective must be 'bottleneck' or 'energy', got {self.objective!r}"
            )

    def observe(self, partition: Partition) -> bool:
        """Record a partition's score and update stagnation state.

        Returns True when the consecutive non-improving call count has reached
        ``patience`` (i.e., the caller should fall back to full DP and reset
        this tracker). Does NOT auto-reset on True — the caller chooses whether
        to escape, so repeated calls without reset keep returning True.
        """
        score = _partition_score(partition, self.objective)
        if score < self._best_score - self.min_delta:
            self._best_score = score
            self._stagnant_calls = 0
        else:
            self._stagnant_calls += 1
        return self._stagnant_calls >= self.patience

    def reset(self) -> None:
        """Clear stagnation state after a fallback. The best score is preserved
        across resets — fallback shouldn't make the tracker forget that a better
        score was already achieved."""
        self._stagnant_calls = 0

    def reset_all(self) -> None:
        """Clear both stagnation counter AND best score. Use when the workload
        or stage set has changed enough that prior scores aren't comparable."""
        self._best_score = math.inf
        self._stagnant_calls = 0

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def stagnant_calls(self) -> int:
        return self._stagnant_calls

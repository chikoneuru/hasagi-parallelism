"""Inter-batch micro-batch scheduler for intra-stage load balancing.

Each node within a pipeline stage hosts the same DNN segment but receives data
proportional to a per-node weight via deficit-Weighted Round Robin.  Three priority
rules adapted from 1F1B pipeline literature keep gradient correctness:

    R1: backward stream takes priority over forward stream (lower memory pressure).
    R2: at the last pipeline stage, forward output triggers immediate loss + backward.
    R3: micro-batch IDs are preserved across forward/backward to keep activations matched.

Literature foundation:
    - Weighted Round-Robin: Katevenis & Sidiropoulos, IEEE JSAC 9(8), 1991.
    - 1F1B pipeline: PipeDream [Narayanan et al., SOSP'19].
    - Heterogeneous-worker resharding: Greyhound [Wu et al., USENIX ATC'25].

Default WRR weight is throughput-share ``w_j ~ mu_j`` (FLOPS).
The energy-aware variant is ``w_j ~ throughput_j / P_j`` (iter-per-joule).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from hasagi.energy.telemetry import WorkerTelemetry


@dataclass
class Node:
    node_id: str
    stage_id: int              # pipeline stage (0-indexed)
    capacity_flops: float
    fwd_queue: deque = field(default_factory=deque)
    bwd_queue: deque = field(default_factory=deque)


def weights_for_stage(nodes: Sequence[Node]) -> dict[str, float]:
    """FLOPS-proportional weights ``w_j = mu_j / sum(mu_k)``.

    Throughput-share baseline (kept for ablation against the iter-per-joule variant).
    """
    total = sum(n.capacity_flops for n in nodes) or 1.0
    return {n.node_id: n.capacity_flops / total for n in nodes}


def energy_weights_for_stage(
    nodes: Sequence[Node],
    telemetry: Mapping[str, WorkerTelemetry],
) -> dict[str, float]:
    """Iter-per-joule weights from live NVML telemetry.

    Weight for node ``j`` is ``throughput_j / P_j`` (iters / joule) normalised so all
    weights sum to 1. Nodes without telemetry — or with zero/negative power_draw_w —
    fall back to ``capacity_flops`` so dispatch keeps working during telemetry warm-up
    and partial-coverage situations.

    Rationale: shifting WRR weight from FLOPS-proportional (which optimises for
    throughput) to iter-per-joule (which optimises for energy efficiency) routes
    more micro-batches to workers that are *currently* most energy-efficient —
    e.g., an A100 power-capped low vs. a T4 at full draw.

    The two weights coincide only when all workers in the stage are homogeneous and
    drawing identical power; on heterogeneous serverless pools (mixed A100 / V100 /
    T4 / power-capped) they diverge meaningfully.

    Args:
        nodes: nodes within a single pipeline stage.
        telemetry: ``worker_id`` → ``WorkerTelemetry`` snapshot (typically the
            orchestrator's most recent scrape from the telemetry sidecar).

    Returns:
        Normalised weight per ``node_id``; sums to 1 (or 0 if all nodes have neither
        usable telemetry nor positive capacity).
    """
    raw: dict[str, float] = {}
    for node in nodes:
        t = telemetry.get(node.node_id)
        if t is not None and t.power_draw_w > 0 and t.throughput_iters_per_s > 0:
            raw[node.node_id] = t.throughput_iters_per_s / t.power_draw_w
        else:
            # Fallback: FLOPS capacity. Keeps dispatch working when NVML is warming
            # up, missing, or throughput counter has not advanced yet.
            raw[node.node_id] = max(node.capacity_flops, 0.0)
    total = sum(raw.values())
    if total <= 0:
        return {nid: 0.0 for nid in raw}
    return {nid: v / total for nid, v in raw.items()}


@dataclass
class PowerSlackGuard:
    """Derate WRR weights for nodes whose power draw is approaching the NVML cap.

    When a worker is power-saturated, dispatching more micro-batches to it backfires:
    NVML clamps the clock to stay within the cap, throughput drops, and the work
    completes worse than if it had been routed to a node with power slack. The guard
    detects "near-cap" workers (utilisation ≥ ``slack_threshold``) and multiplies
    their weight by ``derate_factor``; the freed weight redistributes to nodes with
    headroom via renormalisation.

    Wires after ``energy_weights_for_stage``::

        w = energy_weights_for_stage(nodes, telemetry)
        w = PowerSlackGuard(slack_threshold=0.85).apply(w, telemetry)

    Args:
        slack_threshold: fraction of ``power_cap_w`` at which a node is considered
            saturated. Default 0.85 leaves a 15% headroom buffer below the cap.
        derate_factor: weight multiplier applied to over-threshold nodes. 0.0 fully
            excludes them; 0.5 halves their share; 1.0 disables the guard.

    Telemetry handling:
        - Missing telemetry: node weight passes through unchanged (no derating).
        - ``power_cap_w <= 0``: treated as no-cap / unknown → no derating.
        - ``power_cap_w = math.inf`` (default StageSpec): utilisation = 0 → never
          triggers, so guard is a no-op until real caps are populated.
        - All nodes over threshold: returns the (zero or near-zero) adjusted dict
          rather than dividing by 0; caller decides whether to fall back to FLOPS
          weights or hold the dispatch.
    """

    slack_threshold: float = 0.85
    derate_factor: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.slack_threshold <= 1.0:
            raise ValueError(
                f"slack_threshold must be in (0, 1], got {self.slack_threshold}"
            )
        if not 0.0 <= self.derate_factor <= 1.0:
            raise ValueError(
                f"derate_factor must be in [0, 1], got {self.derate_factor}"
            )

    def apply(
        self,
        weights: Mapping[str, float],
        telemetry: Mapping[str, WorkerTelemetry],
    ) -> dict[str, float]:
        """Return new weights with over-threshold nodes derated and the rest
        renormalised so the sum is 1 (or 0 when every node is over threshold)."""
        adjusted: dict[str, float] = {}
        for nid, w in weights.items():
            t = telemetry.get(nid)
            if t is None or t.power_cap_w <= 0:
                adjusted[nid] = w
                continue
            utilization = t.power_draw_w / t.power_cap_w
            if utilization >= self.slack_threshold:
                adjusted[nid] = w * self.derate_factor
            else:
                adjusted[nid] = w
        total = sum(adjusted.values())
        if total <= 0:
            return adjusted
        return {nid: v / total for nid, v in adjusted.items()}


@dataclass
class WRRScheduler:
    """Weighted Round Robin over a stage's nodes; deficit-based for non-integer weights."""

    nodes: list[Node]
    direction: str  # "fwd" or "bwd"
    deficit: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = weights_for_stage(self.nodes)
        self.deficit = {nid: 0.0 for nid in w}
        self._weights = w

    def pick(self) -> Node | None:
        for n in self.nodes:
            self.deficit[n.node_id] += self._weights[n.node_id]
        chosen = max(self.nodes, key=lambda n: self.deficit[n.node_id])
        if self.deficit[chosen.node_id] < 1.0:
            return None
        self.deficit[chosen.node_id] -= 1.0
        return chosen


@dataclass
class EnergyAwareWRR:
    """Deficit-WRR that refreshes weights from live NVML telemetry on each pick.

    Sibling of ``WRRScheduler``: same deficit-Weighted Round Robin mechanics, but
    weights come from ``energy_weights_for_stage`` (iter-per-joule) instead of
    static FLOPS. Optionally chains through a ``PowerSlackGuard`` to derate
    near-cap workers.

    Refresh strategy: weights recompute every ``refresh_period`` picks. Set to 1
    for max responsiveness (every pick) or higher to amortise the energy-weight
    calculation across many dispatch decisions when the cluster is large.

    Fallback chain: if energy weights + guard collapse to all-zero (e.g., every
    worker saturated and ``derate_factor=0``), the scheduler falls back to FLOPS
    weights so dispatch keeps making progress instead of stalling indefinitely.

    Args:
        nodes: workers in one pipeline stage (all sharing the same stage_id).
        direction: "fwd" or "bwd" — kept separate to honour 1F1B priority.
        telemetry_source: zero-arg callable returning the current telemetry map.
            Called at every weight refresh; the orchestrator typically wires this
            to its sidecar's latest scrape.
        guard: optional ``PowerSlackGuard``; ``None`` skips the derating step.
        refresh_period: number of ``pick()`` calls between weight refreshes
            (>= 1). Default 1 = refresh every pick.
    """

    nodes: list[Node]
    direction: str
    telemetry_source: Callable[[], Mapping[str, WorkerTelemetry]]
    guard: PowerSlackGuard | None = None
    refresh_period: int = 1
    deficit: dict[str, float] = field(default_factory=dict)
    _weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _ticks_since_refresh: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("Need at least one node.")
        if self.direction not in ("fwd", "bwd"):
            raise ValueError(f"direction must be 'fwd' or 'bwd', got {self.direction!r}")
        if self.refresh_period < 1:
            raise ValueError(f"refresh_period must be >= 1, got {self.refresh_period}")
        stage = self.nodes[0].stage_id
        if not all(n.stage_id == stage for n in self.nodes):
            raise ValueError("All nodes must share the same stage_id.")
        self.deficit = {n.node_id: 0.0 for n in self.nodes}
        self._refresh_weights()

    def _refresh_weights(self) -> None:
        tel = self.telemetry_source()
        w = energy_weights_for_stage(self.nodes, tel)
        if self.guard is not None:
            w = self.guard.apply(w, tel)
        if sum(w.values()) <= 0:
            # Energy + guard collapsed to zeros (e.g., every worker over cap).
            # Fall back to FLOPS weights so dispatch keeps moving.
            w = weights_for_stage(self.nodes)
        self._weights = w
        self._ticks_since_refresh = 0

    def pick(self) -> Node | None:
        self._ticks_since_refresh += 1
        if self._ticks_since_refresh >= self.refresh_period:
            self._refresh_weights()
        for n in self.nodes:
            self.deficit[n.node_id] += self._weights.get(n.node_id, 0.0)
        chosen = max(self.nodes, key=lambda n: self.deficit[n.node_id])
        if self.deficit[chosen.node_id] < 1.0:
            return None
        self.deficit[chosen.node_id] -= 1.0
        return chosen


class InterBatchScheduler:
    """Per-stage 1F1B-style scheduler; last-stage nodes auto-trigger backward (rule R2)."""

    def __init__(self, nodes: Sequence[Node], *, is_last_stage: bool = False) -> None:
        if not nodes:
            raise ValueError("Need at least one node per stage.")
        self.nodes = list(nodes)
        self.stage_id = nodes[0].stage_id
        if not all(n.stage_id == self.stage_id for n in nodes):
            raise ValueError("All nodes in a scheduler must share the same stage_id.")
        self.is_last_stage = is_last_stage
        self._fwd = WRRScheduler(self.nodes, direction="fwd")
        self._bwd = WRRScheduler(self.nodes, direction="bwd")

    def enqueue_forward(self, microbatch_id: int) -> None:
        node = self._fwd.pick()
        if node is None:
            node = self.nodes[microbatch_id % len(self.nodes)]
        node.fwd_queue.append(microbatch_id)

    def enqueue_backward(self, microbatch_id: int, target_node_id: str | None = None) -> None:
        if target_node_id is None:
            node = self._bwd.pick() or self.nodes[microbatch_id % len(self.nodes)]
        else:
            matches = [n for n in self.nodes if n.node_id == target_node_id]
            if not matches:
                raise ValueError(f"Unknown node {target_node_id}")
            node = matches[0]
        node.bwd_queue.append(microbatch_id)

    def tick(self) -> list[tuple[str, str, int]]:
        """Drain one micro-batch per node per tick.

        R1 — backward before forward.  R2 — last stage auto-enqueues backward after forward.
        """
        events: list[tuple[str, str, int]] = []
        for node in self.nodes:
            if node.bwd_queue:
                mb = node.bwd_queue.popleft()
                events.append((node.node_id, "bwd", mb))
            elif node.fwd_queue:
                mb = node.fwd_queue.popleft()
                events.append((node.node_id, "fwd", mb))
                if self.is_last_stage:
                    node.bwd_queue.append(mb)
        return events

    def pending(self) -> int:
        return sum(len(n.fwd_queue) + len(n.bwd_queue) for n in self.nodes)

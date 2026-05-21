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

HISE contribution C4: default weight is throughput-share ``w_j ~ mu_j`` (FLOPS).
The energy-aware variant ``w_j ~ throughput_j / P_j`` is the Phase 2 deliverable A4.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hise.energy.telemetry import WorkerTelemetry


@dataclass
class Node:
    node_id: str
    stage_id: int              # pipeline stage (0-indexed)
    capacity_flops: float
    fwd_queue: deque = field(default_factory=deque)
    bwd_queue: deque = field(default_factory=deque)


def weights_for_stage(nodes: Sequence[Node]) -> dict[str, float]:
    """FLOPS-proportional weights ``w_j = mu_j / sum(mu_k)``.

    Phase 1 baseline (kept for ablation as HISE-flops-wrr variant in Phase 4).
    """
    total = sum(n.capacity_flops for n in nodes) or 1.0
    return {n.node_id: n.capacity_flops / total for n in nodes}


def energy_weights_for_stage(
    nodes: Sequence[Node],
    telemetry: Mapping[str, WorkerTelemetry],
) -> dict[str, float]:
    """Iter-per-joule weights from live NVML telemetry (Phase 2 D4.1, contribution C4).

    Weight for node ``j`` is ``throughput_j / P_j`` (iters / joule) normalised so all
    weights sum to 1. Nodes without telemetry — or with zero/negative power_draw_w —
    fall back to ``capacity_flops`` so dispatch keeps working during telemetry warm-up
    and partial-coverage situations.

    Rationale (research-note.md §4.5 C4): WRR weight shifted from FLOPS-proportional
    (which optimises for throughput) to iter-per-joule (which optimises for energy
    efficiency) routes more micro-batches to workers that are *currently* most
    energy-efficient — e.g., an A100 power-capped low vs. a T4 at full draw.

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

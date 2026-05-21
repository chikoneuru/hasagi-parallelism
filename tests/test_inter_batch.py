"""Unit tests for per-stage 1F1B WRR scheduler."""
from __future__ import annotations

from hise.energy.telemetry import WorkerTelemetry
from hise.parallel.inter_batch import (
    InterBatchScheduler,
    Node,
    energy_weights_for_stage,
    weights_for_stage,
)


def _telemetry(worker_id: str, power_w: float, throughput: float,
               gpu_type: str = "A100", stage_id: int = 0) -> WorkerTelemetry:
    return WorkerTelemetry(
        worker_id=worker_id, stage_id=stage_id, gpu_type=gpu_type,
        power_draw_w=power_w, throughput_iters_per_s=throughput,
        energy_cumulative_kwh=0.0, power_cap_w=400.0,
        memory_used_bytes=8 << 30, temperature_c=60.0, timestamp_s=0.0,
    )


def test_weights_sum_to_one() -> None:
    nodes = [
        Node(node_id="w1", stage_id=0, capacity_flops=1.0),
        Node(node_id="w2", stage_id=0, capacity_flops=3.0),
    ]
    w = weights_for_stage(nodes)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["w2"] > w["w1"]


def test_backward_takes_priority() -> None:
    sched = InterBatchScheduler([
        Node(node_id="w1", stage_id=1, capacity_flops=1.0),
        Node(node_id="w2", stage_id=1, capacity_flops=1.0),
    ], is_last_stage=False)
    sched.enqueue_forward(1)
    sched.enqueue_backward(2, target_node_id="w1")
    events = sched.tick()
    assert any(d == "bwd" for _, d, _ in events)


def test_last_stage_auto_triggers_backward() -> None:
    sched = InterBatchScheduler(
        [Node(node_id="w1", stage_id=3, capacity_flops=1.0)],
        is_last_stage=True,
    )
    sched.enqueue_forward(42)
    events1 = sched.tick()
    assert events1 == [("w1", "fwd", 42)]
    events2 = sched.tick()
    assert events2 == [("w1", "bwd", 42)]


def test_non_last_stage_no_auto_backward() -> None:
    sched = InterBatchScheduler(
        [Node(node_id="w1", stage_id=0, capacity_flops=1.0)],
        is_last_stage=False,
    )
    sched.enqueue_forward(42)
    sched.tick()
    assert sched.pending() == 0


def test_proportional_dispatch() -> None:
    sched = InterBatchScheduler([
        Node(node_id="w1", stage_id=0, capacity_flops=1.0),
        Node(node_id="w2", stage_id=0, capacity_flops=3.0),
    ], is_last_stage=False)
    for mb in range(40):
        sched.enqueue_forward(mb)
    n1 = len(next(n.fwd_queue for n in sched.nodes if n.node_id == "w1"))
    n2 = len(next(n.fwd_queue for n in sched.nodes if n.node_id == "w2"))
    assert abs(n2 - 3 * n1) < 6


# --- Energy-aware weights (Phase 2 D4.1) ---

def test_energy_weights_sum_to_one() -> None:
    nodes = [
        Node(node_id="w1", stage_id=0, capacity_flops=1.0),
        Node(node_id="w2", stage_id=0, capacity_flops=1.0),
    ]
    telemetry = {
        "w1": _telemetry("w1", power_w=200.0, throughput=100.0),
        "w2": _telemetry("w2", power_w=400.0, throughput=120.0),
    }
    w = energy_weights_for_stage(nodes, telemetry)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_energy_weights_favor_more_efficient_worker() -> None:
    """w1: 100 iter/s × 200 W = 0.5 iter/J. w2: 120 iter/s × 400 W = 0.3 iter/J.
    w1 is ~67% more efficient → should get higher weight."""
    nodes = [
        Node(node_id="w1", stage_id=0, capacity_flops=1.0),
        Node(node_id="w2", stage_id=0, capacity_flops=1.0),
    ]
    telemetry = {
        "w1": _telemetry("w1", power_w=200.0, throughput=100.0),
        "w2": _telemetry("w2", power_w=400.0, throughput=120.0),
    }
    w = energy_weights_for_stage(nodes, telemetry)
    assert w["w1"] > w["w2"]
    # Sanity ratio: 0.5 / 0.3 ≈ 1.67
    assert 1.5 < (w["w1"] / w["w2"]) < 1.85


def test_energy_weights_diverge_from_flops_weights() -> None:
    """Heterogeneous stage: A100 high-power vs T4 low-power. FLOPS weights say
    A100 wins (more compute capacity); energy weights may say otherwise depending
    on power-cap state. They must produce different distributions."""
    nodes = [
        Node(node_id="a100", stage_id=0, capacity_flops=300e12),
        Node(node_id="t4",   stage_id=0, capacity_flops=8e12),
    ]
    flops_w = weights_for_stage(nodes)

    # A100 throttled to 80 W (power-cap drama) while T4 runs at typical 60 W.
    telemetry = {
        "a100": _telemetry("a100", power_w=80.0, throughput=50.0, gpu_type="A100"),
        "t4":   _telemetry("t4",   power_w=60.0, throughput=40.0, gpu_type="T4"),
    }
    energy_w = energy_weights_for_stage(nodes, telemetry)
    # FLOPS weights heavily favour A100 (~97%); energy weights should be closer
    # because both workers have similar iter-per-J under these conditions.
    assert flops_w["a100"] > 0.95
    assert energy_w["a100"] < flops_w["a100"]


def test_energy_weights_fall_back_to_flops_when_telemetry_missing() -> None:
    nodes = [
        Node(node_id="w1", stage_id=0, capacity_flops=1.0),
        Node(node_id="w2", stage_id=0, capacity_flops=3.0),
    ]
    w = energy_weights_for_stage(nodes, telemetry={})   # empty telemetry
    flops_w = weights_for_stage(nodes)
    # With no telemetry, energy weights collapse to FLOPS weights.
    assert w["w1"] == flops_w["w1"]
    assert w["w2"] == flops_w["w2"]


def test_energy_weights_handle_zero_power_via_fallback() -> None:
    nodes = [
        Node(node_id="w1", stage_id=0, capacity_flops=10.0),
        Node(node_id="w2", stage_id=0, capacity_flops=10.0),
    ]
    telemetry = {
        "w1": _telemetry("w1", power_w=0.0, throughput=0.0),   # idle / not started
        "w2": _telemetry("w2", power_w=200.0, throughput=100.0),
    }
    w = energy_weights_for_stage(nodes, telemetry)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # w1 falls back to FLOPS (10.0); w2 uses iter/J = 100/200 = 0.5.
    # 10 vs 0.5 → w1 dominates. Both produce valid normalised weights.
    assert w["w1"] > w["w2"]


def test_energy_weights_partial_telemetry_mixed_fallback() -> None:
    nodes = [
        Node(node_id="w1", stage_id=0, capacity_flops=100.0),
        Node(node_id="w2", stage_id=0, capacity_flops=100.0),
        Node(node_id="w3", stage_id=0, capacity_flops=100.0),
    ]
    telemetry = {
        # only w1 has telemetry; w2 + w3 fall back to FLOPS.
        "w1": _telemetry("w1", power_w=200.0, throughput=100.0),
    }
    w = energy_weights_for_stage(nodes, telemetry)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    for nid in ("w1", "w2", "w3"):
        assert w[nid] >= 0

"""Unit tests for per-stage 1F1B WRR scheduler."""
from __future__ import annotations

import math

import pytest

from hise.energy.telemetry import WorkerTelemetry
from hise.parallel.inter_batch import (
    InterBatchScheduler,
    Node,
    PowerSlackGuard,
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


# --- PowerSlackGuard ---

def _telem_with_cap(worker_id: str, power_w: float, cap_w: float,
                    throughput: float = 100.0) -> WorkerTelemetry:
    return WorkerTelemetry(
        worker_id=worker_id, stage_id=0, gpu_type="A100",
        power_draw_w=power_w, throughput_iters_per_s=throughput,
        energy_cumulative_kwh=0.0, power_cap_w=cap_w,
        memory_used_bytes=8 << 30, temperature_c=60.0, timestamp_s=0.0,
    )


def test_power_slack_guard_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="slack_threshold"):
        PowerSlackGuard(slack_threshold=0.0)
    with pytest.raises(ValueError, match="slack_threshold"):
        PowerSlackGuard(slack_threshold=1.5)
    with pytest.raises(ValueError, match="derate_factor"):
        PowerSlackGuard(derate_factor=-0.1)
    with pytest.raises(ValueError, match="derate_factor"):
        PowerSlackGuard(derate_factor=1.5)


def test_power_slack_guard_passthrough_when_no_telemetry() -> None:
    """A node without telemetry must come back unchanged after normalisation."""
    guard = PowerSlackGuard(slack_threshold=0.5, derate_factor=0.0)
    weights = {"w1": 0.5, "w2": 0.5}
    out = guard.apply(weights, telemetry={})
    assert out == {"w1": 0.5, "w2": 0.5}


def test_power_slack_guard_all_below_threshold_passes_through() -> None:
    """All nodes well under their cap → weights only renormalised (no change)."""
    guard = PowerSlackGuard(slack_threshold=0.85, derate_factor=0.0)
    weights = {"w1": 0.6, "w2": 0.4}
    tel = {
        "w1": _telem_with_cap("w1", power_w=200.0, cap_w=400.0),  # 50% util
        "w2": _telem_with_cap("w2", power_w=300.0, cap_w=400.0),  # 75% util
    }
    out = guard.apply(weights, tel)
    assert abs(out["w1"] - 0.6) < 1e-9
    assert abs(out["w2"] - 0.4) < 1e-9


def test_power_slack_guard_excludes_saturated_node() -> None:
    """With derate=0.0, an over-threshold node gets zero weight; remainder absorb."""
    guard = PowerSlackGuard(slack_threshold=0.85, derate_factor=0.0)
    weights = {"w_hot": 0.5, "w_cool": 0.5}
    tel = {
        "w_hot":  _telem_with_cap("w_hot",  power_w=380.0, cap_w=400.0),  # 95% util
        "w_cool": _telem_with_cap("w_cool", power_w=200.0, cap_w=400.0),  # 50% util
    }
    out = guard.apply(weights, tel)
    assert out["w_hot"] == 0.0
    assert abs(out["w_cool"] - 1.0) < 1e-9


def test_power_slack_guard_derates_partially() -> None:
    """With derate=0.5, an over-threshold node keeps half its raw weight before renorm."""
    guard = PowerSlackGuard(slack_threshold=0.85, derate_factor=0.5)
    weights = {"w_hot": 0.5, "w_cool": 0.5}
    tel = {
        "w_hot":  _telem_with_cap("w_hot",  power_w=380.0, cap_w=400.0),
        "w_cool": _telem_with_cap("w_cool", power_w=100.0, cap_w=400.0),
    }
    out = guard.apply(weights, tel)
    # Pre-norm: w_hot 0.25, w_cool 0.5 → sum 0.75 → w_hot 1/3, w_cool 2/3
    assert abs(out["w_hot"] - 1.0 / 3.0) < 1e-9
    assert abs(out["w_cool"] - 2.0 / 3.0) < 1e-9


def test_power_slack_guard_all_saturated_returns_zeros() -> None:
    """If every node is over threshold and derate=0, the guard returns the
    zeroed (un-normalised) dict so the caller can detect & fall back."""
    guard = PowerSlackGuard(slack_threshold=0.85, derate_factor=0.0)
    weights = {"w1": 0.4, "w2": 0.6}
    tel = {
        "w1": _telem_with_cap("w1", power_w=390.0, cap_w=400.0),
        "w2": _telem_with_cap("w2", power_w=395.0, cap_w=400.0),
    }
    out = guard.apply(weights, tel)
    assert all(v == 0.0 for v in out.values())


def test_power_slack_guard_infinite_cap_never_triggers() -> None:
    """Default StageSpec.power_cap_w = math.inf means no cap → guard is a no-op
    even at huge absolute power draw."""
    guard = PowerSlackGuard(slack_threshold=0.85, derate_factor=0.0)
    weights = {"w1": 0.5, "w2": 0.5}
    tel = {
        "w1": _telem_with_cap("w1", power_w=5000.0, cap_w=math.inf),
        "w2": _telem_with_cap("w2", power_w=5000.0, cap_w=math.inf),
    }
    out = guard.apply(weights, tel)
    assert abs(out["w1"] - 0.5) < 1e-9
    assert abs(out["w2"] - 0.5) < 1e-9


def test_power_slack_guard_zero_or_negative_cap_passes_through() -> None:
    """Invalid cap (0 or negative) treated as 'no information' → no derating."""
    guard = PowerSlackGuard(slack_threshold=0.85, derate_factor=0.0)
    weights = {"w1": 0.5, "w2": 0.5}
    tel = {
        "w1": _telem_with_cap("w1", power_w=200.0, cap_w=0.0),
        "w2": _telem_with_cap("w2", power_w=200.0, cap_w=-1.0),
    }
    out = guard.apply(weights, tel)
    assert abs(out["w1"] - 0.5) < 1e-9
    assert abs(out["w2"] - 0.5) < 1e-9


def test_power_slack_guard_chains_with_energy_weights() -> None:
    """End-to-end: compute energy weights then run through the guard. A heavily
    saturated efficient worker still gets derated despite its iter-per-joule edge."""
    nodes = [
        Node(node_id="hot",  stage_id=0, capacity_flops=10.0),
        Node(node_id="cool", stage_id=0, capacity_flops=10.0),
    ]
    tel = {
        # "hot" worker: super efficient (high iter/J) but pinned at 95% cap.
        "hot":  _telem_with_cap("hot",  power_w=380.0, cap_w=400.0, throughput=500.0),
        # "cool" worker: less efficient but plenty of headroom.
        "cool": _telem_with_cap("cool", power_w=150.0, cap_w=400.0, throughput=100.0),
    }
    w_energy = energy_weights_for_stage(nodes, tel)
    # Energy alone favours the hot worker (iter/J = 500/380 vs 100/150).
    assert w_energy["hot"] > w_energy["cool"]

    guarded = PowerSlackGuard(slack_threshold=0.85, derate_factor=0.0).apply(w_energy, tel)
    # Guard zeroes the hot worker → cool absorbs all dispatch.
    assert guarded["hot"] == 0.0
    assert abs(guarded["cool"] - 1.0) < 1e-9

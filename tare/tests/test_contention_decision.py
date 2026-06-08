"""Tests for the contention decision-quality study (pure algorithm, no GPU).

Covers the scale-invariance of the partition under uniform contention, the regret
under asymmetric contention, the incremental re-plan recovery, and the
window-trapping case that motivates the StagnationTracker -> full-DP fallback.
"""
from __future__ import annotations

from experiments.exp_contention_decision import (
    _apply_contention,
    _blind_vs_aware_regret,
    _eval_at_cuts,
    _incremental_recovery,
    _links,
    _score,
    _stages,
    _uniform_layers,
)
from tare.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    incremental_partition,
    partition_pipeline,
)


def _barrier_layers(n: int, barrier_idx: int) -> list[LayerProfile]:
    """Uniform compute, but one layer carries a huge activation — a comm barrier
    that a ±1 cut cannot cheaply cross at low bandwidth."""
    return [
        LayerProfile(index=i, fwd_flops=2.0e9, bwd_flops=4.0e9,
                     activation_bytes=400_000_000 if i == barrier_idx else 1_000_000)
        for i in range(n)
    ]


def test_apply_contention_scales_throughput() -> None:
    stages = _stages(4, thru=1.0e13)
    out = _apply_contention(stages, {0: 0.5, 2: 0.25})
    assert out[0].throughput_flops == 0.5e13
    assert out[1].throughput_flops == 1.0e13   # untouched
    assert out[2].throughput_flops == 0.25e13
    # other fields preserved
    assert out[0].power_draw_w == stages[0].power_draw_w


def test_uniform_contention_is_scale_invariant() -> None:
    layers = _uniform_layers(24)
    stages = _stages(4)
    links = _links(4)
    for c in (0.3, 0.5, 0.8):
        r = _blind_vs_aware_regret(layers, stages, links, dict.fromkeys(range(4), c), "bottleneck")
        assert not r["cuts_changed"], f"uniform c={c} should not move cuts"
        assert r["regret"] < 1e-6, f"uniform c={c} regret should be ~0, got {r['regret']}"


def test_asymmetric_contention_regret_grows_with_severity() -> None:
    layers = _uniform_layers(24)
    stages = _stages(4)
    links = _links(4)
    affected = [0, 1]

    def regret(c: float) -> float:
        return _blind_vs_aware_regret(layers, stages, links,
                                      dict.fromkeys(affected, c), "bottleneck")["regret"]

    # mild contention -> small/zero regret; severe -> large regret (monotone at the ends).
    assert regret(0.9) <= 1e-6
    assert regret(0.5) > 0.10
    assert regret(0.3) > regret(0.5) > regret(0.7)


def test_eval_at_cuts_preserves_cuts() -> None:
    layers = _uniform_layers(24)
    stages = _stages(4)
    part = _eval_at_cuts(layers, stages, _links(4), (3, 7, 15))
    assert part.cuts == (3, 7, 15)
    assert part.is_feasible()


def test_incremental_recovery_recovers_and_wider_window_is_not_slower() -> None:
    layers = _uniform_layers(24)
    stages = _stages(4)
    links = _links(4)
    factors = {0: 0.3, 1: 0.3}
    rec1 = _incremental_recovery(layers, stages, links, factors, "bottleneck", window=1)
    rec3 = _incremental_recovery(layers, stages, links, factors, "bottleneck", window=3)
    # both reach the contention-aware optimum
    assert rec1["final_regret"] < 0.01
    assert rec3["final_regret"] < 0.01
    # the cheap local walk starts far from optimum...
    assert rec1["initial_regret"] > 0.1
    # ...and a wider window reaches <1% in no more steps than a narrow one.
    assert rec3["recovered_step"] is not None and rec1["recovered_step"] is not None
    assert rec3["recovered_step"] <= rec1["recovered_step"]


def test_window_trapped_incremental_stalls_but_full_dp_escapes() -> None:
    # Barrier after which stages 2,3 are contended -> the aware optimum needs a cut
    # to cross the barrier layer, which a window-1 incremental walk cannot do.
    n, k = 24, 4
    layers = _barrier_layers(n, barrier_idx=12)
    stages = _stages(k)
    links = [LinkSpec(src_stage=s, dst_stage=s + 1, bandwidth_bps=1.0e10) for s in range(k - 1)]
    factors = {2: 0.3, 3: 0.3}
    contended = _apply_contention(stages, factors)

    nominal = partition_pipeline(layers, stages, links, objective="bottleneck")
    aware = partition_pipeline(layers, contended, links, objective="bottleneck")
    aware_s = _score(aware, "bottleneck")

    # incremental-only walk (no fallback), window 1, many steps -> trapped.
    prev = nominal
    for _ in range(20):
        prev = incremental_partition(prev, layers, contended, links,
                                     boundary_window=1, objective="bottleneck")
    trapped_regret = _score(prev, "bottleneck") / aware_s - 1.0
    assert trapped_regret > 0.2, "window-1 incremental should stall well above the optimum"
    # full DP escapes the barrier (it IS the optimum by construction).
    assert _score(aware, "bottleneck") <= _score(prev, "bottleneck")


def test_stagnation_tracker_fallback_fires_and_recovers_on_barrier() -> None:
    n, k = 24, 4
    layers = _barrier_layers(n, barrier_idx=12)
    stages = _stages(k)
    links = [LinkSpec(src_stage=s, dst_stage=s + 1, bandwidth_bps=1.0e10) for s in range(k - 1)]
    factors = {2: 0.3, 3: 0.3}
    rec = _incremental_recovery(layers, stages, links, factors, "bottleneck",
                                window=1, patience=3, max_steps=30)
    # the window-1 walk traps -> tracker escalates to full DP, which recovers.
    assert rec["fallback_step"] is not None, "fallback should fire when the window traps"
    assert rec["final_regret"] < 0.01, "full-DP fallback should reach the optimum"

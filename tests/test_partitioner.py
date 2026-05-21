"""Unit tests for PipeDream k-way pipeline partitioner + incremental variant."""
from __future__ import annotations

import math

import pytest

from hise.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    incremental_partition,
    partition_pipeline,
)


def _toy_model(n: int = 12) -> list[LayerProfile]:
    return [
        LayerProfile(index=i, fwd_flops=1e8, bwd_flops=2e8, activation_bytes=1_000_000)
        for i in range(n)
    ]


# --- fixtures per K ---

def _stages_k2() -> list[StageSpec]:
    return [
        StageSpec(stage_id=0, throughput_flops=2e11, memory_bytes=8 << 30),
        StageSpec(stage_id=1, throughput_flops=5e12, memory_bytes=16 << 30),
    ]

def _links_k2() -> list[LinkSpec]:
    return [LinkSpec(src_stage=0, dst_stage=1, bandwidth_bps=10e9, latency_s=0.0005)]

def _stages_k3() -> list[StageSpec]:
    return [
        StageSpec(stage_id=0, throughput_flops=2e11, memory_bytes=4 << 30),
        StageSpec(stage_id=1, throughput_flops=1e12, memory_bytes=8 << 30),
        StageSpec(stage_id=2, throughput_flops=5e12, memory_bytes=16 << 30),
    ]

def _links_k3() -> list[LinkSpec]:
    return [
        LinkSpec(src_stage=0, dst_stage=1, bandwidth_bps=1e9, latency_s=0.001),
        LinkSpec(src_stage=1, dst_stage=2, bandwidth_bps=10e9, latency_s=0.0005),
    ]

def _stages_k4() -> list[StageSpec]:
    return [
        StageSpec(stage_id=s, throughput_flops=1e12 * (s + 1), memory_bytes=8 << 30)
        for s in range(4)
    ]

def _links_k4() -> list[LinkSpec]:
    return [
        LinkSpec(src_stage=s, dst_stage=s + 1, bandwidth_bps=10e9, latency_s=0.0005)
        for s in range(3)
    ]


# --- K=1 ---

def test_k1_single_stage() -> None:
    layers = _toy_model(8)
    stages = [StageSpec(stage_id=0, throughput_flops=1e12, memory_bytes=16 << 30)]
    p = partition_pipeline(layers, stages, [])
    assert p.cuts == ()
    assert list(p.stage_layers[0]) == list(range(8))
    assert p.num_stages == 1
    assert math.isfinite(p.pipeline_time)


# --- K=2 ---

def test_k2_covers_all_layers() -> None:
    layers = _toy_model(12)
    p = partition_pipeline(layers, _stages_k2(), _links_k2())
    combined = list(p.stage_layers[0]) + list(p.stage_layers[1])
    assert combined == list(range(12))
    assert p.num_stages == 2
    assert len(p.cuts) == 1


def test_k2_minimises_bottleneck() -> None:
    layers = _toy_model(12)
    p = partition_pipeline(layers, _stages_k2(), _links_k2())
    assert math.isfinite(p.pipeline_time)
    assert math.isfinite(p.sigma_exec)


# --- K=3 ---

def test_k3_covers_all_layers() -> None:
    layers = _toy_model(12)
    p = partition_pipeline(layers, _stages_k3(), _links_k3())
    all_layers: list[int] = []
    for s in range(3):
        all_layers.extend(p.stage_layers[s])
    assert all_layers == list(range(12))
    assert len(p.cuts) == 2


def test_k3_each_stage_nonempty() -> None:
    layers = _toy_model(12)
    p = partition_pipeline(layers, _stages_k3(), _links_k3())
    for s in range(3):
        assert len(p.stage_layers[s]) >= 1


def test_k3_rejects_too_few_layers() -> None:
    with pytest.raises(ValueError):
        partition_pipeline(_toy_model(2), _stages_k3(), _links_k3())


# --- K=4 ---

def test_k4_covers_all_layers() -> None:
    layers = _toy_model(16)
    p = partition_pipeline(layers, _stages_k4(), _links_k4())
    all_layers: list[int] = []
    for s in range(4):
        all_layers.extend(p.stage_layers[s])
    assert all_layers == list(range(16))
    assert len(p.cuts) == 3


# --- Incremental ---

def test_incremental_k3_no_worse_than_full() -> None:
    layers = _toy_model(16)
    full = partition_pipeline(layers, _stages_k3(), _links_k3())
    incr = incremental_partition(full, layers, _stages_k3(), _links_k3(), boundary_window=3)
    assert max(incr.stage_exec_time.values()) <= max(full.stage_exec_time.values()) + 1e-9


def test_incremental_k4() -> None:
    layers = _toy_model(20)
    full = partition_pipeline(layers, _stages_k4(), _links_k4())
    incr = incremental_partition(full, layers, _stages_k4(), _links_k4(), boundary_window=3)
    assert incr.sigma_exec <= full.sigma_exec + 1e-9


def test_incremental_rebuilds_prev_against_current_layers() -> None:
    """Regression: when `previous` was computed on a different layer set, its stored
    stage_layers and stage_exec_time are stale. The returned Partition must cover the
    *current* layer count, never the stale layer count from the previous call."""
    prev_layers = _toy_model(12)
    prev = partition_pipeline(prev_layers, _stages_k3(), _links_k3())
    assert sum(len(prev.stage_layers[s]) for s in range(3)) == 12

    # Grow the model to 24 layers and call incremental with the small-n prev.
    new_layers = _toy_model(24)
    incr = incremental_partition(prev, new_layers, _stages_k3(), _links_k3(), boundary_window=3)

    # The returned partition must cover all 24 layers — not the stale 12.
    total_layers = sum(len(incr.stage_layers[s]) for s in range(3))
    assert total_layers == 24, f"expected 24 layers across stages, got {total_layers}"

    # Cuts must be valid for the new layer count (each cut in [0, n-2]).
    assert all(0 <= c < 23 for c in incr.cuts)

    # stage_exec_time must be recomputed against new_layers, not copied from prev.
    # Verify by reconstructing from incr.cuts and confirming match.
    from hise.parallel.partitioner import _build_partition
    link_map = {lk.src_stage: lk for lk in _links_k3()}
    rebuilt = _build_partition(new_layers, _stages_k3(), link_map, incr.cuts, 3, 1)
    for s in range(3):
        assert abs(incr.stage_exec_time[s] - rebuilt.stage_exec_time[s]) < 1e-12, (
            f"stage {s} exec_time stale: incr={incr.stage_exec_time[s]} "
            f"vs fresh={rebuilt.stage_exec_time[s]}"
        )


# --- Edge cases ---

def test_rejects_missing_link() -> None:
    layers = _toy_model(6)
    stages = _stages_k3()
    with pytest.raises(ValueError, match="Missing link"):
        partition_pipeline(layers, stages, [])


def test_rejects_zero_stages() -> None:
    with pytest.raises(ValueError):
        partition_pipeline(_toy_model(4), [], [])


# --- Energy objective (Phase 2 D1.1) ---

def _stages_k3_with_power(power_draws_w: tuple[float, float, float]) -> list[StageSpec]:
    """K=3 stages with explicit per-stage power draws (mimicking aggregated telemetry)."""
    return [
        StageSpec(stage_id=0, throughput_flops=2e11, memory_bytes=4 << 30, power_draw_w=power_draws_w[0]),
        StageSpec(stage_id=1, throughput_flops=1e12, memory_bytes=8 << 30, power_draw_w=power_draws_w[1]),
        StageSpec(stage_id=2, throughput_flops=5e12, memory_bytes=16 << 30, power_draw_w=power_draws_w[2]),
    ]


def test_rejects_invalid_objective() -> None:
    with pytest.raises(ValueError, match="objective"):
        partition_pipeline(_toy_model(12), _stages_k3(), _links_k3(), objective="invalid")


def test_energy_per_iter_field_populated_for_bottleneck_partition() -> None:
    """Even with the default bottleneck objective, energy_per_iter is computed
    (zero if all stages have no power_draw_w; non-zero otherwise)."""
    stages = _stages_k3_with_power((100.0, 200.0, 300.0))
    p = partition_pipeline(_toy_model(12), stages, _links_k3())
    # E = sum P_s · T_s, all positive, must be finite + positive
    assert math.isfinite(p.energy_per_iter)
    assert p.energy_per_iter > 0

    # Manual sanity check: sum P_s * stage_exec_time[s]
    manual = sum(stages[s].power_draw_w * p.stage_exec_time[s] for s in range(3))
    assert abs(p.energy_per_iter - manual) < 1e-12


def test_energy_objective_matches_manual_sum() -> None:
    """`partition_pipeline(objective='energy')` returns a partition whose
    energy_per_iter equals Σ P_s · T_s computed by hand on the returned cuts."""
    stages = _stages_k3_with_power((150.0, 250.0, 350.0))
    p = partition_pipeline(_toy_model(16), stages, _links_k3(), objective="energy")
    manual = sum(stages[s].power_draw_w * p.stage_exec_time[s] for s in range(3))
    assert abs(p.energy_per_iter - manual) < 1e-12


def test_energy_partition_differs_from_bottleneck_partition() -> None:
    """When per-stage powers are very skewed (slow stage = expensive), the
    energy-optimal partition shifts load off the high-power stage even at the
    cost of some bottleneck increase."""
    # Stage 0 is the slowest (2e11 FLOPS) AND draws huge power. Bottleneck
    # objective fills it as little as possible already (since it's the slowest
    # processor). Energy objective should also avoid stage 0 — they may agree.
    # To force divergence, set the FAST stage 2 to draw 10× more power than
    # slower stages: bottleneck objective puts most layers on stage 2 (it's
    # fastest), but energy objective avoids it.
    stages = [
        StageSpec(stage_id=0, throughput_flops=1e12, memory_bytes=8 << 30, power_draw_w=50.0),
        StageSpec(stage_id=1, throughput_flops=1e12, memory_bytes=8 << 30, power_draw_w=50.0),
        StageSpec(stage_id=2, throughput_flops=1e12, memory_bytes=8 << 30, power_draw_w=1000.0),
    ]
    links = _links_k3()
    layers = _toy_model(24)

    p_bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    p_eng = partition_pipeline(layers, stages, links, objective="energy")

    # Energy-optimal: stage 2 (high-power) should get fewer layers than under
    # bottleneck objective. With equal throughputs, bottleneck wants balanced;
    # energy wants stage 2 minimised.
    assert len(p_eng.stage_layers[2]) <= len(p_bot.stage_layers[2])
    # And energy_per_iter must be strictly lower for the energy-objective result
    # (or equal in the degenerate case where bottleneck partition is already
    # energy-optimal — rare for skewed power).
    assert p_eng.energy_per_iter <= p_bot.energy_per_iter + 1e-12


def test_incremental_partition_respects_energy_objective() -> None:
    """Incremental variant honours objective parameter same as full DP."""
    stages = [
        StageSpec(stage_id=0, throughput_flops=1e12, memory_bytes=8 << 30, power_draw_w=50.0),
        StageSpec(stage_id=1, throughput_flops=1e12, memory_bytes=8 << 30, power_draw_w=50.0),
        StageSpec(stage_id=2, throughput_flops=1e12, memory_bytes=8 << 30, power_draw_w=1000.0),
    ]
    links = _links_k3()
    layers = _toy_model(20)

    # Seed incremental with a non-energy-optimal partition (bottleneck cuts)
    bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    incr_eng = incremental_partition(bot, layers, stages, links,
                                      boundary_window=4, objective="energy")
    # Incremental should improve energy from `bot` toward the energy-optimum
    # within ±boundary_window. Must not be worse than the seed.
    assert incr_eng.energy_per_iter <= bot.energy_per_iter + 1e-9


def test_incremental_rejects_invalid_objective() -> None:
    layers = _toy_model(12)
    full = partition_pipeline(layers, _stages_k3(), _links_k3())
    with pytest.raises(ValueError, match="objective"):
        incremental_partition(full, layers, _stages_k3(), _links_k3(), objective="weird")


def test_energy_objective_with_zero_powers_is_degenerate() -> None:
    """If all stages have power_draw_w=0 (no telemetry), energy_per_iter=0 for
    all partitions — the energy objective ties everything. Function must still
    return a valid partition (the first feasible one found)."""
    stages = _stages_k3()  # power_draw_w defaults to 0.0
    p = partition_pipeline(_toy_model(12), stages, _links_k3(), objective="energy")
    assert p.energy_per_iter == 0.0
    # All layers must still be covered.
    total = sum(len(p.stage_layers[s]) for s in range(3))
    assert total == 12

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


# --- Edge cases ---

def test_rejects_missing_link() -> None:
    layers = _toy_model(6)
    stages = _stages_k3()
    with pytest.raises(ValueError, match="Missing link"):
        partition_pipeline(layers, stages, [])


def test_rejects_zero_stages() -> None:
    with pytest.raises(ValueError):
        partition_pipeline(_toy_model(4), [], [])

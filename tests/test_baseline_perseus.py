"""Tests for the Perseus pipeline-throttling port (experiments/baselines/perseus.py)."""
from __future__ import annotations

import math

import pytest

from experiments.baselines.perseus import perseus_throttle
from hasagi.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    partition_pipeline,
)


def _build_pipeline(stage_flops: list[float], n_layers: int = 12):
    layers = [
        LayerProfile(index=i, fwd_flops=1e9, bwd_flops=2e9, activation_bytes=1024)
        for i in range(n_layers)
    ]
    stages = [
        StageSpec(
            stage_id=i, throughput_flops=f, memory_bytes=10**12, power_draw_w=300.0,
        )
        for i, f in enumerate(stage_flops)
    ]
    links = [LinkSpec(i, i + 1, 1e10) for i in range(len(stages) - 1)]
    partition = partition_pipeline(layers, stages, links)
    return partition, stages


# --- Basic correctness ---


def test_throttle_factors_in_unit_interval() -> None:
    partition, stages = _build_pipeline([1e10, 2e10, 4e10])
    plan = perseus_throttle(partition, stages)
    for r in plan.throttle_factors.values():
        assert 0.0 < r <= 1.0


def test_bottleneck_stage_gets_factor_one() -> None:
    """The slowest stage cannot be throttled; its factor must be exactly 1.0."""
    partition, stages = _build_pipeline([1e10, 2e10, 4e10])
    plan = perseus_throttle(partition, stages)
    slowest_sid = max(partition.stage_exec_time, key=partition.stage_exec_time.get)
    assert plan.throttle_factors[slowest_sid] == 1.0


def test_balanced_pipeline_yields_negligible_savings() -> None:
    """When stage exec times are equal (within edge-stage comm-overhead noise),
    Perseus has no meaningful slack to absorb. Savings should be sub-percent."""
    partition, stages = _build_pipeline([1e10, 1e10, 1e10])
    plan = perseus_throttle(partition, stages)
    # All factors near 1.0 (edge stages have no comm_in or comm_out, so they
    # finish marginally faster than the middle stage — but the difference is
    # too small to matter).
    for r in plan.throttle_factors.values():
        assert abs(r - 1.0) < 1e-3
    assert plan.savings_pct < 0.1     # <0.1% — numerical noise only


def test_imbalanced_pipeline_yields_savings() -> None:
    """An imbalanced partition has positive savings."""
    partition, stages = _build_pipeline([1e10, 2e10, 4e10])
    plan = perseus_throttle(partition, stages)
    assert plan.savings_kwh > 0
    assert plan.savings_pct > 0


def test_savings_monotone_in_voltage_alpha() -> None:
    """Higher voltage exponent → more energy saved per unit of throttling."""
    partition, stages = _build_pipeline([1e10, 2e10, 4e10])
    s_alpha2 = perseus_throttle(partition, stages, voltage_alpha=2.0).savings_pct
    s_alpha3 = perseus_throttle(partition, stages, voltage_alpha=3.0).savings_pct
    assert s_alpha3 > s_alpha2


def test_higher_idle_power_means_more_bloat_to_save() -> None:
    """Larger idle_power_fraction → larger baseline → bigger reduction percentage."""
    partition, stages = _build_pipeline([1e10, 2e10, 4e10])
    s_low = perseus_throttle(partition, stages, idle_power_fraction=0.05).savings_pct
    s_high = perseus_throttle(partition, stages, idle_power_fraction=0.5).savings_pct
    assert s_high > s_low


def test_bottleneck_time_matches_partition_max() -> None:
    partition, stages = _build_pipeline([1e10, 2e10, 4e10])
    plan = perseus_throttle(partition, stages)
    assert plan.bottleneck_time_s == max(partition.stage_exec_time.values())


# --- Edge cases ---


def test_rejects_negative_alpha() -> None:
    partition, stages = _build_pipeline([1e10, 1e10, 1e10])
    with pytest.raises(ValueError, match="voltage_alpha"):
        perseus_throttle(partition, stages, voltage_alpha=0.0)
    with pytest.raises(ValueError, match="voltage_alpha"):
        perseus_throttle(partition, stages, voltage_alpha=-1.0)


def test_rejects_out_of_range_idle_fraction() -> None:
    partition, stages = _build_pipeline([1e10, 1e10, 1e10])
    with pytest.raises(ValueError, match="idle_power_fraction"):
        perseus_throttle(partition, stages, idle_power_fraction=-0.1)
    with pytest.raises(ValueError, match="idle_power_fraction"):
        perseus_throttle(partition, stages, idle_power_fraction=1.5)


def test_rejects_missing_stage_spec() -> None:
    partition, stages = _build_pipeline([1e10, 1e10, 1e10])
    # Drop one StageSpec entry.
    with pytest.raises(ValueError, match="StageSpec set"):
        perseus_throttle(partition, stages[:-1])


def test_savings_nonnegative_even_when_bloat_minimal() -> None:
    """When idle_power_fraction=0 the bloated baseline equals the active-only
    energy term — Perseus may not save anything but never returns negative."""
    partition, stages = _build_pipeline([1e10, 2e10, 4e10])
    plan = perseus_throttle(partition, stages, idle_power_fraction=0.0)
    assert plan.savings_kwh >= 0
    # With voltage_alpha=2 and r<1 there's still active-energy saving.
    assert plan.savings_pct > 0


def test_handles_infeasible_stage_exec_time() -> None:
    """A stage with non-finite exec time passes through with r=1.0; no exception."""
    partition, stages = _build_pipeline([1e10, 2e10, 4e10])
    # Mutate a copy of partition with one infinite stage_exec_time.
    bad_times = dict(partition.stage_exec_time)
    bad_times[1] = math.inf
    bad_partition = type(partition)(
        cuts=partition.cuts,
        stage_layers=partition.stage_layers,
        stage_exec_time=bad_times,
        sigma_exec=partition.sigma_exec,
        pipeline_time=math.inf,
        energy_per_iter=math.inf,
        num_stages=partition.num_stages,
    )
    plan = perseus_throttle(bad_partition, stages)
    assert plan.throttle_factors[1] == 1.0

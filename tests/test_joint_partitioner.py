"""Tests for `hasagi.parallel.joint_partitioner`.

Covers seven test classes:

    1. Weak form (joint never worse than the Perseus rule on a fixed partition).
    2. Strict form on a 2-stage heterogeneous witness.
    3. K-stage embedding preserves the strict gap.
    4. DP regression to ``partition_pipeline(objective="energy")`` at the
       trivial-throttle parameter setting.
    5. Slack regime: gap vanishes when the throughput floor is tight.
    6. Joint energy is monotone non-increasing in the throughput floor.
    7. Feasibility, edge cases, determinism, backtracking validity.
"""
from __future__ import annotations

import math
import random

import pytest

from hasagi.parallel.joint_partitioner import (
    JointPlan,
    _build_throttle_set,
    joint_partition,
)
from hasagi.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    Partition,
    StageSpec,
    partition_pipeline,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _witness_layers(n: int = 10) -> list[LayerProfile]:
    """Equal-FLOPS layers with no activation: fwd=1, bwd=2 per layer."""
    return [
        LayerProfile(index=i, fwd_flops=1.0, bwd_flops=2.0, activation_bytes=0)
        for i in range(n)
    ]


def _witness_stages(theta_1: float = 3.0, power: float = 1.0) -> list[StageSpec]:
    return [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10**18, power_draw_w=power),
        StageSpec(stage_id=1, throughput_flops=theta_1, memory_bytes=10**18, power_draw_w=power),
    ]


def _zero_cost_link(src: int, dst: int) -> LinkSpec:
    return LinkSpec(src_stage=src, dst_stage=dst, bandwidth_bps=1e18, latency_s=0.0)


def _perseus_rule_energy(
    partition: Partition,
    stages: list[StageSpec],
    throttle_min: float,
    voltage_alpha: float = 2.0,
) -> float:
    """E(c, r=T_s/T_max) clamped at throttle_min — sequential Perseus rule on a fixed partition."""
    if not partition.stage_exec_time:
        return math.inf
    t_max = max(partition.stage_exec_time.values())
    if t_max <= 0:
        return math.inf
    e = 0.0
    for s_id, t in partition.stage_exec_time.items():
        r = max(throttle_min, t / t_max)
        e += stages[s_id].power_draw_w * (r ** (voltage_alpha - 1)) * t
    return e


def _discrete_perseus_energy(
    partition: Partition,
    stages: list[StageSpec],
    throttle_set: tuple[float, ...],
    voltage_alpha: float = 2.0,
) -> float:
    """Perseus rule projected onto a discrete throttle set R.

    Continuous Perseus uses ``r_s = T_s/T_max``. For a discrete R, the
    smallest admissible r is the smallest grid point ``r ∈ R`` with
    ``r ≥ T_s/T_max`` (a smaller r would stretch T_s/r past T_max and
    violate the throughput floor at T_floor = T_max). Returns +inf if no
    grid point satisfies the floor — that partition is not representable
    in R.
    """
    if not partition.stage_exec_time:
        return math.inf
    t_max = max(partition.stage_exec_time.values())
    if t_max <= 0:
        return math.inf
    e = 0.0
    for s_id, t in partition.stage_exec_time.items():
        ratio = t / t_max
        admissible = [r for r in throttle_set if r + 1e-12 >= ratio]
        if not admissible:
            return math.inf
        r = min(admissible)
        e += stages[s_id].power_draw_w * (r ** (voltage_alpha - 1)) * t
    return e


# ---------------------------------------------------------------------------
# Throttle set construction (pure helper)
# ---------------------------------------------------------------------------

def test_throttle_set_pinned_at_one() -> None:
    R = _build_throttle_set(0.5, 6)
    assert R[0] == pytest.approx(0.5)
    assert R[-1] == 1.0
    assert len(R) == 6


def test_throttle_set_singleton_when_throttle_min_one() -> None:
    assert _build_throttle_set(1.0, 1) == (1.0,)
    assert _build_throttle_set(1.0, 8) == (1.0,)


# ---------------------------------------------------------------------------
# 1. Weak form — joint ≤ sequential on every workload
# ---------------------------------------------------------------------------

def test_joint_no_worse_than_bottleneck_perseus() -> None:
    rng = random.Random(2026)
    for _ in range(3):
        n = rng.randint(8, 16)
        K = rng.randint(2, 4)
        layers = [
            LayerProfile(
                index=i,
                fwd_flops=rng.uniform(0.5, 2.0),
                bwd_flops=rng.uniform(1.0, 4.0),
                activation_bytes=rng.randint(0, 8),
            )
            for i in range(n)
        ]
        stages = [
            StageSpec(
                stage_id=s,
                throughput_flops=rng.uniform(1.0, 4.0),
                memory_bytes=10**18,
                power_draw_w=rng.uniform(1.0, 5.0),
            )
            for s in range(K)
        ]
        links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(K - 1)]

        bot = partition_pipeline(layers, stages, links, objective="bottleneck")
        t_max_bot = max(bot.stage_exec_time.values())
        granularity = 6
        R = _build_throttle_set(0.5, granularity)
        # T_floor must admit π_BP; pick floor exactly at the partition's bottleneck.
        plan = joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1.0 / t_max_bot,
            voltage_alpha=2.0,
            throttle_min=0.5,
            throttle_granularity=granularity,
        )
        assert plan.is_feasible()
        e_seq = _discrete_perseus_energy(bot, stages, R)
        assert plan.energy_per_iter <= e_seq + 1e-9


def test_joint_no_worse_than_energy_perseus() -> None:
    rng = random.Random(99)
    for _ in range(3):
        n = rng.randint(8, 14)
        K = rng.randint(2, 4)
        layers = [
            LayerProfile(
                index=i,
                fwd_flops=rng.uniform(0.5, 2.0),
                bwd_flops=rng.uniform(1.0, 4.0),
                activation_bytes=rng.randint(0, 8),
            )
            for i in range(n)
        ]
        stages = [
            StageSpec(
                stage_id=s,
                throughput_flops=rng.uniform(1.0, 4.0),
                memory_bytes=10**18,
                power_draw_w=rng.uniform(1.0, 5.0),
            )
            for s in range(K)
        ]
        links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(K - 1)]

        en = partition_pipeline(layers, stages, links, objective="energy")
        t_max_en = max(en.stage_exec_time.values())
        granularity = 6
        R = _build_throttle_set(0.5, granularity)
        plan = joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1.0 / t_max_en,
            voltage_alpha=2.0,
            throttle_min=0.5,
            throttle_granularity=granularity,
        )
        assert plan.is_feasible()
        e_seq = _discrete_perseus_energy(en, stages, R)
        assert plan.energy_per_iter <= e_seq + 1e-9


# ---------------------------------------------------------------------------
# 2. Strict form on the 2-stage heterogeneous witness (c=3, n=10, T_floor=10)
# ---------------------------------------------------------------------------

def test_joint_strictly_better_on_2stage_heterogeneous_witness() -> None:
    layers = _witness_layers(n=10)
    stages = _witness_stages(theta_1=3.0, power=1.0)
    links = [_zero_cost_link(0, 1)]

    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 10.0,
        voltage_alpha=2.0,
        throttle_min=0.5,
        throttle_granularity=6,
    )

    assert plan.is_feasible()
    assert plan.cuts == (0,)
    assert plan.throttle_factors[0] == pytest.approx(0.5)
    assert plan.throttle_factors[1] == pytest.approx(0.9)
    assert plan.energy_per_iter == pytest.approx(9.6, rel=1e-9)
    assert plan.pipeline_time_s == pytest.approx(10.0, rel=1e-9)

    # Strict gain vs energy-then-Perseus: 9.6 < 10.5 = E(π_EP).
    en = partition_pipeline(layers, stages, links, objective="energy")
    e_ep = _perseus_rule_energy(en, stages, throttle_min=0.5)
    assert plan.energy_per_iter <= 0.96 * e_ep  # ≥ 4-percentage-point margin


def test_joint_strict_gain_on_3stage_embed() -> None:
    """Add a trivial singleton-layer stage to the witness; gap must remain."""
    layers = [
        LayerProfile(index=i, fwd_flops=1.0, bwd_flops=2.0, activation_bytes=0)
        for i in range(11)
    ]
    stages = [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10**18, power_draw_w=1.0),
        StageSpec(stage_id=1, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.0),
        StageSpec(stage_id=2, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.0),
    ]
    links = [_zero_cost_link(0, 1), _zero_cost_link(1, 2)]

    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 10.0,
        voltage_alpha=2.0,
        throttle_min=0.5,
        throttle_granularity=6,
    )
    en = partition_pipeline(layers, stages, links, objective="energy")
    e_ep = _perseus_rule_energy(en, stages, throttle_min=0.5)
    assert plan.is_feasible()
    assert plan.energy_per_iter < e_ep - 1e-9


def test_joint_strict_gain_on_4stage_embed() -> None:
    layers = [
        LayerProfile(index=i, fwd_flops=1.0, bwd_flops=2.0, activation_bytes=0)
        for i in range(12)
    ]
    stages = [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10**18, power_draw_w=1.0),
        StageSpec(stage_id=1, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.0),
        StageSpec(stage_id=2, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.0),
        StageSpec(stage_id=3, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.0),
    ]
    links = [_zero_cost_link(s, s + 1) for s in range(3)]

    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 10.0,
        voltage_alpha=2.0,
        throttle_min=0.5,
        throttle_granularity=6,
    )
    en = partition_pipeline(layers, stages, links, objective="energy")
    e_ep = _perseus_rule_energy(en, stages, throttle_min=0.5)
    assert plan.is_feasible()
    assert plan.energy_per_iter < e_ep - 1e-9


# ---------------------------------------------------------------------------
# 3. DP regression — M=1, r=1.0 ≡ partition_pipeline(objective="energy")
# ---------------------------------------------------------------------------

def test_joint_dp_matches_energy_partitioner_when_r_forced_to_one() -> None:
    rng = random.Random(7)
    for _ in range(3):
        n = rng.randint(6, 12)
        K = rng.randint(2, 4)
        layers = [
            LayerProfile(
                index=i,
                fwd_flops=rng.uniform(0.5, 2.0),
                bwd_flops=rng.uniform(1.0, 4.0),
                activation_bytes=rng.randint(0, 8),
            )
            for i in range(n)
        ]
        stages = [
            StageSpec(
                stage_id=s,
                throughput_flops=rng.uniform(1.0, 4.0),
                memory_bytes=10**18,
                power_draw_w=rng.uniform(1.0, 5.0),
            )
            for s in range(K)
        ]
        links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(K - 1)]

        ref = partition_pipeline(layers, stages, links, objective="energy")
        # Loose floor + throttle_min=1.0 forces r ≡ 1, collapsing to energy DP.
        plan = joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1e-9,
            voltage_alpha=2.0,
            throttle_min=1.0,
            throttle_granularity=1,
        )
        assert plan.is_feasible()
        assert plan.cuts == ref.cuts
        assert plan.throttle_factors == (1.0,) * K
        assert plan.energy_per_iter == pytest.approx(ref.energy_per_iter, rel=1e-9)
        for s_id, t in ref.stage_exec_time.items():
            assert plan.stage_exec_time[s_id] == pytest.approx(t, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Slack regime — gap vanishes when T_floor matches the bottleneck partition
# ---------------------------------------------------------------------------

def test_joint_gain_vanishes_when_T_floor_equals_T_max_cB() -> None:
    """At T_floor = T_max(c_B*), joint matches the discrete π_BP plan (no slack to exploit)."""
    layers = _witness_layers(n=10)
    stages = _witness_stages(theta_1=3.0, power=1.0)
    links = [_zero_cost_link(0, 1)]
    bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    t_max_bot = max(bot.stage_exec_time.values())

    granularity = 6
    R = _build_throttle_set(0.5, granularity)
    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / t_max_bot,
        voltage_alpha=2.0,
        throttle_min=0.5,
        throttle_granularity=granularity,
    )
    # Discrete π_BP rounds the Perseus rule up to the nearest grid point with
    # r ≥ T_s/T_max. Joint cannot beat this at the tight floor.
    e_bp_discrete = _discrete_perseus_energy(bot, stages, R)
    assert plan.energy_per_iter == pytest.approx(e_bp_discrete, rel=1e-9)
    # And the joint plan stays at the bottleneck-optimal cycle.
    assert plan.pipeline_time_s == pytest.approx(t_max_bot, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Continuity — gap is monotone in T_floor across the slack regime
# ---------------------------------------------------------------------------

def test_joint_energy_monotone_in_t_floor() -> None:
    """As T_floor relaxes, the joint optimum energy is non-increasing.

    More relaxed floor strictly expands the feasible set (every plan
    feasible at T_floor=t is feasible at T_floor=t+ε), so the optimum
    over the larger set is no larger.
    """
    layers = _witness_layers(n=10)
    stages = _witness_stages(theta_1=3.0, power=1.0)
    links = [_zero_cost_link(0, 1)]

    bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    t_min = max(bot.stage_exec_time.values())

    last_e = math.inf
    for t_floor in (t_min, t_min + 1.0, t_min + 2.0, t_min + 3.0, t_min + 5.0):
        plan = joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1.0 / t_floor,
            voltage_alpha=2.0,
            throttle_min=0.5,
            throttle_granularity=6,
        )
        assert plan.is_feasible()
        assert plan.energy_per_iter <= last_e + 1e-9
        last_e = plan.energy_per_iter


# ---------------------------------------------------------------------------
# 6. Feasibility constraints — each prunes independently
# ---------------------------------------------------------------------------

def test_memory_constraint_blocks_infeasible_workload() -> None:
    layers = [
        LayerProfile(index=i, fwd_flops=1.0, bwd_flops=2.0, activation_bytes=100)
        for i in range(6)
    ]
    stages = [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10, power_draw_w=1.0),
        StageSpec(stage_id=1, throughput_flops=1.0, memory_bytes=10, power_draw_w=1.0),
    ]
    links = [_zero_cost_link(0, 1)]
    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1e-9,
        throttle_min=0.5,
        throttle_granularity=6,
    )
    # Every segment carries > 10 B in activation; no partition fits.
    assert not plan.is_feasible()


def test_power_cap_pruned_by_throttle() -> None:
    """A stage that violates its cap at r=1 may become admissible at r<1."""
    layers = _witness_layers(n=6)
    stages = [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10**18,
                  power_draw_w=2.0, power_cap_w=4.0),
        StageSpec(stage_id=1, throughput_flops=1.0, memory_bytes=10**18,
                  power_draw_w=2.0, power_cap_w=4.0),
    ]
    links = [_zero_cost_link(0, 1)]
    # At r=1, P · r^α = 2.0 ≤ 4.0 (feasible). Tighten the cap to 1.5 to force r<1.
    stages_tight = [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10**18,
                  power_draw_w=2.0, power_cap_w=1.5),
        StageSpec(stage_id=1, throughput_flops=1.0, memory_bytes=10**18,
                  power_draw_w=2.0, power_cap_w=1.5),
    ]
    plan_loose = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 30.0,
        throttle_min=0.5,
        throttle_granularity=6,
    )
    plan_tight = joint_partition(
        layers, stages_tight, links,
        throughput_floor_iters_per_s=1.0 / 30.0,
        throttle_min=0.5,
        throttle_granularity=6,
    )
    assert plan_loose.is_feasible()
    assert plan_tight.is_feasible()
    # Under the tight cap, every throttle r satisfies 2·r² ≤ 1.5 ⇒ r ≤ ~0.866.
    for r in plan_tight.throttle_factors:
        assert 2.0 * r ** 2 <= 1.5 + 1e-9


def test_throughput_floor_infeasible_below_min_pipeline_time() -> None:
    """Setting T_floor below what any partition can achieve at r=1 returns infeasible."""
    layers = _witness_layers(n=10)
    stages = _witness_stages(theta_1=3.0, power=1.0)
    links = [_zero_cost_link(0, 1)]
    # Best-possible pipeline time even at r=1 is bounded below by the
    # slowest single-stage time. Demand a throughput far above what's achievable.
    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=10.0,  # T_floor = 0.1 < any T_s
        throttle_min=0.5,
        throttle_granularity=6,
    )
    assert not plan.is_feasible()


def test_throttle_range_floor_below_one_admits_throttle() -> None:
    layers = _witness_layers(n=10)
    stages = _witness_stages(theta_1=3.0, power=1.0)
    links = [_zero_cost_link(0, 1)]
    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 10.0,
        throttle_min=0.5,
        throttle_granularity=6,
    )
    assert plan.is_feasible()
    # At least one throttle should be strictly below 1.0 (the slack is real).
    assert any(r < 1.0 for r in plan.throttle_factors)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

def test_k1_single_stage_scalar_throttle() -> None:
    layers = _witness_layers(n=10)
    stages = [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10**18, power_draw_w=1.0),
    ]
    plan = joint_partition(
        layers, stages, [],
        throughput_floor_iters_per_s=1.0 / 60.0,  # very loose
        throttle_min=0.5,
        throttle_granularity=6,
    )
    assert plan.is_feasible()
    assert plan.cuts == ()
    assert len(plan.throttle_factors) == 1
    # At loose floor, optimal throttle is r=0.5 (every smaller r still meets floor).
    assert plan.throttle_factors[0] == pytest.approx(0.5)


def test_rejects_too_few_layers() -> None:
    with pytest.raises(ValueError):
        joint_partition(
            _witness_layers(n=2),
            _witness_stages() + [
                StageSpec(stage_id=2, throughput_flops=1.0, memory_bytes=10**18,
                          power_draw_w=1.0)
            ],
            [_zero_cost_link(0, 1), _zero_cost_link(1, 2)],
            throughput_floor_iters_per_s=1.0,
        )


def test_rejects_invalid_voltage_alpha() -> None:
    with pytest.raises(ValueError):
        joint_partition(
            _witness_layers(),
            _witness_stages(),
            [_zero_cost_link(0, 1)],
            throughput_floor_iters_per_s=1.0,
            voltage_alpha=0.5,
        )


def test_rejects_invalid_throttle_min() -> None:
    with pytest.raises(ValueError):
        joint_partition(
            _witness_layers(),
            _witness_stages(),
            [_zero_cost_link(0, 1)],
            throughput_floor_iters_per_s=1.0,
            throttle_min=1.5,
        )


def test_rejects_nonpositive_throughput_floor() -> None:
    with pytest.raises(ValueError):
        joint_partition(
            _witness_layers(),
            _witness_stages(),
            [_zero_cost_link(0, 1)],
            throughput_floor_iters_per_s=0.0,
        )


# ---------------------------------------------------------------------------
# 8. Determinism + backtracking validity
# ---------------------------------------------------------------------------

def test_deterministic_on_repeated_calls() -> None:
    layers = _witness_layers()
    stages = _witness_stages()
    links = [_zero_cost_link(0, 1)]
    a = joint_partition(layers, stages, links,
                        throughput_floor_iters_per_s=1.0 / 10.0,
                        throttle_min=0.5, throttle_granularity=6)
    b = joint_partition(layers, stages, links,
                        throughput_floor_iters_per_s=1.0 / 10.0,
                        throttle_min=0.5, throttle_granularity=6)
    assert a.cuts == b.cuts
    assert a.throttle_factors == b.throttle_factors
    assert a.energy_per_iter == b.energy_per_iter


def test_backtracking_reproduces_dp_energy() -> None:
    """The plan returned by backtracking must score the same as the DP's optimum cell."""
    layers = _witness_layers()
    stages = _witness_stages()
    links = [_zero_cost_link(0, 1)]
    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 10.0,
        voltage_alpha=2.0,
        throttle_min=0.5,
        throttle_granularity=6,
    )
    assert plan.is_feasible()
    # Reconstruct E from the reported (cuts, throttle_factors, stage_exec_time).
    # E = Σ_s P_s · r_s^{α-1} · T_s(c) where T_s(c) = T_s' · r_s (un-throttled).
    e_reconstructed = 0.0
    for s, t_throttled in plan.stage_exec_time.items():
        r = plan.throttle_factors[s]
        t = t_throttled * r
        e_reconstructed += stages[s].power_draw_w * (r ** (2.0 - 1.0)) * t
    assert e_reconstructed == pytest.approx(plan.energy_per_iter, rel=1e-9)


def test_infeasible_plan_reports_not_feasible() -> None:
    plan = JointPlan()
    assert not plan.is_feasible()
    assert math.isinf(plan.energy_per_iter)

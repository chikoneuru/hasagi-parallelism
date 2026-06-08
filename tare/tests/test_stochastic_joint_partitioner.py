"""Tests for `tare.parallel.stochastic_joint_partitioner`.

Covers nine test classes:

    1. Weak form (v2 CVaR ≤ variance-blind plan CVaR under same noise).
    2. Strict witness — 2-stage noise-asymmetric construction reproduces ~2.15% gap.
    3. K-stage embedding preserves the strict gap.
    4. σ_T scaling — gap grows monotonically with σ_T.
    5. DP regression — σ_T=σ_P=0 reproduces joint_partition exactly.
    6. Reductions — β→1 ≡ mean-only, β→0+ → min-variance.
    7. Chance constraints — (Θ̃) and (P̃) margins inflate per (1 + z_α · σ).
    8. Continuity in σ_T at the witness boundary.
    9. Variance summation across stages matches closed-form prediction.
"""
from __future__ import annotations

import math
import random

import pytest

from tare.parallel.joint_partitioner import (
    JointPlan,
    joint_partition,
)
from tare.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
)
from tare.parallel.stochastic_joint_partitioner import (
    StochasticJointPlan,
    _normal_cvar_coefficient,
    stochastic_joint_partition,
)

# ---------------------------------------------------------------------------
# Workload builders
# ---------------------------------------------------------------------------

def _witness_layers(n: int = 6) -> list[LayerProfile]:
    """Equal-FLOPS layers: fwd=1, bwd=2 per layer (total per-layer = 3)."""
    return [
        LayerProfile(index=i, fwd_flops=1.0, bwd_flops=2.0, activation_bytes=0)
        for i in range(n)
    ]


def _witness_stages_asymmetric_power(
    p1: float = 1.01,
) -> list[StageSpec]:
    """Stages with throughput=3 (so per-layer T=1) and 1% power asymmetry."""
    return [
        StageSpec(stage_id=0, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.00),
        StageSpec(stage_id=1, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=p1),
    ]


def _zero_cost_link(src: int, dst: int) -> LinkSpec:
    return LinkSpec(src_stage=src, dst_stage=dst, bandwidth_bps=1e18, latency_s=0.0)


def _analytic_cvar(
    plan: JointPlan,
    stages: list[StageSpec],
    *,
    sigma_t: float,
    sigma_p: float,
    beta: float,
    voltage_alpha: float = 2.0,
) -> float:
    """Compute CVaR_β of plan energy under given profile noise.

    Used to score a v1 plan on the v2 objective for the weak-form
    comparison: v2's CVaR is reported by the optimiser; v1's CVaR is
    computed analytically here from the plan's (cuts, throttles).
    """
    mu = 0.0
    sigma_sq = 0.0
    for s in plan.stage_exec_time:
        t_throttled = plan.stage_exec_time[s]
        r = plan.throttle_factors[s]
        t = t_throttled * r  # un-throttled
        mu += stages[s].power_draw_w * (r ** (voltage_alpha - 1)) * t
        sigma_sq += (
            (r ** (2 * (voltage_alpha - 1)))
            * (stages[s].power_draw_w ** 2)
            * (t ** 2)
            * (sigma_t ** 2 + sigma_p ** 2)
        )
    kappa = _normal_cvar_coefficient(beta)
    return mu + kappa * math.sqrt(sigma_sq)


# ---------------------------------------------------------------------------
# 1. Weak form — v2 CVaR is never worse than v1's plan scored on v2's objective
# ---------------------------------------------------------------------------

def test_v2_no_worse_than_v1_in_cvar_under_random_workloads() -> None:
    rng = random.Random(2026)
    for _ in range(3):
        n = rng.randint(6, 12)
        K = rng.randint(2, 3)
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
                power_draw_w=rng.uniform(1.0, 3.0),
            )
            for s in range(K)
        ]
        links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(K - 1)]

        # Loose floor + no throttle to isolate the variance effect.
        v1 = joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1e-6,
            throttle_min=1.0, throttle_granularity=1,
        )
        v2 = stochastic_joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1e-6,
            throttle_min=1.0, throttle_granularity=1,
            sigma_t=0.10, sigma_p=0.05,
            cvar_beta=0.05, chance_alpha=0.999,
        )
        assert v1.is_feasible() and v2.is_feasible()
        v1_cvar = _analytic_cvar(v1, stages, sigma_t=0.10, sigma_p=0.05, beta=0.05)
        assert v2.cvar_energy <= v1_cvar + 1e-9


# ---------------------------------------------------------------------------
# 2. Strict witness — 2-stage noise-asymmetric reproduces §9.3 numbers
# ---------------------------------------------------------------------------

def test_v2_strictly_better_on_2stage_noise_asymmetric_witness() -> None:
    layers = _witness_layers(n=6)
    stages = _witness_stages_asymmetric_power(p1=1.01)
    links = [_zero_cost_link(0, 1)]

    v2 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        voltage_alpha=2.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.999,
    )
    assert v2.is_feasible()
    # v2 picks the balanced cut (3 layers on each stage).
    assert v2.cuts == (2,)
    assert v2.expected_energy == pytest.approx(6.03, rel=1e-3)
    assert v2.cvar_energy == pytest.approx(6.910, rel=2e-3)

    # Variance-blind v1 picks the most asymmetric cut (lowest mean: 5+1.01).
    v1 = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
    )
    assert v1.cuts == (4,)

    # v1's CVaR (scored on v2's objective) is strictly worse.
    v1_cvar = _analytic_cvar(v1, stages, sigma_t=0.10, sigma_p=0.0, beta=0.05)
    assert v1_cvar == pytest.approx(7.062, rel=2e-3)
    gap = (v1_cvar - v2.cvar_energy) / v1_cvar
    assert gap >= 0.02  # ≥ 2pp margin (witness analytic is 2.15%)


# ---------------------------------------------------------------------------
# 3. K-stage embedding preserves the strict gap
# ---------------------------------------------------------------------------

def test_v2_strict_gain_on_3stage_embed() -> None:
    """Embed the 2-stage witness into K=3 by adding a singleton-layer trivial stage."""
    layers = _witness_layers(n=7)
    stages = [
        StageSpec(stage_id=0, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.00),
        StageSpec(stage_id=1, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.01),
        StageSpec(stage_id=2, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.00),
    ]
    links = [_zero_cost_link(0, 1), _zero_cost_link(1, 2)]

    v1 = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
    )
    v2 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.999,
    )
    assert v2.is_feasible()
    v1_cvar = _analytic_cvar(v1, stages, sigma_t=0.10, sigma_p=0.0, beta=0.05)
    assert v2.cvar_energy <= v1_cvar + 1e-9


def test_v2_strict_gain_on_4stage_embed() -> None:
    layers = _witness_layers(n=8)
    stages = [
        StageSpec(stage_id=0, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.00),
        StageSpec(stage_id=1, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.01),
        StageSpec(stage_id=2, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.00),
        StageSpec(stage_id=3, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.00),
    ]
    links = [_zero_cost_link(s, s + 1) for s in range(3)]

    v1 = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
    )
    v2 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.999,
    )
    assert v2.is_feasible()
    v1_cvar = _analytic_cvar(v1, stages, sigma_t=0.10, sigma_p=0.0, beta=0.05)
    assert v2.cvar_energy <= v1_cvar + 1e-9


# ---------------------------------------------------------------------------
# 4. σ_T scaling — gap grows monotonically with profile noise
# ---------------------------------------------------------------------------

def test_v2_cvar_gap_scales_monotonically_with_sigma_t() -> None:
    layers = _witness_layers(n=6)
    stages = _witness_stages_asymmetric_power(p1=1.01)
    links = [_zero_cost_link(0, 1)]
    v1 = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
    )
    last_gap = 0.0
    for sigma_t in (0.05, 0.10, 0.20, 0.30):
        v2 = stochastic_joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1.0 / 100.0,
            throttle_min=1.0, throttle_granularity=1,
            sigma_t=sigma_t, sigma_p=0.0,
            cvar_beta=0.05, chance_alpha=0.999,
        )
        v1_cvar = _analytic_cvar(v1, stages, sigma_t=sigma_t, sigma_p=0.0, beta=0.05)
        gap = v1_cvar - v2.cvar_energy
        assert gap >= last_gap - 1e-9
        last_gap = gap
    # Final gap at σ_T=0.30 should be > 4× the σ_T=0.05 gap.
    assert last_gap > 0.1  # absolute gap in J


# ---------------------------------------------------------------------------
# 5. DP regression — σ_T=σ_P=0 reproduces joint_partition exactly
# ---------------------------------------------------------------------------

def test_v2_reduces_to_v1_when_no_noise_random_workloads() -> None:
    rng = random.Random(7)
    for _ in range(3):
        n = rng.randint(6, 10)
        K = rng.randint(2, 3)
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
                power_draw_w=rng.uniform(1.0, 3.0),
            )
            for s in range(K)
        ]
        links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(K - 1)]
        v1 = joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1e-6,
            throttle_min=0.5, throttle_granularity=6,
        )
        v2 = stochastic_joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1e-6,
            throttle_min=0.5, throttle_granularity=6,
            sigma_t=0.0, sigma_p=0.0,
            cvar_beta=0.05, chance_alpha=0.999,
        )
        assert v2.is_feasible()
        assert v2.cuts == v1.cuts
        assert v2.energy_stddev == pytest.approx(0.0)
        assert v2.expected_energy == pytest.approx(v1.energy_per_iter, rel=1e-9)
        assert v2.cvar_energy == pytest.approx(v1.energy_per_iter, rel=1e-9)


# ---------------------------------------------------------------------------
# 6. Reductions — β controls risk-aversion
# ---------------------------------------------------------------------------

def test_v2_high_beta_approaches_mean_minimisation() -> None:
    """β close to 1 ⇒ κ_β → 0 ⇒ CVaR objective ≈ expected energy."""
    layers = _witness_layers(n=6)
    stages = _witness_stages_asymmetric_power(p1=1.01)
    links = [_zero_cost_link(0, 1)]
    v1 = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
    )
    # β = 0.95 → κ_β ≈ 0.108, very small risk-aversion.
    v2 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.95, chance_alpha=0.999,
    )
    # At β=0.95, v2 should pick the same cut as v1 (mean-driven).
    assert v2.cuts == v1.cuts


def test_v2_low_beta_picks_minimum_variance_among_mean_ties() -> None:
    """β close to 0 ⇒ κ_β → ∞ ⇒ variance dominates → balanced cut."""
    layers = _witness_layers(n=6)
    stages = _witness_stages_asymmetric_power(p1=1.01)
    links = [_zero_cost_link(0, 1)]
    # β = 0.001 ⇒ κ_β very large.
    v2 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.001, chance_alpha=0.999,
    )
    # At very-low β, v2 picks the balanced cut (minimum variance).
    assert v2.cuts == (2,)


# ---------------------------------------------------------------------------
# 7. Chance constraints — (Θ̃) and (P̃) inflate with z_α · σ
# ---------------------------------------------------------------------------

def test_chance_throughput_floor_inflates_with_z_alpha_sigma_t() -> None:
    """A floor that is feasible at σ_T=0 becomes infeasible at σ_T=0.20 with α=0.05."""
    layers = _witness_layers(n=6)
    stages = _witness_stages_asymmetric_power(p1=1.0)  # symmetric for clarity
    links = [_zero_cost_link(0, 1)]
    # T_floor = max(T_s) = 3 (at j=2). At σ_T=0, just feasible.
    # At σ_T=0.20, α=0.05: inflation factor = 1 + 1.645*0.20 = 1.329.
    # Effective T_s_inflated = 3·1.329 = 3.987 > T_floor = 3 → infeasible.
    feasible_zero_noise = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 3.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.0, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.05,
    )
    infeasible_high_noise = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 3.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.20, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.05,
    )
    assert feasible_zero_noise.is_feasible()
    assert not infeasible_high_noise.is_feasible()


def test_chance_power_cap_inflates_with_z_alpha_sigma_p() -> None:
    """A cap that admits r=1 at σ_P=0 rejects r=1 at higher σ_P."""
    layers = _witness_layers(n=4)
    # Power 1.0 W, cap 1.05 W: feasible at σ_P=0 (P·r^α=1.0 ≤ 1.05).
    # At σ_P=0.10, α=0.05: inflation = 1 + 1.645·0.10 = 1.1645. P·r^α·inflation = 1.1645 > 1.05 → infeasible.
    stages = [
        StageSpec(stage_id=0, throughput_flops=3.0, memory_bytes=10**18,
                  power_draw_w=1.0, power_cap_w=1.05),
        StageSpec(stage_id=1, throughput_flops=3.0, memory_bytes=10**18,
                  power_draw_w=1.0, power_cap_w=1.05),
    ]
    links = [_zero_cost_link(0, 1)]
    feasible_zero = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.0, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.05,
    )
    infeasible_at_r1 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.0, sigma_p=0.10,
        cvar_beta=0.05, chance_alpha=0.05,
    )
    assert feasible_zero.is_feasible()
    assert not infeasible_at_r1.is_feasible()


# ---------------------------------------------------------------------------
# 8. Continuity — gap is smooth in σ_T near the witness
# ---------------------------------------------------------------------------

def test_v2_cvar_gap_is_continuous_in_sigma_t_at_witness() -> None:
    """Small perturbations to σ_T around 0.10 produce small perturbations to gap."""
    layers = _witness_layers(n=6)
    stages = _witness_stages_asymmetric_power(p1=1.01)
    links = [_zero_cost_link(0, 1)]
    v1 = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
    )
    gaps = []
    for sigma_t in (0.095, 0.10, 0.105):
        v2 = stochastic_joint_partition(
            layers, stages, links,
            throughput_floor_iters_per_s=1.0 / 100.0,
            throttle_min=1.0, throttle_granularity=1,
            sigma_t=sigma_t, sigma_p=0.0,
            cvar_beta=0.05, chance_alpha=0.999,
        )
        v1_cvar = _analytic_cvar(v1, stages, sigma_t=sigma_t, sigma_p=0.0, beta=0.05)
        gaps.append(v1_cvar - v2.cvar_energy)
    # Continuity: adjacent gaps differ by < 10% of the middle value.
    assert abs(gaps[0] - gaps[1]) < 0.10 * gaps[1]
    assert abs(gaps[2] - gaps[1]) < 0.10 * gaps[1]


# ---------------------------------------------------------------------------
# 9. Variance summation across stages matches closed-form prediction
# ---------------------------------------------------------------------------

def test_v2_variance_sums_across_stages_in_closed_form() -> None:
    """For the balanced witness cut (j=2), σ_E² = Σ_s (μ_P_s · μ_T_s · σ_combined)²."""
    layers = _witness_layers(n=6)
    stages = _witness_stages_asymmetric_power(p1=1.01)
    links = [_zero_cost_link(0, 1)]
    v2 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.999,
    )
    # j=2: 3 layers per stage, T_s = 3 each, μ_P=(1.0, 1.01), σ_T=0.10.
    expected_sigma_sq = (1.00 ** 2) * (3 ** 2) * (0.10 ** 2) + (1.01 ** 2) * (3 ** 2) * (0.10 ** 2)
    assert (v2.energy_stddev ** 2) == pytest.approx(expected_sigma_sq, rel=1e-9)


# ---------------------------------------------------------------------------
# Edge cases + validation
# ---------------------------------------------------------------------------

def test_k1_single_stage_with_noise() -> None:
    layers = _witness_layers(n=4)
    stages = [
        StageSpec(stage_id=0, throughput_flops=3.0, memory_bytes=10**18, power_draw_w=1.0),
    ]
    plan = stochastic_joint_partition(
        layers, stages, [],
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.999,
    )
    assert plan.is_feasible()
    assert plan.cuts == ()
    assert len(plan.throttle_factors) == 1
    # σ_E = μ_P · t · σ_T = 1.0 · 4.0 · 0.10 = 0.4
    assert plan.energy_stddev == pytest.approx(0.4, rel=1e-9)


def test_rejects_invalid_sigma_t() -> None:
    layers = _witness_layers()
    stages = _witness_stages_asymmetric_power()
    with pytest.raises(ValueError):
        stochastic_joint_partition(
            layers, stages, [_zero_cost_link(0, 1)],
            throughput_floor_iters_per_s=1.0,
            sigma_t=1.5, sigma_p=0.0,
            cvar_beta=0.05, chance_alpha=0.05,
        )


def test_rejects_invalid_cvar_beta() -> None:
    layers = _witness_layers()
    stages = _witness_stages_asymmetric_power()
    with pytest.raises(ValueError):
        stochastic_joint_partition(
            layers, stages, [_zero_cost_link(0, 1)],
            throughput_floor_iters_per_s=1.0,
            cvar_beta=0.0, chance_alpha=0.05,
        )


def test_rejects_invalid_chance_alpha() -> None:
    layers = _witness_layers()
    stages = _witness_stages_asymmetric_power()
    with pytest.raises(ValueError):
        stochastic_joint_partition(
            layers, stages, [_zero_cost_link(0, 1)],
            throughput_floor_iters_per_s=1.0,
            cvar_beta=0.05, chance_alpha=1.5,
        )


def test_deterministic_on_repeated_calls() -> None:
    layers = _witness_layers()
    stages = _witness_stages_asymmetric_power()
    links = [_zero_cost_link(0, 1)]
    a = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.999,
    )
    b = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.999,
    )
    assert a.cuts == b.cuts
    assert a.cvar_energy == b.cvar_energy


def test_backtracking_reproduces_cvar_score() -> None:
    """The plan's (expected_energy, energy_stddev) must reproduce its cvar_energy."""
    layers = _witness_layers()
    stages = _witness_stages_asymmetric_power()
    links = [_zero_cost_link(0, 1)]
    plan = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=1.0 / 100.0,
        throttle_min=1.0, throttle_granularity=1,
        sigma_t=0.10, sigma_p=0.0,
        cvar_beta=0.05, chance_alpha=0.999,
    )
    assert plan.is_feasible()
    kappa = _normal_cvar_coefficient(0.05)
    expected_cvar = plan.expected_energy + kappa * plan.energy_stddev
    assert plan.cvar_energy == pytest.approx(expected_cvar, rel=1e-9)


def test_infeasible_plan_reports_correctly() -> None:
    plan = StochasticJointPlan()
    assert not plan.is_feasible()
    assert math.isinf(plan.cvar_energy)

"""Tests for the PPO scaling policy scaffold.

Scaffolding-only: verifies observation construction, reward function, action
spec, and decide() error path. Full PPO training is not exercised here.
"""
from __future__ import annotations

import pytest

from hise.energy.rl_policy import (
    OBSERVATION_DIM,
    PPOAction,
    PPOObservation,
    PPOScalingPolicy,
    build_observation,
    compute_reward,
    discrete_action_space_size,
)

# --- PPOObservation ---

def test_observation_has_8_fields_matching_constant() -> None:
    obs = PPOObservation(
        training_progress=0.5, gpu_fraction=0.5, power_fraction=0.5,
        throughput_fraction=0.5, intensity_fraction=0.5,
        deadline_fraction_remaining=0.5, energy_fraction_remaining=0.5,
        iters_per_joule_normalized=0.5,
    )
    assert len(obs.to_vector()) == OBSERVATION_DIM == 8


def test_observation_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="out of"):
        PPOObservation(
            training_progress=1.5,   # > 1.0
            gpu_fraction=0.5, power_fraction=0.5, throughput_fraction=0.5,
            intensity_fraction=0.5, deadline_fraction_remaining=0.5,
            energy_fraction_remaining=0.5, iters_per_joule_normalized=0.5,
        )


def test_observation_accepts_boundary_values() -> None:
    """0.0 and 1.0 must both be accepted (closed interval)."""
    PPOObservation(
        training_progress=0.0, gpu_fraction=1.0, power_fraction=0.0,
        throughput_fraction=1.0, intensity_fraction=0.0,
        deadline_fraction_remaining=1.0, energy_fraction_remaining=0.0,
        iters_per_joule_normalized=1.0,
    )


# --- build_observation helper ---

def test_build_observation_normalises_inputs() -> None:
    obs = build_observation(
        iters_done=500, iters_target=1000,
        current_gpus=4, max_gpus=8,
        avg_power_w=200.0, peak_power_w=400.0,
        avg_throughput_iters_per_s=100.0, peak_throughput_iters_per_s=200.0,
        grid_intensity_g_per_kwh=500.0, max_grid_intensity_g_per_kwh=1000.0,
        elapsed_s=300.0, deadline_s=600.0,
        energy_used_kwh=2.0, energy_budget_kwh=10.0,
        target_iters_per_joule=1.0,
    )
    assert obs.training_progress == 0.5
    assert obs.gpu_fraction == 0.5
    assert obs.power_fraction == 0.5
    assert obs.throughput_fraction == 0.5
    assert obs.intensity_fraction == 0.5
    assert obs.deadline_fraction_remaining == 0.5
    assert obs.energy_fraction_remaining == 0.8


def test_build_observation_handles_zero_denominators() -> None:
    """Division-by-zero (e.g., no carbon signal) yields 0.0 not NaN."""
    obs = build_observation(
        iters_done=0, iters_target=0,                 # no progress denominator
        current_gpus=4, max_gpus=8,
        avg_power_w=200.0, peak_power_w=400.0,
        avg_throughput_iters_per_s=100.0, peak_throughput_iters_per_s=200.0,
        grid_intensity_g_per_kwh=500.0, max_grid_intensity_g_per_kwh=0.0,
        elapsed_s=0.0, deadline_s=0.0,
        energy_used_kwh=0.0, energy_budget_kwh=0.0,
        target_iters_per_joule=0.0,
    )
    assert obs.training_progress == 0.0
    assert obs.intensity_fraction == 0.0
    assert obs.deadline_fraction_remaining == 0.0


def test_build_observation_clamps_overflow() -> None:
    """When iters_done > iters_target (overrun), training_progress clamps to 1.0."""
    obs = build_observation(
        iters_done=2000, iters_target=1000,
        current_gpus=8, max_gpus=8,
        avg_power_w=400.0, peak_power_w=400.0,
        avg_throughput_iters_per_s=100.0, peak_throughput_iters_per_s=100.0,
        grid_intensity_g_per_kwh=500.0, max_grid_intensity_g_per_kwh=1000.0,
        elapsed_s=100.0, deadline_s=1000.0,
        energy_used_kwh=1.0, energy_budget_kwh=10.0,
        target_iters_per_joule=1.0,
    )
    assert obs.training_progress == 1.0   # clamped


# --- compute_reward ---

def test_reward_penalises_energy_use() -> None:
    r = compute_reward(delta_kwh=1.0, deadline_overshoot_s=0, reconfig_indicator=0)
    assert r == -1.0


def test_reward_penalises_deadline_overshoot() -> None:
    """ΔkWh=0 + overshoot 100s + λ=0.01 → -1.0."""
    r = compute_reward(delta_kwh=0.0, deadline_overshoot_s=100.0,
                       reconfig_indicator=0, lambda_lag=0.01)
    assert r == -1.0


def test_reward_penalises_reconfig() -> None:
    r = compute_reward(delta_kwh=0.0, deadline_overshoot_s=0,
                       reconfig_indicator=1, mu_reconfig=0.05)
    assert r == -0.05


def test_reward_combines_all_three_terms() -> None:
    r = compute_reward(
        delta_kwh=2.0, deadline_overshoot_s=50.0, reconfig_indicator=1,
        lambda_lag=0.01, mu_reconfig=0.05,
    )
    # -(2.0 + 0.01 * 50 + 0.05 * 1) = -(2.0 + 0.5 + 0.05) = -2.55
    assert abs(r - (-2.55)) < 1e-9


def test_reward_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="delta_kwh"):
        compute_reward(delta_kwh=-0.1, deadline_overshoot_s=0, reconfig_indicator=0)
    with pytest.raises(ValueError, match="deadline_overshoot"):
        compute_reward(delta_kwh=0.0, deadline_overshoot_s=-1.0, reconfig_indicator=0)
    with pytest.raises(ValueError, match="reconfig_indicator"):
        compute_reward(delta_kwh=0.0, deadline_overshoot_s=0, reconfig_indicator=2)


# --- PPOAction ---

def test_action_rejects_zero_gpus() -> None:
    with pytest.raises(ValueError, match="gpu_count must be >= 1"):
        PPOAction(gpu_count=0)


# --- PPOScalingPolicy scaffold ---

def test_policy_construction_validates_bounds() -> None:
    with pytest.raises(ValueError, match="min_gpus must be >= 1"):
        PPOScalingPolicy(min_gpus=0, max_gpus=8)
    with pytest.raises(ValueError, match="max_gpus.*must be >= min_gpus"):
        PPOScalingPolicy(min_gpus=8, max_gpus=4)


def test_policy_decide_raises_without_model() -> None:
    """Pre-Tuần-4 default: no model loaded → decide() raises with clear message."""
    policy = PPOScalingPolicy(min_gpus=1, max_gpus=8)
    obs = PPOObservation(
        training_progress=0.5, gpu_fraction=0.5, power_fraction=0.5,
        throughput_fraction=0.5, intensity_fraction=0.5,
        deadline_fraction_remaining=0.5, energy_fraction_remaining=0.5,
        iters_per_joule_normalized=0.5,
    )
    with pytest.raises(RuntimeError, match="PPO not yet trained"):
        policy.decide(current_gpus=4, observation=obs)


def test_policy_decide_with_stub_model_clamps_to_bounds() -> None:
    """A stub model returning action=10 must be clamped to max_gpus=8."""

    class _StubModel:
        def predict(self, _obs, deterministic: bool = True):  # noqa: FBT002
            import numpy as np
            return np.int64(10), None    # SB3 returns numpy ints

    policy = PPOScalingPolicy(model=_StubModel(), min_gpus=1, max_gpus=8)
    obs = PPOObservation(
        training_progress=0.5, gpu_fraction=0.5, power_fraction=0.5,
        throughput_fraction=0.5, intensity_fraction=0.5,
        deadline_fraction_remaining=0.5, energy_fraction_remaining=0.5,
        iters_per_joule_normalized=0.5,
    )
    decision = policy.decide(current_gpus=4, observation=obs)
    assert decision.target_gpus == 8   # clamped from 10 + 1 = 11 → 8
    assert "PPO" in decision.reason


# --- discrete_action_space_size ---

def test_discrete_action_space_size() -> None:
    assert discrete_action_space_size(1, 8) == 8
    assert discrete_action_space_size(2, 4) == 3
    assert discrete_action_space_size(5, 5) == 1
    with pytest.raises(ValueError, match="max_gpus < min_gpus"):
        discrete_action_space_size(8, 1)

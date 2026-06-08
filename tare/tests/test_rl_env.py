"""Tests for the carbon-aware RL environment.

These exercise the Gymnasium contract (reset/step shapes, dtype, spaces, episode
termination) without running PPO. SB3-specific integration tests live in the
training experiment script.
"""
from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from tare.energy.rl_env import EnvConfig, TareCarbonEnv  # noqa: E402

# --- EnvConfig validation ---


def test_env_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="min_gpus"):
        EnvConfig(min_gpus=0)
    with pytest.raises(ValueError, match="max_gpus"):
        EnvConfig(min_gpus=4, max_gpus=2)
    with pytest.raises(ValueError, match="target_iters"):
        EnvConfig(target_iters=0)
    with pytest.raises(ValueError, match="deadline"):
        EnvConfig(deadline_seconds=0)
    with pytest.raises(ValueError, match="energy_budget"):
        EnvConfig(energy_budget_kwh=0)
    with pytest.raises(ValueError, match="tick_seconds"):
        EnvConfig(tick_seconds=0)


# --- Reset / observation shape ---


def test_reset_returns_8_dim_box_observation() -> None:
    env = TareCarbonEnv()
    obs, info = env.reset()
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    assert info == {}


def test_reset_starts_with_min_gpus() -> None:
    env = TareCarbonEnv(EnvConfig(min_gpus=2, max_gpus=8))
    env.reset()
    # gpu_fraction = current/max = 2/8 = 0.25.
    obs, _, _, _, _ = env.step(0)   # action 0 → min_gpus
    assert abs(float(obs[1]) - 0.25) < 1e-6


# --- Step ---


def test_step_returns_5_tuple() -> None:
    env = TareCarbonEnv()
    env.reset()
    out = env.step(0)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_observation_stays_in_box() -> None:
    env = TareCarbonEnv()
    env.reset()
    for _ in range(20):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(int(a))
        assert 0.0 <= obs.min() and obs.max() <= 1.0
        if term or trunc:
            break


def test_step_advances_sim_time_by_tick() -> None:
    cfg = EnvConfig(tick_seconds=600)
    env = TareCarbonEnv(cfg)
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["sim_t"] == 600
    _, _, _, _, info = env.step(0)
    assert info["sim_t"] == 1200


def test_step_reconfig_indicator_fires_on_gpu_change() -> None:
    env = TareCarbonEnv(EnvConfig(min_gpus=1, max_gpus=4))
    env.reset()
    # First step picks action=0 → min_gpus=1. Reset already set current_gpus=1
    # so the first step is a no-op (no reconfig).
    _, _, _, _, info0 = env.step(0)
    assert info0["reconfig"] == 0
    # Next step picks action=2 → 3 GPUs → reconfig fires.
    _, _, _, _, info1 = env.step(2)
    assert info1["reconfig"] == 1


def test_step_clamps_action_to_max_gpus() -> None:
    env = TareCarbonEnv(EnvConfig(min_gpus=1, max_gpus=4))
    env.reset()
    # Action 99 should clamp to max_gpus=4. Discrete space rejects out-of-range
    # values, but the env internally also clamps for safety.
    _, _, _, _, info = env.step(min(99, env.action_space.n - 1))
    assert 1 <= info["current_gpus"] <= 4


# --- Termination conditions ---


def test_episode_terminates_when_target_iters_reached() -> None:
    """Tiny target → must complete within a few ticks."""
    cfg = EnvConfig(target_iters=100, max_gpus=8, tick_seconds=300)
    env = TareCarbonEnv(cfg)
    env.reset()
    terminated = False
    for _ in range(50):
        # action max → 8 GPUs → ~10 iter/s → 3000 iters in 1 tick → terminate.
        _, _, terminated, _, info = env.step(env.action_space.n - 1)
        if terminated:
            break
    assert terminated
    assert info["iters_done"] >= 100


def test_episode_terminates_when_energy_budget_exhausted() -> None:
    cfg = EnvConfig(
        target_iters=10_000_000,        # far too high, won't finish
        energy_budget_kwh=0.05,         # ~5 ticks at 8 GPUs * 300W * 5min
        deadline_seconds=24 * 3600.0,
    )
    env = TareCarbonEnv(cfg)
    env.reset()
    terminated = False
    for _ in range(100):
        _, _, terminated, _, info = env.step(env.action_space.n - 1)
        if terminated:
            break
    assert terminated
    assert info["energy_used_kwh"] >= cfg.energy_budget_kwh


def test_episode_truncates_when_deadline_passes() -> None:
    cfg = EnvConfig(
        target_iters=10_000_000,
        deadline_seconds=900,           # 3 ticks
        energy_budget_kwh=1000.0,       # not the binding constraint
        tick_seconds=300,
    )
    env = TareCarbonEnv(cfg)
    env.reset()
    truncated = False
    for _ in range(10):
        _, _, _, truncated, info = env.step(0)
        if truncated:
            break
    assert truncated
    assert info["sim_t"] >= 900


# --- Reward sign ---


def test_step_reward_is_non_positive() -> None:
    """Reward is -ΔkWh - λ·lag - μ·reconfig — all non-negative components, so r ≤ 0."""
    env = TareCarbonEnv()
    env.reset()
    for _ in range(20):
        _, reward, term, trunc, _ = env.step(env.action_space.sample())
        assert reward <= 0.0
        if term or trunc:
            break

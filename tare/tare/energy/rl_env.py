"""Gymnasium environment that wraps the carbon-trace simulator for PPO training.

The environment exposes the same observation vector as ``PPOObservation`` (8-dim,
[0, 1] normalised) and the same reward function as ``compute_reward``. One episode
covers one training job from start to either completion (``iters_done >= target``),
deadline expiry, or energy budget exhaustion.

Decoupling the env from ``experiments/exp02_carbon_replay.py`` keeps the SB3 +
gymnasium dependency optional — heavy RL deps live in ``[rl]`` extras, so the
core ``tare`` install stays light. Training scripts import this module; the
orchestrator imports only the lightweight scaffold in ``tare.energy.rl_policy``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover — install tare[rl] to use this module.
    raise ImportError(
        "tare.energy.rl_env requires gymnasium; install tare[rl]"
    ) from exc

from tare.energy.carbon_trace import CarbonTrace, synthetic_solar_trace
from tare.energy.rl_policy import (
    OBSERVATION_DIM,
    build_observation,
    compute_reward,
    discrete_action_space_size,
)


@dataclass
class EnvConfig:
    """Knobs for one training episode. Defaults match the synthetic ResNet-18 / CIFAR-10
    setup used in ``exp02_carbon_replay.py`` so PPO trains in the same regime the
    rule-based and MPC baselines are evaluated on."""

    min_gpus: int = 1
    max_gpus: int = 8
    target_iters: int = 200_000
    deadline_seconds: float = 24 * 3600.0
    energy_budget_kwh: float = 5.0
    tick_seconds: int = 300                  # 5-min decision cadence
    power_per_gpu_w: float = 300.0
    base_throughput_iters_per_s: float = 5.0
    scaling_efficiency: float = 0.85
    reconfig_latency_s: float = 5.0          # cycles spent reconfiguring count as no-progress
    reconfig_energy_kwh: float = 0.002       # per reconfig event
    lambda_lag: float = 0.01                 # reward weight on deadline overshoot
    mu_reconfig: float = 0.05                # reward weight on reconfig indicator
    target_iters_per_joule: float = 50.0     # for obs.iters_per_joule_normalized
    max_grid_intensity_g_per_kwh: float = 800.0  # for obs.intensity_fraction

    def __post_init__(self) -> None:
        if self.min_gpus < 1:
            raise ValueError("min_gpus must be >= 1")
        if self.max_gpus < self.min_gpus:
            raise ValueError("max_gpus must be >= min_gpus")
        if self.target_iters <= 0:
            raise ValueError("target_iters must be > 0")
        if self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be > 0")
        if self.energy_budget_kwh <= 0:
            raise ValueError("energy_budget_kwh must be > 0")
        if self.tick_seconds <= 0:
            raise ValueError("tick_seconds must be > 0")


@dataclass
class _EpisodeState:
    sim_t: float = 0.0
    iter_done: int = 0
    energy_used_kwh: float = 0.0
    current_gpus: int = 1
    avg_power_w: float = 0.0
    avg_throughput_iters_per_s: float = 0.0
    history_gpus: list[int] = field(default_factory=list)


class TareCarbonEnv(gym.Env):
    """Gymnasium env for the carbon-aware elastic-scaling task.

    Observation space: ``Box(0.0, 1.0, (8,), float32)`` matching ``PPOObservation``.
    Action space: ``Discrete(max_gpus - min_gpus + 1)``; action ``a`` ⇒ ``min_gpus + a``.
    Reward: ``-ΔkWh − λ·max(0, deadline_overshoot) − μ·reconfig_indicator`` per tick
        (``compute_reward`` from ``rl_policy.py``).
    Termination: ``iters_done ≥ target`` (success) | ``sim_t ≥ deadline`` (timeout) |
        ``energy_used ≥ budget`` (energy exhausted).

    Args:
        config: ``EnvConfig`` knobs; defaults give a 24h / 200k-iter / 5kWh job.
        trace: ``CarbonTrace`` for grid intensity at each tick. ``None`` uses the
            synthetic-solar 24h cycle from ``synthetic_solar_trace()``.
        seed: optional RNG seed for ``reset()``; ``None`` uses gymnasium's default.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        trace: CarbonTrace | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.trace = trace or synthetic_solar_trace(
            hours=int(self.config.deadline_seconds // 3600) + 1
        )
        n_actions = discrete_action_space_size(self.config.min_gpus, self.config.max_gpus)
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32,
        )
        self._state = _EpisodeState()
        self._seed = seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = _EpisodeState(current_gpus=self.config.min_gpus)
        obs = self._build_obs()
        return obs, {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        prev_gpus = self._state.current_gpus
        target_gpus = max(cfg.min_gpus, min(cfg.max_gpus, int(action) + cfg.min_gpus))
        reconfig = int(target_gpus != prev_gpus)

        effective_tick_s = cfg.tick_seconds - (cfg.reconfig_latency_s if reconfig else 0.0)
        effective_tick_s = max(0.0, effective_tick_s)
        self._state.current_gpus = target_gpus
        self._state.history_gpus.append(target_gpus)

        # Concave throughput model — same as exp02_carbon_replay.
        throughput = cfg.base_throughput_iters_per_s * (target_gpus ** cfg.scaling_efficiency)
        new_iters = int(throughput * effective_tick_s)
        self._state.iter_done += new_iters

        # Energy: full-tick draw + per-reconfig bump.
        active_kwh = (cfg.power_per_gpu_w * target_gpus * cfg.tick_seconds) / 3_600_000.0
        delta_kwh = active_kwh + (cfg.reconfig_energy_kwh if reconfig else 0.0)
        self._state.energy_used_kwh += delta_kwh
        self._state.avg_power_w = (
            cfg.power_per_gpu_w * target_gpus
        )  # instantaneous proxy; production reads NVML
        self._state.avg_throughput_iters_per_s = throughput

        # Project deadline overshoot from current rate (0 if on track).
        remaining_iters = max(0, cfg.target_iters - self._state.iter_done)
        if throughput > 0:
            projected_finish_s = self._state.sim_t + (remaining_iters / throughput)
            overshoot = max(0.0, projected_finish_s - cfg.deadline_seconds)
        else:
            overshoot = cfg.deadline_seconds

        reward = compute_reward(
            delta_kwh=delta_kwh,
            deadline_overshoot_s=overshoot,
            reconfig_indicator=reconfig,
            lambda_lag=cfg.lambda_lag,
            mu_reconfig=cfg.mu_reconfig,
        )

        self._state.sim_t += cfg.tick_seconds
        terminated = (
            self._state.iter_done >= cfg.target_iters
            or self._state.energy_used_kwh >= cfg.energy_budget_kwh
        )
        truncated = self._state.sim_t >= cfg.deadline_seconds

        info = {
            "iters_done": self._state.iter_done,
            "energy_used_kwh": self._state.energy_used_kwh,
            "sim_t": self._state.sim_t,
            "current_gpus": target_gpus,
            "reconfig": reconfig,
            "delta_kwh": delta_kwh,
            "deadline_overshoot_s": overshoot,
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    def _build_obs(self) -> np.ndarray:
        cfg = self.config
        st = self._state
        intensity = self.trace.intensity_at(st.sim_t)
        obs = build_observation(
            iters_done=st.iter_done,
            iters_target=cfg.target_iters,
            current_gpus=st.current_gpus,
            max_gpus=cfg.max_gpus,
            avg_power_w=st.avg_power_w,
            peak_power_w=cfg.power_per_gpu_w * cfg.max_gpus,
            avg_throughput_iters_per_s=st.avg_throughput_iters_per_s,
            peak_throughput_iters_per_s=cfg.base_throughput_iters_per_s * (
                cfg.max_gpus ** cfg.scaling_efficiency
            ),
            grid_intensity_g_per_kwh=intensity,
            max_grid_intensity_g_per_kwh=cfg.max_grid_intensity_g_per_kwh,
            elapsed_s=st.sim_t,
            deadline_s=cfg.deadline_seconds,
            energy_used_kwh=st.energy_used_kwh,
            energy_budget_kwh=cfg.energy_budget_kwh,
            target_iters_per_joule=cfg.target_iters_per_joule,
        )
        return np.asarray(obs.to_vector(), dtype=np.float32)

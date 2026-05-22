"""PPO-based scaling policy scaffold for the energy-aware control loop.

**Status**: scaffolding only (Tuần 3 deliverable D3.3a). Full training lands
Tuần 4 D3.3b with the empirical decide-point (per [Q2-KEEP default decision](docs/phase2-plan.md §10)).
If 1000-episode synthetic-trace training fails to converge → drop PPO, reframe
C1 as MPC-based per [novelty-feasibility-review.md §2.1 R3](docs/novelty-feasibility-review.md).

This module ships:

* ``PPOObservation`` — 8-dim normalized state vector consumed by Stable-Baselines3.
* ``PPOAction`` — discrete GPU count choice; ``MultiDiscrete`` variant left as TODO.
* ``compute_reward`` — energy-aware reward function:
  ``r = -ΔkWh - λ·max(0, deadline_overshoot) - μ·reconfig_indicator``.
* ``PPOScalingPolicy`` — wraps a Stable-Baselines3 PPO model with the same
  ``.decide()`` interface as PowerAwareRulePolicy / MPCPolicy so the orchestrator
  can swap policies via the existing dispatch in ``EnergyAwareControlLoop``.
* ``HISEEnv`` — minimal ``gymnasium.Env`` wiring observation + action + reward
  for training in ``experiments/exp02_carbon_replay.py``.

The Gym environment + actual training loop are NOT in this file (heavy SB3 +
gymnasium imports). They live in ``experiments/`` per the testbed layout.
This file is the lightweight construction interface so the orchestrator can
import ``PPOScalingPolicy`` without dragging SB3 into core hise dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hise.energy.policy import EnergyDecision


@dataclass(frozen=True)
class PPOObservation:
    """8-dim continuous state vector for the PPO agent.

    All fields normalized to ``[0, 1]`` so the policy network sees consistent
    magnitudes across workloads. The orchestrator computes each component
    from job + telemetry state at decide-time.

    Layout matches [docs/phase2-plan.md §D3.3](docs/phase2-plan.md):

    | Index | Field | Source |
    |---|---|---|
    | 0 | training_progress | ``iters_done / iters_target`` |
    | 1 | gpu_fraction | ``current_gpus / max_gpus`` |
    | 2 | power_fraction | ``avg_power / max_power_per_worker`` |
    | 3 | throughput_fraction | ``avg_throughput / peak_throughput`` |
    | 4 | intensity_fraction | ``grid_intensity / max_intensity`` (0 if no carbon) |
    | 5 | deadline_fraction_remaining | ``(deadline - elapsed) / deadline`` |
    | 6 | energy_fraction_remaining | ``(budget - used) / budget`` |
    | 7 | iters_per_joule_normalized | ``current_iter_per_J / target_iter_per_J`` |
    """

    training_progress: float
    gpu_fraction: float
    power_fraction: float
    throughput_fraction: float
    intensity_fraction: float
    deadline_fraction_remaining: float
    energy_fraction_remaining: float
    iters_per_joule_normalized: float

    def to_vector(self) -> tuple[float, ...]:
        """Tuple representation matching SB3 observation space."""
        return (
            self.training_progress,
            self.gpu_fraction,
            self.power_fraction,
            self.throughput_fraction,
            self.intensity_fraction,
            self.deadline_fraction_remaining,
            self.energy_fraction_remaining,
            self.iters_per_joule_normalized,
        )

    def __post_init__(self) -> None:
        for name, v in zip(
            ("training_progress", "gpu_fraction", "power_fraction",
             "throughput_fraction", "intensity_fraction",
             "deadline_fraction_remaining", "energy_fraction_remaining",
             "iters_per_joule_normalized"),
            self.to_vector(),
            strict=True,
        ):
            if not (0.0 <= v <= 1.0 + 1e-6):
                raise ValueError(
                    f"{name}={v} out of [0, 1] — PPO obs must be normalized."
                )


@dataclass(frozen=True)
class PPOAction:
    """Discrete GPU-count choice. ``MultiDiscrete`` variant (GPUs × power_cap)
    is a TODO for Tuần 4 if D3.3 keeps PPO (per Q2 decide-point)."""

    gpu_count: int

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise ValueError(f"gpu_count must be >= 1, got {self.gpu_count}")


def compute_reward(
    delta_kwh: float,
    deadline_overshoot_s: float,
    reconfig_indicator: int,
    *,
    lambda_lag: float = 0.01,
    mu_reconfig: float = 0.05,
) -> float:
    """PPO reward per tick.

    Per [docs/phase2-plan.md §D3.3](docs/phase2-plan.md):

        r_t = -ΔkWh - λ · max(0, deadline_overshoot) - μ · reconfig_indicator

    Three terms in priority order (mirroring the MPC cost decomposition):
    1. ``ΔkWh`` is the energy spent during this tick — negative reward weight.
    2. ``deadline_overshoot`` is the projected overshoot in seconds (0 if on
       track). ``λ`` is the lag weight; set small (~0.01) so PPO doesn't trivially
       max-out GPUs to crush deadline.
    3. ``reconfig_indicator`` is 1 if this tick switched GPU count, 0 otherwise.
       ``μ`` matches MPC's reconfig penalty knob, discouraging flapping.

    Args:
        delta_kwh: energy consumed this tick (kWh); must be >= 0.
        deadline_overshoot_s: max(0, projected_completion_time - deadline) in seconds.
        reconfig_indicator: 0 or 1.
        lambda_lag: lag weight (default 0.01).
        mu_reconfig: reconfig weight (default 0.05).

    Returns:
        Scalar reward, typically negative (we minimize cost → maximize -cost).
    """
    if delta_kwh < 0:
        raise ValueError(f"delta_kwh must be >= 0, got {delta_kwh}")
    if deadline_overshoot_s < 0:
        raise ValueError(f"deadline_overshoot_s must be >= 0, got {deadline_overshoot_s}")
    if reconfig_indicator not in (0, 1):
        raise ValueError(f"reconfig_indicator must be 0 or 1, got {reconfig_indicator}")
    return -(delta_kwh + lambda_lag * deadline_overshoot_s + mu_reconfig * reconfig_indicator)


@dataclass
class PPOScalingPolicy:
    """Wraps a trained Stable-Baselines3 PPO model with the standard ``decide()``
    interface so the orchestrator can swap it in for PowerAwareRule / MPC.

    **Scaffolding state**: this class does NOT train the model — training lives
    in ``experiments/exp02_carbon_replay.py`` per docs/phase2-plan.md §D3.3.
    Tuần 4 decide-point: if 1000-episode synthetic-trace training shows no
    monotone reward improvement, drop PPO entirely (per Q2 empirical decide).

    Args:
        model: a loaded Stable-Baselines3 PPO instance (``stable_baselines3.PPO``).
            ``None`` is accepted for construction-only tests; ``decide()`` will
            then raise ``RuntimeError`` until a model is loaded.
        min_gpus: lower bound on GPU count.
        max_gpus: upper bound on GPU count.
        max_power_per_worker_w: normalization constant for obs.power_fraction.

    Usage (production, after Tuần 4 training)::

        from stable_baselines3 import PPO
        model = PPO.load("ppo_hise_energy.zip")
        policy = PPOScalingPolicy(model=model, min_gpus=1, max_gpus=8,
                                   max_power_per_worker_w=400.0)
        # Inside orchestrator: pass `policy` to EnergyAwareControlLoop.
    """

    model: Any = None    # stable_baselines3.PPO instance
    min_gpus: int = 1
    max_gpus: int = 8
    max_power_per_worker_w: float = 400.0

    def __post_init__(self) -> None:
        if self.min_gpus < 1:
            raise ValueError(f"min_gpus must be >= 1, got {self.min_gpus}")
        if self.max_gpus < self.min_gpus:
            raise ValueError(
                f"max_gpus ({self.max_gpus}) must be >= min_gpus ({self.min_gpus})"
            )

    def decide(self, current_gpus: int, observation: PPOObservation) -> EnergyDecision:
        """Predict next-tick GPU count from PPO policy.

        Signature differs from PowerAwareRule (which takes telemetry directly):
        PPO needs the pre-computed observation vector — the orchestrator
        constructs it from telemetry + carbon + job state before invoking.

        Returns ``EnergyDecision`` with ``target_gpus`` clamped to ``[min, max]``
        and ``reason="PPO predicted N"``. Raises ``RuntimeError`` if model is
        ``None`` (scaffolding default — pre-training).
        """
        if self.model is None:
            raise RuntimeError(
                "PPOScalingPolicy.model is None — PPO not yet trained "
                "(Tuần 4 D3.3b deliverable). Use PowerAwareRulePolicy or "
                "MPCPolicy until PPO training converges."
            )
        obs_vec = observation.to_vector()
        action, _ = self.model.predict(obs_vec, deterministic=True)
        # SB3 returns numpy int for Discrete; convert + bound.
        gpu_choice = int(action) + self.min_gpus
        target = max(self.min_gpus, min(self.max_gpus, gpu_choice))
        return EnergyDecision(target_gpus=target, reason=f"PPO predicted {target}")


def build_observation(
    iters_done: int,
    iters_target: int,
    current_gpus: int,
    max_gpus: int,
    avg_power_w: float,
    peak_power_w: float,
    avg_throughput_iters_per_s: float,
    peak_throughput_iters_per_s: float,
    grid_intensity_g_per_kwh: float,
    max_grid_intensity_g_per_kwh: float,
    elapsed_s: float,
    deadline_s: float,
    energy_used_kwh: float,
    energy_budget_kwh: float,
    target_iters_per_joule: float,
) -> PPOObservation:
    """Construct a normalized PPOObservation from raw orchestrator state.

    Each field is clamped to [0, 1]; division-by-zero (e.g., no carbon signal)
    yields 0.0 for that dimension. Used by the orchestrator's per-tick PPO
    dispatcher to convert telemetry + job state into the agent's observation.
    """
    def _frac(num: float, denom: float) -> float:
        if denom <= 0:
            return 0.0
        return max(0.0, min(1.0, num / denom))

    progress = _frac(iters_done, iters_target)
    gpu_frac = _frac(current_gpus, max_gpus)
    pow_frac = _frac(avg_power_w, peak_power_w)
    tput_frac = _frac(avg_throughput_iters_per_s, peak_throughput_iters_per_s)
    int_frac = _frac(grid_intensity_g_per_kwh, max_grid_intensity_g_per_kwh)
    deadline_frac = _frac(max(0.0, deadline_s - elapsed_s), deadline_s)
    energy_frac = _frac(max(0.0, energy_budget_kwh - energy_used_kwh), energy_budget_kwh)

    if avg_power_w > 0:
        current_ipj = avg_throughput_iters_per_s / avg_power_w
    else:
        current_ipj = 0.0
    ipj_norm = _frac(current_ipj, target_iters_per_joule) if target_iters_per_joule > 0 else 0.0

    return PPOObservation(
        training_progress=progress,
        gpu_fraction=gpu_frac,
        power_fraction=pow_frac,
        throughput_fraction=tput_frac,
        intensity_fraction=int_frac,
        deadline_fraction_remaining=deadline_frac,
        energy_fraction_remaining=energy_frac,
        iters_per_joule_normalized=min(1.0, ipj_norm),
    )


# Export-friendly action space size for SB3 wiring (Tuần 4).
def discrete_action_space_size(min_gpus: int, max_gpus: int) -> int:
    """Number of discrete actions = (max_gpus - min_gpus + 1).

    Used by experiments/exp02_carbon_replay.py to configure SB3's Discrete
    action space before model construction.
    """
    if max_gpus < min_gpus:
        raise ValueError("max_gpus < min_gpus")
    return max_gpus - min_gpus + 1


# Used in Tuần 4 training; kept as a top-level constant so tests can verify
# the spec without instantiating SB3.
OBSERVATION_DIM: int = 8
LOG2_NORMALIZER: float = math.log2(8)   # placeholder for future log-scale obs

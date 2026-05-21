"""Energy-aware scheduling policies — feed ``EnergyDecision`` to the orchestrator each tick.

Two implementations, both consume *energy telemetry* (power.draw, throughput) as primary
signal and optionally *grid intensity proxy* as a secondary weighting:

* ``RuleBasedPolicy`` — fast, deterministic, easy to debug. Threshold-based on a scalar
  "scaling pressure" that combines current power draw and (optionally) grid intensity.
  Currently keyed on intensity for backward-compat with the seed-idea narrative; production
  code should swap in power-per-throughput once profiling is in place.
* ``MPCPolicy`` — Model-Predictive Control over a short horizon; picks the GPU count that
  minimises ``α · energy + γ · carbon_proxy + β · deadline_lag``. When ``carbon_weight = 0``
  the policy is purely energy-aware.

A future ``RLPolicy`` (PPO via Stable-Baselines3) is sketched in ``rl_policy.py``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyDecision:
    target_gpus: int
    pause: bool = False
    reason: str = ""


@dataclass
class RuleBasedPolicy:
    min_gpus: int
    max_gpus: int
    scale_down_above_g_per_kwh: float = 600.0
    scale_up_below_g_per_kwh: float = 250.0
    pause_above_g_per_kwh: float = 800.0

    def decide(self, current_gpus: int, intensity_now: float) -> EnergyDecision:
        if intensity_now >= self.pause_above_g_per_kwh:
            return EnergyDecision(self.min_gpus, pause=True, reason=f"intensity {intensity_now:.0f} ≥ pause threshold")
        if intensity_now >= self.scale_down_above_g_per_kwh:
            target = max(self.min_gpus, current_gpus // 2)
            return EnergyDecision(target, reason=f"intensity {intensity_now:.0f} ≥ scale-down threshold")
        if intensity_now <= self.scale_up_below_g_per_kwh:
            target = min(self.max_gpus, max(current_gpus * 2, self.min_gpus + 1))
            return EnergyDecision(target, reason=f"intensity {intensity_now:.0f} ≤ scale-up threshold")
        return EnergyDecision(current_gpus, reason="steady")


@dataclass
class MPCPolicy:
    """Receding-horizon planner: try GPU counts ``[min, max]``, pick best multi-step value."""

    min_gpus: int
    max_gpus: int
    horizon_steps: int
    step_seconds: float
    power_per_gpu_w: float
    throughput_per_gpu: Callable[[int], float]   # iter/s as fn of gpus
    iterations_remaining: int
    deadline_seconds_remaining: float
    carbon_weight: float = 1.0       # α
    lag_weight: float = 0.01         # β — small so we avoid trivial "always max"

    def decide(self, current_gpus: int, intensity_forecast: list[tuple[float, float]]) -> EnergyDecision:
        """``intensity_forecast`` is a list of ``(t_offset, gCO2/kWh)`` points covering the
        horizon. We greedily evaluate constant-gpu plans across the horizon — a richer MPC
        would optimise over time-varying plans, but constant per-tick keeps the action space
        tractable for the testbed.
        """
        best = current_gpus
        best_cost = float("inf")
        # Stepwise integration of (energy × intensity) plus lag penalty.
        for gpus in range(self.min_gpus, self.max_gpus + 1):
            throughput = self.throughput_per_gpu(gpus)
            iters_done = throughput * self.horizon_steps * self.step_seconds
            iters_left = max(0.0, self.iterations_remaining - iters_done)
            lag = max(0.0, iters_left / max(throughput, 1e-9) - self.deadline_seconds_remaining)

            # Integrate emissions.
            emissions = 0.0
            for _t_off, intensity in intensity_forecast[: self.horizon_steps]:
                kwh = (self.power_per_gpu_w * gpus * self.step_seconds) / 3_600_000.0
                emissions += kwh * intensity
            cost = self.carbon_weight * emissions + self.lag_weight * lag
            if cost < best_cost:
                best_cost = cost
                best = gpus
        reason = f"MPC choose {best} (cost={best_cost:.3f}, intensity_now={intensity_forecast[0][1]:.0f})"
        return EnergyDecision(best, reason=reason)

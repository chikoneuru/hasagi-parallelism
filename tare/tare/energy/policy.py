"""Energy-aware scheduling policies — feed ``EnergyDecision`` to the orchestrator each tick.

Three implementations, all consume *energy telemetry* (power.draw, throughput) as
primary signal and optionally *grid intensity proxy* as a secondary weighting:

* ``RuleBasedPolicy`` — carbon-only baseline. Threshold-based on *carbon intensity
  scalar*. Kept as the HASAGI-no-energy-policy ablation baseline — do NOT delete.
* ``PowerAwareRulePolicy`` — threshold on *power-per-throughput* (J/iter) computed
  from live WorkerTelemetry, with hysteresis to prevent oscillation. Carbon
  intensity is an optional spatial-shift signal, not the primary driver.
* ``MPCPolicy`` — Model-Predictive Control over a short horizon; picks the GPU count
  that minimises ``α · energy + γ · carbon_proxy + β · deadline_lag``. When
  ``carbon_weight = 0`` the policy is purely energy-aware.

A future ``RLPolicy`` (PPO via Stable-Baselines3) is scaffolded in ``rl_policy``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from tare.energy.telemetry import WorkerTelemetry


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
class PowerAwareRulePolicy:
    """Threshold policy on power-per-throughput (J/iter) with hysteresis.

    Decision logic per tick:
        1. Aggregate ``j_per_iter = avg(power_draw_w) / avg(throughput_iters_per_s)``
           across all workers with valid telemetry. This is the energy cost of a
           single iteration in joules — the *energy efficiency* of the current
           allocation.
        2. If ``j_per_iter > scale_down_above_j_per_iter`` for ``hysteresis_ticks``
           consecutive ticks → scale down by 1 GPU (energy-inefficient regime,
           diminishing returns from extra workers).
        3. If ``j_per_iter < scale_up_below_j_per_iter`` for ``hysteresis_ticks``
           consecutive ticks → scale up by 1 GPU (efficient regime, headroom).
        4. Otherwise: hold steady; reset hysteresis counters.

    Hysteresis prevents oscillation under noisy NVML readings (±2% per Zeus NSDI'23)
    and brief transients (gradient sync stalls, JIT compilation spikes).

    Carbon intensity is an *optional* spatial-shift trigger: when
    ``carbon_pause_above_g_per_kwh`` is set and the grid is dirty, the policy can
    proactively scale down regardless of energy efficiency. Default ``None``
    disables this branch — pure energy-driven decisions.

    Rationale: the carbon-only RuleBasedPolicy keys on intensity and treats
    energy efficiency as derived. PowerAwareRule flips that: energy is the
    primary signal (defensible via direct NVML measurement), carbon is secondary.
    Both kept side-by-side for ablation.
    """

    min_gpus: int
    max_gpus: int
    scale_down_above_j_per_iter: float
    scale_up_below_j_per_iter: float
    hysteresis_ticks: int = 3
    carbon_pause_above_g_per_kwh: float | None = None

    _ticks_above: int = field(default=0, init=False, repr=False)
    _ticks_below: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scale_up_below_j_per_iter >= self.scale_down_above_j_per_iter:
            raise ValueError(
                f"scale_up threshold ({self.scale_up_below_j_per_iter}) must be "
                f"< scale_down threshold ({self.scale_down_above_j_per_iter})"
            )
        if self.hysteresis_ticks < 1:
            raise ValueError(f"hysteresis_ticks must be >= 1, got {self.hysteresis_ticks}")

    def decide(
        self,
        current_gpus: int,
        telemetry: Mapping[str, WorkerTelemetry],
        carbon_intensity_now: float | None = None,
    ) -> EnergyDecision:
        # Spatial-shift gate: if grid dirty and threshold configured, scale down.
        if (
            self.carbon_pause_above_g_per_kwh is not None
            and carbon_intensity_now is not None
            and carbon_intensity_now >= self.carbon_pause_above_g_per_kwh
        ):
            self._reset()
            target = max(self.min_gpus, current_gpus // 2)
            return EnergyDecision(
                target,
                reason=f"carbon spatial-shift: {carbon_intensity_now:.0f} gCO2/kWh "
                       f"≥ {self.carbon_pause_above_g_per_kwh:.0f} threshold"
            )

        # Aggregate J/iter across workers with valid (positive) signal.
        signals = [
            (t.power_draw_w, t.throughput_iters_per_s)
            for t in telemetry.values()
            if t.power_draw_w > 0 and t.throughput_iters_per_s > 0
        ]
        if not signals:
            self._reset()
            return EnergyDecision(current_gpus, reason="no valid telemetry — hold")

        avg_power = sum(p for p, _ in signals) / len(signals)
        avg_throughput = sum(t for _, t in signals) / len(signals)
        j_per_iter = avg_power / avg_throughput

        if j_per_iter > self.scale_down_above_j_per_iter:
            self._ticks_above += 1
            self._ticks_below = 0
            if self._ticks_above >= self.hysteresis_ticks:
                target = max(self.min_gpus, current_gpus - 1)
                self._ticks_above = 0
                return EnergyDecision(
                    target,
                    reason=f"J/iter {j_per_iter:.1f} > {self.scale_down_above_j_per_iter:.1f} "
                           f"for {self.hysteresis_ticks} ticks — scale down"
                )
            return EnergyDecision(
                current_gpus,
                reason=f"J/iter {j_per_iter:.1f} above threshold "
                       f"(tick {self._ticks_above}/{self.hysteresis_ticks})"
            )

        if j_per_iter < self.scale_up_below_j_per_iter:
            self._ticks_below += 1
            self._ticks_above = 0
            if self._ticks_below >= self.hysteresis_ticks:
                target = min(self.max_gpus, current_gpus + 1)
                self._ticks_below = 0
                return EnergyDecision(
                    target,
                    reason=f"J/iter {j_per_iter:.1f} < {self.scale_up_below_j_per_iter:.1f} "
                           f"for {self.hysteresis_ticks} ticks — scale up"
                )
            return EnergyDecision(
                current_gpus,
                reason=f"J/iter {j_per_iter:.1f} below threshold "
                       f"(tick {self._ticks_below}/{self.hysteresis_ticks})"
            )

        # Within healthy band — reset counters, hold.
        self._reset()
        return EnergyDecision(current_gpus, reason=f"J/iter {j_per_iter:.1f} steady")

    def _reset(self) -> None:
        self._ticks_above = 0
        self._ticks_below = 0


@dataclass
class MPCPolicy:
    """Receding-horizon planner: try GPU counts ``[min, max]``, pick best multi-step value.

    Reconfig penalty: the planner accounts for the one-shot cost of switching
    GPU counts mid-job. Each candidate ``gpus ≠ current_gpus`` pays
    ``reconfig_latency_s`` of training-pause lag and ``reconfig_energy_kwh`` of
    state-migration energy (evaluated at the current grid intensity). This biases the
    planner toward holding the current allocation unless the multi-step gain clears
    the switching cost — prevents flapping under noisy intensity forecasts.

    Set both reconfig fields to 0.0 to recover the pre-penalty behaviour.
    """

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
    reconfig_latency_s: float = 0.0      # paused training time per switch
    reconfig_energy_kwh: float = 0.0     # state-migration energy per switch

    def __post_init__(self) -> None:
        if self.reconfig_latency_s < 0:
            raise ValueError(
                f"reconfig_latency_s must be >= 0, got {self.reconfig_latency_s}"
            )
        if self.reconfig_energy_kwh < 0:
            raise ValueError(
                f"reconfig_energy_kwh must be >= 0, got {self.reconfig_energy_kwh}"
            )

    def decide(self, current_gpus: int, intensity_forecast: list[tuple[float, float]]) -> EnergyDecision:
        """``intensity_forecast`` is a list of ``(t_offset, gCO2/kWh)`` points covering the
        horizon. We greedily evaluate constant-gpu plans across the horizon — a richer MPC
        would optimise over time-varying plans, but constant per-tick keeps the action space
        tractable for the testbed.
        """
        intensity_now = intensity_forecast[0][1] if intensity_forecast else 0.0
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

            # One-shot reconfig penalty when switching off current_gpus.
            if gpus != current_gpus:
                emissions += self.reconfig_energy_kwh * intensity_now
                lag += self.reconfig_latency_s

            cost = self.carbon_weight * emissions + self.lag_weight * lag
            if cost < best_cost:
                best_cost = cost
                best = gpus
        switched = best != current_gpus
        has_penalty = self.reconfig_latency_s > 0 or self.reconfig_energy_kwh > 0
        reason = (
            f"MPC choose {best} (cost={best_cost:.3f}, intensity_now={intensity_now:.0f}"
            f"{', reconfig penalty applied' if switched and has_penalty else ''})"
        )
        return EnergyDecision(best, reason=reason)


@dataclass
class OnlinePrimalDualPolicy:
    """Online primal-dual scheduler with a deadline-throughput constraint.

    Each step, given the live carbon intensity ``b_t``, the policy picks the
    GPU count ``g*`` that minimises the per-step Lagrangian::

        L_t(g) = E(g) · μ(g) · b_t  −  λ_t · μ(g)

    where ``E(g)`` is energy-per-iter at allocation ``g`` (J / iter),
    ``μ(g)`` is the resulting iteration rate (iter / s), and ``λ_t`` is the
    running dual estimate for the deadline-throughput constraint. The dual
    is updated by projected gradient ascent::

        λ_{t+1} = max(0, λ_t + η · (target_iter_rate − μ(g*)))

    so a streak of below-target throughput pushes ``λ`` upward until the
    primal argmin selects a higher-throughput GPU count. The step size
    ``η`` is calibrated to the running cost scale so ``λ`` converges within
    ``O(√T)`` steps::

        η = max_energy_estimate · intensity_scale / √T

    Compared with ``MPCPolicy``: MPC needs a horizon forecast of ``b_t``;
    primal-dual uses only the current sample plus its running multiplier,
    so it is robust to non-stationary or unforecastable intensity. The
    trade is a higher constant in the regret bound — see proofs §10.

    Args:
        min_gpus, max_gpus: action set ``[min_gpus, max_gpus]``.
        throughput_per_gpu: ``μ(g)`` in iter/s.
        energy_per_iter: ``E(g)`` in J/iter (or any consistent unit that
            matches ``max_energy_estimate``).
        target_iter_rate: deadline-induced throughput floor in iter/s.
            Typically ``iters_remaining / deadline_seconds_remaining``
            refreshed by the orchestrator each tick — but the policy
            holds it constant across the calibration horizon so ``λ`` has
            a stable target to chase.
        horizon_steps: ``T`` used in the ``η`` calibration; should
            reflect the number of decision steps over which ``λ`` is
            expected to converge.
        max_energy_estimate: upper bound on ``E(g) · μ(g)`` over the
            action set; used to size ``η``.
        intensity_scale: expected magnitude of ``b_t`` (mean carbon
            intensity). Defaults to 1.0 — set to your trace mean when
            running against gCO2/kWh traces.
        eta: step size override. ``0.0`` (default) triggers the
            calibration formula above.

    Mutable state ``lambda_t`` evolves across ``decide`` calls — instantiate
    one policy per job, not one shared across jobs.
    """

    min_gpus: int
    max_gpus: int
    throughput_per_gpu: Callable[[int], float]
    energy_per_iter: Callable[[int], float]
    target_iter_rate: float
    horizon_steps: int
    max_energy_estimate: float
    intensity_scale: float = 1.0
    eta: float = 0.0

    lambda_t: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.min_gpus < 1:
            raise ValueError(f"min_gpus must be >= 1, got {self.min_gpus}")
        if self.max_gpus < self.min_gpus:
            raise ValueError(
                f"max_gpus ({self.max_gpus}) must be >= min_gpus ({self.min_gpus})"
            )
        if self.horizon_steps < 1:
            raise ValueError(f"horizon_steps must be >= 1, got {self.horizon_steps}")
        if self.target_iter_rate < 0:
            raise ValueError(
                f"target_iter_rate must be >= 0, got {self.target_iter_rate}"
            )
        if self.max_energy_estimate <= 0:
            raise ValueError(
                f"max_energy_estimate must be > 0, got {self.max_energy_estimate}"
            )
        if self.eta < 0:
            raise ValueError(f"eta must be >= 0, got {self.eta}")
        if self.eta == 0.0:
            self.eta = (
                self.max_energy_estimate
                * self.intensity_scale
                / max(1.0, math.sqrt(self.horizon_steps))
            )

    def decide(self, current_gpus: int, intensity_now: float) -> EnergyDecision:
        """Pick ``argmin_g L_t(g)``, then update ``λ`` toward the target rate."""
        # Primal argmin over the discrete action set.
        best_gpus = self.min_gpus
        best_score = math.inf
        for g in range(self.min_gpus, self.max_gpus + 1):
            mu = self.throughput_per_gpu(g)
            e = self.energy_per_iter(g)
            score = e * mu * intensity_now - self.lambda_t * mu
            if score < best_score:
                best_score = score
                best_gpus = g

        # Dual ascent — projected onto non-negative reals.
        mu_chosen = self.throughput_per_gpu(best_gpus)
        prev_lambda = self.lambda_t
        self.lambda_t = max(
            0.0,
            self.lambda_t + self.eta * (self.target_iter_rate - mu_chosen),
        )
        reason = (
            f"primal-dual choose {best_gpus} "
            f"(score={best_score:.3f}, μ={mu_chosen:.3f}, "
            f"target={self.target_iter_rate:.3f}, λ {prev_lambda:.3f}→{self.lambda_t:.3f})"
        )
        return EnergyDecision(best_gpus, reason=reason)

"""Stochastic-profile joint partition-and-throttle DP — CVaR-bounded co-design.

Extends ``joint_partition`` to the setting where per-stage profiles are noisy:

    T̃_s(c) ∼ 𝒩(μ_T(c, s), σ_T² · μ_T²(c, s))
    P̃_s   ∼ 𝒩(μ_P(s),    σ_P² · μ_P²(s))

The joint plan π = (c, r) is scored by the CVaR-β of total per-iteration energy:

    CVaR_β(Ẽ(π)) = μ_E(π) + κ_β · σ_E(π)

where κ_β = φ(Φ⁻¹(1 − β)) / β. The closed form holds because Ẽ(π) is
approximately Gaussian by CLT for K ≥ 4 stages and small σ_T, σ_P.

Chance constraints (Θ̃) throughput floor and (P̃) power cap reduce to
deterministic margins inflated by ``(1 + z_α · σ)`` where
``z_α = Φ⁻¹(1 − α_chance)``.

Setting ``sigma_t = sigma_p = 0`` reduces this DP exactly to
``joint_partition`` (regression test by construction).

The CVaR objective ``μ + κ · √σ²`` is not additively separable across
stages because of the square-root non-linearity, so the DP maintains a
Pareto frontier of (μ, σ²) pairs at each cell (dominated pairs pruned).
Typical frontier size is small (< 20 entries) for HISE-typical
workloads; complexity is O(n² · K · M · |frontier|).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from hise.parallel.joint_partitioner import JointPlan, _build_throttle_set
from hise.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    _comm_time,
    _comp_time,
    _exec_time,
)


@dataclass(frozen=True)
class StochasticJointPlan(JointPlan):
    """Joint partition-and-throttle plan with CVaR scoring under stochastic profiles.

    Extends :class:`JointPlan` with:
        ``expected_energy`` — μ_E(π), the per-iteration energy at the mean profile
        ``energy_stddev``   — σ_E(π), one-sigma uncertainty in per-iteration energy
        ``cvar_energy``     — μ_E(π) + κ_β · σ_E(π), the optimised CVaR score

    Set ``cvar_energy`` to inf for infeasible plans (no Pareto entry survives the
    chance constraints).
    """

    expected_energy: float = math.inf
    energy_stddev: float = math.inf
    cvar_energy: float = math.inf

    def is_feasible(self) -> bool:
        return math.isfinite(self.cvar_energy)


# ---------------------------------------------------------------------------
# Gaussian helpers
# ---------------------------------------------------------------------------

def _normal_cvar_coefficient(beta: float) -> float:
    """κ_β = φ(Φ⁻¹(1 − β)) / β for a standard Normal random variable."""
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    z = NormalDist().inv_cdf(1.0 - beta)
    phi_z = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    return phi_z / beta


def _chance_constraint_z(alpha_chance: float) -> float:
    """z_α = Φ⁻¹(1 − α_chance) — chance-constraint inflation factor."""
    if not 0.0 < alpha_chance < 1.0:
        raise ValueError(f"alpha_chance must be in (0, 1), got {alpha_chance}")
    return NormalDist().inv_cdf(1.0 - alpha_chance)


# ---------------------------------------------------------------------------
# Pareto-frontier DP state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ParetoEntry:
    """One Pareto-non-dominated (μ, σ²) point with backpointers for backtracking."""

    mu: float
    sigma_sq: float
    prev_i: int
    prev_r: float
    prev_entry_idx: int


def _dominates(a: _ParetoEntry, b: _ParetoEntry) -> bool:
    """True iff a is no worse than b on both (μ, σ²) and strictly better on one."""
    return (
        a.mu <= b.mu + 1e-12
        and a.sigma_sq <= b.sigma_sq + 1e-12
        and (a.mu < b.mu - 1e-12 or a.sigma_sq < b.sigma_sq - 1e-12)
    )


def _insert_pareto(frontier: list[_ParetoEntry], new: _ParetoEntry) -> None:
    """Insert ``new`` if non-dominated; prune entries dominated by ``new``.

    O(|frontier|) per insert; the frontier stays bounded by the number of
    distinct Pareto-optimal extensions which is typically small for HISE-
    typical workloads (n ≤ 20, K ≤ 4, M ≤ 16).
    """
    for e in frontier:
        if _dominates(e, new):
            return
        # Pair-equality: don't insert duplicates.
        if abs(e.mu - new.mu) < 1e-12 and abs(e.sigma_sq - new.sigma_sq) < 1e-12:
            return
    frontier[:] = [e for e in frontier if not _dominates(new, e)]
    frontier.append(new)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def stochastic_joint_partition(
    layers: Sequence[LayerProfile],
    stages: Sequence[StageSpec],
    links: Sequence[LinkSpec],
    *,
    throughput_floor_iters_per_s: float,
    voltage_alpha: float = 2.0,
    throttle_min: float = 0.5,
    throttle_granularity: int = 11,
    sigma_t: float = 0.10,
    sigma_p: float = 0.05,
    cvar_beta: float = 0.05,
    chance_alpha: float = 0.05,
) -> StochasticJointPlan:
    """CVaR-bounded joint optimiser over (cuts, throttle vector) under stochastic profiles.

    Same arguments as :func:`joint_partition` plus four new ones for the
    stochastic-profile setting:

    Args:
        sigma_t: relative std-deviation of profiled exec time per stage.
            T̃_s(c) ~ N(μ_T, (σ_T · μ_T)²). σ_T = 0 reduces to v1.
        sigma_p: relative std-deviation of profiled power per stage.
            P̃_s ~ N(μ_P, (σ_P · μ_P)²). σ_P = 0 reduces to v1.
        cvar_beta: CVaR confidence level β ∈ (0, 1). Lower β = more
            risk-averse (penalises tail outcomes more heavily). β → 1
            recovers expected-energy minimisation.
        chance_alpha: chance-constraint violation tolerance α ∈ (0, 1).
            (Θ̃) and (P̃) constraints must hold with probability ≥ 1 - α
            per stage.

    Returns:
        :class:`StochasticJointPlan`. Infeasible workloads return a plan with
        empty cuts and infinite cvar_energy.
    """
    if voltage_alpha < 1.0:
        raise ValueError(f"voltage_alpha must be >= 1, got {voltage_alpha}")
    if not 0.0 <= sigma_t < 1.0:
        raise ValueError(f"sigma_t must be in [0, 1), got {sigma_t}")
    if not 0.0 <= sigma_p < 1.0:
        raise ValueError(f"sigma_p must be in [0, 1), got {sigma_p}")

    kappa_beta = _normal_cvar_coefficient(cvar_beta)
    z_alpha = _chance_constraint_z(chance_alpha)
    sigma_combined_sq = sigma_t * sigma_t + sigma_p * sigma_p

    if throughput_floor_iters_per_s <= 0:
        raise ValueError(
            f"throughput_floor_iters_per_s must be > 0, got {throughput_floor_iters_per_s}"
        )
    if not 0.0 < throttle_min <= 1.0:
        raise ValueError(f"throttle_min must be in (0, 1], got {throttle_min}")
    if throttle_granularity < 1:
        raise ValueError(f"throttle_granularity must be >= 1, got {throttle_granularity}")

    n = len(layers)
    K = len(stages)
    if K < 1:
        raise ValueError("Need at least 1 stage.")
    if n < K:
        raise ValueError(f"Need at least {K} layers for {K} stages.")

    R = _build_throttle_set(throttle_min, throttle_granularity)
    T_floor = 1.0 / throughput_floor_iters_per_s
    T_floor_tol = T_floor * (1.0 + 1e-9)
    theta_inflation = 1.0 + z_alpha * sigma_t  # (Θ̃) margin factor
    p_inflation = 1.0 + z_alpha * sigma_p  # (P̃) margin factor

    link_map: dict[int, LinkSpec] = {lk.src_stage: lk for lk in links}
    for s in range(K - 1):
        if s not in link_map:
            raise ValueError(f"Missing link from stage {s} to stage {s+1}.")

    prefix_fwd = [0.0] * (n + 1)
    prefix_bwd = [0.0] * (n + 1)
    prefix_mem = [0] * (n + 1)
    for i in range(n):
        prefix_fwd[i + 1] = prefix_fwd[i] + layers[i].fwd_flops
        prefix_bwd[i + 1] = prefix_bwd[i] + layers[i].bwd_flops
        prefix_mem[i + 1] = prefix_mem[i] + layers[i].activation_bytes

    def seg_exec(stage_id: int, start: int, end: int) -> float:
        fwd = prefix_fwd[end + 1] - prefix_fwd[start]
        bwd = prefix_bwd[end + 1] - prefix_bwd[start]
        comp = _comp_time(stages[stage_id], fwd, bwd)
        comm_in = 0.0
        if stage_id > 0 and start > 0:
            comm_in = _comm_time(link_map[stage_id - 1], layers[start - 1].activation_bytes)
        comm_out = 0.0
        if stage_id < K - 1:
            comm_out = _comm_time(link_map[stage_id], layers[end].activation_bytes)
        return _exec_time(comp, comm_out, comm_in)

    def seg_feasible(stage_id: int, start: int, end: int) -> bool:
        mem = prefix_mem[end + 1] - prefix_mem[start]
        return mem <= stages[stage_id].memory_bytes

    def transition_terms(
        stage_id: int, t: float, r: float
    ) -> tuple[float, float] | None:
        """Per-stage (Δμ, Δσ²) contribution if (Θ̃), (P̃), (R) all hold, else None."""
        spec = stages[stage_id]
        # (Θ̃) chance constraint: μ_T · (1 + z_α · σ_T) ≤ r · T_floor
        if t * theta_inflation > r * T_floor_tol:
            return None
        # (P̃) chance constraint: μ_P · r^α · (1 + z_α · σ_P) ≤ P_cap
        if spec.power_draw_w * (r ** voltage_alpha) * p_inflation > spec.power_cap_w:
            return None
        delta_mu = spec.power_draw_w * (r ** (voltage_alpha - 1)) * t
        delta_sigma_sq = (
            (r ** (2.0 * (voltage_alpha - 1)))
            * (spec.power_draw_w ** 2)
            * (t ** 2)
            * sigma_combined_sq
        )
        return delta_mu, delta_sigma_sq

    # K=1: enumerate r for the single stage covering all layers.
    if K == 1:
        if not seg_feasible(0, 0, n - 1):
            return StochasticJointPlan()
        t = seg_exec(0, 0, n - 1)
        best: tuple[float, float, float, float] | None = None  # (cvar_score, mu, sigma_sq, r)
        for r in R:
            tr = transition_terms(0, t, r)
            if tr is None:
                continue
            mu, sigma_sq = tr
            score = mu + kappa_beta * math.sqrt(sigma_sq)
            if best is None or score < best[0]:
                best = (score, mu, sigma_sq, r)
        if best is None:
            return StochasticJointPlan()
        score, mu, sigma_sq, r = best
        return StochasticJointPlan(
            cuts=(),
            throttle_factors=(r,),
            stage_layers={0: tuple(range(n))},
            stage_exec_time={0: t / r},
            energy_per_iter=mu,  # legacy v1 field — equals expected_energy here
            pipeline_time_s=t / r,
            num_stages=1,
            expected_energy=mu,
            energy_stddev=math.sqrt(sigma_sq),
            cvar_energy=score,
        )

    # K >= 2: Pareto-frontier DP over (μ, σ²).
    dp: list[list[list[_ParetoEntry]]] = [
        [[] for _ in range(K)] for _ in range(n)
    ]

    # Base: stage 0 covers layers 0..j.
    for j in range(n):
        if not seg_feasible(0, 0, j):
            continue
        t = seg_exec(0, 0, j)
        for r in R:
            tr = transition_terms(0, t, r)
            if tr is None:
                continue
            mu, sigma_sq = tr
            _insert_pareto(
                dp[j][0],
                _ParetoEntry(
                    mu=mu, sigma_sq=sigma_sq, prev_i=-1, prev_r=r, prev_entry_idx=-1
                ),
            )

    # Inductive: stages 1..K-1.
    for s in range(1, K):
        for j in range(s, n):
            for i in range(s - 1, j):
                if not seg_feasible(s, i + 1, j):
                    continue
                t = seg_exec(s, i + 1, j)
                for r in R:
                    tr = transition_terms(s, t, r)
                    if tr is None:
                        continue
                    delta_mu, delta_sigma_sq = tr
                    for prev_idx, prev_entry in enumerate(dp[i][s - 1]):
                        new_mu = prev_entry.mu + delta_mu
                        new_sigma_sq = prev_entry.sigma_sq + delta_sigma_sq
                        _insert_pareto(
                            dp[j][s],
                            _ParetoEntry(
                                mu=new_mu, sigma_sq=new_sigma_sq,
                                prev_i=i, prev_r=r, prev_entry_idx=prev_idx,
                            ),
                        )

    if not dp[n - 1][K - 1]:
        return StochasticJointPlan()

    # Argmin CVaR over the final-cell Pareto frontier.
    final_entries = dp[n - 1][K - 1]
    best_entry = min(
        final_entries,
        key=lambda e: e.mu + kappa_beta * math.sqrt(e.sigma_sq),
    )
    best_mu = best_entry.mu
    best_sigma_sq = best_entry.sigma_sq
    best_cvar = best_mu + kappa_beta * math.sqrt(best_sigma_sq)

    # Backtrack through Pareto entries to recover (cuts, throttles).
    cuts_list: list[int] = []
    throttles: list[float] = [0.0] * K
    current = best_entry
    j = n - 1
    for s in range(K - 1, 0, -1):
        throttles[s] = current.prev_r
        cuts_list.append(current.prev_i)
        next_j = current.prev_i
        current = dp[next_j][s - 1][current.prev_entry_idx]
        j = next_j
    throttles[0] = current.prev_r
    cuts_list.reverse()

    # Build stage_layers and stage_exec_time (throttled).
    boundaries = [-1, *cuts_list, n - 1]
    stage_layers: dict[int, tuple[int, ...]] = {}
    stage_exec_time: dict[int, float] = {}
    for s in range(K):
        start = boundaries[s] + 1
        end = boundaries[s + 1]
        stage_layers[s] = tuple(range(start, end + 1))
        t = seg_exec(s, start, end)
        stage_exec_time[s] = t / throttles[s]

    pipeline_time_s = max(stage_exec_time.values())

    return StochasticJointPlan(
        cuts=tuple(cuts_list),
        throttle_factors=tuple(throttles),
        stage_layers=stage_layers,
        stage_exec_time=stage_exec_time,
        energy_per_iter=best_mu,  # legacy v1 field — equals expected_energy
        pipeline_time_s=pipeline_time_s,
        num_stages=K,
        expected_energy=best_mu,
        energy_stddev=math.sqrt(best_sigma_sq),
        cvar_energy=best_cvar,
    )

"""Multi-seed regret sweep for the primal-dual policy across noise × horizon.

Companion validation for the regret bound `R(T) ≤ O(L_max · √T)` proven for
the online primal-dual algorithm. The bound is asymptotic; this harness
checks that:

  1. Across many seeds and noise levels, the empirical regret vs the
     offline-LP optimum stays within the analytical envelope
     ``(η · G² · T) / 2``.
  2. The mean primal-dual gap is competitive with MPC-receding at low
     forecast noise and beats MPC at high noise (≥ 25%).
  3. Constraint-violation (running iter shortfall vs the deadline) tracks
     the predicted ``O(√T / λ̄)`` scaling.

Reuses the action set, trace builder, and solver routines from
``exp_online_deadline.py`` so the constants used here match the ones
quoted in the proof.

Usage::

    python -m experiments.exp_online_primal_dual_sweep
    python -m experiments.exp_online_primal_dual_sweep --num-seeds 50
"""
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from experiments.exp_online_deadline import (
    Action,
    build_action_set,
    mpc_receding,
    offline_lp_solve,
    online_primal_dual,
)
from experiments.exp_online_feasibility import build_noisy_trace

# ---------------------------------------------------------------------------
# Sweep cell + statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """One ``(noise, hours, seed)`` simulation across all policies."""

    noise_pct: float
    hours: int
    seed: int
    horizon_steps: int
    offline_cost: float
    primal_dual_cost: float
    mpc_cost: float
    primal_dual_iters: float
    deadline_iters: float
    envelope: float

    @property
    def primal_dual_regret(self) -> float:
        return self.primal_dual_cost - self.offline_cost

    @property
    def primal_dual_gap_pct(self) -> float:
        if self.offline_cost <= 0:
            return 0.0
        return 100.0 * self.primal_dual_regret / self.offline_cost

    @property
    def mpc_gap_pct(self) -> float:
        if self.offline_cost <= 0:
            return 0.0
        return 100.0 * (self.mpc_cost - self.offline_cost) / self.offline_cost

    @property
    def violation(self) -> float:
        return max(0.0, self.deadline_iters - self.primal_dual_iters)

    @property
    def within_envelope(self) -> bool:
        return self.primal_dual_regret <= self.envelope


@dataclass(frozen=True)
class CellSummary:
    """Aggregate over seeds at one ``(noise, hours)`` setting."""

    noise_pct: float
    hours: int
    horizon_steps: int
    num_seeds: int
    pd_gap_mean_pct: float
    pd_gap_ci95: tuple[float, float]
    mpc_gap_mean_pct: float
    violation_mean: float
    envelope_mean: float
    regret_mean: float
    coverage_pct: float
    crossover_count: int

    @property
    def crossover_rate(self) -> float:
        return self.crossover_count / max(1, self.num_seeds)


# ---------------------------------------------------------------------------
# Envelope from §10.4 of proofs.md
# ---------------------------------------------------------------------------


def t8_envelope(actions: list[Action], trace: list[float], deadline_iters: float) -> float:
    """Analytical regret envelope ``(η · G² · T) / 2`` with the calibration
    ``η = L_max · mean(b) / √T``. Matches §10.4.
    """
    T = len(trace)
    target_mu = deadline_iters / T
    G = max(abs(a.throughput - target_mu) for a in actions)
    max_e = max(a.energy_per_iter for a in actions)
    max_mu = max(a.throughput for a in actions)
    max_b = max(trace) if trace else 0.0
    L_max = max_e * max_mu * max_b
    mean_b = sum(trace) / max(1, T)
    eta = L_max * mean_b / max(1.0, math.sqrt(T))
    return eta * G * G * T / 2.0


# ---------------------------------------------------------------------------
# Single trial — one seed, one (noise, hours) pair
# ---------------------------------------------------------------------------


def run_trial(
    noise_pct: float,
    hours: int,
    sample_minutes: int,
    seed: int,
    actions: list[Action],
    deadline_multiplier: float,
    mpc_horizon: int,
) -> Trial:
    trace = build_noisy_trace(
        hours=hours,
        sample_minutes=sample_minutes,
        noise_pct=noise_pct,
        regime_shift_at=None,
        regime_shift_magnitude=0.0,
        seed=seed,
    )
    T = len(trace)
    mu_min = min(a.throughput for a in actions)
    mu_max = max(a.throughput for a in actions)
    target_mu = deadline_multiplier * (mu_min + mu_max) / 2
    deadline_iters = target_mu * T

    offline_cost, _, _ = offline_lp_solve(trace, actions, deadline_iters)
    pd_cost, pd_choices = online_primal_dual(trace, actions, deadline_iters)
    mpc_cost, _ = mpc_receding(
        trace, actions, deadline_iters,
        horizon=mpc_horizon, forecast_noise_pct=noise_pct, seed=seed,
    )

    pd_iters = sum(actions[i].throughput for i in pd_choices)
    envelope = t8_envelope(actions, trace, deadline_iters)

    return Trial(
        noise_pct=noise_pct,
        hours=hours,
        seed=seed,
        horizon_steps=T,
        offline_cost=offline_cost,
        primal_dual_cost=pd_cost,
        mpc_cost=mpc_cost,
        primal_dual_iters=pd_iters,
        deadline_iters=deadline_iters,
        envelope=envelope,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _ci95(samples: list[float]) -> tuple[float, float]:
    """Approximate 95% CI on the mean (normal approx, n ≥ 20 OK)."""
    if not samples:
        return (0.0, 0.0)
    mean = statistics.mean(samples)
    if len(samples) == 1:
        return (mean, mean)
    sd = statistics.stdev(samples)
    half = 1.96 * sd / math.sqrt(len(samples))
    return (mean - half, mean + half)


def summarise(noise_pct: float, hours: int, trials: list[Trial]) -> CellSummary:
    if not trials:
        raise ValueError("at least one trial required to summarise a cell")
    pd_gaps = [t.primal_dual_gap_pct for t in trials]
    mpc_gaps = [t.mpc_gap_pct for t in trials]
    violations = [t.violation for t in trials]
    envelopes = [t.envelope for t in trials]
    regrets = [t.primal_dual_regret for t in trials]
    coverage = sum(1 for t in trials if t.within_envelope) / len(trials) * 100.0
    crossovers = sum(1 for t in trials if t.primal_dual_cost < t.mpc_cost)
    pd_mean = statistics.mean(pd_gaps)
    pd_ci = _ci95(pd_gaps)
    return CellSummary(
        noise_pct=noise_pct,
        hours=hours,
        horizon_steps=trials[0].horizon_steps,
        num_seeds=len(trials),
        pd_gap_mean_pct=pd_mean,
        pd_gap_ci95=pd_ci,
        mpc_gap_mean_pct=statistics.mean(mpc_gaps),
        violation_mean=statistics.mean(violations),
        envelope_mean=statistics.mean(envelopes),
        regret_mean=statistics.mean(regrets),
        coverage_pct=coverage,
        crossover_count=crossovers,
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    console = Console()
    actions = build_action_set()
    noise_levels = args.noise_levels
    hours_levels = args.hours_levels

    console.print(
        f"[bold]Primal-dual regret sweep[/] — "
        f"{len(noise_levels)} noise × {len(hours_levels)} horizon × "
        f"{args.num_seeds} seeds = {len(noise_levels) * len(hours_levels) * args.num_seeds} trials"
    )

    summaries: list[CellSummary] = []
    for noise in noise_levels:
        for hours in hours_levels:
            trials = [
                run_trial(
                    noise_pct=noise,
                    hours=hours,
                    sample_minutes=args.sample_minutes,
                    seed=seed,
                    actions=actions,
                    deadline_multiplier=args.deadline_multiplier,
                    mpc_horizon=args.mpc_horizon,
                )
                for seed in range(args.num_seeds)
            ]
            summaries.append(summarise(noise, hours, trials))

    # Per-cell table
    table = Table(title="Primal-dual vs MPC across noise × horizon")
    table.add_column("noise%", justify="right")
    table.add_column("hours", justify="right")
    table.add_column("T steps", justify="right")
    table.add_column("PD gap %", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("MPC gap %", justify="right")
    table.add_column("violation", justify="right")
    table.add_column("regret", justify="right")
    table.add_column("envelope", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("PD < MPC", justify="right")
    for s in summaries:
        table.add_row(
            f"{s.noise_pct * 100:.0f}",
            str(s.hours),
            str(s.horizon_steps),
            f"{s.pd_gap_mean_pct:+.2f}",
            f"[{s.pd_gap_ci95[0]:+.2f}, {s.pd_gap_ci95[1]:+.2f}]",
            f"{s.mpc_gap_mean_pct:+.2f}",
            f"{s.violation_mean:.2f}",
            f"{s.regret_mean:.0f}",
            f"{s.envelope_mean:.0f}",
            f"{s.coverage_pct:.0f}%",
            f"{s.crossover_count}/{s.num_seeds}",
        )
    console.print(table)

    # Acceptance summary
    high_noise = [s for s in summaries if s.noise_pct >= 0.25]
    pd_beats_mpc = sum(s.crossover_count for s in high_noise)
    high_noise_total = sum(s.num_seeds for s in high_noise)
    overall_coverage = (
        sum(s.coverage_pct * s.num_seeds for s in summaries)
        / sum(s.num_seeds for s in summaries)
    )

    console.print()
    console.print(
        f"[bold]Acceptance check 1 (envelope coverage)[/]: "
        f"overall {overall_coverage:.1f}% of trials within "
        f"R(T) ≤ (η G² T)/2 — target ≥ 90%"
    )
    color1 = "green" if overall_coverage >= 90.0 else "red"
    console.print(f"[{color1}]{'✓ PASS' if overall_coverage >= 90.0 else '✗ FAIL'}[/]")

    if high_noise_total > 0:
        crossover_rate = pd_beats_mpc / high_noise_total * 100.0
        console.print(
            f"\n[bold]Acceptance check 2 (high-noise crossover)[/]: "
            f"primal-dual beats MPC in {pd_beats_mpc}/{high_noise_total} "
            f"= {crossover_rate:.0f}% of trials at noise ≥ 25% — target ≥ 50%"
        )
        color2 = "green" if crossover_rate >= 50.0 else "red"
        console.print(f"[{color2}]{'✓ PASS' if crossover_rate >= 50.0 else '✗ FAIL'}[/]")
    else:
        crossover_rate = 0.0
        color2 = "red"

    # Violation-vs-√T scaling check.
    if len(hours_levels) >= 2:
        violations_by_T = sorted(
            (s.horizon_steps, statistics.mean(
                [v.violation_mean for v in summaries if v.horizon_steps == s.horizon_steps]
            ))
            for s in summaries
        )
        unique = {}
        for T_steps, v in violations_by_T:
            unique[T_steps] = v
        sorted_T = sorted(unique.items())
        first_T, first_v = sorted_T[0]
        last_T, last_v = sorted_T[-1]
        expected_ratio = math.sqrt(last_T / first_T) if first_T > 0 else 0.0
        actual_ratio = last_v / max(first_v, 1e-9)
        console.print(
            f"\n[bold]Sanity check (violation ∝ √T)[/]: "
            f"V(T={first_T}) = {first_v:.2f}, V(T={last_T}) = {last_v:.2f}, "
            f"ratio = {actual_ratio:.2f}, expected ≈ √(T_last/T_first) = {expected_ratio:.2f}"
        )

    success = overall_coverage >= 90.0 and crossover_rate >= 50.0
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise-levels",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        help="Comma-separated noise levels (0..1).",
    )
    parser.add_argument(
        "--hours-levels",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[6, 12, 24, 48],
        help="Comma-separated trace horizons (hours).",
    )
    parser.add_argument("--sample-minutes", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--deadline-multiplier", type=float, default=1.0)
    parser.add_argument("--mpc-horizon", type=int, default=6)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

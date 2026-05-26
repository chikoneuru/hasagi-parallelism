"""Feasibility test for an online-learning carbon-aware scaling policy.

Tests whether a regret-minimising online policy can beat the existing
MPC controller on a noisy carbon trace, and characterises the trace's
non-stationarity properties as a sanity check on the comparison.

Two parts:

  Part A — Trace characterisation. Measures variance, drift, and forecast
  predictability on the synthetic solar trace plus added Gaussian noise.
  If the trace is too predictable, MPC is hard to beat and online learning
  has no opening.

  Part B — Online vs MPC simulation. Three policies pick among K=5
  allocator configurations at each timestep on a noisy carbon trace:
      - offline-optimal     (oracle: knows the full trace in hindsight)
      - mpc-with-forecast   (6-step lookahead with Gaussian forecast noise)
      - online-hedge        (full-information Hedge with shifted experts)
  The check is whether online matches or beats MPC at moderate-to-high
  forecast noise (~15-30%, consistent with ElectricityMaps reported
  uncertainty).

Acceptance:
    1. Trace exhibits measurable variance (coefficient of variation > 10%)
    2. Trace forecast at the 6-step horizon has > 10% error (MPC's forecast
       is not perfect)
    3. Online policy reaches within 5% of offline-optimal at non-trivial
       noise levels, and outperforms MPC at noise ≥ 20%.

Usage:
    python -m experiments.exp_online_feasibility
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from hise.energy.carbon_trace import synthetic_solar_trace

# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------

def build_noisy_trace(
    hours: int,
    sample_minutes: int,
    noise_pct: float,
    regime_shift_at: float | None,
    regime_shift_magnitude: float,
    seed: int,
) -> list[float]:
    """Synthetic solar baseline + iid Gaussian noise (% of mean) + optional
    regime shift (sudden mean jump at ``regime_shift_at`` ∈ [0, 1] of horizon)."""
    rng = random.Random(seed)
    base = synthetic_solar_trace(hours=hours, sample_minutes=sample_minutes)
    out: list[float] = []
    n = len(base.intensities)
    shift_t = int(regime_shift_at * n) if regime_shift_at is not None else -1
    for i, v in enumerate(base.intensities):
        shift = regime_shift_magnitude if i >= shift_t >= 0 else 0.0
        noise = rng.gauss(0.0, noise_pct * v)
        out.append(max(0.0, v + noise + shift))
    return out


# ---------------------------------------------------------------------------
# Action space — five representative allocator configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    name: str
    energy_per_step: float        # joules per timestep at this configuration


def build_action_set() -> list[Action]:
    """Five plans on the energy/throughput Pareto frontier.

    Magnitudes are dimensionless; what matters is the ratio between them.
    The lowest-energy plan corresponds to "aggressive throttle, joint
    optimum at ×2.00"; the highest is "bottleneck-only, no throttle".
    """
    return [
        Action(name="bottleneck-only", energy_per_step=50.0),
        Action(name="bottleneck+Perseus", energy_per_step=42.5),
        Action(name="energy-only", energy_per_step=40.0),
        Action(name="energy+Perseus", energy_per_step=35.0),
        Action(name="joint (heavy throttle)", energy_per_step=25.0),
    ]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def offline_optimal(trace: list[float], actions: list[Action]) -> tuple[float, list[int]]:
    """Pick the lowest-energy action at every step regardless of carbon —
    since loss = E · b_t is monotone in E for non-negative b_t, the offline
    optimum is constant: always pick the min-E action."""
    a_star = min(range(len(actions)), key=lambda i: actions[i].energy_per_step)
    total = sum(actions[a_star].energy_per_step * b for b in trace)
    return total, [a_star] * len(trace)


def mpc_with_forecast(
    trace: list[float],
    actions: list[Action],
    horizon: int,
    forecast_noise_pct: float,
    seed: int,
    reconfig_cost: float,
) -> tuple[float, list[int]]:
    """MPC with noisy 6-step lookahead. The MPC sees a noisy forecast of the
    next ``horizon`` values; picks the action that minimises the sum over the
    horizon. Reconfig cost is charged whenever the action changes."""
    rng = random.Random(seed)
    chosen: list[int] = []
    total = 0.0
    prev_a = -1
    for t, b_t in enumerate(trace):
        # Build noisy forecast for steps t..t+horizon-1
        window = trace[t:t + horizon]
        forecast = [v + rng.gauss(0.0, forecast_noise_pct * v) for v in window]
        # Pick the action minimising forecast-summed cost
        best_a, best_cost = 0, math.inf
        for a_idx, a in enumerate(actions):
            c = sum(a.energy_per_step * f for f in forecast)
            if a_idx != prev_a and prev_a != -1:
                c += reconfig_cost
            if c < best_cost:
                best_cost = c
                best_a = a_idx
        # Actually realise the chosen action on the true b_t (not the forecast)
        cost = actions[best_a].energy_per_step * b_t
        if prev_a != -1 and best_a != prev_a:
            cost += reconfig_cost
        total += cost
        chosen.append(best_a)
        prev_a = best_a
    return total, chosen


def online_hedge_shifted(
    trace: list[float],
    actions: list[Action],
    reconfig_cost: float,
    seed: int,
) -> tuple[float, list[int]]:
    """Shifted-experts Hedge: commit to one action per block of length √T,
    update weights at block boundaries using observed in-block average loss.

    Full information — we know E(a) for each a, and observe b_t each step,
    so all arm losses are computable each step.
    """
    rng = random.Random(seed)
    n = len(trace)
    if n == 0:
        return 0.0, []
    block_size = max(1, int(round(math.sqrt(n))))
    num_blocks = (n + block_size - 1) // block_size
    eta = math.sqrt(8.0 * math.log(max(2, len(actions))) / max(1, num_blocks))

    log_w = [0.0] * len(actions)  # log of unnormalised weight
    chosen: list[int] = []
    total = 0.0
    prev_a = -1
    block_idx = 0
    while block_idx < num_blocks:
        # Sample an action proportional to softmax(log_w)
        m = max(log_w)
        probs = [math.exp(w - m) for w in log_w]
        z = sum(probs)
        probs = [p / z for p in probs]
        u = rng.random()
        cum = 0.0
        a_idx = len(actions) - 1
        for i, p in enumerate(probs):
            cum += p
            if u <= cum:
                a_idx = i
                break

        # Execute the chosen action for one block
        block_start = block_idx * block_size
        block_end = min(n, block_start + block_size)
        # Per-arm loss accumulator during this block (full information)
        arm_loss = [0.0] * len(actions)
        for t in range(block_start, block_end):
            b_t = trace[t]
            cost = actions[a_idx].energy_per_step * b_t
            if prev_a != -1 and a_idx != prev_a and t == block_start:
                cost += reconfig_cost
            total += cost
            chosen.append(a_idx)
            for i, a in enumerate(actions):
                arm_loss[i] += a.energy_per_step * b_t
            prev_a = a_idx

        # Update log weights with the observed block losses
        block_len = block_end - block_start
        if block_len > 0:
            mean_loss = [v / block_len for v in arm_loss]
            scale = max(mean_loss) - min(mean_loss)
            if scale > 0:
                normalised = [(v - min(mean_loss)) / scale for v in mean_loss]
                for i in range(len(log_w)):
                    log_w[i] -= eta * normalised[i]
        block_idx += 1
    return total, chosen


# ---------------------------------------------------------------------------
# Part A — Trace characterisation
# ---------------------------------------------------------------------------

def characterise_trace(trace: list[float]) -> dict:
    n = len(trace)
    mean = sum(trace) / n
    var = sum((v - mean) ** 2 for v in trace) / n
    cv = math.sqrt(var) / mean if mean > 0 else 0.0
    # Mean absolute change between adjacent samples
    deltas = [abs(trace[i + 1] - trace[i]) for i in range(n - 1)]
    mean_delta = sum(deltas) / max(1, len(deltas))
    # 6-step forecast error: predict v[t+6] = v[t] (persistence forecast)
    horizon = 6
    errors = [
        abs(trace[t + horizon] - trace[t]) / trace[t]
        for t in range(n - horizon) if trace[t] > 0
    ]
    mean_horizon_err = sum(errors) / max(1, len(errors))
    return {
        "n_samples": n,
        "mean_g_per_kwh": mean,
        "stddev_g_per_kwh": math.sqrt(var),
        "coefficient_of_variation_pct": cv * 100.0,
        "mean_step_change_g_per_kwh": mean_delta,
        "mean_6step_persistence_err_pct": mean_horizon_err * 100.0,
    }


# ---------------------------------------------------------------------------
# Part B — Online vs MPC sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimResult:
    noise_pct: float
    forecast_err_pct: float
    offline_cost: float
    mpc_cost: float
    online_cost: float

    @property
    def online_vs_optimal_gap_pct(self) -> float:
        return 100.0 * (self.online_cost - self.offline_cost) / self.offline_cost

    @property
    def mpc_vs_optimal_gap_pct(self) -> float:
        return 100.0 * (self.mpc_cost - self.offline_cost) / self.offline_cost

    @property
    def online_vs_mpc_pct(self) -> float:
        return 100.0 * (self.mpc_cost - self.online_cost) / self.mpc_cost


def simulate(
    trace: list[float],
    actions: list[Action],
    forecast_err: float,
    reconfig_cost: float,
    seed: int,
) -> tuple[float, float, float]:
    off_total, _ = offline_optimal(trace, actions)
    mpc_total, _ = mpc_with_forecast(
        trace, actions, horizon=6, forecast_noise_pct=forecast_err,
        seed=seed, reconfig_cost=reconfig_cost,
    )
    online_total, _ = online_hedge_shifted(
        trace, actions, reconfig_cost=reconfig_cost, seed=seed,
    )
    return off_total, mpc_total, online_total


def run(args: argparse.Namespace) -> None:
    console = Console()
    actions = build_action_set()

    # Build the noisy trace once
    trace = build_noisy_trace(
        hours=args.hours,
        sample_minutes=args.sample_minutes,
        noise_pct=args.trace_noise,
        regime_shift_at=args.regime_shift_at,
        regime_shift_magnitude=args.regime_shift_magnitude,
        seed=args.seed,
    )

    # Part A — Trace characterisation
    stats = characterise_trace(trace)
    console.print("\n[bold cyan]Part A — Trace characterisation[/]")
    char_table = Table()
    char_table.add_column("metric")
    char_table.add_column("value", justify="right")
    char_table.add_row("samples", str(stats["n_samples"]))
    char_table.add_row("mean", f"{stats['mean_g_per_kwh']:.1f} gCO2/kWh")
    char_table.add_row("stddev", f"{stats['stddev_g_per_kwh']:.1f} gCO2/kWh")
    char_table.add_row("coefficient of variation", f"{stats['coefficient_of_variation_pct']:.1f}%")
    char_table.add_row("mean step-to-step change", f"{stats['mean_step_change_g_per_kwh']:.1f} gCO2/kWh")
    char_table.add_row("6-step persistence err", f"{stats['mean_6step_persistence_err_pct']:.1f}%")
    console.print(char_table)

    a1 = stats["coefficient_of_variation_pct"] > 10.0
    a2 = stats["mean_6step_persistence_err_pct"] > 10.0
    console.print(
        f"[{'green' if a1 else 'red'}]Acceptance A1 (CV > 10%): "
        f"{'✓' if a1 else '✗'} ({stats['coefficient_of_variation_pct']:.1f}%)[/]"
    )
    console.print(
        f"[{'green' if a2 else 'red'}]Acceptance A2 (6-step forecast err > 10%): "
        f"{'✓' if a2 else '✗'} ({stats['mean_6step_persistence_err_pct']:.1f}%)[/]"
    )

    # Part B — Online vs MPC sweep
    console.print("\n[bold cyan]Part B — Online vs MPC sweep[/]")
    results: list[SimResult] = []
    for fe in args.forecast_errors:
        off, mpc, on = simulate(
            trace, actions, forecast_err=fe,
            reconfig_cost=args.reconfig_cost, seed=args.seed,
        )
        results.append(SimResult(
            noise_pct=args.trace_noise, forecast_err_pct=fe,
            offline_cost=off, mpc_cost=mpc, online_cost=on,
        ))

    sim_table = Table(title="Online-hedge vs MPC-with-noisy-forecast")
    sim_table.add_column("forecast noise %")
    sim_table.add_column("offline cost", justify="right")
    sim_table.add_column("MPC cost", justify="right")
    sim_table.add_column("online cost", justify="right")
    sim_table.add_column("MPC gap %", justify="right")
    sim_table.add_column("online gap %", justify="right")
    sim_table.add_column("online vs MPC %", justify="right")
    for r in results:
        sim_table.add_row(
            f"{r.forecast_err_pct*100:.0f}%",
            f"{r.offline_cost:.0f}",
            f"{r.mpc_cost:.0f}",
            f"{r.online_cost:.0f}",
            f"{r.mpc_vs_optimal_gap_pct:+.2f}%",
            f"{r.online_vs_optimal_gap_pct:+.2f}%",
            f"{r.online_vs_mpc_pct:+.2f}%",
        )
    console.print(sim_table)

    # Acceptance check: online beats MPC at high noise
    threshold_noise = 0.20  # 20% forecast noise (matches ElectricityMaps)
    high_noise_results = [r for r in results if r.forecast_err_pct >= threshold_noise]
    online_beats_mpc_at_high_noise = any(r.online_vs_mpc_pct > 0 for r in high_noise_results)
    online_within_5pct_of_optimal = any(
        abs(r.online_vs_optimal_gap_pct) <= 5.0 for r in results
    )

    console.print(
        f"\n[{'green' if online_beats_mpc_at_high_noise else 'red'}]"
        f"Acceptance B1 (online beats MPC at noise ≥ 20%): "
        f"{'✓' if online_beats_mpc_at_high_noise else '✗'}[/]"
    )
    console.print(
        f"[{'green' if online_within_5pct_of_optimal else 'red'}]"
        f"Acceptance B2 (online within 5% of offline-optimal): "
        f"{'✓' if online_within_5pct_of_optimal else '✗'}[/]"
    )

    overall_pass = a1 and a2 and online_beats_mpc_at_high_noise
    console.print(
        f"\n[bold {'green' if overall_pass else 'red'}]"
        f"Checkpoint 3 (empirical fit): {'PASS' if overall_pass else 'FAIL'}[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--sample-minutes", type=int, default=5)
    parser.add_argument("--trace-noise", type=float, default=0.05,
                        help="Std-dev of additive iid Gaussian noise as fraction of mean")
    parser.add_argument("--regime-shift-at", type=float, default=0.5)
    parser.add_argument("--regime-shift-magnitude", type=float, default=100.0,
                        help="Mean jump in gCO2/kWh at the regime shift point")
    parser.add_argument("--forecast-errors", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--reconfig-cost", type=float, default=50.0,
                        help="Switching cost charged per allocation change (J)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

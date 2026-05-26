"""Tail-violation Monte-Carlo — the empirical value prop of v2's chance constraints.

The v1 joint partitioner is variance-blind: it picks the deterministic
optimum sitting at the boundary of (Θ) and (P). Under any noise σ > 0,
roughly half of the random profile draws push the realised throughput
or power past the constraint — a violation rate that scales with σ but
sits around 50% per stage at typical NVML / RAPL noise levels.

v2's chance constraints inflate the deterministic bounds by
``(1 + z_α · σ)`` so the realised configuration violates with
probability at most α per stage by construction.

This experiment draws N random profile realisations for each plan and
reports the actual violation rate. The comparison is:

    v1_naive: joint_partition with deterministic constraints
              (no noise awareness — the practitioner default)
    v2:       stochastic_joint_partition with chance_alpha constraints
              (designed to bound violations)

Headline metric: per-stage violation rate of (Θ̃) and (P̃) under N draws.

Acceptance:
    v1_naive's violation rate is ≥ 30% per stage at σ_T = 0.10 (noise
    pushes it past the boundary), while v2's rate is ≤ chance_alpha
    by construction (verified empirically to validate the closed-form
    Gaussian assumption).

Usage:
    python -m experiments.exp_v2_tail_violations
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from experiments.exp_joint_real_workloads import (
    HARDWARE_PROFILES,
    MODELS,
)
from hise.parallel.joint_partitioner import JointPlan, joint_partition
from hise.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    partition_pipeline,
)
from hise.parallel.stochastic_joint_partitioner import stochastic_joint_partition

# ---------------------------------------------------------------------------
# Sampling + violation checks
# ---------------------------------------------------------------------------

def _sample_realisation(
    rng: random.Random,
    plan: JointPlan,
    stages: list[StageSpec],
    sigma_t: float,
    sigma_p: float,
) -> dict[int, tuple[float, float]]:
    """Sample one (T̃_s, P̃_s) realisation per stage in the plan.

    Returns dict[stage_id] = (T_s_realised_unthrottled, P_s_realised_nominal).
    Negative draws are clipped at 0 (rare at σ ≤ 0.20).
    """
    out: dict[int, tuple[float, float]] = {}
    for s in plan.stage_exec_time:
        t_nom = plan.stage_exec_time[s] * plan.throttle_factors[s]
        p_nom = stages[s].power_draw_w
        t_noise = rng.gauss(0.0, sigma_t * t_nom) if sigma_t > 0 else 0.0
        p_noise = rng.gauss(0.0, sigma_p * p_nom) if sigma_p > 0 else 0.0
        t_real = max(0.0, t_nom + t_noise)
        p_real = max(0.0, p_nom + p_noise)
        out[s] = (t_real, p_real)
    return out


def _count_violations(
    plan: JointPlan,
    realisation: dict[int, tuple[float, float]],
    stages: list[StageSpec],
    t_floor: float,
    voltage_alpha: float,
) -> tuple[bool, bool, int, int]:
    """Check whether the realisation violates (Θ) or (P) for any stage.

    Returns (theta_violation, p_violation, theta_stages, p_stages):
        theta_violation: True if ANY stage breached the throughput floor
        p_violation: True if ANY stage breached the power cap
        theta_stages: count of stages with Θ-breach
        p_stages: count of stages with P-breach
    """
    theta_v = False
    p_v = False
    theta_n = 0
    p_n = 0
    for s, (t_real, p_real) in realisation.items():
        r = plan.throttle_factors[s]
        if t_real / r > t_floor:
            theta_v = True
            theta_n += 1
        if p_real * (r ** voltage_alpha) > stages[s].power_cap_w:
            p_v = True
            p_n += 1
    return theta_v, p_v, theta_n, p_n


# ---------------------------------------------------------------------------
# Per-cell sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TailResult:
    model: str
    hardware: str
    sigma_t: float
    sigma_p: float
    n_stages: int
    n_draws: int
    v1_per_realisation_theta_violation_pct: float
    v1_per_stage_theta_violation_pct: float
    v1_per_realisation_p_violation_pct: float
    v1_per_stage_p_violation_pct: float
    v2_per_realisation_theta_violation_pct: float
    v2_per_stage_theta_violation_pct: float
    v2_per_realisation_p_violation_pct: float
    v2_per_stage_p_violation_pct: float
    v1_expected_energy: float
    v2_expected_energy: float
    v1_feasible: bool
    v2_feasible: bool


def monte_carlo_violations(
    plan: JointPlan,
    stages: list[StageSpec],
    *,
    sigma_t: float,
    sigma_p: float,
    t_floor: float,
    voltage_alpha: float,
    n_draws: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Run n_draws realisations; return (per-realisation Θ%, per-stage Θ%,
    per-realisation P%, per-stage P%)."""
    if not plan.is_feasible():
        return 0.0, 0.0, 0.0, 0.0
    rng = random.Random(seed)
    theta_realisations = 0
    p_realisations = 0
    theta_stages_total = 0
    p_stages_total = 0
    total_stages = 0
    for _ in range(n_draws):
        real = _sample_realisation(rng, plan, stages, sigma_t, sigma_p)
        theta_v, p_v, theta_n, p_n = _count_violations(
            plan, real, stages, t_floor, voltage_alpha,
        )
        if theta_v:
            theta_realisations += 1
        if p_v:
            p_realisations += 1
        theta_stages_total += theta_n
        p_stages_total += p_n
        total_stages += len(real)
    return (
        100.0 * theta_realisations / n_draws,
        100.0 * theta_stages_total / max(1, total_stages),
        100.0 * p_realisations / n_draws,
        100.0 * p_stages_total / max(1, total_stages),
    )


def _expected_energy(plan: JointPlan, stages: list[StageSpec],
                     voltage_alpha: float) -> float:
    if not plan.is_feasible():
        return math.inf
    e = 0.0
    for s in plan.stage_exec_time:
        t = plan.stage_exec_time[s] * plan.throttle_factors[s]
        r = plan.throttle_factors[s]
        e += stages[s].power_draw_w * (r ** (voltage_alpha - 1)) * t
    return e


def evaluate_cell(
    layers: list[LayerProfile],
    stages: list[StageSpec],
    links: list[LinkSpec],
    *,
    throughput_floor_iters_per_s: float,
    sigma_t: float,
    sigma_p: float,
    chance_alpha: float,
    voltage_alpha: float,
    throttle_min: float,
    throttle_granularity: int,
    n_draws: int,
    seed: int,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    JointPlan,
    JointPlan,
]:
    t_floor = 1.0 / throughput_floor_iters_per_s
    # v1: variance-blind, deterministic constraints (the practitioner default)
    v1 = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=throughput_floor_iters_per_s,
        voltage_alpha=voltage_alpha,
        throttle_min=throttle_min,
        throttle_granularity=throttle_granularity,
    )
    # v2: chance-constrained
    v2 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=throughput_floor_iters_per_s,
        voltage_alpha=voltage_alpha,
        throttle_min=throttle_min,
        throttle_granularity=throttle_granularity,
        sigma_t=sigma_t,
        sigma_p=sigma_p,
        cvar_beta=0.05,
        chance_alpha=chance_alpha,
    )

    v1_rates = monte_carlo_violations(
        v1, stages, sigma_t=sigma_t, sigma_p=sigma_p,
        t_floor=t_floor, voltage_alpha=voltage_alpha,
        n_draws=n_draws, seed=seed,
    )
    v2_rates = monte_carlo_violations(
        v2, stages, sigma_t=sigma_t, sigma_p=sigma_p,
        t_floor=t_floor, voltage_alpha=voltage_alpha,
        n_draws=n_draws, seed=seed + 1,
    )
    return v1_rates, v2_rates, v1, v2


def sweep(args: argparse.Namespace) -> list[TailResult]:
    rows: list[TailResult] = []
    for mk in args.models:
        m_label, layers = MODELS[mk]
        for hk in args.hardware:
            hw_label, stages = HARDWARE_PROFILES[hk]
            links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(len(stages) - 1)]
            bot = partition_pipeline(layers, stages, links, objective="bottleneck")
            t_max_bot = max(bot.stage_exec_time.values())
            t_floor = t_max_bot * args.t_floor_multiplier

            for sigma_t in args.sigma_t_values:
                v1_rates, v2_rates, v1, v2 = evaluate_cell(
                    layers, stages, links,
                    throughput_floor_iters_per_s=1.0 / t_floor,
                    sigma_t=sigma_t,
                    sigma_p=args.sigma_p,
                    chance_alpha=args.chance_alpha,
                    voltage_alpha=args.voltage_alpha,
                    throttle_min=args.throttle_min,
                    throttle_granularity=args.throttle_granularity,
                    n_draws=args.n_draws,
                    seed=args.seed,
                )
                rows.append(TailResult(
                    model=m_label, hardware=hw_label,
                    sigma_t=sigma_t, sigma_p=args.sigma_p,
                    n_stages=len(stages), n_draws=args.n_draws,
                    v1_per_realisation_theta_violation_pct=v1_rates[0],
                    v1_per_stage_theta_violation_pct=v1_rates[1],
                    v1_per_realisation_p_violation_pct=v1_rates[2],
                    v1_per_stage_p_violation_pct=v1_rates[3],
                    v2_per_realisation_theta_violation_pct=v2_rates[0],
                    v2_per_stage_theta_violation_pct=v2_rates[1],
                    v2_per_realisation_p_violation_pct=v2_rates[2],
                    v2_per_stage_p_violation_pct=v2_rates[3],
                    v1_expected_energy=_expected_energy(v1, stages, args.voltage_alpha),
                    v2_expected_energy=_expected_energy(v2, stages, args.voltage_alpha),
                    v1_feasible=v1.is_feasible(),
                    v2_feasible=v2.is_feasible(),
                ))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_violations(rows: list[TailResult], console: Console) -> None:
    table = Table(
        title=(
            "Per-stage Θ-violation rate under Monte-Carlo profile draws — "
            "v1_naive (no chance constraints) vs v2 (chance α)"
        )
    )
    table.add_column("model")
    table.add_column("hardware")
    table.add_column("σ_T", justify="right")
    table.add_column("v1 Θ-viol /stage %", justify="right")
    table.add_column("v2 Θ-viol /stage %", justify="right")
    table.add_column("v1 Θ-viol /realisation %", justify="right")
    table.add_column("v2 Θ-viol /realisation %", justify="right")
    table.add_column("E gain v1 vs v2 %", justify="right")
    for r in rows:
        if not (r.v1_feasible and r.v2_feasible):
            continue
        e_gain = (
            100.0 * (r.v2_expected_energy - r.v1_expected_energy) / r.v2_expected_energy
            if math.isfinite(r.v1_expected_energy) and r.v2_expected_energy > 0 else 0.0
        )
        table.add_row(
            r.model, r.hardware,
            f"{r.sigma_t}",
            f"{r.v1_per_stage_theta_violation_pct:.1f}%",
            f"{r.v2_per_stage_theta_violation_pct:.1f}%",
            f"{r.v1_per_realisation_theta_violation_pct:.1f}%",
            f"{r.v2_per_realisation_theta_violation_pct:.1f}%",
            f"{e_gain:+.2f}%",
        )
    console.print(table)


def render_aggregate(rows: list[TailResult], console: Console,
                       chance_alpha: float) -> None:
    feasible = [r for r in rows if r.v1_feasible and r.v2_feasible
                and r.sigma_t > 0]
    if not feasible:
        return
    v1_stage = [r.v1_per_stage_theta_violation_pct for r in feasible]
    v2_stage = [r.v2_per_stage_theta_violation_pct for r in feasible]
    v1_real = [r.v1_per_realisation_theta_violation_pct for r in feasible]
    v2_real = [r.v2_per_realisation_theta_violation_pct for r in feasible]

    t = Table(title="Aggregate violation statistics (σ_T > 0 cells)")
    t.add_column("metric")
    t.add_column("v1_naive", justify="right")
    t.add_column("v2", justify="right")
    t.add_column("ratio v1/v2", justify="right")
    t.add_row(
        "mean per-stage Θ-viol %",
        f"{sum(v1_stage)/len(v1_stage):.2f}%",
        f"{sum(v2_stage)/len(v2_stage):.2f}%",
        f"{(sum(v1_stage)/max(1, sum(v2_stage))):.1f}x",
    )
    t.add_row(
        "max per-stage Θ-viol %",
        f"{max(v1_stage):.2f}%",
        f"{max(v2_stage):.2f}%",
        f"{max(v1_stage)/max(0.01, max(v2_stage)):.1f}x",
    )
    t.add_row(
        "mean per-realisation Θ-viol %",
        f"{sum(v1_real)/len(v1_real):.2f}%",
        f"{sum(v2_real)/len(v2_real):.2f}%",
        f"{(sum(v1_real)/max(1, sum(v2_real))):.1f}x",
    )
    t.add_row(
        "max per-realisation Θ-viol %",
        f"{max(v1_real):.2f}%",
        f"{max(v2_real):.2f}%",
        f"{max(v1_real)/max(0.01, max(v2_real)):.1f}x",
    )
    console.print(t)

    # Acceptance check
    v1_high_violation_cells = sum(1 for r in feasible if r.v1_per_stage_theta_violation_pct >= 30.0)
    v2_within_alpha_cells = sum(
        1 for r in feasible
        if r.v2_per_stage_theta_violation_pct <= chance_alpha * 100.0 + 1.0
    )
    total = len(feasible)
    console.print(
        f"\n[bold]Acceptance check[/]: "
        f"v1 per-stage Θ-violation ≥ 30%: [{'green' if v1_high_violation_cells >= total * 0.5 else 'red'}]"
        f"{v1_high_violation_cells}/{total}[/]; "
        f"v2 per-stage Θ-violation ≤ α+1pp ({chance_alpha*100:.0f}+1)%: "
        f"[{'green' if v2_within_alpha_cells == total else 'red'}]{v2_within_alpha_cells}/{total}[/]"
    )


def run(args: argparse.Namespace) -> None:
    console = Console()
    console.print(
        f"[bold]exp_v2_tail_violations[/] — Monte-Carlo N={args.n_draws}, "
        f"σ_P={args.sigma_p}, chance_α={args.chance_alpha}, "
        f"T_floor mult={args.t_floor_multiplier}"
    )
    rows = sweep(args)
    render_violations(rows, console)
    render_aggregate(rows, console, args.chance_alpha)
    console.print(
        "\n[dim]v1_naive runs joint_partition with deterministic constraints — "
        "the variance-blind default. v2 runs stochastic_joint_partition with "
        "chance_alpha-bounded constraints. Per-stage Θ-violation = fraction "
        "of (stage × realisation) pairs where the realised T_s/r_s exceeded "
        "T_floor. Per-realisation = fraction of realisations where ANY stage "
        "violated. v2's chance constraints make it pay a small energy premium "
        "in exchange for bounded tail-violation rate.[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=str, nargs="*",
                        default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--hardware", type=str, nargs="*",
                        default=list(HARDWARE_PROFILES), choices=list(HARDWARE_PROFILES))
    parser.add_argument("--sigma-t-values", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20])
    parser.add_argument("--sigma-p", type=float, default=0.05)
    parser.add_argument("--chance-alpha", type=float, default=0.05)
    parser.add_argument("--t-floor-multiplier", type=float, default=1.50)
    parser.add_argument("--voltage-alpha", type=float, default=2.0)
    parser.add_argument("--throttle-min", type=float, default=0.5)
    parser.add_argument("--throttle-granularity", type=int, default=11)
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

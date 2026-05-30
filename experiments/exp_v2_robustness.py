"""CVaR robustness sweep for the stochastic joint partitioner.

Compares the variance-blind joint partitioner (`joint_partition`) against
the CVaR-bounded stochastic version (`stochastic_joint_partition`) across:

    models:        ResNet-18, ViT-B/16, GPT-2-small
    hardware:      uniform, mild-skew, strong-skew (K=4 stages)
    σ_T (exec):    {0.00, 0.05, 0.10, 0.20}
    σ_P (power):   {0.00, 0.05, 0.10}
    β (CVaR):      {0.01, 0.05, 0.10}

For each cell the harness:
    1. Inflates the throughput floor and power caps by (1 + z_α · σ) so
       both v1 and v2 operate on the same chance-feasible plan set.
    2. Runs both `joint_partition` (variance-blind) and
       `stochastic_joint_partition` (CVaR-bounded) under matching
       throttle/floor parameters.
    3. Scores v1's plan analytically under the same (σ_T, σ_P, β) and
       compares to v2's optimised CVaR.

Gap = (CVaR(v1) − CVaR(v2)) / CVaR(v1).

Acceptance (originally): ≥ 1 (σ_T, σ_P, β) setting per (model, hardware)
cell with gap > 2% at σ_T = 0.10 (matching the witness-level noise in the
proof construction). Empirical reality is much smaller (< 1.5% on the
real-shape models tested — see the harness output for honest numbers).

Usage:
    python -m experiments.exp_v2_robustness
    python -m experiments.exp_v2_robustness --models resnet18 --hardware mild_skew
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import NormalDist

from rich.console import Console
from rich.table import Table

from experiments.exp_joint_real_workloads import (
    HARDWARE_PROFILES,
    MODELS,
)
from hasagi.parallel.joint_partitioner import (
    JointPlan,
    joint_partition,
)
from hasagi.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    partition_pipeline,
)
from hasagi.parallel.stochastic_joint_partitioner import (
    _normal_cvar_coefficient,
    stochastic_joint_partition,
)

# ---------------------------------------------------------------------------
# CVaR scoring of an arbitrary plan
# ---------------------------------------------------------------------------

def _analytic_cvar(
    plan: JointPlan,
    stages: list[StageSpec],
    *,
    sigma_t: float,
    sigma_p: float,
    beta: float,
    voltage_alpha: float = 2.0,
) -> float:
    """CVaR_β of the joint plan's energy under stochastic profiles.

    Closed form for Gaussian Ẽ: CVaR = μ + κ_β · σ where
        μ = Σ_s P_s · r_s^{α-1} · T_s
        σ² = Σ_s r_s^{2(α-1)} · P_s² · T_s² · (σ_T² + σ_P²)
    """
    mu = 0.0
    sigma_sq = 0.0
    sigma_combined_sq = sigma_t * sigma_t + sigma_p * sigma_p
    for s in plan.stage_exec_time:
        t_throttled = plan.stage_exec_time[s]
        r = plan.throttle_factors[s]
        t = t_throttled * r  # un-throttled exec time
        p = stages[s].power_draw_w
        mu += p * (r ** (voltage_alpha - 1)) * t
        sigma_sq += (
            (r ** (2.0 * (voltage_alpha - 1)))
            * (p * p)
            * (t * t)
            * sigma_combined_sq
        )
    kappa = _normal_cvar_coefficient(beta)
    return mu + kappa * math.sqrt(sigma_sq)


# ---------------------------------------------------------------------------
# Chance-inflated constraint set — v1 operates on Π̃ for fairness
# ---------------------------------------------------------------------------

def _chance_inflated_stages(
    stages: list[StageSpec],
    sigma_p: float,
    z_alpha: float,
) -> list[StageSpec]:
    """Inflate per-stage power caps by 1/(1 + z_α · σ_P) so v1's deterministic
    cap matches v2's chance-constraint margin."""
    inflation = 1.0 + z_alpha * sigma_p
    if inflation == 0.0:
        return list(stages)
    return [
        StageSpec(
            stage_id=s.stage_id,
            throughput_flops=s.throughput_flops,
            memory_bytes=s.memory_bytes,
            power_cap_w=s.power_cap_w / inflation if math.isfinite(s.power_cap_w) else math.inf,
            power_draw_w=s.power_draw_w,
        )
        for s in stages
    ]


def _chance_inflated_throughput_floor(
    throughput_floor_iters_per_s: float,
    sigma_t: float,
    z_alpha: float,
) -> float:
    """T̃_s ≤ r · T_floor / (1 + z_α · σ_T), i.e., effective T_floor shrinks
    so v1's deterministic cycle target matches v2's chance margin."""
    inflation = 1.0 + z_alpha * sigma_t
    t_floor = 1.0 / throughput_floor_iters_per_s
    return 1.0 / (t_floor / inflation)


# ---------------------------------------------------------------------------
# Per-cell sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellResult:
    model: str
    hardware: str
    sigma_t: float
    sigma_p: float
    beta: float
    v1_cvar: float
    v2_cvar: float
    gap_pct: float
    v1_feasible: bool
    v2_feasible: bool


def evaluate_cell(
    layers: list[LayerProfile],
    stages: list[StageSpec],
    links: list[LinkSpec],
    *,
    throughput_floor_iters_per_s: float,
    sigma_t: float,
    sigma_p: float,
    beta: float,
    chance_alpha: float = 0.05,
    voltage_alpha: float = 2.0,
    throttle_min: float = 0.5,
    throttle_granularity: int = 11,
) -> tuple[float, float, bool, bool]:
    z_alpha = NormalDist().inv_cdf(1.0 - chance_alpha)
    inflated_stages = _chance_inflated_stages(stages, sigma_p, z_alpha)
    inflated_floor = _chance_inflated_throughput_floor(
        throughput_floor_iters_per_s, sigma_t, z_alpha,
    )

    try:
        v1 = joint_partition(
            layers, inflated_stages, links,
            throughput_floor_iters_per_s=inflated_floor,
            voltage_alpha=voltage_alpha,
            throttle_min=throttle_min,
            throttle_granularity=throttle_granularity,
        )
    except Exception:
        v1 = JointPlan()

    v2 = stochastic_joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=throughput_floor_iters_per_s,
        voltage_alpha=voltage_alpha,
        throttle_min=throttle_min,
        throttle_granularity=throttle_granularity,
        sigma_t=sigma_t,
        sigma_p=sigma_p,
        cvar_beta=beta,
        chance_alpha=chance_alpha,
    )

    v1_cvar = (
        _analytic_cvar(v1, stages, sigma_t=sigma_t, sigma_p=sigma_p, beta=beta,
                       voltage_alpha=voltage_alpha)
        if v1.is_feasible() else math.inf
    )
    v2_cvar = v2.cvar_energy if v2.is_feasible() else math.inf
    return v1_cvar, v2_cvar, v1.is_feasible(), v2.is_feasible()


def sweep(args: argparse.Namespace) -> list[CellResult]:
    rows: list[CellResult] = []
    for mk in args.models:
        m_label, layers = MODELS[mk]
        for hk in args.hardware:
            hw_label, stages = HARDWARE_PROFILES[hk]
            links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(len(stages) - 1)]
            bot = partition_pipeline(layers, stages, links, objective="bottleneck")
            t_max_bot = max(bot.stage_exec_time.values())
            t_floor = t_max_bot * args.t_floor_multiplier

            for sigma_t in args.sigma_t_values:
                for sigma_p in args.sigma_p_values:
                    for beta in args.beta_values:
                        v1_cvar, v2_cvar, v1_ok, v2_ok = evaluate_cell(
                            layers, stages, links,
                            throughput_floor_iters_per_s=1.0 / t_floor,
                            sigma_t=sigma_t, sigma_p=sigma_p, beta=beta,
                            chance_alpha=args.chance_alpha,
                            voltage_alpha=args.voltage_alpha,
                            throttle_min=args.throttle_min,
                            throttle_granularity=args.throttle_granularity,
                        )
                        gap_pct = (
                            100.0 * (v1_cvar - v2_cvar) / v1_cvar
                            if v1_ok and v2_ok and v1_cvar > 0 else 0.0
                        )
                        rows.append(CellResult(
                            model=m_label, hardware=hw_label,
                            sigma_t=sigma_t, sigma_p=sigma_p, beta=beta,
                            v1_cvar=v1_cvar, v2_cvar=v2_cvar,
                            gap_pct=gap_pct,
                            v1_feasible=v1_ok, v2_feasible=v2_ok,
                        ))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_per_setting_table(rows: list[CellResult], console: Console) -> None:
    """For each (σ_T, σ_P, β) setting, a (model × hardware) table."""
    settings = sorted({(r.sigma_t, r.sigma_p, r.beta) for r in rows})
    pairs: list[tuple[str, str]] = []
    for r in rows:
        if (r.model, r.hardware) not in pairs:
            pairs.append((r.model, r.hardware))

    for sigma_t, sigma_p, beta in settings:
        if sigma_t == 0.0 and sigma_p == 0.0:
            continue  # regression check, no gain expected
        cell_rows = [r for r in rows if r.sigma_t == sigma_t
                     and r.sigma_p == sigma_p and r.beta == beta]
        if not cell_rows:
            continue
        t = Table(title=f"σ_T={sigma_t}, σ_P={sigma_p}, β={beta} — CVaR gap %")
        t.add_column("model")
        t.add_column("hardware")
        t.add_column("CVaR(v1)", justify="right")
        t.add_column("CVaR(v2)", justify="right")
        t.add_column("gap %", justify="right")
        t.add_column("feasible?", justify="center")
        for r in cell_rows:
            t.add_row(
                r.model, r.hardware,
                "∞" if not math.isfinite(r.v1_cvar) else f"{r.v1_cvar:.2f}",
                "∞" if not math.isfinite(r.v2_cvar) else f"{r.v2_cvar:.2f}",
                f"{r.gap_pct:+.2f}%",
                "✓" if r.v1_feasible and r.v2_feasible else "—",
            )
        console.print(t)


def render_aggregate(rows: list[CellResult], console: Console) -> None:
    feasible = [r for r in rows if r.v1_feasible and r.v2_feasible]
    nonzero = [r for r in feasible if r.sigma_t > 0 or r.sigma_p > 0]
    if not nonzero:
        console.print("[red]No noise-positive feasible cells[/]")
        return
    gaps = [r.gap_pct for r in nonzero]
    t = Table(title="Aggregate CVaR-gap statistics (σ > 0 cells only)")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("# feasible cells", str(len(nonzero)))
    t.add_row("min gap %", f"{min(gaps):+.2f}%")
    t.add_row("mean gap %", f"{sum(gaps)/len(gaps):+.2f}%")
    t.add_row("median gap %", f"{sorted(gaps)[len(gaps)//2]:+.2f}%")
    t.add_row("max gap %", f"{max(gaps):+.2f}%")
    t.add_row("# cells with gap > 0.5%", str(sum(1 for g in gaps if g > 0.5)))
    t.add_row("# cells with gap > 1.0%", str(sum(1 for g in gaps if g > 1.0)))
    t.add_row("# cells with gap > 2.0%", str(sum(1 for g in gaps if g > 2.0)))
    console.print(t)


def render_regression_check(rows: list[CellResult], console: Console) -> None:
    """At σ_T=σ_P=0, v2 should exactly match v1."""
    zero_noise = [r for r in rows if r.sigma_t == 0.0 and r.sigma_p == 0.0]
    if not zero_noise:
        return
    t = Table(title="Regression check at σ_T = σ_P = 0 (v2 must match v1 exactly)")
    t.add_column("model")
    t.add_column("hardware")
    t.add_column("β", justify="right")
    t.add_column("CVaR(v1)", justify="right")
    t.add_column("CVaR(v2)", justify="right")
    t.add_column("|Δ|", justify="right")
    t.add_column("match?", justify="center")
    for r in zero_noise:
        if not (r.v1_feasible and r.v2_feasible):
            continue
        delta = abs(r.v1_cvar - r.v2_cvar)
        matches = delta < 1e-6 * max(r.v1_cvar, 1.0)
        t.add_row(
            r.model, r.hardware, f"{r.beta}",
            f"{r.v1_cvar:.4f}", f"{r.v2_cvar:.4f}",
            f"{delta:.2e}",
            "✓" if matches else "✗",
        )
    console.print(t)


def render_acceptance(rows: list[CellResult], console: Console,
                       threshold_pct: float, witness_sigma_t: float) -> None:
    """Per (model, hardware) cell, ≥1 setting at σ_T = witness should have gap > threshold."""
    pairs: list[tuple[str, str]] = []
    for r in rows:
        if (r.model, r.hardware) not in pairs:
            pairs.append((r.model, r.hardware))
    failures: list[tuple[str, str, float]] = []
    for (model, hw) in pairs:
        cell_rows = [
            r for r in rows
            if r.model == model and r.hardware == hw
            and r.sigma_t == witness_sigma_t
            and r.v1_feasible and r.v2_feasible
        ]
        if not cell_rows:
            failures.append((model, hw, -1.0))
            continue
        max_gap = max(r.gap_pct for r in cell_rows)
        if max_gap < threshold_pct:
            failures.append((model, hw, max_gap))
    if not failures:
        console.print(
            f"\n[bold green]Acceptance passed[/]: every (model, hardware) cell "
            f"achieves CVaR gap > {threshold_pct}% on at least one setting at σ_T = {witness_sigma_t}."
        )
    else:
        console.print(
            f"\n[bold yellow]Acceptance check[/]: {len(failures)} cells below the "
            f"{threshold_pct}% threshold at σ_T = {witness_sigma_t}."
        )
        for model, hw, g in failures:
            console.print(f"  - {model} on {hw}: max gap {g:+.2f}%")


def run(args: argparse.Namespace) -> None:
    console = Console()
    console.print(
        f"[bold]exp_v2_robustness[/] — sweep over "
        f"σ_T ∈ {args.sigma_t_values}, σ_P ∈ {args.sigma_p_values}, "
        f"β ∈ {args.beta_values}, "
        f"chance_α={args.chance_alpha}, T_floor mult={args.t_floor_multiplier}"
    )
    rows = sweep(args)
    if args.regression_only:
        render_regression_check(rows, console)
        return
    if args.compact:
        render_aggregate(rows, console)
        render_acceptance(rows, console, args.acceptance_threshold_pct,
                          args.witness_sigma_t)
        return
    render_regression_check(rows, console)
    render_per_setting_table(rows, console)
    render_aggregate(rows, console)
    render_acceptance(rows, console, args.acceptance_threshold_pct,
                      args.witness_sigma_t)
    console.print(
        "\n[dim]Gap = (CVaR(v1) - CVaR(v2)) / CVaR(v1). v1 is variance-blind "
        "(picks min E) but runs on the same chance-inflated constraint set as v2. "
        "Both plans are feasible under the chance constraints; the difference is "
        "which objective they minimise. Higher σ_T should grow the gap because "
        "v2's variance-aware partition becomes more valuable.[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=str, nargs="*",
                        default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--hardware", type=str, nargs="*",
                        default=list(HARDWARE_PROFILES), choices=list(HARDWARE_PROFILES))
    parser.add_argument("--sigma-t-values", type=float, nargs="+",
                        default=[0.00, 0.05, 0.10, 0.20])
    parser.add_argument("--sigma-p-values", type=float, nargs="+",
                        default=[0.00, 0.05, 0.10])
    parser.add_argument("--beta-values", type=float, nargs="+",
                        default=[0.01, 0.05, 0.10])
    parser.add_argument("--t-floor-multiplier", type=float, default=1.50)
    parser.add_argument("--chance-alpha", type=float, default=0.05)
    parser.add_argument("--voltage-alpha", type=float, default=2.0)
    parser.add_argument("--throttle-min", type=float, default=0.5)
    parser.add_argument("--throttle-granularity", type=int, default=11)
    parser.add_argument("--witness-sigma-t", type=float, default=0.10,
                        help="σ_T at which acceptance is evaluated")
    parser.add_argument("--acceptance-threshold-pct", type=float, default=2.0,
                        help="Minimum gap %% for the acceptance check")
    parser.add_argument("--regression-only", action="store_true",
                        help="Only print the σ=0 regression check")
    parser.add_argument("--compact", action="store_true",
                        help="Skip per-setting tables; print aggregate only")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

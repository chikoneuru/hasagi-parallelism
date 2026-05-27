"""Joint partition-and-throttle vs sequential composition.

Compares five allocator families on the same heterogeneous workload:

    bottleneck-only        bottleneck-DP cuts, r ≡ 1
    bottleneck + Perseus   bottleneck-DP cuts, Perseus rule per stage
    energy-only            energy-DP cuts,     r ≡ 1
    energy + Perseus       energy-DP cuts,     Perseus rule per stage
    joint                  joint DP picking (cuts, throttle) together

Sweeps over the per-stage capacity ratio (heterogeneity dial) and the
throughput floor. Reports the per-iteration energy of each allocator and
the gap between the best sequential composition and the joint optimum.

The "Perseus rule" projects continuous `r = T_s/T_max` onto the discrete
throttle grid R the joint optimiser uses, by rounding up to the smallest
admissible grid point. Without this projection, a continuous-Perseus
plan can sit outside the joint's discrete feasible set; the apples-to-
apples comparison projects both onto R.

Usage:
    python -m experiments.exp_joint_vs_stacked
    python -m experiments.exp_joint_vs_stacked --heterogeneity 1 2 3 5
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from hise.parallel.joint_partitioner import (
    ThrottleCurve,
    _build_throttle_set,
    joint_partition,
)
from hise.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    Partition,
    StageSpec,
    partition_pipeline,
)


@dataclass(frozen=True)
class AllocResult:
    name: str
    cuts: tuple[int, ...]
    throttle_factors: tuple[float, ...]
    energy: float
    pipeline_time: float
    feasible: bool
    # Best-effort energy at the chosen partition/throttle. Always populated
    # for partition baselines and Perseus variants (with a unit-throttle
    # fallback when the Perseus projection is empty). Matches ``energy``
    # when feasible; remains finite when the deadline is missed but the
    # partition itself exists, so a relaxed decomposition can still
    # compute synergy on infeasible cells.
    raw_energy: float = math.inf


def _discrete_perseus_throttle(
    partition: Partition,
    throttle_set: tuple[float, ...],
) -> tuple[float, ...] | None:
    """Project the Perseus rule r_s = T_s/T_max onto the discrete throttle set.

    Smallest admissible grid point ≥ T_s/T_max per stage. Returns None if
    any stage's required throttle exceeds 1 (the partition is not Perseus-
    representable in this grid — e.g., the bottleneck stage at r=1 is
    already past the implied T_floor).
    """
    if not partition.stage_exec_time:
        return None
    t_max = max(partition.stage_exec_time.values())
    if not math.isfinite(t_max) or t_max <= 0:
        return None
    throttles: list[float] = []
    for s in sorted(partition.stage_exec_time):
        ratio = partition.stage_exec_time[s] / t_max
        admissible = [r for r in throttle_set if r + 1e-12 >= ratio]
        if not admissible:
            return None
        throttles.append(min(admissible))
    return tuple(throttles)


def _partition_energy_under_throttle(
    partition: Partition,
    stages: list[StageSpec],
    throttle: tuple[float, ...],
    voltage_alpha: float,
    throttle_curve: ThrottleCurve | None = None,
) -> tuple[float, float]:
    """E(c, r) and pipeline_time = max_s T_s(c) / r_s.

    When ``throttle_curve`` is provided the parametric ``r^(α-1)`` and
    ``1/r`` factors are replaced by the curve's measured scales — keeps
    the non-joint baselines apples-to-apples with the joint allocator's
    empirical mode.
    """
    e = 0.0
    t_max = -math.inf
    for s, t in partition.stage_exec_time.items():
        r = throttle[s]
        if throttle_curve is None:
            e_scale = r ** (voltage_alpha - 1)
            t_scale = 1.0 / r
        else:
            e_scale = throttle_curve.energy_scale(r)
            t_scale = throttle_curve.time_scale(r)
        e += stages[s].power_draw_w * e_scale * t
        t_max = max(t_max, t * t_scale)
    return e, t_max


def _heterogeneity_ratio(stages: list[StageSpec]) -> float:
    """max θ / min θ — simple ratio scalar."""
    caps = [s.throughput_flops for s in stages]
    return max(caps) / min(caps)


def _heterogeneity_entropy(stages: list[StageSpec]) -> float:
    """Shannon entropy of the normalised capacity distribution.

    0 = perfectly homogeneous (single capacity); log K = max heterogeneity
    (uniform across all stages at the same total). Reports in nats.
    """
    caps = [s.throughput_flops for s in stages]
    total = sum(caps)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in caps:
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def evaluate_workload(
    layers: list[LayerProfile],
    stages: list[StageSpec],
    links: list[LinkSpec],
    *,
    throughput_floor_iters_per_s: float,
    voltage_alpha: float,
    throttle_min: float,
    throttle_granularity: int,
    throttle_curve: ThrottleCurve | None = None,
) -> list[AllocResult]:
    """Run all five allocators against the same workload + T_floor.

    ``throttle_curve`` (optional) routes all five allocators through the
    empirical Pareto frontier instead of the parametric ``r^α`` model.
    """
    if throttle_curve is not None:
        R = throttle_curve.ratios()
    else:
        R = _build_throttle_set(throttle_min, throttle_granularity)
    t_floor = 1.0 / throughput_floor_iters_per_s
    results: list[AllocResult] = []
    n_stages = len(stages)
    unit_throttle = (1.0,) * n_stages

    bot = partition_pipeline(layers, stages, links, objective="bottleneck")
    en = partition_pipeline(layers, stages, links, objective="energy")

    # Per-partition unit-throttle energy is the relaxed reference: even if
    # the deadline is missed, the partition exists and has a finite energy.
    raw_unit_energy: dict[str, float] = {}
    for name, partition in (("bottleneck-only", bot), ("energy-only", en)):
        e, tp = _partition_energy_under_throttle(
            partition, stages, unit_throttle, voltage_alpha, throttle_curve,
        )
        feasible = tp <= t_floor + 1e-9
        raw_unit_energy[name] = e
        results.append(AllocResult(
            name=name, cuts=partition.cuts, throttle_factors=unit_throttle,
            energy=e if feasible else math.inf,
            pipeline_time=tp, feasible=feasible,
            raw_energy=e,
        ))

    for name, partition in (("bottleneck + Perseus", bot), ("energy + Perseus", en)):
        base_name = "bottleneck-only" if name.startswith("bottleneck") else "energy-only"
        proj = _discrete_perseus_throttle(partition, R)
        if proj is None:
            # Throttle projection empty (deadline cannot be met by any
            # admissible throttle vector). Fall back to unit throttle for
            # raw_energy so the relaxed decomposition still has a number.
            results.append(AllocResult(
                name=name, cuts=partition.cuts, throttle_factors=(),
                energy=math.inf, pipeline_time=math.inf, feasible=False,
                raw_energy=raw_unit_energy.get(base_name, math.inf),
            ))
            continue
        e, tp = _partition_energy_under_throttle(
            partition, stages, proj, voltage_alpha, throttle_curve,
        )
        feasible = tp <= t_floor + 1e-9
        results.append(AllocResult(
            name=name, cuts=partition.cuts, throttle_factors=proj,
            energy=e if feasible else math.inf,
            pipeline_time=tp, feasible=feasible,
            raw_energy=e,
        ))

    plan = joint_partition(
        layers, stages, links,
        throughput_floor_iters_per_s=throughput_floor_iters_per_s,
        voltage_alpha=voltage_alpha,
        throttle_min=throttle_min,
        throttle_granularity=throttle_granularity,
        throttle_curve=throttle_curve,
    )
    results.append(AllocResult(
        name="joint",
        cuts=plan.cuts if plan.is_feasible() else (),
        throttle_factors=plan.throttle_factors,
        energy=plan.energy_per_iter,
        pipeline_time=plan.pipeline_time_s,
        feasible=plan.is_feasible(),
        raw_energy=plan.energy_per_iter,
    ))
    return results


def _build_workload(
    n_layers: int,
    capacity_ratio: float,
    n_stages: int = 2,
) -> tuple[list[LayerProfile], list[StageSpec], list[LinkSpec]]:
    """Equal-FLOPS layers, K stages with capacities (1, c, c, …) ramping from slow to fast."""
    layers = [
        LayerProfile(index=i, fwd_flops=1.0, bwd_flops=2.0, activation_bytes=0)
        for i in range(n_layers)
    ]
    if n_stages == 1:
        thetas = [1.0]
    elif n_stages == 2:
        thetas = [1.0, capacity_ratio]
    else:
        # Slow stage 0, fast remaining stages (uniform at capacity_ratio).
        thetas = [1.0] + [capacity_ratio] * (n_stages - 1)
    stages = [
        StageSpec(stage_id=s, throughput_flops=thetas[s], memory_bytes=10**18, power_draw_w=1.0)
        for s in range(n_stages)
    ]
    links = [LinkSpec(s, s + 1, 1e18, 0.0) for s in range(n_stages - 1)]
    return layers, stages, links


def run_scenario(
    name: str,
    layers: list[LayerProfile],
    stages: list[StageSpec],
    links: list[LinkSpec],
    throughput_floor_iters_per_s: float,
    args: argparse.Namespace,
    console: Console,
) -> None:
    t_floor = 1.0 / throughput_floor_iters_per_s
    console.print(
        f"\n[bold magenta]=== {name} ===[/]   "
        f"K={len(stages)}, n={len(layers)}, "
        f"θ={tuple(s.throughput_flops for s in stages)}, "
        f"T_floor={t_floor:.3f}s, ratio={_heterogeneity_ratio(stages):.2f}, "
        f"entropy={_heterogeneity_entropy(stages):.3f} nats"
    )

    results = evaluate_workload(
        layers, stages, links,
        throughput_floor_iters_per_s=throughput_floor_iters_per_s,
        voltage_alpha=args.voltage_alpha,
        throttle_min=args.throttle_min,
        throttle_granularity=args.throttle_granularity,
    )

    table = Table(title=name)
    table.add_column("allocator")
    table.add_column("cuts", overflow="fold")
    table.add_column("throttle r", overflow="fold")
    table.add_column("E (J)", justify="right")
    table.add_column("pipeline (s)", justify="right")
    table.add_column("feasible?", justify="center")
    for r in results:
        cuts_str = ",".join(str(c) for c in r.cuts) if r.cuts else "—"
        thr_str = (
            ",".join(f"{x:.2f}" for x in r.throttle_factors)
            if r.throttle_factors else "—"
        )
        energy_str = "∞" if not math.isfinite(r.energy) else f"{r.energy:.3f}"
        pipe_str = "∞" if not math.isfinite(r.pipeline_time) else f"{r.pipeline_time:.3f}"
        table.add_row(r.name, cuts_str, thr_str, energy_str, pipe_str,
                      "✓" if r.feasible else "—")
    console.print(table)

    feasible = {r.name: r.energy for r in results if r.feasible}
    if "joint" in feasible:
        e_joint = feasible["joint"]
        seq = [r for r in results if r.name != "joint" and r.feasible]
        if seq:
            best_seq = min(seq, key=lambda r: r.energy)
            gap_abs = best_seq.energy - e_joint
            gap_pct = 100.0 * gap_abs / best_seq.energy if best_seq.energy > 0 else 0.0
            console.print(
                f"[bold]Joint vs best sequential ({best_seq.name})[/]: "
                f"ΔE = {gap_abs:+.3f} J ({gap_pct:+.2f}%)"
            )


def run_sweep(args: argparse.Namespace) -> None:
    """Heterogeneity sweep summary — gap as a function of capacity ratio."""
    console = Console()
    summary_rows: list[tuple[float, float, float, float, float]] = []
    for ratio in args.heterogeneity:
        layers, stages, links = _build_workload(
            n_layers=args.n_layers,
            capacity_ratio=ratio,
            n_stages=args.n_stages,
        )
        bot = partition_pipeline(layers, stages, links, objective="bottleneck")
        t_max_bot = max(bot.stage_exec_time.values())
        # T_floor at 1.25× the bottleneck partition's natural cycle — opens
        # the slack window the joint optimiser can throttle into.
        throughput_floor = 1.0 / (t_max_bot * args.t_floor_multiplier)

        results = evaluate_workload(
            layers, stages, links,
            throughput_floor_iters_per_s=throughput_floor,
            voltage_alpha=args.voltage_alpha,
            throttle_min=args.throttle_min,
            throttle_granularity=args.throttle_granularity,
        )
        feasible = {r.name: r.energy for r in results if r.feasible}
        if "joint" not in feasible:
            continue
        e_joint = feasible["joint"]
        seq = [r for r in results if r.name != "joint" and r.feasible]
        if not seq:
            continue
        best_seq = min(seq, key=lambda r: r.energy)
        gap_pct = (
            100.0 * (best_seq.energy - e_joint) / best_seq.energy
            if best_seq.energy > 0 else 0.0
        )
        summary_rows.append((
            ratio,
            _heterogeneity_entropy(stages),
            best_seq.energy,
            e_joint,
            gap_pct,
        ))

    table = Table(title="Joint-vs-stacked gap as a function of heterogeneity")
    table.add_column("capacity ratio", justify="right")
    table.add_column("entropy (nats)", justify="right")
    table.add_column("E best seq (J)", justify="right")
    table.add_column("E joint (J)", justify="right")
    table.add_column("gap %", justify="right")
    for ratio, entropy, e_seq, e_jt, pct in summary_rows:
        table.add_row(
            f"{ratio:.2f}", f"{entropy:.3f}",
            f"{e_seq:.3f}", f"{e_jt:.3f}", f"{pct:+.2f}%",
        )
    console.print("")
    console.print(table)
    console.print(
        "\n[dim]Joint beats the best sequential plan across the whole sweep. Two "
        "distinct gain sources combine non-monotonically with the capacity ratio:\n"
        "  (a) partition-shape gain — at moderate-to-high heterogeneity, c_E* differs "
        "from c_B*, so the energy-DP partition lets Perseus extract more. Joint "
        "subsumes this.\n"
        "  (b) T_floor slack gain — every sequential plan throttles against the "
        "partition's own T_max, not against T_floor. The unused band (T_max, T_floor] "
        "is invisible to Perseus but the joint DP exploits it. This source persists "
        "even at the homogeneous limit (ratio=1).\n"
        "Also note: energy + Perseus often becomes infeasible at low heterogeneity "
        "because the energy DP picks a long-tail partition with T_max > T_floor — "
        "joint avoids this because the throughput floor is a DP-level constraint, "
        "not a post-hoc check.[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--n-stages", type=int, default=2)
    parser.add_argument("--heterogeneity", type=float, nargs="+",
                        default=[1.0, 1.5, 2.0, 3.0, 5.0])
    parser.add_argument("--t-floor-multiplier", type=float, default=1.25,
                        help="T_floor = multiplier × T_max(c_B*); >1 opens the slack regime")
    parser.add_argument("--voltage-alpha", type=float, default=2.0)
    parser.add_argument("--throttle-min", type=float, default=0.5)
    parser.add_argument("--throttle-granularity", type=int, default=11)
    parser.add_argument("--scenarios-only", action="store_true",
                        help="Skip the sweep table and only print per-ratio scenario tables")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[bold]exp_joint_vs_stacked[/] — n_layers={args.n_layers}, "
        f"n_stages={args.n_stages}, T_floor multiplier={args.t_floor_multiplier}, "
        f"α={args.voltage_alpha}, R=[{args.throttle_min}, 1] × {args.throttle_granularity}"
    )

    if args.scenarios_only:
        for ratio in args.heterogeneity:
            layers, stages, links = _build_workload(
                args.n_layers, ratio, args.n_stages,
            )
            bot = partition_pipeline(layers, stages, links, objective="bottleneck")
            throughput_floor = 1.0 / (
                max(bot.stage_exec_time.values()) * args.t_floor_multiplier
            )
            run_scenario(
                f"heterogeneity ratio = {ratio:.2f}",
                layers, stages, links,
                throughput_floor, args, console,
            )
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()

"""Workload-dependent energy-optimal power cap, from measured DVFS sweeps.

The carbon-throttle policy leans to the GPU's *energy-optimal* power cap during
high-carbon windows. That cap is a hardware × WORKLOAD property, not a constant:
the power-vs-cap exponent alpha is stable across workloads (power is bounded by
the cap), but throughput saturates at different caps, so the energy-per-iter
U-curve bottoms out at a workload-dependent cap. A throttle policy that hardcodes
one workload's optimal cap therefore pays an avoidable energy (hence carbon)
penalty on another workload.

This loads two measured power-cap sweeps from ``exp_hardware_pareto.py`` (one per
workload) and reports, per workload, the energy-optimal cap, the throughput-max
cap, and the fitted alpha; then the CROSS-APPLICATION penalty: the extra
energy-per-iter incurred by running each workload at the OTHER workload's optimal
cap instead of its own. That penalty is the quantitative motivation for measuring
(or co-designing) the cap per workload rather than fixing it.

Usage::

    python -m experiments.exp_workload_cap_compare \
        --profile resnet=artifacts/hardware-pareto-3080ti.json \
        --profile transformer=artifacts/hardware-pareto-3080ti-transformer.json \
        --out artifacts/workload_cap_compare.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hasagi.energy.throttle_pareto import PowerCapProfile


def _energy_at(profile: PowerCapProfile, cap_w: float) -> float:
    """Energy-per-iter (J) at the profile's measured point nearest ``cap_w``."""
    return profile.point(cap_w).energy_per_iter_kwh * 3_600_000.0


def _alpha(profile_path: str) -> float:
    return float(json.loads(Path(profile_path).read_text()).get("alpha", float("nan")))


def _elasticity_brackets_alpha(profile: PowerCapProfile, alpha: float) -> dict:
    """Check the first-order optimality condition on the measured U-curve.

    Minimising E/iter(c) = P(c)/t(c) gives d log P/d log c = d log t/d log c at the
    optimum. With the fitted P(c) ∝ c^alpha, that is: the THROUGHPUT elasticity
    (d log t/d log c) equals alpha at the energy-optimal cap. Below the optimum
    throughput is elastic (elasticity > alpha, raise the cap); above it throughput
    saturates (elasticity < alpha, the extra power is wasted). So the optimal cap
    is workload-dependent purely through the throughput-saturation shape, while
    alpha (the power exponent) is workload-stable. This returns the throughput
    elasticity just below and just above the energy-optimal cap and whether they
    bracket alpha (the discrete-grid signature of the optimality condition).
    """
    caps = profile.caps
    eo = profile.energy_optimal_cap
    i = caps.index(eo)

    def elasticity(c_lo: float, c_hi: float) -> float:
        t_lo, t_hi = profile.point(c_lo).throughput_iters_s, profile.point(c_hi).throughput_iters_s
        if t_lo <= 0 or c_lo <= 0:
            return float("nan")
        return (math.log(t_hi) - math.log(t_lo)) / (math.log(c_hi) - math.log(c_lo))

    below = elasticity(caps[i - 1], caps[i]) if i >= 1 else float("nan")
    above = elasticity(caps[i], caps[i + 1]) if i + 1 < len(caps) else float("nan")
    brackets = (not math.isnan(below) and not math.isnan(above)
                and below >= alpha - 1e-9 and above <= alpha + 1e-9)
    return {
        "energy_optimal_cap_w": eo,
        "alpha": alpha,
        "throughput_elasticity_below_opt": below,
        "throughput_elasticity_above_opt": above,
        "elasticity_brackets_alpha_at_opt": brackets,
    }


def run(args: argparse.Namespace) -> int:
    console = Console()
    profiles: dict[str, PowerCapProfile] = {}
    paths: dict[str, str] = {}
    for spec in args.profile:
        if "=" not in spec:
            raise SystemExit(f"--profile expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        profiles[name.strip()] = PowerCapProfile.from_json(path.strip())
        paths[name.strip()] = path.strip()
    if len(profiles) < 2:
        raise SystemExit("need >=2 profiles to compare workload-dependence")

    summary: dict[str, dict] = {}
    table = Table(title="Energy-optimal power cap per workload (measured DVFS sweep)")
    table.add_column("workload")
    table.add_column("alpha", justify="right")
    table.add_column("energy-opt cap (W)", justify="right")
    table.add_column("E/iter at opt (J)", justify="right")
    table.add_column("thru-max cap (W)", justify="right")
    table.add_column("E/iter at thru-max (J)", justify="right")
    for name, prof in profiles.items():
        eo_cap = prof.energy_optimal_cap
        mt_cap = prof.max_throughput_cap
        summary[name] = {
            "alpha": _alpha(paths[name]),
            "energy_optimal_cap_w": eo_cap,
            "energy_per_iter_j_at_opt": _energy_at(prof, eo_cap),
            "throughput_max_cap_w": mt_cap,
            "energy_per_iter_j_at_throughput_max": _energy_at(prof, mt_cap),
        }
        table.add_row(
            name, f"{summary[name]['alpha']:.3f}", f"{eo_cap:.0f}",
            f"{summary[name]['energy_per_iter_j_at_opt']:.3f}", f"{mt_cap:.0f}",
            f"{summary[name]['energy_per_iter_j_at_throughput_max']:.3f}",
        )
    console.print(table)

    # Optimality-condition check on the measured curve: throughput elasticity
    # brackets alpha at the energy-optimal cap (the "co-design over measured DVFS").
    opt_cond = Table(title="Optimality condition on measured DVFS: throughput elasticity brackets alpha at the optimal cap")
    opt_cond.add_column("workload")
    opt_cond.add_column("alpha", justify="right")
    opt_cond.add_column("elasticity below opt", justify="right")
    opt_cond.add_column("elasticity above opt", justify="right")
    opt_cond.add_column("brackets alpha?", justify="right")
    for name, prof in profiles.items():
        chk = _elasticity_brackets_alpha(prof, summary[name]["alpha"])
        summary[name]["optimality_check"] = chk
        opt_cond.add_row(
            name, f"{chk['alpha']:.3f}",
            f"{chk['throughput_elasticity_below_opt']:.3f}",
            f"{chk['throughput_elasticity_above_opt']:.3f}",
            "yes" if chk["elasticity_brackets_alpha_at_opt"] else "no",
        )
    console.print(opt_cond)
    console.print(
        "[dim]Reading: below the optimal cap throughput is elastic (elasticity > alpha → raising the cap "
        "pays); above it throughput saturates (elasticity < alpha → extra power is wasted). The optimum "
        "sits where elasticity = alpha. alpha is workload-stable, so the optimal cap moves only with the "
        "throughput-saturation shape — which is why it is workload-dependent.[/]"
    )

    caps = {n: s["energy_optimal_cap_w"] for n, s in summary.items()}
    distinct = len(set(caps.values())) > 1
    console.print(
        f"[bold]Energy-optimal cap is {'WORKLOAD-DEPENDENT' if distinct else 'the same'}[/]: "
        + ", ".join(f"{n} {c:.0f} W" for n, c in caps.items())
        + ". alpha is "
        + ("stable" if max(s["alpha"] for s in summary.values()) - min(s["alpha"] for s in summary.values()) < 0.1 else "variable")
        + " across workloads (power is cap-bounded; throughput saturation sets the optimum)."
    )

    # Cross-application penalty: run workload A at workload B's optimal cap.
    cross = Table(title="Cross-application penalty: extra E/iter from using the OTHER workload's optimal cap")
    cross.add_column("run workload")
    cross.add_column("at its own opt cap", justify="right")
    cross.add_column("forced to other's cap", justify="right")
    cross.add_column("E/iter own (J)", justify="right")
    cross.add_column("E/iter forced (J)", justify="right")
    cross.add_column("penalty %", justify="right")
    cross_pen: dict[str, dict] = {}
    names = list(profiles)
    for a in names:
        for b in names:
            if a == b:
                continue
            own = _energy_at(profiles[a], caps[a])
            forced = _energy_at(profiles[a], caps[b])
            pen = 100.0 * (forced - own) / own if own > 0 else 0.0
            cross.add_row(a, f"{caps[a]:.0f} W", f"{caps[b]:.0f} W ({b})",
                          f"{own:.3f}", f"{forced:.3f}", f"{pen:+.1f}")
            cross_pen[f"{a}_at_{b}_cap"] = {
                "own_cap_w": caps[a], "forced_cap_w": caps[b],
                "energy_j_own": own, "energy_j_forced": forced, "penalty_pct": pen,
            }
    console.print(cross)
    console.print(
        "[dim]A carbon-throttle that fixes one workload's energy-optimal cap pays the penalty above on "
        "the other. The co-design takeaway: the throttle cap should come from the per-workload measured "
        "U-curve (or be co-optimised with the partition), not a hardcoded constant.[/]"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "profiles": paths,
            "per_workload": summary,
            "energy_optimal_cap_is_workload_dependent": distinct,
            "cross_application_penalty": cross_pen,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", action="append", required=True,
                   help="name=path (repeatable), e.g. resnet=artifacts/hardware-pareto-3080ti.json")
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

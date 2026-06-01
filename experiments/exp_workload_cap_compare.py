"""Workload-dependent energy-optimal power cap, from measured DVFS sweeps.

SUPERSEDED FOR THE WORKLOAD-DEPENDENCE CLAIM: this tool compares SINGLE sweeps and
reports an apparent ResNet 200 W vs transformer 250 W split. A rigorous re-measure
(``exp_cap_robustness``: 3 repeats per cap × both cap-orders) found that split is
within run-to-run noise — the optimal plateaus are broad and OVERLAP at ~250 W
(ResNet ~200–250 W, transformer ~250–300 W), so the strong "workload-dependent
optimum" conclusion is withdrawn. The surviving effect is asymmetric LOW-SIDE
sensitivity (under-capping hurts the compute-heavy transformer far more). Read the
cross-application penalty here as "under-capping is costly", and treat
``exp_cap_robustness`` as the authoritative result. Kept for the per-sweep
decomposition detail (eco-lever ratios, the interior-minimum check).


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


def _alpha_binding(profile: PowerCapProfile) -> float:
    """Refit the power exponent ``P ∝ cap^alpha`` over the BINDING caps only —
    rows where the drawn power is at or below the cap. The stored global alpha
    includes sub-floor caps (where power exceeds the request and the model breaks),
    which distorts the slope; this excludes them. Reported only as a workload-
    similarity statistic, not as the optimality threshold."""
    pts = [profile.point(c) for c in profile.caps]
    pts = [p for p in pts if p.avg_power_w > 0 and p.cap_w > 0 and p.avg_power_w <= p.cap_w * 1.02]
    if len(pts) < 2:
        return float("nan")
    p_max = max(p.avg_power_w for p in pts)
    xs = [math.log(p.cap_w / p_max) for p in pts]
    ys = [math.log(p.avg_power_w / p_max) for p in pts]
    den = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys, strict=True)) / den if den > 0 else float("nan")


def _local_elasticity(profile: PowerCapProfile, c_lo: float, c_hi: float, attr: str) -> float:
    """Local log-log elasticity ``d log <attr>/d log cap`` over the segment
    ``[c_lo, c_hi]`` from the measured points (``attr`` is a CapPoint field)."""
    v_lo = getattr(profile.point(c_lo), attr)
    v_hi = getattr(profile.point(c_hi), attr)
    if v_lo <= 0 or c_lo <= 0 or c_hi <= 0 or c_hi == c_lo:
        return float("nan")
    return (math.log(v_hi) - math.log(v_lo)) / (math.log(c_hi) - math.log(c_lo))


def _optimality_check(profile: PowerCapProfile) -> dict:
    """INTERIOR-MINIMUM check on the measured U-curve (NOT an independent law).

    Honesty note: because energy-per-iter is stored as ``E/iter = avg_power/throughput``
    exactly, comparing the throughput elasticity to the local power elasticity on
    the below/above segments is mathematically IDENTICAL to checking that the
    recorded energy-optimal cap is lower than both its grid neighbours, i.e. that
    the minimum is interior to the swept caps. (On an ascending segment,
    ``t_elast >= p_elast`` iff ``E_lo >= E_hi``.) So ``optimum_is_interior`` is a
    restatement of "the argmin is not at a grid boundary", not corroboration of a
    first-order optimality law — it can only fail when the optimum sits at the
    lowest or highest swept cap. The elasticities are reported as descriptive
    detail; do not read them as an independent validation. The genuine
    workload-dependent finding is the throughput-saturation shape that moves the
    U-curve minimum, not this check.
    """
    caps = profile.caps
    eo = profile.energy_optimal_cap
    i = caps.index(eo)
    t_below = _local_elasticity(profile, caps[i - 1], caps[i], "throughput_iters_s") if i >= 1 else float("nan")
    t_above = _local_elasticity(profile, caps[i], caps[i + 1], "throughput_iters_s") if i + 1 < len(caps) else float("nan")
    p_below = _local_elasticity(profile, caps[i - 1], caps[i], "avg_power_w") if i >= 1 else float("nan")
    p_above = _local_elasticity(profile, caps[i], caps[i + 1], "avg_power_w") if i + 1 < len(caps) else float("nan")
    brackets = (not any(math.isnan(x) for x in (t_below, t_above, p_below, p_above))
                and t_below >= p_below - 1e-9 and t_above <= p_above + 1e-9)
    return {
        "energy_optimal_cap_w": eo,
        "throughput_elasticity_below_opt": t_below,
        "throughput_elasticity_above_opt": t_above,
        "power_elasticity_below_opt": p_below,
        "power_elasticity_above_opt": p_above,
        "optimum_is_interior": brackets,
    }


def _plateau(profile: PowerCapProfile, tol_frac: float = 0.02) -> tuple[float, float]:
    """Caps whose energy-per-iter is within ``tol_frac`` of the minimum — the flat
    bottom of the U-curve. A single measurement cannot resolve the optimum within
    this band, so the optimum is reported as the plateau range, not a point."""
    e_min = min(profile.point(c).energy_per_iter_kwh for c in profile.caps)
    flat = [c for c in profile.caps if profile.point(c).energy_per_iter_kwh <= e_min * (1.0 + tol_frac)]
    return (min(flat), max(flat))


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
    table = Table(title="Energy-optimal power cap per workload (single measured DVFS sweep, no repeats)")
    table.add_column("workload")
    table.add_column("alpha (binding rng)", justify="right")
    table.add_column("energy-opt cap (W)", justify="right")
    table.add_column("opt plateau (±2%)", justify="right")
    table.add_column("E/iter at opt (J)", justify="right")
    table.add_column("thru-max cap (W)", justify="right")
    for name, prof in profiles.items():
        eo_cap = prof.energy_optimal_cap
        mt_cap = prof.max_throughput_cap
        plo, phi = _plateau(prof)
        summary[name] = {
            "alpha_binding_fit": _alpha_binding(prof),
            "energy_optimal_cap_w": eo_cap,
            "energy_optimal_plateau_w": [plo, phi],
            "energy_per_iter_j_at_opt": _energy_at(prof, eo_cap),
            "throughput_max_cap_w": mt_cap,
            "energy_per_iter_j_at_throughput_max": _energy_at(prof, mt_cap),
        }
        table.add_row(
            name, f"{summary[name]['alpha_binding_fit']:.2f}", f"{eo_cap:.0f}",
            f"{plo:.0f}-{phi:.0f}", f"{summary[name]['energy_per_iter_j_at_opt']:.3f}", f"{mt_cap:.0f}",
        )
    console.print(table)

    # Interior-minimum check: is the measured energy optimum interior to the swept
    # caps? Because E/iter = avg_power/throughput exactly, the elasticity bracket is
    # mathematically identical to "argmin is not at a grid boundary" — NOT an
    # independent optimality law. Elasticities shown as descriptive detail only.
    opt_cond = Table(title="Interior-minimum check (is the measured optimum inside the swept caps? — elasticities are descriptive only)")
    opt_cond.add_column("workload")
    opt_cond.add_column("t-elast below/above", justify="right")
    opt_cond.add_column("P-elast below/above", justify="right")
    opt_cond.add_column("interior?", justify="right")
    for name, prof in profiles.items():
        chk = _optimality_check(prof)
        summary[name]["interior_minimum_check"] = chk
        opt_cond.add_row(
            name,
            f"{chk['throughput_elasticity_below_opt']:.2f}/{chk['throughput_elasticity_above_opt']:.2f}",
            f"{chk['power_elasticity_below_opt']:.2f}/{chk['power_elasticity_above_opt']:.2f}",
            "yes" if chk["optimum_is_interior"] else "no",
        )
    console.print(opt_cond)
    console.print(
        "[dim]Reading: 'interior' means the measured energy-per-iter minimum is below both its grid "
        "neighbours, so a denser sweep cannot move it outside the bracketing caps. Since E/iter = "
        "power/throughput exactly, this elasticity bracket is identically the interior-argmin test — it "
        "is NOT independent corroboration of an optimality law, only that the optimum is not clipped at a "
        "grid edge.[/]"
    )

    caps = {n: s["energy_optimal_cap_w"] for n, s in summary.items()}
    distinct = len(set(caps.values())) > 1
    console.print(
        f"[bold]Energy-optimal cap is {'WORKLOAD-DEPENDENT' if distinct else 'the same'}[/] across these "
        f"two single-GPU microbenchmarks: "
        + ", ".join(f"{n} {c:.0f} W (plateau {s['energy_optimal_plateau_w'][0]:.0f}-{s['energy_optimal_plateau_w'][1]:.0f})"
                    for (n, c), s in zip(caps.items(), summary.values(), strict=True))
        + ". The optimum moves through each workload's throughput-saturation shape. (The power-vs-cap "
        "slope is ~1 by construction in the cap-binding regime — power tracks the cap — so it carries no "
        "cross-workload information and is not part of this claim.)"
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
        "the other. The forward penalty (forcing the transformer DOWN to the ResNet cap) is large — "
        "~9x the reverse penalty — while the reverse is small and depends on which point of the flat "
        "transformer plateau is taken, so treat it as a lower bound. Both are single-sweep, no repeats: "
        "directional. Co-design takeaway: the throttle cap should come from the per-workload measured "
        "U-curve (or be co-optimised with the partition), not a hardcoded constant.[/]"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "_SUPERSEDED": ("Single-sweep comparison; the workload-dependent-optimum claim did NOT "
                            "survive repeats and is WITHDRAWN. Authoritative: cap_robustness.json "
                            "(3 repeats x both cap-orders). Do not cite the field below or the "
                            "single-sweep cross-application penalty as a live result."),
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

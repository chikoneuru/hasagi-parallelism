"""H5-C policy comparison: HASAGI-threshold vs GREEN-temporal (NSDI'25).

Apples-to-apples comparison of carbon-aware temporal-shift policies on the
parametric 16-zone trace. Four policies are evaluated per zone and seed:

  - **HASAGI-offline** — full-trace median as the reference threshold.
    Requires offline knowledge; only fair against ``green-offline``.
  - **HASAGI-online** — rolling 24 h median as the reference threshold.
    Same information set as ``green-online``; the headline fairness fix.
  - **GREEN-offline** (oracle) — knows the full trace and pauses the
    top-K highest-intensity ticks at the matched pause fraction.
  - **GREEN-online-percentile** — rolling 24 h percentile estimator at
    the matched pause fraction.

To make the comparison fair, the two GREEN flavours are run with a pause
budget matched to the pause fraction the corresponding HASAGI flavour
*emerges with* on each zone. The headline gap is **HASAGI-online minus
GREEN-online**: both see the same rolling 24 h window with no offline
lookahead, so any savings difference reflects the policy decision rule
(median-threshold vs percentile-pause) rather than information advantage.

For each (zone, seed, policy) we report active ticks, pause fraction,
total kWh under the Zeus reference power model (210 W active, 0 W when
paused), and total emissions in grams of CO2. Savings are reported
relative to a ``constant-N=1`` reference that runs every tick.

Usage::

    python -m experiments.exp_h5c_vs_green --zones DE US-CA FR GB NO ZA \\
        --days 14 --seeds 0 1 2 --out artifacts/h5c_vs_green.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.baselines.green import (
    green_offline_optimal_mask,
    green_online_percentile_mask,
    hasagi_threshold_mask,
    hasagi_threshold_online_mask,
    pause_fraction,
)
from hasagi.energy.carbon_trace import published_grid_trace

# Zeus reference single-GPU power model (V100), matches exp_h5c_real_trace.py.
ACTIVE_POWER_W = 210.0


@dataclass(frozen=True)
class PolicyResult:
    zone: str
    seed: int
    policy: str
    active_ticks: int
    pause_fraction: float
    energy_kwh: float
    emissions_g: float


def _evaluate_mask(
    mask: tuple[int, ...],
    intensities: list[float],
    sample_minutes: int,
) -> tuple[float, float]:
    """(energy_kwh, emissions_g) under the Zeus reference model."""
    tick_seconds = sample_minutes * 60.0
    base_kwh = ACTIVE_POWER_W * tick_seconds / 3_600_000.0
    energy_kwh = 0.0
    emissions_g = 0.0
    for m, i in zip(mask, intensities, strict=True):
        if m == 1:
            energy_kwh += base_kwh
            emissions_g += base_kwh * i
    return energy_kwh, emissions_g


def run_zone_seed(
    zone: str,
    seed: int,
    days: int,
    sample_minutes: int,
    hasagi_threshold_multiplier: float,
) -> list[PolicyResult]:
    trace = published_grid_trace(zone, days=days, sample_minutes=sample_minutes, seed=seed)
    intensities = list(trace.intensities)
    n = len(intensities)
    window_ticks = max(1, 24 * 60 // sample_minutes)

    # Reference: all active.
    const_mask = (1,) * n
    const_e, const_em = _evaluate_mask(const_mask, intensities, sample_minutes)

    # HASAGI-offline: full-trace median threshold. Offline information.
    hasagi_off_mask = hasagi_threshold_mask(intensities, hasagi_threshold_multiplier)
    hasagi_off_pf = pause_fraction(hasagi_off_mask)
    hasagi_off_e, hasagi_off_em = _evaluate_mask(hasagi_off_mask, intensities, sample_minutes)

    # HASAGI-online: rolling 24 h median threshold. Same horizon as GREEN-online.
    hasagi_on_mask = hasagi_threshold_online_mask(
        intensities, hasagi_threshold_multiplier, window_size=window_ticks,
    )
    hasagi_on_pf = pause_fraction(hasagi_on_mask)
    hasagi_on_e, hasagi_on_em = _evaluate_mask(hasagi_on_mask, intensities, sample_minutes)

    # GREEN-offline at HASAGI-offline's pause budget (legacy reference; expected
    # to be byte-identical to HASAGI-offline by construction).
    green_off_mask = green_offline_optimal_mask(intensities, pause_fraction=hasagi_off_pf)
    green_off_e, green_off_em = _evaluate_mask(green_off_mask, intensities, sample_minutes)

    # GREEN-online at HASAGI-online's emergent pause budget. The fair head-to-head.
    green_on_mask = green_online_percentile_mask(
        intensities, pause_fraction=hasagi_on_pf, window_size=window_ticks,
    )
    green_on_e, green_on_em = _evaluate_mask(green_on_mask, intensities, sample_minutes)

    return [
        PolicyResult(zone, seed, "constant-N", n, 0.0, const_e, const_em),
        PolicyResult(zone, seed, "hasagi-offline", sum(hasagi_off_mask), hasagi_off_pf,
                     hasagi_off_e, hasagi_off_em),
        PolicyResult(zone, seed, "hasagi-online", sum(hasagi_on_mask), hasagi_on_pf,
                     hasagi_on_e, hasagi_on_em),
        PolicyResult(zone, seed, "green-offline", sum(green_off_mask),
                     pause_fraction(green_off_mask), green_off_e, green_off_em),
        PolicyResult(zone, seed, "green-online", sum(green_on_mask),
                     pause_fraction(green_on_mask), green_on_e, green_on_em),
    ]


def _pct_savings(emissions: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - emissions) / reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zones", nargs="+",
        default=["NO", "FR", "BR", "GB", "US-CA", "DE", "AE", "SG",
                 "VN", "KR", "JP", "AU", "CN", "IN", "PL", "ZA"],
    )
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--hasagi-threshold-multiplier", type=float, default=1.10)
    parser.add_argument("--out", default="artifacts/h5c_vs_green.json")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[bold]H5-C policy comparison[/]: {len(args.zones)} zones × "
        f"{len(args.seeds)} seeds × {args.days} days × {args.sample_minutes}-min cadence; "
        f"HASAGI threshold = median × {args.hasagi_threshold_multiplier}"
    )

    all_results: list[PolicyResult] = []
    for zone in args.zones:
        for seed in args.seeds:
            all_results.extend(run_zone_seed(
                zone, seed, args.days, args.sample_minutes,
                args.hasagi_threshold_multiplier,
            ))

    # Aggregate per (zone, policy): mean savings across seeds.
    by_zone_policy_seed: dict[tuple[str, str, int], PolicyResult] = {
        (r.zone, r.policy, r.seed): r for r in all_results
    }

    def saves_for(zone: str, policy: str) -> list[float]:
        return [
            _pct_savings(
                by_zone_policy_seed[(zone, policy, s)].emissions_g,
                by_zone_policy_seed[(zone, "constant-N", s)].emissions_g,
            )
            for s in args.seeds
        ]

    # Fair head-to-head: HASAGI-online vs GREEN-online.
    table = Table(title="Per-zone savings vs constant-N=1 (3 seeds; HASAGI-online vs GREEN-online is the fair head-to-head)")
    table.add_column("zone")
    table.add_column("HASAGI-on pause%", justify="right")
    table.add_column("HASAGI-on Δ% (μ±σ)", justify="right")
    table.add_column("GREEN-on Δ% (μ±σ)", justify="right")
    table.add_column("HASAGI-off Δ% (μ)", justify="right")
    table.add_column("GREEN-off Δ% (μ)", justify="right")
    table.add_column("HASAGI-on − GREEN-on (pp, μ)", justify="right")

    fair_pp_deltas: list[float] = []
    fair_wins = 0
    for zone in args.zones:
        h_on = saves_for(zone, "hasagi-online")
        g_on = saves_for(zone, "green-online")
        h_off = saves_for(zone, "hasagi-offline")
        g_off = saves_for(zone, "green-offline")
        pp = [h - g for h, g in zip(h_on, g_on, strict=True)]
        mean_pp = statistics.mean(pp)
        fair_pp_deltas.append(mean_pp)
        if mean_pp >= 0:
            fair_wins += 1
        pause_pcts = [
            by_zone_policy_seed[(zone, "hasagi-online", s)].pause_fraction * 100
            for s in args.seeds
        ]

        def fmt(values: list[float]) -> str:
            mu = statistics.mean(values)
            if len(values) >= 2:
                sd = statistics.stdev(values)
                return f"{mu:+.2f}±{sd:.2f}"
            return f"{mu:+.2f}"

        table.add_row(
            zone,
            f"{statistics.mean(pause_pcts):.1f}%",
            fmt(h_on),
            fmt(g_on),
            f"{statistics.mean(h_off):+.2f}",
            f"{statistics.mean(g_off):+.2f}",
            f"{mean_pp:+.2f}",
        )
    console.print(table)

    summary = Table(title="Aggregate across zones (fair head-to-head: HASAGI-online vs GREEN-online)")
    summary.add_column("metric")
    summary.add_column("value", justify="right")

    def grand_mean(policy: str) -> float:
        return statistics.mean(
            v for z in args.zones for v in saves_for(z, policy)
        )

    grand_pp = statistics.mean(fair_pp_deltas)
    summary.add_row("HASAGI-online mean savings", f"{grand_mean('hasagi-online'):+.2f}%")
    summary.add_row("GREEN-online mean savings", f"{grand_mean('green-online'):+.2f}%")
    summary.add_row("HASAGI-offline (uses full trace) mean savings",
                    f"{grand_mean('hasagi-offline'):+.2f}%")
    summary.add_row("GREEN-offline (oracle) mean savings",
                    f"{grand_mean('green-offline'):+.2f}%")
    summary.add_row("HASAGI-online − GREEN-online gap (pp)", f"{grand_pp:+.2f}")
    summary.add_row("zones where HASAGI-online ≥ GREEN-online",
                    f"{fair_wins}/{len(args.zones)}")
    summary.add_row("HASAGI-offline ≡ GREEN-offline (by construction)",
                    "yes — same matched-budget set")
    console.print(summary)

    # Pairwise per-seed gaps for downstream stats.
    per_seed_gaps: dict[str, list[float]] = defaultdict(list)
    for zone in args.zones:
        for s in args.seeds:
            h = _pct_savings(
                by_zone_policy_seed[(zone, "hasagi-online", s)].emissions_g,
                by_zone_policy_seed[(zone, "constant-N", s)].emissions_g,
            )
            g = _pct_savings(
                by_zone_policy_seed[(zone, "green-online", s)].emissions_g,
                by_zone_policy_seed[(zone, "constant-N", s)].emissions_g,
            )
            per_seed_gaps[zone].append(h - g)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "results": [asdict(r) for r in all_results],
        "per_zone_fair_pp_gap_per_seed": dict(per_seed_gaps),
    }, indent=2))
    console.print(f"\n[dim]wrote {out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

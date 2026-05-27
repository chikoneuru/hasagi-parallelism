"""H5-C policy comparison: HISE-threshold vs GREEN-temporal (NSDI'25).

Closes Extension 2 of the pre-paper review: the original
``exp_h5c_real_trace.py`` compares HISE's threshold-pause policy against
``constant-N=1``. That comparison shows the *upper bound* on savings but does
not isolate HISE's contribution from prior carbon-aware schedulers. This
harness adds two GREEN-style temporal-shift baselines:

  - **GREEN-offline-optimal** — oracle that knows the full trace and pauses
    the top-K highest-intensity ticks. Theoretical max savings.
  - **GREEN-online-percentile** — rolling 24h window estimator that decides
    pause/active per tick based on percentile of recent history. Realistic
    online policy mirroring the GREEN paper's deployment.

To make the comparison fair, the two GREEN flavours are run with a pause
budget matched to the pause fraction HISE-threshold *emerges with* on each
zone — so all three policies pause the same total share of ticks per zone.
What differs is *which* ticks they pause: GREEN-offline picks optimally,
GREEN-online picks based on rolling history, HISE-threshold picks based on
the all-time median.

For each (zone, policy) we report:
    - active ticks (count) and pause fraction
    - total kWh assumed (rectangular integration under the Zeus reference
      power model: 210 W active, 0 W when paused)
    - total emissions gCO2 = sum_t (1 - mask_t) * 0 + mask_t * P * tick_seconds * intensity_t

The headline metric is *emissions relative to constant-N=1*; HISE vs GREEN
gap is reported as the second-order quality of the policy.

Usage::

    python -m experiments.exp_h5c_vs_green --zones DE US-CA FR GB NO ZA \\
        --days 14 --out artifacts/h5c_vs_green.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

from rich.console import Console
from rich.table import Table

from experiments.baselines.green import (
    green_offline_optimal_mask,
    green_online_percentile_mask,
    hise_threshold_mask,
    pause_fraction,
)
from hise.energy.carbon_trace import published_grid_trace

# Zeus reference single-GPU power model (V100), matches exp_h5c_real_trace.py.
ACTIVE_POWER_W = 210.0


@dataclass(frozen=True)
class PolicyResult:
    zone: str
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
    """(energy_kwh, emissions_g) under the Zeus reference model.

    energy contribution per active tick = ACTIVE_POWER_W × tick_seconds / 3.6e6
    emission contribution = energy × intensity_at_tick
    """
    tick_seconds = sample_minutes * 60.0
    base_kwh = ACTIVE_POWER_W * tick_seconds / 3_600_000.0
    energy_kwh = 0.0
    emissions_g = 0.0
    for m, i in zip(mask, intensities, strict=True):
        if m == 1:
            energy_kwh += base_kwh
            emissions_g += base_kwh * i
    return energy_kwh, emissions_g


def run_zone(zone: str, days: int, sample_minutes: int, seed: int,
             hise_threshold_multiplier: float) -> list[PolicyResult]:
    trace = published_grid_trace(zone, days=days, sample_minutes=sample_minutes, seed=seed)
    intensities = list(trace.intensities)
    n = len(intensities)

    # Constant-N=1 reference: all active.
    const_mask = (1,) * n
    const_e, const_em = _evaluate_mask(const_mask, intensities, sample_minutes)

    # HISE threshold: pause if intensity > median × multiplier.
    hise_mask = hise_threshold_mask(intensities, hise_threshold_multiplier)
    hise_pf = pause_fraction(hise_mask)
    hise_e, hise_em = _evaluate_mask(hise_mask, intensities, sample_minutes)

    # GREEN-offline: oracle, matched pause budget to HISE.
    green_off_mask = green_offline_optimal_mask(intensities, pause_fraction=hise_pf)
    green_off_e, green_off_em = _evaluate_mask(green_off_mask, intensities, sample_minutes)

    # GREEN-online: rolling 24h window, matched pause budget to HISE.
    # Window is in ticks; for hourly data 24 ticks = 24h.
    green_on_mask = green_online_percentile_mask(
        intensities, pause_fraction=hise_pf, window_size=24 * 60 // sample_minutes,
    )
    green_on_e, green_on_em = _evaluate_mask(green_on_mask, intensities, sample_minutes)

    return [
        PolicyResult(zone, "constant-N", n, 0.0, const_e, const_em),
        PolicyResult(zone, "hise-threshold", sum(hise_mask), hise_pf, hise_e, hise_em),
        PolicyResult(zone, "green-offline", sum(green_off_mask),
                     pause_fraction(green_off_mask), green_off_e, green_off_em),
        PolicyResult(zone, "green-online", sum(green_on_mask),
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hise-threshold-multiplier", type=float, default=1.10)
    parser.add_argument("--out", default="artifacts/h5c_vs_green.json")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[bold]H5-C policy comparison[/]: {len(args.zones)} zones × {args.days} days × "
        f"{args.sample_minutes}-min cadence; HISE threshold = median × {args.hise_threshold_multiplier}"
    )

    all_results: list[PolicyResult] = []
    for zone in args.zones:
        all_results.extend(run_zone(zone, args.days, args.sample_minutes, args.seed,
                                    args.hise_threshold_multiplier))

    # Per-zone table.
    table = Table(title="Per-zone savings vs constant-N=1 (matched pause-budget across HISE/GREEN)")
    table.add_column("zone")
    table.add_column("pause %", justify="right")
    table.add_column("HISE Δ %", justify="right")
    table.add_column("GREEN-online Δ %", justify="right")
    table.add_column("GREEN-offline (oracle) Δ %", justify="right")
    table.add_column("HISE-vs-online (pp)", justify="right")

    by_zone: dict[str, dict[str, PolicyResult]] = {}
    for r in all_results:
        by_zone.setdefault(r.zone, {})[r.policy] = r

    hise_pp_deltas: list[float] = []
    for zone in args.zones:
        z = by_zone[zone]
        const_em = z["constant-N"].emissions_g
        hise_em = z["hise-threshold"].emissions_g
        green_off_em = z["green-offline"].emissions_g
        green_on_em = z["green-online"].emissions_g
        hise_save = _pct_savings(hise_em, const_em)
        green_off_save = _pct_savings(green_off_em, const_em)
        green_on_save = _pct_savings(green_on_em, const_em)
        # Quality delta: HISE-online gap (positive means HISE saves *more* than GREEN-online).
        pp_delta = hise_save - green_on_save
        hise_pp_deltas.append(pp_delta)
        table.add_row(
            zone,
            f"{z['hise-threshold'].pause_fraction * 100:.1f}%",
            f"{hise_save:.2f}%",
            f"{green_on_save:.2f}%",
            f"{green_off_save:.2f}%",
            f"{pp_delta:+.2f}",
        )
    console.print(table)

    # Aggregate roll-up.
    summary = Table(title="Aggregate across zones")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    hise_saves = [
        _pct_savings(by_zone[z]["hise-threshold"].emissions_g, by_zone[z]["constant-N"].emissions_g)
        for z in args.zones
    ]
    green_on_saves = [
        _pct_savings(by_zone[z]["green-online"].emissions_g, by_zone[z]["constant-N"].emissions_g)
        for z in args.zones
    ]
    green_off_saves = [
        _pct_savings(by_zone[z]["green-offline"].emissions_g, by_zone[z]["constant-N"].emissions_g)
        for z in args.zones
    ]
    summary.add_row("HISE mean savings", f"{statistics.mean(hise_saves):.2f}%")
    summary.add_row("GREEN-online mean savings", f"{statistics.mean(green_on_saves):.2f}%")
    summary.add_row("GREEN-offline (oracle) mean savings",
                    f"{statistics.mean(green_off_saves):.2f}%")
    summary.add_row("HISE vs GREEN-online gap (pp)", f"{statistics.mean(hise_pp_deltas):+.2f}")
    summary.add_row("HISE vs GREEN-offline gap (pp)",
                    f"{statistics.mean(hise_saves) - statistics.mean(green_off_saves):+.2f}")
    summary.add_row("zones where HISE ≥ GREEN-online",
                    f"{sum(1 for d in hise_pp_deltas if d >= 0)}/{len(hise_pp_deltas)}")
    console.print(summary)

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "results": [asdict(r) for r in all_results],
    }, indent=2))
    console.print(f"\n[dim]wrote {out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""H5-C policy comparison on REAL (downloaded) carbon-intensity traces.

SCOPE / SUPERSEDED: like ``exp_h5c_vs_green.py``, this is the earlier
DECISION-RULE micro-study (median-threshold vs percentile, both PAUSE) on a flat
per-tick energy model with a single deterministic trace per zone. For carbon
HEADLINE claims use ``exp_realtrace_pareto`` (measured power-cap energy model,
throttle-vs-pause on a shared substrate, GREEN across its full capability range,
diurnal-offset replication, zone-clustered CIs). "HASAGI-online" here is a
pause-threshold rule, not the throttle mechanism.

Mirror of ``exp_h5c_vs_green.py`` (which uses the 16-zone parametric
trace), but routes every policy through a CSV trace loaded via
``load_electricitymaps_csv``. The CSV schema matches the
ElectricityMaps Data Portal export so the existing
``fetch_real_carbon_traces.py`` (Energy-Charts / ENTSO-E mirror) or any
manually-downloaded ElectricityMaps CSV plugs in unchanged.

Reuses the four-allocator pipeline:
    HASAGI-offline, HASAGI-online (rolling 24h), GREEN-offline (oracle),
    GREEN-online (rolling 24h percentile)

Per-zone the pause budget for GREEN is matched to HASAGI-online's
emergent pause fraction so the head-to-head is apples-to-apples (same
information set, same pause count). Reports HASAGI-online minus
GREEN-online as the fair quality gap.

Unlike the parametric harness, this consumer uses a single
deterministic trace per zone (no seed sweep — the data is the data).
The aggregate test is therefore: does the +1.99 pp parametric gap
hold up on the actual recorded grid?

Usage::

    python -m experiments.exp_h5c_real_trace_vs_green \\
        --csv DE=data_cache/real_traces/de_2024-07-01_2024-07-15_hourly.csv \\
              NO=data_cache/real_traces/no_2024-07-01_2024-07-15_hourly.csv \\
        --out artifacts/h5c_real_trace.json
"""
from __future__ import annotations

import argparse
import json
import statistics
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
from hasagi.energy.carbon_trace import load_electricitymaps_csv

ACTIVE_POWER_W = 210.0


@dataclass(frozen=True)
class RealTracePolicyResult:
    zone: str
    policy: str
    n_ticks: int
    active_ticks: int
    pause_fraction: float
    energy_kwh: float
    emissions_g: float


def _evaluate_mask(
    mask: tuple[int, ...],
    intensities: list[float],
    sample_minutes: int,
) -> tuple[float, float]:
    tick_seconds = sample_minutes * 60.0
    base_kwh = ACTIVE_POWER_W * tick_seconds / 3_600_000.0
    energy_kwh = 0.0
    emissions_g = 0.0
    for m, i in zip(mask, intensities, strict=True):
        if m == 1:
            energy_kwh += base_kwh
            emissions_g += base_kwh * i
    return energy_kwh, emissions_g


def _infer_sample_minutes(intensities_n: int, span_seconds: float) -> int:
    """Return the per-tick cadence in minutes inferred from the trace span."""
    if intensities_n < 2 or span_seconds <= 0:
        return 60
    seconds_per_tick = span_seconds / (intensities_n - 1)
    return max(1, int(round(seconds_per_tick / 60.0)))


def run_zone(csv_path: Path, zone: str, hasagi_threshold_multiplier: float) -> list[RealTracePolicyResult]:
    trace = load_electricitymaps_csv(csv_path)
    intensities = list(trace.intensities)
    timestamps = trace.timestamps
    n = len(intensities)
    if n < 2:
        raise ValueError(f"trace {csv_path} has only {n} samples")
    span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    sample_minutes = _infer_sample_minutes(n, span_seconds)
    window_ticks = max(1, 24 * 60 // sample_minutes)

    const_mask = (1,) * n
    const_e, const_em = _evaluate_mask(const_mask, intensities, sample_minutes)

    hasagi_off_mask = hasagi_threshold_mask(intensities, hasagi_threshold_multiplier)
    hasagi_off_pf = pause_fraction(hasagi_off_mask)
    hasagi_off_e, hasagi_off_em = _evaluate_mask(hasagi_off_mask, intensities, sample_minutes)

    hasagi_on_mask = hasagi_threshold_online_mask(
        intensities, hasagi_threshold_multiplier, window_size=window_ticks,
    )
    hasagi_on_pf = pause_fraction(hasagi_on_mask)
    hasagi_on_e, hasagi_on_em = _evaluate_mask(hasagi_on_mask, intensities, sample_minutes)

    green_off_mask = green_offline_optimal_mask(intensities, pause_fraction=hasagi_off_pf)
    green_off_e, green_off_em = _evaluate_mask(green_off_mask, intensities, sample_minutes)

    green_on_mask = green_online_percentile_mask(
        intensities, pause_fraction=hasagi_on_pf, window_size=window_ticks,
    )
    green_on_e, green_on_em = _evaluate_mask(green_on_mask, intensities, sample_minutes)

    return [
        RealTracePolicyResult(zone, "constant-N", n, n, 0.0, const_e, const_em),
        RealTracePolicyResult(zone, "hasagi-offline", n, sum(hasagi_off_mask),
                              hasagi_off_pf, hasagi_off_e, hasagi_off_em),
        RealTracePolicyResult(zone, "hasagi-online", n, sum(hasagi_on_mask),
                              hasagi_on_pf, hasagi_on_e, hasagi_on_em),
        RealTracePolicyResult(zone, "green-offline", n, sum(green_off_mask),
                              pause_fraction(green_off_mask), green_off_e, green_off_em),
        RealTracePolicyResult(zone, "green-online", n, sum(green_on_mask),
                              pause_fraction(green_on_mask), green_on_e, green_on_em),
    ]


def _pct_savings(emissions: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - emissions) / reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="Zone=path pairs, e.g. 'DE=data_cache/real_traces/de_*.csv'",
    )
    parser.add_argument("--hasagi-threshold-multiplier", type=float, default=1.10)
    parser.add_argument("--out", default="artifacts/h5c_real_trace.json")
    args = parser.parse_args()

    pairs: list[tuple[str, Path]] = []
    for spec in args.csv:
        if "=" not in spec:
            raise SystemExit(f"--csv expects Zone=path, got {spec!r}")
        zone, path = spec.split("=", 1)
        pairs.append((zone.strip().upper(), Path(path.strip())))

    console = Console()
    console.print(
        f"[bold]H5-C on real traces[/]: {len(pairs)} zones; "
        f"HASAGI threshold = median × {args.hasagi_threshold_multiplier}"
    )

    all_results: list[RealTracePolicyResult] = []
    for zone, path in pairs:
        if not path.exists():
            raise SystemExit(f"trace file not found: {path}")
        all_results.extend(run_zone(path, zone, args.hasagi_threshold_multiplier))

    by_zone_policy = {(r.zone, r.policy): r for r in all_results}

    table = Table(title="Per-zone savings (real-trace HASAGI-online vs GREEN-online is the head-to-head)")
    table.add_column("zone")
    table.add_column("n ticks", justify="right")
    table.add_column("HASAGI-on pause%", justify="right")
    table.add_column("HASAGI-on Δ%", justify="right")
    table.add_column("GREEN-on Δ%", justify="right")
    table.add_column("HASAGI-off Δ%", justify="right")
    table.add_column("GREEN-off Δ%", justify="right")
    table.add_column("HASAGI-on − GREEN-on (pp)", justify="right")

    pp_gaps: list[float] = []
    wins = 0
    for zone, _ in pairs:
        const_em = by_zone_policy[(zone, "constant-N")].emissions_g
        h_on = _pct_savings(by_zone_policy[(zone, "hasagi-online")].emissions_g, const_em)
        g_on = _pct_savings(by_zone_policy[(zone, "green-online")].emissions_g, const_em)
        h_off = _pct_savings(by_zone_policy[(zone, "hasagi-offline")].emissions_g, const_em)
        g_off = _pct_savings(by_zone_policy[(zone, "green-offline")].emissions_g, const_em)
        pp = h_on - g_on
        pp_gaps.append(pp)
        if pp >= 0:
            wins += 1
        table.add_row(
            zone,
            f"{by_zone_policy[(zone, 'hasagi-online')].n_ticks}",
            f"{by_zone_policy[(zone, 'hasagi-online')].pause_fraction * 100:.1f}%",
            f"{h_on:+.2f}",
            f"{g_on:+.2f}",
            f"{h_off:+.2f}",
            f"{g_off:+.2f}",
            f"{pp:+.2f}",
        )
    console.print(table)

    summary = Table(title="Aggregate across real-trace zones")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("HASAGI-online mean savings",
                    f"{statistics.mean([_pct_savings(by_zone_policy[(z, 'hasagi-online')].emissions_g, by_zone_policy[(z, 'constant-N')].emissions_g) for z, _ in pairs]):+.2f}%")
    summary.add_row("GREEN-online mean savings",
                    f"{statistics.mean([_pct_savings(by_zone_policy[(z, 'green-online')].emissions_g, by_zone_policy[(z, 'constant-N')].emissions_g) for z, _ in pairs]):+.2f}%")
    summary.add_row("HASAGI-online − GREEN-online gap (pp)",
                    f"{statistics.mean(pp_gaps):+.2f}")
    summary.add_row("zones where HASAGI-online ≥ GREEN-online",
                    f"{wins}/{len(pairs)}")
    console.print(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "results": [asdict(r) for r in all_results],
        "pp_gaps_per_zone": dict(zip([z for z, _ in pairs], pp_gaps, strict=True)),
    }, indent=2))
    console.print(f"\n[dim]wrote {out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

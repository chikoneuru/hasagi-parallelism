"""H5-C — carbon-aware scale-to-zero shift over a grid carbon-intensity trace.

Two trace sources, recorded per run in the ``trace_source`` field:

  - ``synthetic-parametric`` (DEFAULT, no ``--csv``): a SYNTHETIC multi-harmonic
    trace from ``hise.energy.carbon_trace.published_grid_trace``. Its per-zone
    mean/amplitude parameters are loosely modelled on published annual grid
    statistics, but the diurnal phase is identical across all zones and the fit
    is NOT validated against source data. Treat these zones as stress-test grid
    SHAPES, not as real measurements.
  - ``real-csv`` (with ``--csv``): a real ElectricityMaps/Energy-Charts CSV
    export. Only DE and NO currently have real CSV fixtures in the repo, so the
    real-trace claim is restricted to those two zones.

The harness is *pure simulation* — it does not poke Knative — so the
carbon-shift wiring can be exercised across regions/thresholds without the K8s
reconfig cost.

For each (zone, threshold-multiplier) cell we report total energy (kWh, Zeus
reference power model), total emissions (gCO2 = energy x intensity_at_tick),
pause fraction, and savings vs the constant-N=1 baseline.

IMPORTANT: artifacts written without ``--csv`` are synthetic and must be
labelled as such in any table or prose; do not present them as real-trace
results. The genuinely real-trace result lives in the single-pair DE+NO run.

Usage::

    # SYNTHETIC stress-test shapes (label as synthetic):
    python -m experiments.exp_h5c_real_trace --zones DE US-CA FR PL --days 7
    # REAL trace (DE or NO only):
    python -m experiments.exp_h5c_real_trace --zones DE --csv /path/to/em_export.csv
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

from rich.console import Console
from rich.table import Table

from hise.energy.carbon_trace import (
    CarbonTrace,
    load_electricitymaps_csv,
    published_grid_trace,
)

# Same Zeus reference power model as ``exp_h5c_carbon_shift.py`` so the two
# harnesses produce comparable kWh numbers.
ACTIVE_POWER_W = 210.0
COLD_START_POWER_W = 130.0
COLD_START_SECONDS = 4.7


@dataclass(frozen=True)
class PolicyRun:
    """Outcome of one (zone, threshold) cell."""

    name: str
    zone: str
    threshold_g_per_kwh: float
    energy_kwh: float
    emissions_g: float
    pause_minutes: float
    cold_starts: int
    # "synthetic-parametric" or "real-csv:<path>" — provenance of the carbon
    # trace this run was computed on. Never drop this from emitted artifacts.
    trace_source: str = "synthetic-parametric"


def _simulate(
    trace: CarbonTrace,
    sample_minutes: float,
    target_fn,
    name: str,
    zone: str,
    threshold: float,
    trace_source: str = "synthetic-parametric",
) -> PolicyRun:
    """Pure-Python replay of the carbon-aware policy over a trace.

    No K8s, no Knative — the policy decides ``target ∈ {0, 1}`` per tick and we
    tally energy + emissions under the same Zeus reference power model the
    on-cluster harness uses.
    """
    energy_kwh = 0.0
    emissions_g = 0.0
    cold_starts = 0
    pause_ticks = 0
    current = 0
    tick_seconds = sample_minutes * 60.0
    for intensity in trace.intensities:
        target = target_fn(intensity)
        cold_start = target > current and current == 0
        if cold_start:
            cold_starts += 1
        if target == 0:
            tick_kwh = (
                COLD_START_POWER_W * COLD_START_SECONDS / 3_600_000.0
                if cold_start else 0.0
            )
            pause_ticks += 1
        else:
            base_kwh = (target * ACTIVE_POWER_W * tick_seconds) / 3_600_000.0
            cold_kwh = (
                COLD_START_POWER_W * COLD_START_SECONDS / 3_600_000.0
                if cold_start else 0.0
            )
            tick_kwh = base_kwh + cold_kwh
        energy_kwh += tick_kwh
        emissions_g += tick_kwh * intensity
        current = target
    return PolicyRun(
        name=name,
        zone=zone,
        threshold_g_per_kwh=threshold,
        energy_kwh=energy_kwh,
        emissions_g=emissions_g,
        pause_minutes=pause_ticks * sample_minutes,
        cold_starts=cold_starts,
        trace_source=trace_source,
    )


def _carbon_aware_target(intensity: float, threshold: float) -> int:
    return 0 if intensity > threshold else 1


def _load_or_generate(
    args: argparse.Namespace, zone: str,
) -> tuple[CarbonTrace, float, str]:
    """Either load the user-supplied CSV or generate the parametric trace.

    Returns ``(trace, sample_minutes, trace_source)``. ``trace_source`` is
    ``"real-csv:<path>"`` when a CSV is supplied, else ``"synthetic-parametric"``.
    """
    if args.csv:
        trace = load_electricitymaps_csv(args.csv)
        source = f"real-csv:{args.csv}"
        if len(trace.timestamps) < 2:
            return trace, args.sample_minutes, source
        gap_s = (trace.timestamps[1] - trace.timestamps[0]).total_seconds()
        return trace, gap_s / 60.0, source
    trace = published_grid_trace(
        zone, days=args.days, sample_minutes=args.sample_minutes, seed=args.seed,
    )
    return trace, args.sample_minutes, "synthetic-parametric"


def run(args: argparse.Namespace) -> int:
    console = Console()
    zones = args.zones or ["DE"]
    all_runs: list[PolicyRun] = []

    real_zones = {"DE", "NO"}
    for zone in zones:
        trace, sample_minutes, trace_source = _load_or_generate(args, zone)
        if trace_source == "synthetic-parametric":
            console.print(
                f"[yellow]NOTE[/] zone {zone}: SYNTHETIC parametric trace "
                f"(not real grid data); label any result as synthetic."
            )
        elif zone not in real_zones:
            console.print(
                f"[yellow]WARN[/] zone {zone}: real-CSV trace supplied but only "
                f"{sorted(real_zones)} have validated real fixtures in-repo."
            )
        median_intensity = statistics.median(trace.intensities)
        threshold = median_intensity * args.threshold_multiplier
        console.print(
            f"[bold]{zone}[/] — {len(trace.intensities)} samples × "
            f"{sample_minutes:g} min, median={median_intensity:.0f} g, "
            f"threshold={threshold:.0f} g (median × {args.threshold_multiplier}) "
            f"[{trace_source}]"
        )
        aware = _simulate(
            trace, sample_minutes,
            lambda intensity, th=threshold: _carbon_aware_target(intensity, th),
            f"{zone}/carbon-aware", zone, threshold, trace_source,
        )
        baseline = _simulate(
            trace, sample_minutes,
            lambda intensity: 1,
            f"{zone}/constant-N", zone, float("nan"), trace_source,
        )
        all_runs.extend([aware, baseline])

    # Summary: paired comparison per zone.
    table = Table(title="H5-C real-trace comparison (carbon-aware vs constant-N=1)")
    table.add_column("zone")
    table.add_column("aware kWh", justify="right")
    table.add_column("const kWh", justify="right")
    table.add_column("Δ kWh", justify="right")
    table.add_column("aware gCO2", justify="right")
    table.add_column("const gCO2", justify="right")
    table.add_column("Δ gCO2 %", justify="right")
    table.add_column("pause min", justify="right")
    table.add_column("cold-starts", justify="right")

    paired: dict[str, tuple[PolicyRun, PolicyRun]] = {}
    for r in all_runs:
        slot = paired.setdefault(r.zone, [None, None])
        if "carbon-aware" in r.name:
            slot[0] = r
        else:
            slot[1] = r

    for zone, (aware, baseline) in paired.items():
        d_kwh = aware.energy_kwh - baseline.energy_kwh
        d_em_pct = (
            100.0 * (aware.emissions_g - baseline.emissions_g) / baseline.emissions_g
            if baseline.emissions_g > 0 else 0.0
        )
        table.add_row(
            zone,
            f"{aware.energy_kwh:.4f}",
            f"{baseline.energy_kwh:.4f}",
            f"{d_kwh:+.4f}",
            f"{aware.emissions_g:.1f}",
            f"{baseline.emissions_g:.1f}",
            f"{d_em_pct:+.2f}%",
            f"{aware.pause_minutes:.0f}",
            str(aware.cold_starts),
        )
    console.print(table)

    if args.out:
        from pathlib import Path
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(r) for r in all_runs], indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zones", nargs="+", default=["DE", "US-CA", "FR", "PL"],
        help="ElectricityMaps zone codes for the parametric trace.",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional: bypass the parametric model and load an EM CSV export.",
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threshold-multiplier", type=float, default=1.10,
        help="Pause when intensity > median × this multiplier.",
    )
    parser.add_argument("--out", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

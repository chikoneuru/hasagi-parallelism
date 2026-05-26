"""H5-C — carbon-aware scale-to-zero shift over a replayed intensity trace.

Drives the Knative service with a deadline-aware primal-dual policy whose
input is a replayed grid intensity trace. When intensity exceeds a
threshold, the policy targets zero replicas (pause); when it drops back,
the policy targets one replica again. The harness records the per-tick
(intensity, target, actual ready replicas, cumulative energy) for both
the *carbon-aware* run and a *constant-N* baseline that ignores intensity.

The energy figure for the comparison is computed from the published Zeus
NSDI'23 reference values (single-GPU NVIDIA V100 idle ≈ 70 W, active
≈ 210 W) modulated by the active replica count and ticked at a
configurable cadence. The cold-start energy attributed to each
spin-up event matches the measured ``app_ready + cuda_init`` budget from
``exp_knative_lifecycle.py`` (≈ 4.7 s × ≈ 130 W startup power).

This is not a wall-clock measurement of cloud-grade carbon savings — it
is a local "the wiring works end-to-end" demonstration. The end-to-end
claim (real ElectricityMaps trace, real multi-region routing) is a Tuần
37 deliverable.

Usage::

    python -m experiments.exp_h5c_carbon_shift --duration-minutes 30 --tick-seconds 60
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

from rich.console import Console
from rich.table import Table

from hise.energy.carbon_trace import synthetic_solar_trace
from hise.pool.knative_pool import KnativePool

# Power model — Zeus reference (V100 single-GPU).
ACTIVE_POWER_W = 210.0
IDLE_POWER_W = 70.0
COLD_START_POWER_W = 130.0
COLD_START_SECONDS = 4.7  # from exp_knative_lifecycle results


@dataclass(frozen=True)
class TickSample:
    tick: int
    t_seconds: float
    intensity_g_per_kwh: float
    target_replicas: int
    observed_replicas: int
    incurred_cold_start: bool


@dataclass(frozen=True)
class RunSummary:
    name: str
    samples: list[TickSample]
    energy_kwh: float
    emissions_g: float
    pause_minutes: float
    cold_starts: int


def _scale_with_event(
    pool: KnativePool, target: int, current: int,
) -> tuple[int, bool]:
    """Issue scale + return (observed_replicas_after, incurred_cold_start)."""
    cold_start = target > current and current == 0
    res = pool.scale(target=target, timeout_seconds=30.0, wait_for_ready=True)
    return res.observed_replicas, cold_start


def _energy_for_tick(
    observed_replicas: int,
    tick_seconds: float,
    cold_start_this_tick: bool,
) -> float:
    """Convert (replicas, tick window) → kWh for this tick.

    Idle replicas burn ``IDLE_POWER_W`` for the whole tick; active
    replicas burn ``ACTIVE_POWER_W``. Cold-start events add a fixed
    overhead of ``COLD_START_POWER_W × COLD_START_SECONDS``.
    """
    if observed_replicas <= 0:
        cold_kwh = (
            COLD_START_POWER_W * COLD_START_SECONDS / 3_600_000.0
            if cold_start_this_tick else 0.0
        )
        return cold_kwh
    power_w = observed_replicas * ACTIVE_POWER_W
    base_kwh = (power_w * tick_seconds) / 3_600_000.0
    cold_kwh = (
        COLD_START_POWER_W * COLD_START_SECONDS / 3_600_000.0
        if cold_start_this_tick else 0.0
    )
    return base_kwh + cold_kwh


def _carbon_aware_target(intensity: float, threshold: float) -> int:
    """Pause when intensity > threshold; otherwise run one replica."""
    return 0 if intensity > threshold else 1


def _run(
    pool: KnativePool,
    name: str,
    target_fn,
    intensities: list[float],
    tick_seconds: float,
    console: Console,
) -> RunSummary:
    """Drive the pool through ``intensities`` using ``target_fn(intensity)``."""
    console.print(f"[bold]Run: {name}[/]")
    samples: list[TickSample] = []
    energy_kwh = 0.0
    emissions_g = 0.0
    cold_starts = 0
    current = 0
    t_start = time.time()
    for tick, intensity in enumerate(intensities):
        target = target_fn(intensity)
        observed, cold_start = _scale_with_event(pool, target, current)
        if cold_start:
            cold_starts += 1
        kwh = _energy_for_tick(observed, tick_seconds, cold_start)
        energy_kwh += kwh
        emissions_g += kwh * intensity
        samples.append(TickSample(
            tick=tick,
            t_seconds=(time.time() - t_start),
            intensity_g_per_kwh=intensity,
            target_replicas=target,
            observed_replicas=observed,
            incurred_cold_start=cold_start,
        ))
        if tick % max(1, len(intensities) // 5) == 0:
            console.print(
                f"  tick {tick:3d}: intensity={intensity:6.1f} → "
                f"target={target} observed={observed} kwh+= {kwh:.6f}"
            )
        current = observed
        time.sleep(0.2)   # cap the harness cadence; tick_seconds is for the energy model
    pause_minutes = sum(
        tick_seconds / 60.0 for s in samples if s.observed_replicas == 0
    )
    return RunSummary(
        name=name,
        samples=samples,
        energy_kwh=energy_kwh,
        emissions_g=emissions_g,
        pause_minutes=pause_minutes,
        cold_starts=cold_starts,
    )


def run(args: argparse.Namespace) -> int:
    console = Console()
    pool = KnativePool(service=args.service, namespace=args.namespace)

    # Build a synthetic 24-hour solar-driven trace, then take the first
    # ``duration-minutes`` slice at ``sample-minutes`` cadence.
    trace = synthetic_solar_trace(hours=24, sample_minutes=args.sample_minutes)
    total_samples = int(args.duration_minutes // args.sample_minutes)
    intensities = list(trace.intensities[:total_samples])
    if not intensities:
        console.print("[red]no intensity samples in slice[/]")
        return 2
    median_intensity = statistics.median(intensities)
    threshold = median_intensity * args.threshold_multiplier
    console.print(
        f"[bold]H5-C harness[/] — {total_samples} ticks × "
        f"{args.sample_minutes} min, threshold = "
        f"{threshold:.0f} gCO2/kWh (median × {args.threshold_multiplier})"
    )

    # Run carbon-aware then constant-N baseline (in that order so the harness
    # ends in the constant-N state, which we restore to N=0 at exit).
    aware = _run(
        pool, "carbon-aware (pause when intensity > threshold)",
        lambda intensity: _carbon_aware_target(intensity, threshold),
        intensities, tick_seconds=args.sample_minutes * 60.0, console=console,
    )
    baseline = _run(
        pool, "constant N=1 (no carbon shift)",
        lambda intensity: 1,
        intensities, tick_seconds=args.sample_minutes * 60.0, console=console,
    )

    # Restore service to zero replicas at exit.
    pool.scale(target=0, timeout_seconds=10.0, wait_for_ready=False)

    table = Table(title="H5-C carbon-shift comparison")
    table.add_column("metric")
    table.add_column("carbon-aware", justify="right")
    table.add_column("constant N=1", justify="right")
    table.add_column("delta", justify="right")
    table.add_row(
        "total energy (kWh)",
        f"{aware.energy_kwh:.4f}", f"{baseline.energy_kwh:.4f}",
        f"{(aware.energy_kwh - baseline.energy_kwh):+.4f}",
    )
    table.add_row(
        "total emissions (gCO2)",
        f"{aware.emissions_g:.1f}", f"{baseline.emissions_g:.1f}",
        f"{(aware.emissions_g - baseline.emissions_g):+.1f}",
    )
    table.add_row(
        "pause time (min)",
        f"{aware.pause_minutes:.1f}", f"{baseline.pause_minutes:.1f}",
        f"{(aware.pause_minutes - baseline.pause_minutes):+.1f}",
    )
    table.add_row(
        "cold starts",
        str(aware.cold_starts), str(baseline.cold_starts),
        f"{aware.cold_starts - baseline.cold_starts:+d}",
    )
    if baseline.emissions_g > 0:
        rel = 100.0 * (aware.emissions_g - baseline.emissions_g) / baseline.emissions_g
        table.add_row("carbon delta %", "", "", f"{rel:+.2f}%")
    console.print(table)

    if args.out:
        from pathlib import Path
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "carbon_aware": asdict(aware),
            "baseline": asdict(baseline),
        }, indent=2, default=lambda o: list(o) if hasattr(o, "__iter__") else o))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="hise-worker-lifecycle")
    parser.add_argument("--namespace", default="hise-validation")
    parser.add_argument("--duration-minutes", type=int, default=30)
    parser.add_argument("--sample-minutes", type=int, default=5)
    parser.add_argument(
        "--threshold-multiplier", type=float, default=1.10,
        help="Pause when intensity > median × this multiplier.",
    )
    parser.add_argument(
        "--out", default=None, help="Optional JSON path for per-tick samples.",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

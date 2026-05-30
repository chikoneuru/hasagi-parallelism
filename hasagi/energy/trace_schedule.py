"""Turn a real carbon trace into schedules for the carbon-policy simulators.

The WS1 Pareto/deferral simulators take a per-window intensity list cycled over
time. This adapts a real ``CarbonTrace`` (e.g. an ElectricityMaps zone export)
into that form, and — crucially for honest statistics — yields multiple
schedules that start at *different* points in the diurnal cycle.

The earlier synthetic traces shared one diurnal phase across every seed, so the
seeds were not independent draws (effective N collapsed). Sampling a real trace
at staggered start offsets gives genuinely different real-day realisations: the
start offset is the replication unit, decoupled from a single fixed phase.
"""
from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hasagi.energy.carbon_trace import (
    _GRID_ZONES,
    CarbonTrace,
    load_electricitymaps_csv,
    published_grid_trace,
)

#: ElectricityMaps zone IDs the project reports on (the agreed 16-zone set).
GRID_ZONE_IDS: tuple[str, ...] = tuple(_GRID_ZONES)


def trace_to_hourly(trace: CarbonTrace) -> list[float]:
    """Resample a trace to one intensity per hour over its full span."""
    n_hours = int(trace.duration_seconds // 3600) + 1
    return [trace.intensity_at(h * 3600.0) for h in range(n_hours)]


def rotate(values: Sequence[float], offset: int) -> list[float]:
    """Cyclically rotate ``values`` by ``offset`` positions (start phase shift)."""
    if not values:
        return []
    k = offset % len(values)
    return list(values[k:]) + list(values[:k])


def quantile_threshold(values: Sequence[float], q: float = 0.5) -> float:
    """The ``q``-quantile of ``values`` — a zone-relative dirty/clean cutoff.

    A zone-relative threshold is necessary because absolute intensities differ
    by an order of magnitude across zones (e.g. ~570 gCO2/kWh in DE vs ~30 in
    NO); a fixed cutoff would mark NO as never-dirty and DE as always-dirty.
    """
    if not values:
        raise ValueError("quantile_threshold needs a non-empty sequence")
    xs = sorted(values)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def diurnal_offsets(n_hours: int, stride_hours: int = 24, span_hours: int = 24) -> list[int]:
    """Start offsets (hours) for staggered samples over an ``n_hours`` trace.

    Each offset begins the job at a different diurnal phase. ``stride_hours``
    controls spacing (24 = one sample per day); ``span_hours`` is how much trace
    a single sample needs after its start, so offsets that would not leave room
    are dropped (the schedule still cycles, but staggering within the real span
    keeps samples close to genuinely-distinct days).
    """
    if n_hours <= 0:
        return []
    last_start = max(0, n_hours - span_hours)
    return list(range(0, last_start + 1, max(1, stride_hours)))


def zone_stats(values: Sequence[float]) -> dict[str, float]:
    """Quick descriptive stats for a zone's intensity series (for reporting)."""
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "swing": max(values) - min(values),
    }


@dataclass(frozen=True)
class ZoneTrace:
    """A zone's carbon trace plus where it came from.

    ``source`` is ``"real-csv"`` when a dropped-in ElectricityMaps export was
    found for the zone, else ``"synthetic-parametric"`` (the labelled fallback).
    Cross-zone significance should be computed over the real zones only.
    """

    zone: str
    trace: CarbonTrace
    source: str
    csv_path: str | None


def find_zone_csv(real_dir: str | Path, zone: str) -> Path | None:
    """First CSV in ``real_dir`` whose name starts with ``<zone-id>_`` (case-insensitive).

    Drop ElectricityMaps zone exports in named like ``de_2024-...csv``,
    ``us-ca_2024-...csv`` — the zone id (lower-cased, hyphens kept) then an
    underscore. Returns ``None`` if no match.
    """
    d = Path(real_dir)
    if not d.is_dir():
        return None
    prefix = zone.lower() + "_"
    for f in sorted(d.glob("*.csv")):
        if f.name.lower().startswith(prefix):
            return f
    return None


def load_zone_traces(
    real_dir: str | Path,
    zones: Sequence[str] = GRID_ZONE_IDS,
    *,
    synthetic_days: int = 14,
) -> dict[str, ZoneTrace]:
    """Load each zone's trace: a dropped-in real CSV if present, else synthetic.

    This is the multi-zone drop-in: place real ElectricityMaps CSVs in
    ``real_dir`` (one per zone, named ``<zone>_*.csv``) and those zones load as
    ``real-csv``; the rest fall back to the labelled ``synthetic-parametric``
    generator. As more real CSVs are added, more zones flip to real and the
    cross-zone (clustered) inference over real zones strengthens.
    """
    out: dict[str, ZoneTrace] = {}
    for z in zones:
        csv = find_zone_csv(real_dir, z)
        if csv is not None:
            out[z] = ZoneTrace(z, load_electricitymaps_csv(csv), "real-csv", str(csv))
        else:
            out[z] = ZoneTrace(
                z, published_grid_trace(z, days=synthetic_days), "synthetic-parametric", None,
            )
    return out

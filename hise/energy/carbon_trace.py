"""Carbon intensity trace replay — secondary, proxy-only input to the control loop.

**Status**: carbon is a *derived* metric in HISE, not a measured one. Use these traces to
report carbon proxy alongside the primary kWh measurement; never as the sole control signal.
See research-note.md §2.3 for the measurement methodology and uncertainty bounds.

Two ways to build a ``CarbonTrace``:

* ``load_csv_trace`` — read a CSV with columns ``timestamp_iso, intensity_g_per_kwh``. The
  ElectricityMaps "history" export already matches this schema. WattTime exports need a
  light pre-conversion (their column is ``MOER`` in lbs/MWh, divide by 2.205 → gCO2/kWh).
* ``synthetic_solar_trace`` — deterministic 24h cycle for offline experiments / unit tests.

For paper-grade reporting, evaluate **all three** of ElectricityMaps, WattTime, and the
IEA static grid-mix table in parallel — disagreement >20% between sources should be
reported explicitly rather than cherry-picked.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class CarbonTrace:
    """Carbon intensity time-series with linear interpolation between samples."""

    timestamps: list[datetime]
    intensities: list[float]   # gCO2 / kWh

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.intensities) or not self.timestamps:
            raise ValueError("trace requires non-empty, aligned timestamps and intensities")
        # Pre-compute seconds-since-start for fast lookups.
        self._t0 = self.timestamps[0]
        self._seconds = [(ts - self._t0).total_seconds() for ts in self.timestamps]

    @property
    def duration_seconds(self) -> float:
        return self._seconds[-1]

    def intensity_at(self, seconds_from_start: float) -> float:
        """Linearly interpolate intensity at ``seconds_from_start``; clamp at endpoints."""
        if seconds_from_start <= 0:
            return self.intensities[0]
        if seconds_from_start >= self._seconds[-1]:
            return self.intensities[-1]
        # Binary search.
        lo, hi = 0, len(self._seconds) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._seconds[mid] <= seconds_from_start:
                lo = mid
            else:
                hi = mid
        t0, t1 = self._seconds[lo], self._seconds[hi]
        i0, i1 = self.intensities[lo], self.intensities[hi]
        if t1 == t0:
            return i0
        frac = (seconds_from_start - t0) / (t1 - t0)
        return i0 + frac * (i1 - i0)

    def forecast(self, horizon_seconds: float, step_seconds: float = 300.0) -> Iterable[tuple[float, float]]:
        """Yield ``(seconds_from_now, intensity)`` samples over ``[0, horizon_seconds]``.

        Real deployments should plug in WattTime / ElectricityMaps forecast endpoint here.
        """
        t = 0.0
        while t <= horizon_seconds:
            yield t, self.intensity_at(t)
            t += step_seconds


def load_csv_trace(path: str | Path, *, ts_col: str = "timestamp", val_col: str = "intensity_g_per_kwh") -> CarbonTrace:
    rows = []
    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = datetime.fromisoformat(row[ts_col])
            val = float(row[val_col])
            rows.append((ts, val))
    rows.sort(key=lambda r: r[0])
    return CarbonTrace(
        timestamps=[r[0] for r in rows],
        intensities=[r[1] for r in rows],
    )


def synthetic_solar_trace(
    hours: int = 24,
    *,
    sample_minutes: int = 5,
    baseline_g_per_kwh: float = 450.0,
    solar_swing_g_per_kwh: float = 250.0,
    start: datetime | None = None,
) -> CarbonTrace:
    """A simple daily cycle: lowest at noon (solar peak), highest around 19:00 (evening peak).

    Used by ``traces/synthetic_solar.csv`` and the smoke test so we have a deterministic
    fixture for tests / CI.
    """
    start = start or datetime(2026, 5, 19, 0, 0, 0)
    timestamps: list[datetime] = []
    intensities: list[float] = []
    n_samples = hours * 60 // sample_minutes
    for k in range(n_samples + 1):
        ts = start + timedelta(minutes=k * sample_minutes)
        hour_of_day = (ts - start).total_seconds() / 3600.0
        # Sinusoid: minimum at hour=12, maximum at hour=0 / 24.
        phase = math.cos((hour_of_day / 24.0) * 2 * math.pi)
        intensity = baseline_g_per_kwh + solar_swing_g_per_kwh * phase
        timestamps.append(ts)
        intensities.append(intensity)
    return CarbonTrace(timestamps=timestamps, intensities=intensities)

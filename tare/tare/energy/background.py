"""Time-varying background power for marginal attribution on a shared GPU.

A single up-front background calibration cannot track a co-tenant whose load
drifts over minutes — which is exactly what biased the first deferral run
(active power read 107 W vs 46 W for the same workload). The fix is to anchor
the background at every moment our job is *off* the GPU — before and after a run
(brackets) and during every pause (the device draw then *is* the background) —
and interpolate between those anchors, then re-integrate the recorded device
power trace against this time-varying baseline.

This makes the comparison robust to slow drift without needing exclusive GPU
access; only absolute per-process attribution during long unbroken active spans
(no anchors inside them) still relies on interpolation.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field


@dataclass
class BackgroundModel:
    """Piecewise-linear background power (W) from timestamped anchor samples.

    Anchors ``(t_s, watts)`` are observed when our job is not on the GPU. Between
    anchors the background is linearly interpolated; outside the anchor span it
    is held flat at the nearest anchor. With one anchor it is constant (the old
    fixed-background behaviour); with none it is zero.
    """

    anchors: list[tuple[float, float]] = field(default_factory=list)

    def add(self, t_s: float, watts: float) -> None:
        """Record a background observation, keeping anchors time-sorted."""
        self.anchors.append((t_s, watts))
        self.anchors.sort(key=lambda a: a[0])

    def at(self, t_s: float) -> float:
        """Interpolated background power at time ``t_s``."""
        if not self.anchors:
            return 0.0
        if len(self.anchors) == 1:
            return self.anchors[0][1]
        if t_s <= self.anchors[0][0]:
            return self.anchors[0][1]
        if t_s >= self.anchors[-1][0]:
            return self.anchors[-1][1]
        times = [a[0] for a in self.anchors]
        i = bisect.bisect_right(times, t_s)
        t0, w0 = self.anchors[i - 1]
        t1, w1 = self.anchors[i]
        if t1 == t0:
            return w1
        return w0 + (w1 - w0) * (t_s - t0) / (t1 - t0)

    @property
    def mean_w(self) -> float:
        return sum(w for _, w in self.anchors) / len(self.anchors) if self.anchors else 0.0

    @property
    def drift_w(self) -> float:
        """Peak-to-peak spread of the anchors — how much the background moved."""
        if not self.anchors:
            return 0.0
        ws = [w for _, w in self.anchors]
        return max(ws) - min(ws)


def marginal_kwh_from_trace(
    trace: list[tuple[float, float]],
    background: BackgroundModel,
    t_start: float,
    t_end: float,
) -> float:
    """Integrate ``max(0, P_device(t) − background(t))`` over ``[t_start, t_end]``.

    ``trace`` is a time-sorted list of ``(t_s, device_watts)`` samples. Each
    consecutive pair defines a slice; the slice's overlap with ``[t_start, t_end]``
    contributes ``max(0, device_w − background(t)) × overlap_seconds`` using the
    device power at the slice's right endpoint (matching the online meter's
    left-to-right integration). Returns kWh.
    """
    if t_end <= t_start or len(trace) < 2:
        return 0.0
    total_kwh = 0.0
    for (t_prev, _p_prev), (t_cur, p_cur) in zip(trace, trace[1:], strict=False):
        lo = max(t_prev, t_start)
        hi = min(t_cur, t_end)
        if hi <= lo:
            continue
        marginal_w = max(0.0, p_cur - background.at(t_cur))
        total_kwh += marginal_w * (hi - lo) / 3_600_000.0
    return total_kwh

"""Carbon-throttle vs pause vs always-on, from a measured power-cap profile.

Power-capping a GPU is a *device-wide* setting, so it cannot be exercised live on
a shared card without throttling the co-tenant. But the per-cap behaviour
(throughput, average power, energy-per-iteration) was already measured on this
exact GPU as a power-cap sweep. This module feeds that measured profile into a
forward simulation of three policies over a clean/dirty grid schedule, so the
throttle-vs-pause-vs-deadline trade-off is grounded in real per-cap data without
any live capping:

  - always-on : run at the throughput-maximising cap through every window.
  - pause     : run at full cap in clean windows; scale to zero in dirty windows
                and defer that work, paying a resume cost on each wake-up. Idle
                power during the pause is billed per freed-GPU regime.
  - throttle  : run at full cap in clean windows; drop to a low (energy-leaner)
                cap in dirty windows and keep training — no pause, no resume.

Each policy completes the same total iterations; the output is total carbon and
makespan, i.e. the carbon-vs-latency Pareto.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CapPoint:
    """Measured behaviour at one power cap."""

    cap_w: float
    throughput_iters_s: float
    avg_power_w: float
    energy_per_iter_kwh: float


@dataclass
class PowerCapProfile:
    """A measured power-cap sweep for one GPU + workload."""

    gpu_name: str
    points: dict[float, CapPoint]

    @classmethod
    def from_json(cls, path: str | Path) -> PowerCapProfile:
        data = json.loads(Path(path).read_text())
        points: dict[float, CapPoint] = {}
        for r in data["rows"]:
            cap = float(r["cap_w_observed"])
            points[cap] = CapPoint(
                cap_w=cap,
                throughput_iters_s=float(r["throughput_iters_per_s"]),
                avg_power_w=float(r["avg_power_w"]),
                energy_per_iter_kwh=float(r["energy_per_iter_kwh"]),
            )
        return cls(gpu_name=str(data.get("gpu_name", "?")), points=points)

    @property
    def caps(self) -> list[float]:
        return sorted(self.points)

    @property
    def max_throughput_cap(self) -> float:
        return max(self.points.values(), key=lambda p: p.throughput_iters_s).cap_w

    @property
    def energy_optimal_cap(self) -> float:
        """Cap minimising energy-per-iteration (the U-curve minimum)."""
        return min(self.points.values(), key=lambda p: p.energy_per_iter_kwh).cap_w

    def point(self, cap_w: float) -> CapPoint:
        """Nearest measured cap point to ``cap_w``."""
        return self.points[min(self.points, key=lambda c: abs(c - cap_w))]


@dataclass(frozen=True)
class PolicyResult:
    name: str
    total_carbon_g: float
    makespan_s: float
    iters: int
    active_energy_kwh: float
    idle_carbon_g: float
    resume_carbon_g: float


def simulate_policy(
    profile: PowerCapProfile,
    *,
    name: str,
    total_iters: int,
    window_s: float,
    schedule_g: list[float],
    clean_cap_w: float,
    dirty_cap_w: float | None,
    threshold_g: float,
    resume_energy_kwh: float = 0.0,
    idle_power_w: float = 0.0,
    max_windows: int = 100_000,
) -> PolicyResult:
    """Forward-simulate one policy to ``total_iters`` over a cycled grid schedule.

    ``dirty_cap_w`` is the cap used while intensity exceeds ``threshold_g``;
    pass ``None`` to PAUSE in dirty windows instead (deferring the work). Clean
    windows always run at ``clean_cap_w``. ``window_s`` is each window's wall
    duration; ``schedule_g`` is the per-window intensity, cycled.

    Carbon is ``energy_kwh × intensity`` per window. A pause bills
    ``idle_power_w`` over the window at that window's intensity, and a resume
    cost on the first active window after any pause.
    """
    iters = 0
    carbon_g = 0.0
    active_kwh = 0.0
    idle_carbon_g = 0.0
    resume_carbon_g = 0.0
    makespan_s = 0.0
    paused_last = False
    w = 0
    while iters < total_iters and w < max_windows:
        intensity = schedule_g[w % len(schedule_g)]
        w += 1
        dirty = intensity > threshold_g
        if dirty and dirty_cap_w is None:
            # Pause: defer this window's work; bill idle power for the regime.
            idle_kwh = idle_power_w * window_s / 3_600_000.0
            idle_carbon_g += idle_kwh * intensity
            carbon_g += idle_kwh * intensity
            makespan_s += window_s
            paused_last = True
            continue
        cap = dirty_cap_w if (dirty and dirty_cap_w is not None) else clean_cap_w
        pt = profile.point(cap)
        if paused_last and resume_energy_kwh > 0.0:
            resume_carbon_g += resume_energy_kwh * intensity
            carbon_g += resume_energy_kwh * intensity
        paused_last = False
        # Train at this cap for the window, or less if we finish mid-window.
        iters_full = pt.throughput_iters_s * window_s
        iters_this = min(iters_full, total_iters - iters)
        wall_this = iters_this / pt.throughput_iters_s if pt.throughput_iters_s > 0 else window_s
        e_kwh = iters_this * pt.energy_per_iter_kwh
        active_kwh += e_kwh
        carbon_g += e_kwh * intensity
        iters += int(round(iters_this))
        makespan_s += wall_this
    return PolicyResult(
        name=name, total_carbon_g=carbon_g, makespan_s=makespan_s, iters=iters,
        active_energy_kwh=active_kwh, idle_carbon_g=idle_carbon_g,
        resume_carbon_g=resume_carbon_g,
    )


def simulate_masked_policy(
    profile: PowerCapProfile,
    *,
    name: str,
    total_iters: int,
    window_s: float,
    schedule_g: list[float],
    active_mask: Sequence[int],
    full_cap_w: float,
    off_cap_w: float | None = None,
    resume_energy_kwh: float = 0.0,
    idle_power_w: float = 0.0,
    max_windows: int = 100_000,
) -> PolicyResult:
    """Forward-simulate one policy whose per-window response is a precomputed mask.

    Unlike :func:`simulate_policy` (which decides dirty/clean from a scalar
    ``threshold_g``), the response here is dictated by ``active_mask`` — a 0/1
    series aligned to ``schedule_g`` and cycled with it. A ``1`` runs the window
    at ``full_cap_w``. A ``0`` is the carbon response: PAUSE (defer the work, pay
    ``resume_energy_kwh`` on the next active window and bill ``idle_power_w`` over
    the window) when ``off_cap_w is None``, or THROTTLE to ``off_cap_w`` and keep
    training otherwise.

    This lets an external decision rule — e.g. a rolling-window percentile mask
    that uses no lookahead — drive the same measured-energy substrate as the
    threshold policies. Holding the energy model and a given mask fixed, a
    head-to-head between two responses on that mask reflects the pause-vs-throttle
    difference; it does not by itself disentangle that difference from the
    cap-efficiency a throttle banks on the off-windows (see the carbon-signal
    decomposition in the real-trace experiment for that).
    """
    if not active_mask:
        raise ValueError("active_mask must be non-empty")
    iters = 0
    carbon_g = 0.0
    active_kwh = 0.0
    idle_carbon_g = 0.0
    resume_carbon_g = 0.0
    makespan_s = 0.0
    paused_last = False
    w = 0
    while iters < total_iters and w < max_windows:
        intensity = schedule_g[w % len(schedule_g)]
        active = active_mask[w % len(active_mask)]
        w += 1
        if not active and off_cap_w is None:
            # Pause: defer this window's work; bill idle power for the regime.
            idle_kwh = idle_power_w * window_s / 3_600_000.0
            idle_carbon_g += idle_kwh * intensity
            carbon_g += idle_kwh * intensity
            makespan_s += window_s
            paused_last = True
            continue
        cap = off_cap_w if (not active and off_cap_w is not None) else full_cap_w
        pt = profile.point(cap)
        if paused_last and resume_energy_kwh > 0.0:
            resume_carbon_g += resume_energy_kwh * intensity
            carbon_g += resume_energy_kwh * intensity
        paused_last = False
        iters_full = pt.throughput_iters_s * window_s
        iters_this = min(iters_full, total_iters - iters)
        wall_this = iters_this / pt.throughput_iters_s if pt.throughput_iters_s > 0 else window_s
        e_kwh = iters_this * pt.energy_per_iter_kwh
        active_kwh += e_kwh
        carbon_g += e_kwh * intensity
        iters += int(round(iters_this))
        makespan_s += wall_this
    return PolicyResult(
        name=name, total_carbon_g=carbon_g, makespan_s=makespan_s, iters=iters,
        active_energy_kwh=active_kwh, idle_carbon_g=idle_carbon_g,
        resume_carbon_g=resume_carbon_g,
    )


def pareto(
    profile: PowerCapProfile,
    *,
    total_iters: int,
    window_s: float,
    schedule_g: list[float],
    threshold_g: float,
    dirty_throttle_cap_w: float | None = None,
    resume_energy_kwh: float = 0.0,
    idle_power_w: float = 0.0,
) -> dict[str, PolicyResult]:
    """always-on / pause / throttle on the same job and schedule.

    ``dirty_throttle_cap_w`` defaults to the profile's energy-optimal cap.
    ``idle_power_w`` is the pause regime's idle floor (0 for reallocated/
    powered-down, the card idle floor for dedicated).
    """
    full = profile.max_throughput_cap
    throttle_cap = dirty_throttle_cap_w if dirty_throttle_cap_w is not None else profile.energy_optimal_cap
    common = dict(
        total_iters=total_iters, window_s=window_s, schedule_g=schedule_g,
        clean_cap_w=full, threshold_g=threshold_g,
    )
    eco_common = dict(common)
    eco_common["clean_cap_w"] = throttle_cap   # energy-optimal cap in EVERY window
    return {
        "always_on": simulate_policy(
            profile, name="always-on", dirty_cap_w=full, **common,
        ),
        # Carbon-BLIND efficiency baseline: run at the energy-optimal cap always.
        # The gap between this and always-on is the U-curve efficiency win that a
        # carbon-unaware operator already gets; the gap between this and
        # throttle/pause is the part attributable to carbon-AWARENESS.
        "always_on_eco": simulate_policy(
            profile, name=f"always-on@{throttle_cap:.0f}W(eco)", dirty_cap_w=throttle_cap,
            **eco_common,
        ),
        "pause": simulate_policy(
            profile, name="pause", dirty_cap_w=None,
            resume_energy_kwh=resume_energy_kwh, idle_power_w=idle_power_w, **common,
        ),
        "throttle": simulate_policy(
            profile, name=f"throttle@{throttle_cap:.0f}W", dirty_cap_w=throttle_cap, **common,
        ),
    }

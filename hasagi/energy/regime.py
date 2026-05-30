"""What happens to the freed GPU decides whether carbon-aware pause wins.

WS0 measured a paused job on a *shared* card and (with marginal attribution)
found its idle cost is ~0 — because the co-tenant draws the idle either way.
But that is only one of three regimes for the GPU a paused job releases:

  - ``dedicated``    — the card stays ours and idles. We are billed its idle
                       floor for the whole pause, at the (dirty) pause-window
                       intensity. Pause often loses here.
  - ``reallocated``  — the freed card runs another tenant's work. Our job is
                       billed ~0 idle (the co-tenant pays). This is the shared
                       GPU we are actually on.
  - ``powered_down`` — the card is released / spun down. ~0 idle, plus an
                       optional one-off spin-down/up energy.

The measured marginal ledger is regime-independent for the active and cold-start
phases (that is our job's own draw). The regimes differ only in how the *idle*
phases are charged. ``regime_carbon`` recomputes the total under a given regime;
``break_even_window_s`` gives the dirty-window length beyond which pausing beats
riding through, for that regime.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from hasagi.energy.pod_ledger import PHASE_IDLE, LedgerReport


class GpuRegime(str, enum.Enum):
    DEDICATED = "dedicated"
    REALLOCATED = "reallocated"
    POWERED_DOWN = "powered_down"


@dataclass(frozen=True)
class RegimeCarbon:
    """Total carbon for a run under one freed-GPU regime.

    ``measured_carbon_g`` is the regime-independent part (our job's active +
    cold-start marginal carbon). ``idle_carbon_g`` is what the regime charges
    for the pause windows. ``total_carbon_g`` is their sum.
    """

    regime: GpuRegime
    total_carbon_g: float
    measured_carbon_g: float
    idle_carbon_g: float


def regime_carbon(
    report: LedgerReport,
    regime: GpuRegime,
    *,
    dedicated_idle_w: float = 0.0,
    spin_down_kwh: float = 0.0,
) -> RegimeCarbon:
    """Recompute a measured marginal ledger's total carbon under ``regime``.

    Args:
        report: the measured (marginal) ledger.
        regime: which freed-GPU regime to bill the idle phases under.
        dedicated_idle_w: the card's idle-floor power, charged across every idle
            interval only in the ``dedicated`` regime (measure on a clean GPU, or
            use a labelled spec value).
        spin_down_kwh: one-off energy charged once per idle interval in the
            ``powered_down`` regime (spin-down + later spin-up). Billed at the
            idle interval's intensity.
    """
    measured_non_idle = sum(
        iv.carbon_g for iv in report.intervals if iv.phase != PHASE_IDLE
    )
    idle_carbon = 0.0
    for iv in report.intervals:
        if iv.phase != PHASE_IDLE:
            continue
        intensity = iv.intensity_g_per_kwh or 0.0
        if regime is GpuRegime.DEDICATED:
            idle_kwh = dedicated_idle_w * iv.duration_s / 3_600_000.0
            idle_carbon += idle_kwh * intensity
        elif regime is GpuRegime.POWERED_DOWN:
            idle_carbon += spin_down_kwh * intensity
        # REALLOCATED: idle charged to the co-tenant → 0.
    return RegimeCarbon(
        regime=regime,
        total_carbon_g=measured_non_idle + idle_carbon,
        measured_carbon_g=measured_non_idle,
        idle_carbon_g=idle_carbon,
    )


def regime_breakdown(
    report: LedgerReport,
    *,
    dedicated_idle_w: float = 0.0,
    spin_down_kwh: float = 0.0,
) -> dict[str, RegimeCarbon]:
    """All three regimes' totals for one measured marginal ledger."""
    return {
        r.value: regime_carbon(
            report, r, dedicated_idle_w=dedicated_idle_w, spin_down_kwh=spin_down_kwh,
        )
        for r in GpuRegime
    }


def break_even_window_s(
    *,
    active_power_w: float,
    intensity_dirty: float,
    intensity_clean: float,
    resume_energy_kwh: float,
    idle_power_w: float = 0.0,
    resume_intensity: float | None = None,
) -> float:
    """Dirty-window length T* beyond which pause+resume beats ride-through.

    A fixed chunk of work that would run during a dirty window of length ``T``
    costs, if ridden through, ``E_active(T) · I_dirty``. If instead paused, that
    work is deferred to clean time (``E_active(T) · I_clean``), the idle card is
    billed for the window (``E_idle(T) · I_dirty``), and a one-off resume cost is
    paid (``resume_energy_kwh · I_resume``). Setting the two equal and solving
    for ``T``::

        T* = resume · I_resume · 3.6e6
             ----------------------------------------------------
             active · (I_dirty − I_clean)  −  idle · I_dirty

    Returns ``+inf`` when the denominator is ≤ 0 — pausing never wins for that
    regime (e.g. dedicated idle so costly it outweighs the carbon arbitrage).
    Units: power in W, intensity in gCO2/kWh, energy in kWh, result in seconds.
    """
    i_resume = intensity_clean if resume_intensity is None else resume_intensity
    denom = active_power_w * (intensity_dirty - intensity_clean) - idle_power_w * intensity_dirty
    if denom <= 0.0:
        return math.inf
    return resume_energy_kwh * i_resume * 3_600_000.0 / denom

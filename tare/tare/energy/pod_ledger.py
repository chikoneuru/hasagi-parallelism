"""Attribute *measured* GPU energy to a serverless pod's lifecycle phases.

The serverless training carbon ledger needs to answer: when a job is paused on
a high-carbon hour and later resumed, how much energy (and therefore carbon) did
the *resume* actually cost? Stateless-function carbon papers never pay this,
because a function has no optimiser state, no CUDA context, and no first-iter
warmup to rebuild. A training job does.

This module records lifecycle transitions and attributes the host NVML energy
stream to the phase that was active between consecutive transitions:

  - ``cold_start`` — from a resume request to the first useful training
    iteration. Folds in pod schedule latency, container start, ``app_ready``,
    CUDA-context init, checkpoint + optimiser-state reload, and first-iteration
    warmup. This is the *training-specific resume cost*.
  - ``active``    — useful training between cold-start completion and the next
    pause.
  - ``idle``      — wall-clock while the job is paused / scaled to zero. On a
    dedicated GPU this is near-zero incremental energy; on a shared GPU it is
    whatever the card draws while our job holds no work.

The ledger is deliberately decoupled from NVML: it reads cumulative energy
through a ``energy_kwh_fn`` callable and time through a ``clock`` callable, so it
is unit-testable on CPU. ``nvml_cumulative_kwh_fn`` wires it to a live
``NvmlTelemetrySource`` in production.

Carbon for a phase is ``energy_kwh × intensity_g_per_kwh`` evaluated at the
intensity recorded when the phase began — carbon is a strict derivative of the
measured energy, never an independent quantity.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from tare.energy.background import BackgroundModel, marginal_kwh_from_trace

PHASE_COLD_START = "cold_start"
PHASE_ACTIVE = "active"
PHASE_IDLE = "idle"

# Internal sentinel phase used only to seal the final open interval at report().
_END = "__end__"


class _SupportsSnapshot(Protocol):
    def snapshot(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class LedgerMark:
    """One recorded lifecycle transition.

    ``cumulative_kwh`` is the meter's running integral at this instant; energy
    for the interval *starting* here is the next mark's reading minus this one.
    ``intensity_g_per_kwh`` is the grid intensity in force during this interval.
    """

    phase: str
    t_s: float
    cumulative_kwh: float
    intensity_g_per_kwh: float | None


@dataclass(frozen=True)
class PhaseInterval:
    """A measured slice of one lifecycle phase."""

    phase: str
    start_s: float
    end_s: float
    energy_kwh: float
    intensity_g_per_kwh: float | None
    carbon_g: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class LedgerReport:
    """Per-phase and aggregate attribution of measured energy to carbon."""

    intervals: tuple[PhaseInterval, ...]
    total_energy_kwh: float
    total_carbon_g: float
    energy_by_phase_kwh: dict[str, float]
    carbon_by_phase_g: dict[str, float]
    duration_by_phase_s: dict[str, float]
    phase_counts: dict[str, int]

    @property
    def resume_energy_kwh(self) -> float:
        """Total energy charged to cold-start / resume across the run."""
        return self.energy_by_phase_kwh.get(PHASE_COLD_START, 0.0)

    @property
    def resume_carbon_g(self) -> float:
        """Total carbon charged to cold-start / resume across the run."""
        return self.carbon_by_phase_g.get(PHASE_COLD_START, 0.0)

    @property
    def active_energy_kwh(self) -> float:
        return self.energy_by_phase_kwh.get(PHASE_ACTIVE, 0.0)

    @property
    def cold_starts(self) -> int:
        """Number of resume events (cold-start intervals) in the run."""
        return self.phase_counts.get(PHASE_COLD_START, 0)


@dataclass
class PodEnergyLedger:
    """Accumulate lifecycle marks; attribute measured energy to phases.

    Args:
        energy_kwh_fn: zero-arg callable returning the meter's *cumulative* kWh
            now (monotonic non-decreasing). In production wire to
            ``nvml_cumulative_kwh_fn(source)``.
        clock: zero-arg monotonic clock; defaults to ``time.monotonic``.

    Usage::

        ledger = PodEnergyLedger(nvml_cumulative_kwh_fn(src))
        ledger.mark(PHASE_COLD_START, intensity)   # resume requested
        # ... pod cold-starts, checkpoint reloads, first iter warms up ...
        ledger.mark(PHASE_ACTIVE, intensity)       # first useful iteration
        # ... train ...
        ledger.mark(PHASE_IDLE, intensity)         # pause / scale-to-zero
        report = ledger.report()
    """

    energy_kwh_fn: Callable[[], float]
    clock: Callable[[], float] = time.monotonic
    _marks: list[LedgerMark] = field(default_factory=list, init=False, repr=False)

    def mark(
        self,
        phase: str,
        intensity_g_per_kwh: float | None = None,
        *,
        t_s: float | None = None,
        cumulative_kwh: float | None = None,
    ) -> LedgerMark:
        """Record the start of ``phase``.

        ``t_s`` / ``cumulative_kwh`` override the clock / meter reads (used by
        tests and by replays from logged timestamps).
        """
        t = self.clock() if t_s is None else t_s
        e = self.energy_kwh_fn() if cumulative_kwh is None else cumulative_kwh
        m = LedgerMark(phase, t, e, intensity_g_per_kwh)
        self._marks.append(m)
        return m

    def report(
        self,
        *,
        t_s: float | None = None,
        cumulative_kwh: float | None = None,
    ) -> LedgerReport:
        """Seal the trailing phase with a terminal read and attribute energy.

        The final open interval runs from the last mark to *now* (or the
        supplied ``t_s`` / ``cumulative_kwh``), so callers never need a closing
        mark of their own.
        """
        if not self._marks:
            return self._assemble([])

        t = self.clock() if t_s is None else t_s
        e = self.energy_kwh_fn() if cumulative_kwh is None else cumulative_kwh
        sealed = [*self._marks, LedgerMark(_END, t, e, None)]
        raw: list[tuple[str, float, float, float | None, float]] = []
        for cur, nxt in zip(sealed, sealed[1:], strict=False):
            # Cumulative energy is monotonic; clamp guards meter jitter / resets.
            energy = max(0.0, nxt.cumulative_kwh - cur.cumulative_kwh)
            raw.append((cur.phase, cur.t_s, nxt.t_s, cur.intensity_g_per_kwh, energy))
        return self._assemble(raw)

    def report_from_trace(
        self,
        trace: list[tuple[float, float]],
        background: BackgroundModel,
        *,
        t_end: float | None = None,
    ) -> LedgerReport:
        """Re-attribute from a recorded power trace and a time-varying background.

        Uses the marks' timestamps (not their cumulative reads) as phase
        boundaries and integrates ``max(0, P_device − background(t))`` over each
        interval. This corrects for a co-tenant whose load drifts during the run,
        which a single fixed background cannot. The trailing phase runs to
        ``t_end`` (or the last trace timestamp).
        """
        if not self._marks:
            return self._assemble([])
        if t_end is None:
            t_end = trace[-1][0] if trace else self._marks[-1].t_s
        boundaries = [*self._marks, LedgerMark(_END, t_end, 0.0, None)]
        raw: list[tuple[str, float, float, float | None, float]] = []
        for cur, nxt in zip(boundaries, boundaries[1:], strict=False):
            energy = marginal_kwh_from_trace(trace, background, cur.t_s, nxt.t_s)
            raw.append((cur.phase, cur.t_s, nxt.t_s, cur.intensity_g_per_kwh, energy))
        return self._assemble(raw)

    @staticmethod
    def _assemble(
        raw: list[tuple[str, float, float, float | None, float]],
    ) -> LedgerReport:
        """Build a ``LedgerReport`` from ``(phase, start_s, end_s, intensity, energy_kwh)`` rows."""
        intervals: list[PhaseInterval] = []
        energy_by_phase: dict[str, float] = {}
        carbon_by_phase: dict[str, float] = {}
        duration_by_phase: dict[str, float] = {}
        phase_counts: dict[str, int] = {}
        total_energy = 0.0
        total_carbon = 0.0
        for phase, start_s, end_s, intensity, energy in raw:
            carbon = energy * intensity if intensity is not None else 0.0
            intervals.append(PhaseInterval(
                phase=phase, start_s=start_s, end_s=end_s, energy_kwh=energy,
                intensity_g_per_kwh=intensity, carbon_g=carbon,
            ))
            energy_by_phase[phase] = energy_by_phase.get(phase, 0.0) + energy
            carbon_by_phase[phase] = carbon_by_phase.get(phase, 0.0) + carbon
            duration_by_phase[phase] = duration_by_phase.get(phase, 0.0) + (end_s - start_s)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            total_energy += energy
            total_carbon += carbon
        return LedgerReport(
            intervals=tuple(intervals),
            total_energy_kwh=total_energy,
            total_carbon_g=total_carbon,
            energy_by_phase_kwh=energy_by_phase,
            carbon_by_phase_g=carbon_by_phase,
            duration_by_phase_s=duration_by_phase,
            phase_counts=phase_counts,
        )


def nvml_cumulative_kwh_fn(source: _SupportsSnapshot) -> Callable[[], float]:
    """Build a cumulative-kWh reader summed across a telemetry source's workers.

    Works with any source exposing ``snapshot() -> {worker_id: WorkerTelemetry}``
    (the real ``NvmlTelemetrySource`` or a fake). Summing across workers gives
    the whole-node energy for a single- or multi-GPU job.
    """

    def _fn() -> float:
        return sum(
            getattr(t, "energy_cumulative_kwh", 0.0)
            for t in source.snapshot().values()
        )

    return _fn

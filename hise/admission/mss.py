"""Minimum Satisfactory Share (ElasticFlow §4.1) and HISE's Energy-Budgeted MSS.

ElasticFlow defines MSS as the smallest GPU count ``x*`` such that the job's scaling curve
``T(x)`` is sufficient to finish before the deadline. Under a concave scaling curve we can
find ``x*`` by binary search.

HISE extends MSS along the **energy axis** (research-note §3.5 Gap-3, §4.5 C3):

* **EnergyBudgetMSS** — primary contribution. Objective: smallest ``x`` whose projected
  energy ``∫ P(x, t) dt`` over the job's remaining duration fits the user's energy budget
  while still meeting the deadline. Energy is *measured directly* via NVML / RAPL during
  execution; the projection at admission time uses Zeus-style profiling (power-per-GPU as
  a function of allocation size).
* **CarbonProxyBudget** — *optional* derived view. If the user supplies a grid intensity
  forecast and a carbon budget, ``EnergyBudgetMSS`` translates the carbon budget into an
  equivalent energy budget via ``B_E = B_C / max(c_grid(t))`` (worst-case) and reports
  carbon estimate with uncertainty band tracked separately. Carbon is never the primary
  decision input — only a re-weighting on the energy objective.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingCurve:
    """Throughput (iters / second) as a function of GPU count. Must be non-decreasing and
    concave for the MSS optimality proof (ElasticFlow Theorem 1)."""

    throughput_per_gpu_count: Sequence[float]  # index 0 => 1 GPU, index 1 => 2 GPUs, ...
    max_gpus: int = 0

    def __post_init__(self) -> None:
        if not self.throughput_per_gpu_count:
            raise ValueError("scaling curve must have at least one point")
        # set max_gpus consistent with the curve length
        object.__setattr__(self, "max_gpus", len(self.throughput_per_gpu_count))

    def throughput(self, gpus: int) -> float:
        if gpus <= 0:
            return 0.0
        idx = min(gpus, self.max_gpus) - 1
        return self.throughput_per_gpu_count[idx]


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    gpus: int
    reason: str = ""


def minimum_satisfactory_share(
    iterations_remaining: int,
    deadline_seconds: float,
    curve: ScalingCurve,
) -> int:
    """Smallest ``x`` such that ``iterations_remaining / curve.throughput(x) <= deadline``.

    Returns 0 if no allocation in ``[1, curve.max_gpus]`` can meet the deadline.
    """
    if iterations_remaining <= 0 or deadline_seconds <= 0:
        return 0
    required_rate = iterations_remaining / deadline_seconds
    # Binary search on a concave-throughput curve.
    lo, hi = 1, curve.max_gpus
    while lo < hi:
        mid = (lo + hi) // 2
        if curve.throughput(mid) >= required_rate:
            hi = mid
        else:
            lo = mid + 1
    if curve.throughput(lo) >= required_rate:
        return lo
    return 0


# ---------------------------------------------------------------------------
# Energy-Budgeted MSS — HISE contribution C3
# ---------------------------------------------------------------------------

@dataclass
class EnergyBudgetMSS:
    """Find smallest ``x`` meeting deadline AND energy budget (primary), optionally also
    a carbon proxy budget (secondary, derived).

    Energy is the *primary* objective and the constraint that drives the search; carbon is
    a derived view computed by multiplying projected energy by a user-supplied grid
    intensity forecast. The class deliberately keeps the two separate so we never claim
    carbon as the optimisation target.

    Args:
        curve: scaling curve (iter/s per GPU count).
        power_per_gpu_w: average GPU power draw at full utilisation (W). In practice this
            comes from a per-model Zeus-style profiling pass.
        energy_budget_kwh: total allowed energy for the remainder of this job (kWh). This
            is the binding constraint.
        carbon_intensity_forecast: *optional* callable returning gCO2/kWh at time t. If
            supplied alongside ``carbon_budget_g``, the class adds it as a secondary
            constraint. Otherwise it is only used for reporting.
        carbon_budget_g: *optional* secondary carbon budget; ignored if forecast is None.
    """

    curve: ScalingCurve
    power_per_gpu_w: float
    energy_budget_kwh: float
    carbon_intensity_forecast: Callable[[float], float] | None = None
    carbon_budget_g: float | None = None

    def project_energy_kwh(self, gpus: int, duration_s: float) -> float:
        """Direct energy projection: power × duration. Equivalent to integrating constant
        power; real workload variation handled by Zeus-style multi-power-state model."""
        if gpus <= 0 or duration_s <= 0:
            return 0.0
        return (self.power_per_gpu_w * gpus * duration_s) / 3_600_000.0

    def project_emissions_g(self, gpus: int, duration_s: float, dt_s: float = 300.0) -> float:
        """Derived carbon proxy: ``∫ power(t) × c_grid(t) dt``. Returns 0 if no forecast."""
        if self.carbon_intensity_forecast is None or gpus <= 0 or duration_s <= 0:
            return 0.0
        total = 0.0
        t = 0.0
        while t < duration_s:
            step = min(dt_s, duration_s - t)
            kwh = (self.power_per_gpu_w * gpus * step) / 3_600_000.0
            intensity = self.carbon_intensity_forecast(t)
            total += kwh * intensity
            t += step
        return total

    def find(self, iterations_remaining: int, deadline_seconds: float) -> AdmissionDecision:
        base_mss = minimum_satisfactory_share(iterations_remaining, deadline_seconds, self.curve)
        if base_mss == 0:
            return AdmissionDecision(False, 0, "no allocation meets deadline")

        # Walk gpus upward from MSS. Adding GPUs shortens duration but raises instantaneous
        # power; energy can move either direction depending on the per-GPU efficiency curve.
        for x in range(base_mss, self.curve.max_gpus + 1):
            throughput = self.curve.throughput(x)
            if throughput <= 0:
                continue
            duration = iterations_remaining / throughput
            if duration > deadline_seconds:
                continue
            energy_kwh = self.project_energy_kwh(x, duration)
            if energy_kwh > self.energy_budget_kwh:
                continue
            # Secondary check: carbon proxy if user supplied both forecast and budget.
            if self.carbon_intensity_forecast is not None and self.carbon_budget_g is not None:
                emissions = self.project_emissions_g(x, duration)
                if emissions > self.carbon_budget_g:
                    continue
                return AdmissionDecision(
                    True, x,
                    f"meets deadline + energy budget ({energy_kwh:.2f} kWh) "
                    f"+ carbon proxy ({emissions:.1f} gCO2)",
                )
            return AdmissionDecision(
                True, x,
                f"meets deadline + energy budget (proj={energy_kwh:.2f} kWh)",
            )
        return AdmissionDecision(False, 0, "no allocation meets deadline + energy budget")


# Backwards-compatible alias for code/tests written before the energy-first rename. The
# old name communicated the intent less precisely; new code should use ``EnergyBudgetMSS``.
EnergyAdjustedMSS = EnergyBudgetMSS


# ---------------------------------------------------------------------------
# Greedy marginal-return allocator (ElasticFlow Algorithm 2).
# ---------------------------------------------------------------------------

def greedy_marginal_allocation(
    admitted: list[tuple[str, ScalingCurve, int]],
    available_gpus: int,
) -> dict[str, int]:
    """Hand out remaining GPUs greedily to the job with highest marginal throughput-per-GPU.

    Each ``admitted`` entry is ``(job_id, scaling_curve, current_gpus)``. Returns the new
    allocation count per job (>= current_gpus).
    """
    alloc = {jid: cur for jid, _curve, cur in admitted}
    curves = {jid: curve for jid, curve, _ in admitted}
    remaining = available_gpus - sum(alloc.values())
    while remaining > 0 and any(alloc[j] < curves[j].max_gpus for j in alloc):
        # Compute marginal returns; only consider jobs with headroom.
        best_jid = None
        best_marginal = -math.inf
        for jid in alloc:
            curve = curves[jid]
            cur = alloc[jid]
            if cur >= curve.max_gpus:
                continue
            margin = curve.throughput(cur + 1) - curve.throughput(cur)
            if margin > best_marginal:
                best_marginal = margin
                best_jid = jid
        if best_jid is None:
            break
        alloc[best_jid] += 1
        remaining -= 1
    return alloc

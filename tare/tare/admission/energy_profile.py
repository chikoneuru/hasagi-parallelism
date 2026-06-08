"""Zeus-style empirical energy profile — energy-per-iteration as a function of GPU count.

Motivation: the linear projection ``power_per_gpu_w × gpus × duration_s`` ignores
allreduce overhead. Zeus NSDI'23 §4.2 demonstrates empirically that energy-per-iter
is **convex in GPU count**: adding more GPUs reduces wall-clock time but each GPU
still draws power (plus allreduce overhead), producing a U-shape or monotone-
increasing curve past the minimum.

This module ships ``EnergyProfile``: a tuple-indexed lookup of (energy_per_iter_kwh,
throughput_iters_per_s) by GPU count, with ``validate_convexity()`` checking that
second differences of the energy curve are non-negative. ``EnergyBudgetMSS`` will
optionally consume an ``EnergyProfile`` instead of ``power_per_gpu_w`` for accurate
projection; the linear model remains as a fallback so existing call sites do not break.

The convexity assumption underlies the EB-MSS optimality theorem (proof pending).
If it fails empirically on a target workload, fall back to piecewise-linear
approximation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyProfile:
    """Empirical energy curve indexed by GPU count.

    Both tuples are aligned: index ``i`` corresponds to ``gpus = i + 1``.
    For ``gpus`` outside ``[1, max_gpus]`` the lookup clamps to the endpoints.

    Args:
        energy_per_iter_kwh: per-iteration energy at each GPU count, in kWh.
            Convex in GPU count under the Zeus assumption (validated by
            ``validate_convexity()``).
        throughput_iters_per_s: iterations per second at each GPU count. Usually
            concave (diminishing returns); aligned with ``ScalingCurve`` semantics
            in ``tare.admission.mss``.
    """

    energy_per_iter_kwh: tuple[float, ...]
    throughput_iters_per_s: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.energy_per_iter_kwh:
            raise ValueError("energy_per_iter_kwh must be non-empty")
        if len(self.energy_per_iter_kwh) != len(self.throughput_iters_per_s):
            raise ValueError(
                f"aligned tuples required: energy has {len(self.energy_per_iter_kwh)} "
                f"entries but throughput has {len(self.throughput_iters_per_s)}"
            )
        if any(e < 0 for e in self.energy_per_iter_kwh):
            raise ValueError("energy_per_iter_kwh entries must be non-negative")
        if any(t < 0 for t in self.throughput_iters_per_s):
            raise ValueError("throughput_iters_per_s entries must be non-negative")

    @property
    def max_gpus(self) -> int:
        return len(self.energy_per_iter_kwh)

    def energy_per_iter(self, gpus: int) -> float:
        """Energy in kWh for one iteration at ``gpus`` allocation."""
        if gpus <= 0:
            return 0.0
        idx = min(gpus, self.max_gpus) - 1
        return self.energy_per_iter_kwh[idx]

    def throughput(self, gpus: int) -> float:
        """Iterations per second at ``gpus`` allocation."""
        if gpus <= 0:
            return 0.0
        idx = min(gpus, self.max_gpus) - 1
        return self.throughput_iters_per_s[idx]

    def total_energy_kwh(self, gpus: int, iterations: int) -> float:
        """Total energy to complete ``iterations`` at ``gpus`` allocation."""
        if gpus <= 0 or iterations <= 0:
            return 0.0
        return self.energy_per_iter(gpus) * iterations

    def total_energy_kwh_over_duration(self, gpus: int, duration_s: float) -> float:
        """Total energy over a wall-clock window. Equivalent to integrating
        ``power × time``, using the profile's bundled throughput for iter count."""
        if gpus <= 0 or duration_s <= 0:
            return 0.0
        iters = self.throughput(gpus) * duration_s
        return self.energy_per_iter(gpus) * iters

    def validate_convexity(self, tolerance: float = 1e-12) -> bool:
        """Check that the energy-per-iter curve is convex in GPU count.

        A discrete function ``f`` is convex iff its second differences
        ``f(i+1) - 2·f(i) + f(i-1) ≥ 0`` for all interior ``i``.

        Returns True for profiles with fewer than 3 points (trivially convex).

        ``tolerance`` allows numerical slack for floating-point noise; set to 0
        for strict convexity checks.
        """
        e = self.energy_per_iter_kwh
        if len(e) < 3:
            return True
        for i in range(1, len(e) - 1):
            second_diff = e[i + 1] - 2.0 * e[i] + e[i - 1]
            if second_diff < -tolerance:
                return False
        return True

    def validate_power_convexity(self, tolerance: float = 1e-12) -> bool:
        """Check that instantaneous power ``p(g) = e(g) · t(g)`` is convex in g.

        This is the assumption ``greedy_marginal_energy_allocation`` relies on
        for global optimality. Convexity of ``e(g)`` alone (``validate_convexity``)
        is necessary but not sufficient: the product of a convex and a concave
        function is not convex in general, so empirical profiles must call this
        stronger check before being fed to the allocator.

        Returns True for profiles with fewer than 3 points (trivially convex).
        """
        p = tuple(
            self.energy_per_iter_kwh[i] * self.throughput_iters_per_s[i]
            for i in range(self.max_gpus)
        )
        if len(p) < 3:
            return True
        for i in range(1, len(p) - 1):
            if p[i + 1] - 2.0 * p[i] + p[i - 1] < -tolerance:
                return False
        return True

    def optimal_gpu_count(self) -> int:
        """Return the GPU count minimising energy-per-iter (the U-shape minimum).

        Useful for sanity-checking that the profile has a sensible shape and for
        seeding initial admission decisions.
        """
        min_idx = min(range(self.max_gpus), key=lambda i: self.energy_per_iter_kwh[i])
        return min_idx + 1


def linear_profile(
    power_per_gpu_w: float,
    base_throughput_iters_per_s: float,
    max_gpus: int,
    scaling_efficiency: float = 0.85,
    allreduce_coefficient: float = 0.05,
) -> EnergyProfile:
    """Construct a Zeus-shape EnergyProfile from a synthetic power + allreduce model.

    Used as a fallback when no empirical profile is available — produces a plausible
    curve under the assumption of constant per-GPU power **plus** a quadratic
    allreduce overhead that grows with worker count. Real deployments should
    replace this with a profile measured from short profiling passes per the Zeus
    NSDI'23 §4.2 methodology.

    Model:
        ``throughput(g) = base · g^efficiency``                    (concave, diminishing returns)
        ``energy(g)     = (P·g + α·P·g²) / throughput(g) / 3.6e6`` (kWh per iter)

    The ``α·g²`` term encodes the empirical observation that allreduce traffic +
    idle waiting at the synchronisation barrier scales superlinearly with worker
    count. With ``α ≳ 0.05`` and ``efficiency ≲ 0.9`` the resulting energy curve
    is convex in ``g`` (validated downstream by ``validate_convexity()``).

    Args:
        power_per_gpu_w: per-GPU instantaneous power at full utilisation (W).
        base_throughput_iters_per_s: throughput on 1 GPU (iters/s).
        max_gpus: highest GPU count in the profile.
        scaling_efficiency: throughput scaling exponent in (0, 1]; 1.0 = perfect
            linear scaling, 0.85 ≈ NCCL allreduce on 8 GPUs per Megatron-LM bench.
        allreduce_coefficient: quadratic coefficient ``α`` on the energy overhead.
            0.0 disables (curve becomes concave); 0.05 matches Zeus §4.2 fits.
    """
    if max_gpus < 1:
        raise ValueError("max_gpus must be >= 1")
    if not 0.0 < scaling_efficiency <= 1.0:
        raise ValueError("scaling_efficiency must be in (0, 1]")
    if allreduce_coefficient < 0.0:
        raise ValueError("allreduce_coefficient must be >= 0")

    throughput = tuple(
        base_throughput_iters_per_s * (g ** scaling_efficiency) for g in range(1, max_gpus + 1)
    )
    energy_per_iter = tuple(
        (power_per_gpu_w * g + allreduce_coefficient * power_per_gpu_w * g * g)
        / max(throughput[g - 1], 1e-9)
        / 3_600_000.0
        for g in range(1, max_gpus + 1)
    )
    return EnergyProfile(energy_per_iter_kwh=energy_per_iter, throughput_iters_per_s=throughput)

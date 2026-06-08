"""Perseus port (Chung et al., NSDI'24) — pipeline-energy throttling baseline.

Perseus observes that in a steady-state pipeline-parallel training schedule
the bottleneck stage's exec time ``T_max`` determines the pipeline rate; every
non-bottleneck stage has slack ``T_max - T_s`` it spends idling. Idle GPUs still
draw power, so this is *energy bloat*. Perseus closes the bloat by reducing the
freq (and proportionally the power) of fast stages so their exec time stretches
to ``T_max`` — same pipeline throughput, lower energy.

Quantitative model (voltage-frequency scaling, common DVFS approximation):

    throttle_factor r_s = T_s / T_max,        r_s ∈ (0, 1]
    P_s' = P_s · r_s^α,                       α ≈ 2 (quadratic) … 3 (cubic)
    T_s' = T_max
    E_s_per_iter = P_s' · T_max               (no idle period left)

vs the bloated baseline where stage s computes for T_s then idles for T_max-T_s
at some baseline idle power ``P_idle_s``:

    E_s_per_iter_bloated = P_s · T_s + P_idle_s · (T_max - T_s)

The port operates over a HASAGI ``Partition`` + ``StageSpec`` set so it can be
applied to any partition produced by ``tare.parallel.partition_pipeline``.

Reference:
    Chung, Lyu, Choi, Park, Kim, Yang, Chowdhury, "Perseus: Reducing Energy
    Bloat in Large Model Training," NSDI 2024.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tare.parallel.partitioner import Partition, StageSpec


@dataclass(frozen=True)
class PerseusPlan:
    """Per-stage throttle factors and projected energy savings vs the bloated baseline."""

    throttle_factors: dict[int, float]          # stage_id → r_s in (0, 1]
    bottleneck_time_s: float                    # T_max
    baseline_energy_per_iter_kwh: float         # bloated (no throttling)
    throttled_energy_per_iter_kwh: float        # Perseus-throttled
    savings_kwh: float                          # baseline − throttled (always ≥ 0)
    savings_pct: float                          # 100 × savings / baseline


def perseus_throttle(
    partition: Partition,
    stages: Sequence[StageSpec],
    *,
    voltage_alpha: float = 2.0,
    idle_power_fraction: float = 0.3,
) -> PerseusPlan:
    """Compute the Perseus throttling plan for a pipeline partition.

    Args:
        partition: ``Partition`` from ``tare.parallel.partition_pipeline``.
        stages: ``StageSpec`` per stage (must align by ``stage_id`` with ``partition``).
        voltage_alpha: power-frequency scaling exponent. 2.0 ≈ quadratic (V²
            constant); 3.0 ≈ cubic (V² scales with f). Real NVIDIA DVFS is in
            between (~2.3); 2.0 is a conservative lower-bound on savings.
        idle_power_fraction: fraction of `P_s` drawn when the stage is idle in
            the bloated baseline. Empirical: ~30% (A100 idle ≈ 80 W of 300 W
            cap). 0.0 zeroes out idle and removes the bloat-baseline term;
            1.0 assumes idle draws the full power (pessimistic baseline).

    Returns:
        ``PerseusPlan`` with per-stage throttle factors and energy delta.

    Constraints:
        - Stages with infinite / non-finite ``stage_exec_time`` are passed
          through with throttle_factor=1.0 (infeasible — no savings).
        - The bottleneck stage gets ``r_s = 1.0`` (cannot throttle further).
        - When all stages tie for the bottleneck (perfectly balanced
          partition), every ``r_s = 1.0`` and savings are zero.
    """
    if not 0.0 <= idle_power_fraction <= 1.0:
        raise ValueError(f"idle_power_fraction must be in [0, 1], got {idle_power_fraction}")
    if voltage_alpha <= 0:
        raise ValueError(f"voltage_alpha must be > 0, got {voltage_alpha}")

    stage_lookup = {s.stage_id: s for s in stages}
    if any(sid not in stage_lookup for sid in partition.stage_exec_time):
        raise ValueError("StageSpec set does not cover all stages in the partition.")

    valid_times = {
        sid: t for sid, t in partition.stage_exec_time.items() if math.isfinite(t) and t > 0
    }
    if not valid_times:
        return PerseusPlan(
            throttle_factors={sid: 1.0 for sid in partition.stage_exec_time},
            bottleneck_time_s=math.inf,
            baseline_energy_per_iter_kwh=math.inf,
            throttled_energy_per_iter_kwh=math.inf,
            savings_kwh=0.0,
            savings_pct=0.0,
        )

    t_max = max(valid_times.values())
    throttle: dict[int, float] = {}
    baseline_energy_kwh = 0.0
    throttled_energy_kwh = 0.0

    for sid, t_s in partition.stage_exec_time.items():
        spec = stage_lookup[sid]
        p_s = spec.power_draw_w
        if not math.isfinite(t_s) or t_s <= 0:
            throttle[sid] = 1.0
            continue
        r_s = t_s / t_max
        throttle[sid] = r_s
        p_idle = p_s * idle_power_fraction

        # Bloated baseline: compute window draws full power, idle window draws fractional
        baseline_ws = p_s * t_s + p_idle * (t_max - t_s)
        throttled_ws = p_s * (r_s ** voltage_alpha) * t_max

        baseline_energy_kwh += baseline_ws / 3_600_000.0
        throttled_energy_kwh += throttled_ws / 3_600_000.0

    savings = max(0.0, baseline_energy_kwh - throttled_energy_kwh)
    savings_pct = (
        100.0 * savings / baseline_energy_kwh if baseline_energy_kwh > 0 else 0.0
    )
    return PerseusPlan(
        throttle_factors=throttle,
        bottleneck_time_s=t_max,
        baseline_energy_per_iter_kwh=baseline_energy_kwh,
        throttled_energy_per_iter_kwh=throttled_energy_kwh,
        savings_kwh=savings,
        savings_pct=savings_pct,
    )

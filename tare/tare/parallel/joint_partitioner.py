"""Joint pipeline partition and per-stage throttle co-design.

Picks layer cuts `c = (c_1, …, c_{K-1})` and per-stage throttle factors
`r = (r_0, …, r_{K-1}) ∈ [r_min, 1]^K` jointly, minimising per-iteration
energy

    E(π) = Σ_s P_s · r_s^{α-1} · T_s(c)

subject to a hard throughput floor `max_s T_s(c)/r_s ≤ T_floor`,
per-stage power cap `P_s · r_s^α ≤ P_s^{cap}`, segment memory fit, and the
hardware throttle range `r ∈ [r_min, 1]`.

Reduces to existing primitives at parameter boundaries:

    throttle_min=1.0, throttle_granularity=1, throughput_floor → 0
        ≡ partition_pipeline(objective="energy")
    cuts fixed by caller (single-cell DP)
        ≡ perseus_throttle on that partition
    throttle_min=1.0 with cuts fixed at bottleneck-optimal
        ≡ Perseus on bottleneck partition

Complexity: O(n² · K · M) time, O(n · K) space, where M =
``throttle_granularity``.

Two throttle models are supported:

* **Parametric (default)** — closed-form ``P' = P · r^α``, parameterised
  by ``voltage_alpha``. The original analytic surrogate used by the T6
  proof. ``α ≥ 1`` makes the energy curve convex.
* **Empirical curve** — caller passes a measured Pareto frontier as a
  ``ThrottleCurve`` (typically built from a power-cap DVFS sweep such
  as ``exp_hardware_pareto.py``). The DP replaces ``r^α``-style scaling
  with the measured ``(time_scale, energy_scale)`` per throttle level,
  so the optimiser sees the real U-shaped energy-per-iter curve that
  real hardware exhibits (Zeus NSDI'23 reports the same shape across
  A40, V100, RTX6000, P100).

The empirical mode preserves the DP's correctness argument (principle of
optimality only needs each stage's cost to depend on the stage's local
``(cut, throttle)`` choice — both throttle models satisfy that). The
parametric T6 strong-form witness no longer applies to empirical curves;
the gap depends on the measured curve's shape.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tare.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    StageSpec,
    _comm_time,
    _comp_time,
    _exec_time,
)


@dataclass(frozen=True)
class ThrottleCurve:
    """Empirical throttle curve from a real-hardware DVFS sweep.

    Each point maps a throttle ratio ``r ∈ (0, 1]`` (= measured cap /
    max cap) to:

      - ``time_scale``: ``t_throttled / t_baseline`` ≥ 1. The baseline
        is the ``r = 1.0`` measurement.
      - ``energy_scale``: ``E_per_iter(r) / E_per_iter(r=1.0)``. Can be
        below 1 (energy savings) or above 1 (U-shape overshoot).

    The ``r = 1.0`` entry is required (it defines the baseline). Other
    ratios are typically ``cap_w / max_cap_w`` from the sweep.

    Use ``ThrottleCurve.from_pareto_json`` to construct from an
    ``exp_hardware_pareto.py`` JSON output. Use ``ThrottleCurve.from_alpha``
    to round-trip a parametric curve as an empirical lookup (handy for
    backward-compat tests).
    """

    points: tuple[tuple[float, float, float], ...]
    """Sorted by throttle ratio ascending: ``((r, time_scale, energy_scale), ...)``."""

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("ThrottleCurve requires at least one point")
        rs = [p[0] for p in self.points]
        if any(r <= 0.0 for r in rs):
            raise ValueError(f"throttle ratios must be > 0, got {rs}")
        if 1.0 not in rs and not any(abs(r - 1.0) < 1e-9 for r in rs):
            raise ValueError(
                f"ThrottleCurve must include r=1.0 baseline; got ratios {rs}"
            )
        for i in range(1, len(rs)):
            if rs[i] <= rs[i - 1]:
                raise ValueError(
                    f"ThrottleCurve points must be sorted ascending in r; got {rs}"
                )

    def ratios(self) -> tuple[float, ...]:
        return tuple(p[0] for p in self.points)

    def time_scale(self, r: float) -> float:
        for ratio, t_scale, _ in self.points:
            if abs(ratio - r) < 1e-9:
                return t_scale
        raise KeyError(f"throttle ratio {r} not in curve; available {self.ratios()}")

    def energy_scale(self, r: float) -> float:
        for ratio, _, e_scale in self.points:
            if abs(ratio - r) < 1e-9:
                return e_scale
        raise KeyError(f"throttle ratio {r} not in curve; available {self.ratios()}")

    @classmethod
    def from_alpha(
        cls,
        voltage_alpha: float,
        throttle_min: float = 0.5,
        granularity: int = 8,
    ) -> ThrottleCurve:
        """Sample the parametric ``r^α`` model at the given grid. Useful for
        regression-by-construction: a partitioner run with this curve must
        match a run with ``voltage_alpha=...`` and the matching grid."""
        if voltage_alpha < 1.0:
            raise ValueError(f"voltage_alpha must be >= 1, got {voltage_alpha}")
        ratios = _build_throttle_set(throttle_min, granularity)
        points: list[tuple[float, float, float]] = []
        for r in ratios:
            time_scale = 1.0 / r
            energy_scale = r ** (voltage_alpha - 1.0)
            points.append((r, time_scale, energy_scale))
        return cls(points=tuple(points))

    @classmethod
    def from_pareto_json(cls, path: str | Path) -> ThrottleCurve:
        """Load a ``ThrottleCurve`` from an ``exp_hardware_pareto.py`` JSON file.

        Expects keys ``rows`` with entries ``cap_w_observed``,
        ``wall_seconds``, ``avg_power_w``. The throttle ratio is
        ``cap_w_observed / max(cap_w_observed)``; ``time_scale`` is
        derived from ``wall_seconds`` normalised to the r=1.0 entry;
        ``energy_scale`` from ``avg_power_w * wall_seconds`` similarly.
        """
        raw = json.loads(Path(path).read_text())
        rows = raw["rows"]
        if not rows:
            raise ValueError(f"no rows in {path}")
        max_cap = max(r["cap_w_observed"] for r in rows)
        baseline = max(rows, key=lambda r: r["cap_w_observed"])
        t_base = baseline["wall_seconds"]
        e_base = baseline["avg_power_w"] * baseline["wall_seconds"]
        points: list[tuple[float, float, float]] = []
        for row in rows:
            r = row["cap_w_observed"] / max_cap
            t_scale = row["wall_seconds"] / t_base
            e_scale = (row["avg_power_w"] * row["wall_seconds"]) / e_base
            points.append((r, t_scale, e_scale))
        points.sort(key=lambda p: p[0])
        return cls(points=tuple(points))


@dataclass(frozen=True)
class JointPlan:
    """Joint partition-and-throttle plan over `K` pipeline stages.

    ``stage_exec_time`` reports the *throttled* exec time `T_s/r_s`
    (what the pipeline cycle sees). ``pipeline_time_s`` is the maximum
    over all stages — the pipeline's steady-state cycle.
    ``energy_per_iter`` is in Joules.

    A plan with non-finite ``energy_per_iter`` indicates the problem was
    infeasible under the supplied constraints; callers should check
    :meth:`is_feasible` before consuming the other fields.
    """

    cuts: tuple[int, ...] = ()
    throttle_factors: tuple[float, ...] = ()
    stage_layers: dict[int, tuple[int, ...]] = field(default_factory=dict)
    stage_exec_time: dict[int, float] = field(default_factory=dict)
    energy_per_iter: float = math.inf
    pipeline_time_s: float = math.inf
    num_stages: int = 0

    def is_feasible(self) -> bool:
        return math.isfinite(self.energy_per_iter)


def _build_throttle_set(throttle_min: float, granularity: int) -> tuple[float, ...]:
    """Discretise the throttle interval [throttle_min, 1.0].

    granularity == 1 yields a single value: 1.0 if throttle_min == 1.0,
    otherwise throttle_min (the user is forcing a single off-1.0 throttle).
    granularity >= 2 includes both endpoints; the top endpoint is pinned to
    exactly 1.0 to avoid floating-point drift in equality comparisons.
    """
    if granularity == 1:
        return (1.0 if throttle_min >= 1.0 else throttle_min,)
    if throttle_min >= 1.0:
        return (1.0,)
    step = (1.0 - throttle_min) / (granularity - 1)
    body = tuple(throttle_min + i * step for i in range(granularity - 1))
    return body + (1.0,)


def joint_partition(
    layers: Sequence[LayerProfile],
    stages: Sequence[StageSpec],
    links: Sequence[LinkSpec],
    *,
    throughput_floor_iters_per_s: float,
    voltage_alpha: float = 2.0,
    throttle_min: float = 0.5,
    throttle_granularity: int = 8,
    throttle_curve: ThrottleCurve | None = None,
) -> JointPlan:
    """Joint optimiser over (cuts, throttle vector).

    Args:
        layers: n LayerProfile objects, indexed 0..n-1.
        stages: K StageSpec objects, ordered by stage_id 0..K-1.
        links: K-1 LinkSpec objects for consecutive stage pairs.
        throughput_floor_iters_per_s: minimum acceptable pipeline
            throughput. The corresponding cycle ceiling is ``T_floor =
            1/throughput_floor_iters_per_s``; every joint plan satisfies
            ``max_s T_s/r_s ≤ T_floor``. Lower values relax the floor and
            grant the optimiser more room to throttle.
        voltage_alpha: power-frequency scaling exponent for the
            parametric model. 2.0 ≈ quadratic (NVIDIA conservative
            bound), 3.0 ≈ cubic. Must be ≥ 1. Ignored when
            ``throttle_curve`` is supplied.
        throttle_min: hardware-imposed minimum throttle for the
            parametric model. Ignored when ``throttle_curve`` is
            supplied — the curve already encodes the legal range.
        throttle_granularity: discretisation of [throttle_min, 1.0] for
            the parametric model. Ignored when ``throttle_curve`` is
            supplied.
        throttle_curve: optional empirical Pareto frontier from a
            real-hardware DVFS sweep. When provided, the DP replaces the
            parametric ``r^α`` model with the measured
            ``(time_scale, energy_scale)`` per throttle ratio. See
            ``ThrottleCurve.from_pareto_json``.

    Returns:
        ``JointPlan``. If the constraints cannot be satisfied, returns a
        plan with empty fields and ``energy_per_iter = inf``; callers
        should check :meth:`JointPlan.is_feasible`.
    """
    if throughput_floor_iters_per_s <= 0:
        raise ValueError(
            f"throughput_floor_iters_per_s must be > 0, got {throughput_floor_iters_per_s}"
        )
    if throttle_curve is None:
        if voltage_alpha < 1.0:
            raise ValueError(f"voltage_alpha must be >= 1, got {voltage_alpha}")
        if not 0.0 < throttle_min <= 1.0:
            raise ValueError(f"throttle_min must be in (0, 1], got {throttle_min}")
        if throttle_granularity < 1:
            raise ValueError(
                f"throttle_granularity must be >= 1, got {throttle_granularity}"
            )

    n = len(layers)
    K = len(stages)
    if K < 1:
        raise ValueError("Need at least 1 stage.")
    if n < K:
        raise ValueError(f"Need at least {K} layers for {K} stages.")

    if throttle_curve is not None:
        R = throttle_curve.ratios()

        def t_mul(r: float) -> float:
            return throttle_curve.time_scale(r)

        def e_mul(r: float) -> float:
            return throttle_curve.energy_scale(r)
    else:
        R = _build_throttle_set(throttle_min, throttle_granularity)

        def t_mul(r: float) -> float:
            return 1.0 / r

        def e_mul(r: float) -> float:
            return r ** (voltage_alpha - 1.0)

    T_floor = 1.0 / throughput_floor_iters_per_s

    link_map: dict[int, LinkSpec] = {lk.src_stage: lk for lk in links}
    for s in range(K - 1):
        if s not in link_map:
            raise ValueError(f"Missing link from stage {s} to stage {s+1}.")

    prefix_fwd = [0.0] * (n + 1)
    prefix_bwd = [0.0] * (n + 1)
    prefix_mem = [0] * (n + 1)
    for i in range(n):
        prefix_fwd[i + 1] = prefix_fwd[i] + layers[i].fwd_flops
        prefix_bwd[i + 1] = prefix_bwd[i] + layers[i].bwd_flops
        prefix_mem[i + 1] = prefix_mem[i] + layers[i].activation_bytes

    def seg_exec(stage_id: int, start: int, end: int) -> float:
        fwd = prefix_fwd[end + 1] - prefix_fwd[start]
        bwd = prefix_bwd[end + 1] - prefix_bwd[start]
        comp = _comp_time(stages[stage_id], fwd, bwd)
        comm_in = 0.0
        if stage_id > 0 and start > 0:
            comm_in = _comm_time(
                link_map[stage_id - 1], layers[start - 1].activation_bytes
            )
        comm_out = 0.0
        if stage_id < K - 1:
            comm_out = _comm_time(link_map[stage_id], layers[end].activation_bytes)
        return _exec_time(comp, comm_out, comm_in)

    def seg_feasible(stage_id: int, start: int, end: int) -> bool:
        mem = prefix_mem[end + 1] - prefix_mem[start]
        return mem <= stages[stage_id].memory_bytes

    # Throttle search at a single transition: yields (energy_delta, max_t_over_r, r)
    # for each r ∈ R that survives (R), (P), (Θ-local) constraints.
    # The relative tolerance on the (Θ) check absorbs the ulp-level drift that
    # accumulates when T_floor is computed as 1/(1/T_max_bot) and T_s/r lands
    # within a few ulps of T_floor at the active grid point.
    t_floor_tol = T_floor * (1.0 + 1e-9)

    def best_local(stage_id: int, t: float, prev_max: float) -> tuple[float, float, float] | None:
        spec = stages[stage_id]
        p = spec.power_draw_w
        p_cap = spec.power_cap_w
        best: tuple[float, float, float] | None = None
        for r in R:
            if r <= 0.0:
                continue
            t_mul_r = t_mul(r)
            t_throttled = t * t_mul_r
            if t_throttled > t_floor_tol:
                continue
            if max(prev_max, t_throttled) > t_floor_tol:
                continue
            energy_factor = e_mul(r)
            # Average power at throttle = baseline power × (energy_scale / time_scale).
            # Parametric model: p · r^α / r = p · r^(α-1) · (1/r)... wait, p · r^α (uniform).
            # In α form: e_mul=r^(α-1), t_mul=1/r → avg_power = p · r^α (matches original).
            avg_power = p * energy_factor / t_mul_r
            if avg_power > p_cap:
                continue
            e_delta = p * energy_factor * t
            if best is None or e_delta < best[0]:
                best = (e_delta, t_throttled, r)
        return best

    INF = float("inf")

    # K=1: just throttle the single stage covering all layers.
    if K == 1:
        if not seg_feasible(0, 0, n - 1):
            return JointPlan()
        t = seg_exec(0, 0, n - 1)
        local = best_local(0, t, 0.0)
        if local is None:
            return JointPlan()
        e_delta, t_throttled, r = local
        return JointPlan(
            cuts=(),
            throttle_factors=(r,),
            stage_layers={0: tuple(range(n))},
            stage_exec_time={0: t_throttled},
            energy_per_iter=e_delta,
            pipeline_time_s=t_throttled,
            num_stages=1,
        )

    # dp[j][s] = (energy_to_here, max_throttled_exec_to_here, prev_i, prev_r)
    dp: list[list[tuple[float, float, int, float]]] = [
        [(INF, INF, -1, 0.0) for _ in range(K)] for _ in range(n)
    ]

    # Base case: stage 0 covers [0..j]
    for j in range(n):
        if not seg_feasible(0, 0, j):
            continue
        t = seg_exec(0, 0, j)
        local = best_local(0, t, 0.0)
        if local is None:
            continue
        e_delta, t_throttled, r = local
        dp[j][0] = (e_delta, t_throttled, -1, r)

    # Inductive: stages 1..K-1
    for s in range(1, K):
        for j in range(s, n):
            best: tuple[float, float, int, float] = (INF, INF, -1, 0.0)
            for i in range(s - 1, j):
                prev_e, prev_max, _, _ = dp[i][s - 1]
                if prev_e >= INF:
                    continue
                if not seg_feasible(s, i + 1, j):
                    continue
                t = seg_exec(s, i + 1, j)
                local = best_local(s, t, prev_max)
                if local is None:
                    continue
                e_delta, t_throttled, r = local
                e_new = prev_e + e_delta
                max_new = max(prev_max, t_throttled)
                if e_new < best[0]:
                    best = (e_new, max_new, i, r)
            dp[j][s] = best

    final_e, _, _, _ = dp[n - 1][K - 1]
    if final_e >= INF:
        return JointPlan()

    # Backtrack: recover (cuts, throttle_factors) by walking dp[n-1][K-1] back.
    cuts_list: list[int] = []
    throttles: list[float] = [0.0] * K
    j = n - 1
    for s in range(K - 1, 0, -1):
        _, _, prev_i, r = dp[j][s]
        throttles[s] = r
        cuts_list.append(prev_i)
        j = prev_i
    throttles[0] = dp[j][0][3]
    cuts_list.reverse()

    boundaries = [-1, *cuts_list, n - 1]
    stage_layers: dict[int, tuple[int, ...]] = {}
    stage_exec_time: dict[int, float] = {}
    for s in range(K):
        start = boundaries[s] + 1
        end = boundaries[s + 1]
        stage_layers[s] = tuple(range(start, end + 1))
        t = seg_exec(s, start, end)
        stage_exec_time[s] = t * t_mul(throttles[s])

    pipeline_time_s = max(stage_exec_time.values())

    return JointPlan(
        cuts=tuple(cuts_list),
        throttle_factors=tuple(throttles),
        stage_layers=stage_layers,
        stage_exec_time=stage_exec_time,
        energy_per_iter=final_e,
        pipeline_time_s=pipeline_time_s,
        num_stages=K,
    )

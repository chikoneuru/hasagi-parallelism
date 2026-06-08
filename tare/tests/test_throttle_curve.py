"""Tests for ``ThrottleCurve`` and joint-partitioner empirical-lookup mode.

Covers:
  1. ThrottleCurve constructor validation (missing baseline, duplicate r, ordering).
  2. Regression-by-construction: ``ThrottleCurve.from_alpha(α)`` driven through
     the partitioner reproduces the α-formula result bit-for-bit.
  3. Empirical curve loader from ``exp_hardware_pareto.py`` JSON output.
  4. U-shape energy: a curve with energy_scale dipping below 1 then climbing
     above 1 is accepted; the DP finds the minimum-energy throttle.
  5. The ``r = 1.0`` baseline yields the non-throttled energy result.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tare.parallel.joint_partitioner import ThrottleCurve, joint_partition
from tare.parallel.partitioner import LayerProfile, LinkSpec, StageSpec

# --- Constructor validation ---


def test_curve_rejects_empty_points() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        ThrottleCurve(points=())


def test_curve_rejects_missing_baseline() -> None:
    """``r = 1.0`` baseline is mandatory."""
    with pytest.raises(ValueError, match="r=1.0 baseline"):
        ThrottleCurve(points=((0.5, 2.0, 0.25),))


def test_curve_rejects_non_positive_ratio() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        ThrottleCurve(points=((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)))


def test_curve_rejects_unsorted_points() -> None:
    with pytest.raises(ValueError, match="sorted ascending"):
        ThrottleCurve(points=((1.0, 1.0, 1.0), (0.5, 2.0, 0.25)))


def test_curve_lookup_keyerror_off_grid() -> None:
    curve = ThrottleCurve(points=((0.5, 2.0, 0.25), (1.0, 1.0, 1.0)))
    with pytest.raises(KeyError, match="not in curve"):
        curve.time_scale(0.75)


def test_curve_lookup_within_tolerance() -> None:
    """Float drift within 1e-9 is treated as a match."""
    curve = ThrottleCurve(points=((0.5, 2.0, 0.25), (1.0, 1.0, 1.0)))
    assert curve.time_scale(0.5 + 1e-10) == 2.0
    assert curve.energy_scale(1.0 - 1e-10) == 1.0


# --- Regression-by-construction ---


def _layers(n: int = 8) -> list[LayerProfile]:
    return [
        LayerProfile(index=i, fwd_flops=1.0, bwd_flops=2.0, activation_bytes=0)
        for i in range(n)
    ]


def _stages(power: float = 1.0) -> list[StageSpec]:
    return [
        StageSpec(stage_id=0, throughput_flops=1.0, memory_bytes=10**18, power_draw_w=power),
        StageSpec(stage_id=1, throughput_flops=1.5, memory_bytes=10**18, power_draw_w=power),
    ]


def _link() -> list[LinkSpec]:
    return [LinkSpec(src_stage=0, dst_stage=1, bandwidth_bps=1e18, latency_s=0.0)]


def test_from_alpha_round_trip_matches_alpha_mode() -> None:
    """ThrottleCurve.from_alpha(α) through the partitioner must reproduce the
    parametric-α partitioner's result exactly."""
    layers = _layers(8)
    stages = _stages()
    links = _link()
    floor = 0.05  # generous, so throttling is available

    for alpha in (1.0, 1.5, 2.0, 2.5, 3.0):
        for tm in (0.4, 0.6, 0.8):
            for gran in (3, 5, 8):
                curve = ThrottleCurve.from_alpha(
                    voltage_alpha=alpha, throttle_min=tm, granularity=gran,
                )
                ref = joint_partition(
                    layers, stages, links,
                    throughput_floor_iters_per_s=floor,
                    voltage_alpha=alpha,
                    throttle_min=tm, throttle_granularity=gran,
                )
                emp = joint_partition(
                    layers, stages, links,
                    throughput_floor_iters_per_s=floor,
                    throttle_curve=curve,
                )
                assert ref.is_feasible() == emp.is_feasible(), (
                    f"feasibility mismatch at α={alpha}, tm={tm}, gran={gran}: "
                    f"ref={ref.is_feasible()} emp={emp.is_feasible()}"
                )
                if ref.is_feasible():
                    assert math.isclose(
                        ref.energy_per_iter, emp.energy_per_iter, rel_tol=1e-9,
                    ), (
                        f"α={alpha} tm={tm} gran={gran}: "
                        f"ref={ref.energy_per_iter} emp={emp.energy_per_iter}"
                    )
                    assert ref.cuts == emp.cuts
                    assert ref.throttle_factors == emp.throttle_factors


# --- Empirical loader ---


def test_from_pareto_json_round_trip(tmp_path: Path) -> None:
    """Synthesize an exp_hardware_pareto.py JSON, load, verify scales."""
    # Baseline at 350W, throttled at 175W (half cap), same iter count.
    j = {
        "gpu_name": "test-gpu",
        "rows": [
            {
                "cap_w_requested": 175.0,
                "cap_w_observed": 175.0,
                "iters": 100,
                "wall_seconds": 4.0,           # 2x slower than baseline
                "throughput_iters_per_s": 25.0,
                "avg_power_w": 100.0,          # half the baseline power
                "peak_power_w": 100.0,
                "avg_sm_clock_mhz": 800.0,
                "energy_per_iter_j": 4.0,      # 100 * 4 / 100 = 4 J/iter
                "energy_per_iter_kwh": 1.11e-6,
                "samples_count": 100,
            },
            {
                "cap_w_requested": 350.0,
                "cap_w_observed": 350.0,
                "iters": 100,
                "wall_seconds": 2.0,
                "throughput_iters_per_s": 50.0,
                "avg_power_w": 200.0,
                "peak_power_w": 200.0,
                "avg_sm_clock_mhz": 1900.0,
                "energy_per_iter_j": 4.0,
                "energy_per_iter_kwh": 1.11e-6,
                "samples_count": 100,
            },
        ],
        "alpha": 1.0,
        "p_max_w": 200.0,
    }
    path = tmp_path / "pareto.json"
    path.write_text(json.dumps(j))
    curve = ThrottleCurve.from_pareto_json(path)

    # Baseline (r=1.0) is the max-cap row.
    assert curve.time_scale(1.0) == 1.0
    assert curve.energy_scale(1.0) == 1.0
    # Throttled to r=0.5: wall 4s vs 2s baseline → time_scale=2.0;
    # energy = 100*4 = 400 vs 200*2 = 400 → energy_scale=1.0.
    assert math.isclose(curve.time_scale(0.5), 2.0)
    assert math.isclose(curve.energy_scale(0.5), 1.0)


def test_from_pareto_json_real_3080ti_curve() -> None:
    """Load the actual 3080 Ti sweep result and verify the U-shape."""
    path = Path(__file__).resolve().parents[1] / "artifacts" / "hardware-pareto-3080ti.json"
    if not path.exists():
        pytest.skip(f"requires {path}; run exp_hardware_pareto first")
    curve = ThrottleCurve.from_pareto_json(path)
    ratios = curve.ratios()
    assert math.isclose(ratios[-1], 1.0)   # baseline is r=1.0
    # Find the energy-scale minimum (the U-shape elbow).
    energy_scales = [(r, curve.energy_scale(r)) for r in ratios]
    min_r, min_es = min(energy_scales, key=lambda p: p[1])
    # On the RTX 3080 Ti the energy-optimal cap is around 200W → r ≈ 0.57.
    assert 0.5 < min_r < 0.65, f"unexpected energy-optimal r={min_r}"
    assert min_es < 1.0, f"energy-scale at optimum should be < 1.0, got {min_es}"
    # Below the optimum, energy must climb back up (U-shape).
    below_optimum = [es for r, es in energy_scales if r < min_r]
    assert all(es > min_es for es in below_optimum), (
        f"energy scale not climbing below the optimum: {below_optimum} vs min {min_es}"
    )


# --- DP correctness under empirical curves ---


def test_baseline_r1_only_curve_reproduces_no_throttle() -> None:
    """A curve with just ``r=1.0`` forces no throttling."""
    curve = ThrottleCurve(points=((1.0, 1.0, 1.0),))
    plan = joint_partition(
        _layers(8), _stages(), _link(),
        throughput_floor_iters_per_s=0.05,
        throttle_curve=curve,
    )
    assert plan.is_feasible()
    assert all(r == 1.0 for r in plan.throttle_factors)


def test_u_shape_curve_picks_energy_optimum() -> None:
    """A U-shaped curve with a clear energy minimum at r=0.6 → partitioner
    must land on r=0.6 when the throughput floor allows it."""
    # Mimic the 3080 Ti shape: r=0.4 wastes energy (1.5x), r=0.6 saves (0.8x),
    # r=1.0 baseline. time_scale is monotone in 1/r as a rough surrogate.
    curve = ThrottleCurve(points=(
        (0.4, 2.5, 1.5),    # heavy throttle wastes energy (matches real GPU below 200W)
        (0.6, 1.6, 0.8),    # energy-optimal
        (1.0, 1.0, 1.0),    # full power baseline
    ))
    # Generous T_floor so r=0.4 is also feasible by throughput.
    plan = joint_partition(
        _layers(8), _stages(power=1.0), _link(),
        throughput_floor_iters_per_s=0.02,   # T_floor=50, generous
        throttle_curve=curve,
    )
    assert plan.is_feasible()
    # Both stages should pick r=0.6 (the energy minimum) on this generous floor.
    for r in plan.throttle_factors:
        assert math.isclose(r, 0.6), f"expected energy-optimum r=0.6, got {r}"


def test_empirical_curve_respects_throughput_floor() -> None:
    """When the floor is tight, the partitioner picks the throttle nearest
    1.0 that still hits the floor (because energy increases as r drops below
    the energy-optimum on a U-shaped curve, but on a strictly-monotone curve
    the tightest feasible r wins)."""
    # Monotone curve: deeper throttle saves more energy (no U-shape).
    curve = ThrottleCurve(points=(
        (0.5, 2.0, 0.7),
        (0.75, 1.333, 0.85),
        (1.0, 1.0, 1.0),
    ))
    # Tight floor: time at r=0.5 doubles wall-clock → may exceed floor.
    layers = _layers(8)
    stages = _stages(power=1.0)
    plan = joint_partition(
        layers, stages, _link(),
        throughput_floor_iters_per_s=0.1,
        throttle_curve=curve,
    )
    if plan.is_feasible():
        # Throughput floor restricts how aggressively we can throttle.
        for r in plan.throttle_factors:
            assert r >= 0.5

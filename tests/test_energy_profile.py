"""Unit tests for EnergyProfile."""
from __future__ import annotations

import pytest

from hise.admission.energy_profile import EnergyProfile, linear_profile

# --- Construction + validation ---

def test_rejects_empty_tuples() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        EnergyProfile(energy_per_iter_kwh=(), throughput_iters_per_s=())


def test_rejects_mismatched_tuple_lengths() -> None:
    with pytest.raises(ValueError, match="aligned tuples"):
        EnergyProfile(
            energy_per_iter_kwh=(1.0, 2.0),
            throughput_iters_per_s=(10.0, 20.0, 30.0),
        )


def test_rejects_negative_energy() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        EnergyProfile(
            energy_per_iter_kwh=(1.0, -0.5, 2.0),
            throughput_iters_per_s=(10.0, 20.0, 30.0),
        )


def test_rejects_negative_throughput() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        EnergyProfile(
            energy_per_iter_kwh=(1.0, 2.0, 3.0),
            throughput_iters_per_s=(10.0, 20.0, -5.0),
        )


# --- Lookup ---

def test_max_gpus_matches_tuple_length() -> None:
    p = EnergyProfile(
        energy_per_iter_kwh=(1e-5, 8e-6, 6e-6, 7e-6),
        throughput_iters_per_s=(10.0, 18.0, 25.0, 30.0),
    )
    assert p.max_gpus == 4


def test_lookup_clamps_below_one() -> None:
    p = EnergyProfile(
        energy_per_iter_kwh=(1e-5, 8e-6),
        throughput_iters_per_s=(10.0, 18.0),
    )
    assert p.energy_per_iter(0) == 0.0
    assert p.throughput(-1) == 0.0


def test_lookup_clamps_above_max() -> None:
    p = EnergyProfile(
        energy_per_iter_kwh=(1e-5, 8e-6, 6e-6),
        throughput_iters_per_s=(10.0, 18.0, 25.0),
    )
    # gpus=99 clamps to max=3
    assert p.energy_per_iter(99) == 6e-6
    assert p.throughput(99) == 25.0


def test_total_energy_scales_linearly_with_iterations() -> None:
    p = EnergyProfile(
        energy_per_iter_kwh=(1e-5, 8e-6),
        throughput_iters_per_s=(10.0, 18.0),
    )
    assert p.total_energy_kwh(gpus=1, iterations=1000) == pytest.approx(1e-2)
    assert p.total_energy_kwh(gpus=1, iterations=2000) == pytest.approx(2e-2)


def test_total_energy_over_duration_uses_bundled_throughput() -> None:
    p = EnergyProfile(
        energy_per_iter_kwh=(1e-5, 8e-6),
        throughput_iters_per_s=(10.0, 20.0),
    )
    # gpus=2, duration=5s → iters = 20 × 5 = 100; total = 8e-6 × 100 = 8e-4
    assert p.total_energy_kwh_over_duration(gpus=2, duration_s=5.0) == pytest.approx(8e-4)


# --- Convexity ---

def test_strictly_convex_profile_validates() -> None:
    # f(x) = x², second diff = 2 > 0
    p = EnergyProfile(
        energy_per_iter_kwh=(1.0, 4.0, 9.0, 16.0, 25.0),
        throughput_iters_per_s=(10.0, 20.0, 30.0, 40.0, 50.0),
    )
    assert p.validate_convexity() is True


def test_linear_profile_validates_as_convex() -> None:
    # f(x) = x, second diff = 0 (boundary)
    p = EnergyProfile(
        energy_per_iter_kwh=(1.0, 2.0, 3.0, 4.0),
        throughput_iters_per_s=(10.0, 20.0, 30.0, 40.0),
    )
    assert p.validate_convexity() is True


def test_u_shaped_profile_validates_as_convex() -> None:
    # Zeus-style U-shape: cheapest at 4 GPUs
    p = EnergyProfile(
        energy_per_iter_kwh=(10.0, 6.0, 4.5, 4.0, 4.5, 6.0, 9.0),
        throughput_iters_per_s=(10.0, 18.0, 25.0, 30.0, 33.0, 35.0, 36.0),
    )
    assert p.validate_convexity() is True


def test_concave_profile_fails_validation() -> None:
    # f(x) = -x², second diff = -2 < 0 → concave
    p = EnergyProfile(
        energy_per_iter_kwh=(1.0, 4.0, 5.0, 4.0, 1.0),
        throughput_iters_per_s=(10.0, 20.0, 25.0, 28.0, 30.0),
    )
    assert p.validate_convexity() is False


def test_two_point_profile_trivially_convex() -> None:
    p = EnergyProfile(
        energy_per_iter_kwh=(1.0, 2.0),
        throughput_iters_per_s=(10.0, 20.0),
    )
    assert p.validate_convexity() is True


def test_single_point_profile_trivially_convex() -> None:
    p = EnergyProfile(
        energy_per_iter_kwh=(1.0,),
        throughput_iters_per_s=(10.0,),
    )
    assert p.validate_convexity() is True


# --- Optimal GPU count ---

def test_optimal_gpu_at_u_shape_minimum() -> None:
    # Minimum at 4 GPUs (index 3 → 4-th entry, gpus=4)
    p = EnergyProfile(
        energy_per_iter_kwh=(10.0, 6.0, 4.5, 4.0, 4.5, 6.0, 9.0),
        throughput_iters_per_s=(10.0, 18.0, 25.0, 30.0, 33.0, 35.0, 36.0),
    )
    assert p.optimal_gpu_count() == 4


def test_optimal_gpu_at_one_when_monotone_increasing() -> None:
    p = EnergyProfile(
        energy_per_iter_kwh=(1.0, 2.0, 3.0, 4.0),
        throughput_iters_per_s=(10.0, 20.0, 30.0, 40.0),
    )
    assert p.optimal_gpu_count() == 1


# --- linear_profile helper ---

def test_linear_profile_produces_valid_profile() -> None:
    """Synthetic helper returns a valid (non-empty, aligned) EnergyProfile.

    Convexity is NOT guaranteed for the simple synthetic model — the concave
    ``g^0.15`` useful-work term dominates the quadratic allreduce term for small
    α. Real Zeus-measured profiles exhibit U-shape; the helper is for smoke
    testing only.
    """
    p = linear_profile(power_per_gpu_w=300.0, base_throughput_iters_per_s=10.0,
                       max_gpus=8, scaling_efficiency=0.85)
    assert p.max_gpus == 8
    # Energy must be positive, monotone-non-decreasing for sane synthesis.
    assert all(e > 0 for e in p.energy_per_iter_kwh)
    assert all(t > 0 for t in p.throughput_iters_per_s)


def test_linear_profile_with_heavy_allreduce_becomes_convex() -> None:
    """Sufficient allreduce penalty recovers convexity over the practical range."""
    # α=10 makes the quadratic term dominate from g=1 onwards.
    p = linear_profile(power_per_gpu_w=300.0, base_throughput_iters_per_s=10.0,
                       max_gpus=8, scaling_efficiency=0.85,
                       allreduce_coefficient=10.0)
    assert p.validate_convexity() is True


def test_linear_profile_rejects_invalid_efficiency() -> None:
    with pytest.raises(ValueError, match="scaling_efficiency"):
        linear_profile(power_per_gpu_w=300.0, base_throughput_iters_per_s=10.0,
                       max_gpus=4, scaling_efficiency=0.0)


def test_linear_profile_rejects_invalid_max_gpus() -> None:
    with pytest.raises(ValueError, match="max_gpus"):
        linear_profile(power_per_gpu_w=300.0, base_throughput_iters_per_s=10.0,
                       max_gpus=0)

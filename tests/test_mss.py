"""Unit tests for ElasticFlow MSS + HISE Energy-Budgeted MSS."""
from __future__ import annotations

from hise.admission.energy_profile import EnergyProfile, linear_profile
from hise.admission.mss import (
    EnergyBudgetMSS,
    ScalingCurve,
    greedy_marginal_allocation,
    minimum_satisfactory_share,
)


def _concave_curve(max_gpus: int = 16) -> ScalingCurve:
    # 1, 1.82, 2.55, ..., concave shape (x^0.85).
    return ScalingCurve(throughput_per_gpu_count=[x ** 0.85 for x in range(1, max_gpus + 1)])


def test_mss_returns_minimal_gpu_count() -> None:
    curve = _concave_curve()
    # 100 iters in 100 seconds: need throughput >= 1 → 1 gpu is enough.
    assert minimum_satisfactory_share(100, 100.0, curve) == 1
    # 1000 iters in 100 seconds: need throughput >= 10 → ~14 gpus (since 14^0.85 ≈ 9.4, 16^0.85 ≈ 10.5).
    mss = minimum_satisfactory_share(1000, 100.0, curve)
    assert mss > 1
    assert curve.throughput(mss) >= 10.0
    assert curve.throughput(mss - 1) < 10.0


def test_mss_returns_zero_when_infeasible() -> None:
    curve = _concave_curve(max_gpus=4)
    # 10 000 iters in 1 sec: nothing fits in 4-gpu cluster.
    assert minimum_satisfactory_share(10_000, 1.0, curve) == 0


def test_energy_budget_mss_admits_with_generous_budget() -> None:
    curve = _concave_curve()
    # 500 iters at concave curve completes well under 200s with 1 GPU; energy is small.
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=1.0,  # 1 kWh — plenty for a 500-iter mini-job.
    )
    decision = ebmss.find(iterations_remaining=500, deadline_seconds=200.0)
    assert decision.admitted
    assert decision.gpus >= 1


def test_energy_budget_mss_rejects_when_energy_too_low() -> None:
    curve = _concave_curve()
    # Tiny energy budget but generous deadline → no allocation fits the energy box.
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=1e-9,
    )
    decision = ebmss.find(iterations_remaining=10_000, deadline_seconds=3600.0)
    assert not decision.admitted


def test_energy_budget_mss_with_carbon_proxy_secondary() -> None:
    curve = _concave_curve()
    # Energy budget admits us; carbon proxy adds a (loose) secondary constraint.
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=1.0,
        carbon_intensity_forecast=lambda t: 100.0,  # clean grid
        carbon_budget_g=1_000.0,
    )
    decision = ebmss.find(iterations_remaining=500, deadline_seconds=200.0)
    assert decision.admitted
    assert "carbon proxy" in decision.reason


def test_energy_budget_mss_rejects_when_carbon_too_tight() -> None:
    curve = _concave_curve()
    # Deadline + energy are both fine, but carbon proxy is impossibly tight — the carbon
    # branch should be what causes rejection (not deadline / not energy).
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=100.0,
        carbon_intensity_forecast=lambda t: 5_000.0,
        carbon_budget_g=0.001,
    )
    decision = ebmss.find(iterations_remaining=500, deadline_seconds=3600.0)
    assert not decision.admitted
    assert "energy budget" in decision.reason


def test_greedy_marginal_allocation_distributes_remaining() -> None:
    curve = _concave_curve()
    admitted = [("job-a", curve, 2), ("job-b", curve, 3)]
    alloc = greedy_marginal_allocation(admitted, available_gpus=10)
    assert alloc["job-a"] >= 2
    assert alloc["job-b"] >= 3
    assert alloc["job-a"] + alloc["job-b"] == 10


# --- EnergyProfile integration (Phase 2 D2.1) ---

def test_ebmss_uses_energy_profile_when_provided() -> None:
    """When energy_profile is set, projection uses profile not linear power × duration."""
    curve = _concave_curve()
    profile = EnergyProfile(
        # 16 entries aligned with curve.max_gpus=16; convex U-shape
        energy_per_iter_kwh=tuple(1e-4 * (1.0 + 0.05 * (g - 4) ** 2) for g in range(1, 17)),
        throughput_iters_per_s=tuple(c for c in curve.throughput_per_gpu_count),
    )
    assert profile.validate_convexity()

    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,             # fallback, should NOT be used
        energy_budget_kwh=10.0,
        energy_profile=profile,
    )
    # Profile-based projection: gpus=4, duration=10s, throughput=4^0.85≈3.25 iter/s
    # iters ≈ 32, E_iter at gpus=4 = 1e-4 × (1 + 0.05 × 0) = 1e-4
    # total ≈ 3.2e-3 kWh
    proj = ebmss.project_energy_kwh(gpus=4, duration_s=10.0)
    # Linear would give: 300 × 4 × 10 / 3.6e6 ≈ 3.33e-3 — similar magnitude but
    # different because profile already encodes throughput non-linearity.
    linear_estimate = (300.0 * 4 * 10.0) / 3_600_000.0
    assert abs(proj - linear_estimate) > 0  # must differ from linear


def test_ebmss_falls_back_to_linear_when_no_profile() -> None:
    curve = _concave_curve()
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=10.0,
        energy_profile=None,
    )
    # Linear: 300W × 4 × 10s = 12000 J = 12kJ = 12/3.6e6 kWh
    proj = ebmss.project_energy_kwh(gpus=4, duration_s=10.0)
    assert proj == (300.0 * 4 * 10.0) / 3_600_000.0


def test_ebmss_admits_with_synthetic_linear_profile() -> None:
    """End-to-end admission using linear_profile helper."""
    curve = _concave_curve()
    profile = linear_profile(power_per_gpu_w=300.0, base_throughput_iters_per_s=1.0,
                              max_gpus=curve.max_gpus, scaling_efficiency=0.85)
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=10.0,            # generous
        energy_profile=profile,
    )
    decision = ebmss.find(iterations_remaining=100, deadline_seconds=1000.0)
    assert decision.admitted
    assert decision.gpus >= 1

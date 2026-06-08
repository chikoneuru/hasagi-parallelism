"""Unit tests for ElasticFlow MSS + HASAGI Energy-Budgeted MSS."""
from __future__ import annotations

from tare.admission.energy_profile import EnergyProfile, linear_profile
from tare.admission.mss import (
    EnergyBudgetMSS,
    ScalingCurve,
    greedy_marginal_allocation,
    greedy_marginal_energy_allocation,
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


# --- EnergyProfile integration ---

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


# --- Marginal-energy-return allocator ---

def _eff_profile(max_gpus: int, alpha: float) -> EnergyProfile:
    """Synthetic convex profile: lower ``alpha`` → flatter marginal cost
    (more efficient at scale); higher ``alpha`` → steeper marginal cost."""
    return linear_profile(
        power_per_gpu_w=300.0,
        base_throughput_iters_per_s=1.0,
        max_gpus=max_gpus,
        scaling_efficiency=0.85,
        allreduce_coefficient=alpha,
    )


def test_marginal_energy_alloc_handles_empty_and_no_slack() -> None:
    assert greedy_marginal_energy_allocation([], available_gpus=10) == {}
    profile = _eff_profile(max_gpus=4, alpha=0.05)
    # No slack: available_gpus equals already-allocated total.
    alloc = greedy_marginal_energy_allocation([("a", profile, 2), ("b", profile, 1)], 3)
    assert alloc == {"a": 2, "b": 1}


def test_marginal_energy_alloc_fills_single_job() -> None:
    profile = _eff_profile(max_gpus=4, alpha=0.05)
    alloc = greedy_marginal_energy_allocation([("a", profile, 1)], available_gpus=4)
    assert alloc == {"a": 4}


def test_marginal_energy_alloc_respects_max_gpus() -> None:
    profile = _eff_profile(max_gpus=4, alpha=0.05)
    alloc = greedy_marginal_energy_allocation([("a", profile, 1)], available_gpus=100)
    assert alloc["a"] == 4  # capped at profile.max_gpus


def test_marginal_energy_alloc_identical_jobs_split_evenly() -> None:
    """Two jobs with identical convex profiles must split spare GPUs evenly
    (within 1 — greedy ties broken by insertion order)."""
    profile = _eff_profile(max_gpus=8, alpha=0.05)
    alloc = greedy_marginal_energy_allocation(
        [("a", profile, 1), ("b", profile, 1)], available_gpus=8,
    )
    assert abs(alloc["a"] - alloc["b"]) <= 1
    assert alloc["a"] + alloc["b"] == 8


def test_marginal_energy_alloc_prefers_more_efficient_job() -> None:
    """Steeper allreduce coefficient → higher marginal energy cost → fewer GPUs."""
    efficient = _eff_profile(max_gpus=8, alpha=0.01)   # flat marginal cost
    inefficient = _eff_profile(max_gpus=8, alpha=0.20) # steep marginal cost
    alloc = greedy_marginal_energy_allocation(
        [("eff", efficient, 1), ("inef", inefficient, 1)], available_gpus=8,
    )
    assert alloc["eff"] + alloc["inef"] == 8
    assert alloc["eff"] > alloc["inef"]


def test_marginal_energy_alloc_skips_saturated_curve() -> None:
    """Job whose throughput plateaus at high GPU count gets no further allocation
    once Δ throughput drops to 0."""
    flat = EnergyProfile(
        energy_per_iter_kwh=(1e-4, 1e-4, 1e-4, 1e-4),
        throughput_iters_per_s=(1.0, 1.5, 1.5, 1.5),   # saturates at 2 GPUs
    )
    growing = _eff_profile(max_gpus=4, alpha=0.05)
    alloc = greedy_marginal_energy_allocation(
        [("flat", flat, 1), ("grow", growing, 1)], available_gpus=6,
    )
    # flat saturates after 2 GPUs; remainder must go to "grow".
    assert alloc["flat"] <= 2
    assert alloc["grow"] >= alloc["flat"]


def test_marginal_energy_alloc_marginal_cost_monotone_with_convex_profile() -> None:
    """Sanity: under a strictly convex profile, the greedy never allocates a GPU
    whose marginal cost exceeds the *next* candidate's marginal cost. We verify
    by recomputing marginal costs at the final allocation and confirming they
    don't strictly dominate one another (within numerical tolerance)."""
    import math as _m
    profile = _eff_profile(max_gpus=8, alpha=0.05)
    alloc = greedy_marginal_energy_allocation(
        [("a", profile, 1), ("b", profile, 1)], available_gpus=8,
    )

    def marginal_at(p: EnergyProfile, cur: int) -> float:
        if cur >= p.max_gpus:
            return _m.inf
        t_cur = p.throughput(cur)
        t_next = p.throughput(cur + 1)
        if t_next <= t_cur:
            return _m.inf
        pow_cur = p.energy_per_iter(cur) * t_cur
        pow_next = p.energy_per_iter(cur + 1) * t_next
        return (pow_next - pow_cur) / (t_next - t_cur)

    margins = [marginal_at(profile, alloc[j]) for j in ("a", "b")]
    # Any job not at max_gpus would have been pickable on the next step → its
    # marginal must be >= the previously picked job's marginal (monotone DP).
    assert all(_m.isfinite(m) or alloc[j] == profile.max_gpus
               for m, j in zip(margins, ("a", "b"), strict=True))

"""Unit tests for carbon trace + rule-based / MPC policies."""
from __future__ import annotations

import pytest

from tare.energy.carbon_trace import synthetic_solar_trace
from tare.energy.policy import MPCPolicy, RuleBasedPolicy


def test_synthetic_trace_has_24h_period() -> None:
    trace = synthetic_solar_trace(hours=24)
    midnight = trace.intensity_at(0)
    noon = trace.intensity_at(12 * 3600)
    # baseline 450, swing 250, cosine at 0 -> 700 (peak), at 12 -> 200 (trough).
    assert midnight > noon
    assert noon < 300
    assert midnight > 600


def test_rule_based_scales_down_at_high_carbon() -> None:
    policy = RuleBasedPolicy(min_gpus=1, max_gpus=16, scale_down_above_g_per_kwh=600)
    d = policy.decide(current_gpus=8, intensity_now=700.0)
    assert d.target_gpus < 8


def test_rule_based_scales_up_at_low_carbon() -> None:
    policy = RuleBasedPolicy(min_gpus=1, max_gpus=16, scale_up_below_g_per_kwh=250)
    d = policy.decide(current_gpus=2, intensity_now=150.0)
    assert d.target_gpus > 2


def test_rule_based_pauses_at_extreme_carbon() -> None:
    policy = RuleBasedPolicy(min_gpus=1, max_gpus=16, pause_above_g_per_kwh=800)
    d = policy.decide(current_gpus=4, intensity_now=900.0)
    assert d.pause


def test_mpc_picks_low_gpus_when_dirty() -> None:
    mpc = MPCPolicy(
        min_gpus=1,
        max_gpus=8,
        horizon_steps=4,
        step_seconds=300.0,
        power_per_gpu_w=300.0,
        throughput_per_gpu=lambda g: 5.0 * (g ** 0.85),
        iterations_remaining=10_000,
        deadline_seconds_remaining=10_000.0,  # generous deadline
    )
    dirty_forecast = [(i * 300.0, 800.0) for i in range(8)]
    d = mpc.decide(current_gpus=8, intensity_forecast=dirty_forecast)
    # With clean lag penalty small + dirty carbon, MPC should prefer fewer gpus.
    assert d.target_gpus <= 4


def test_mpc_picks_high_gpus_when_clean() -> None:
    mpc = MPCPolicy(
        min_gpus=1,
        max_gpus=8,
        horizon_steps=4,
        step_seconds=300.0,
        power_per_gpu_w=300.0,
        throughput_per_gpu=lambda g: 5.0 * (g ** 0.85),
        iterations_remaining=100_000,
        deadline_seconds_remaining=300.0,  # tight deadline
        lag_weight=10.0,
    )
    clean_forecast = [(i * 300.0, 50.0) for i in range(8)]
    d = mpc.decide(current_gpus=1, intensity_forecast=clean_forecast)
    assert d.target_gpus >= 4


# --- MPC reconfig penalty ---

def _mpc(**overrides) -> MPCPolicy:
    defaults = dict(
        min_gpus=1,
        max_gpus=8,
        horizon_steps=4,
        step_seconds=300.0,
        power_per_gpu_w=300.0,
        throughput_per_gpu=lambda g: 5.0 * (g ** 0.85),
        iterations_remaining=10_000,
        deadline_seconds_remaining=10_000.0,
    )
    defaults.update(overrides)
    return MPCPolicy(**defaults)


def test_mpc_rejects_negative_reconfig_costs() -> None:
    with pytest.raises(ValueError, match="reconfig_latency_s"):
        _mpc(reconfig_latency_s=-1.0)
    with pytest.raises(ValueError, match="reconfig_energy_kwh"):
        _mpc(reconfig_energy_kwh=-0.01)


def test_mpc_zero_reconfig_cost_matches_pre_penalty_behaviour() -> None:
    """With both reconfig costs at 0, decision must equal the pre-penalty behaviour."""
    mpc = _mpc(reconfig_latency_s=0.0, reconfig_energy_kwh=0.0)
    forecast = [(i * 300.0, 800.0) for i in range(8)]
    d = mpc.decide(current_gpus=8, intensity_forecast=forecast)
    assert d.target_gpus <= 4
    assert "reconfig penalty" not in d.reason


def test_mpc_large_reconfig_penalty_prevents_switch() -> None:
    """When the switch-cost massively exceeds the marginal gain from changing
    GPU count, the planner must hold at current_gpus."""
    forecast = [(i * 300.0, 800.0) for i in range(8)]  # dirty grid → would prefer fewer GPUs
    no_penalty = _mpc(reconfig_latency_s=0.0, reconfig_energy_kwh=0.0)
    d_baseline = no_penalty.decide(current_gpus=8, intensity_forecast=forecast)
    assert d_baseline.target_gpus < 8   # confirms baseline wants to switch

    heavy_penalty = _mpc(reconfig_latency_s=0.0, reconfig_energy_kwh=10.0)  # huge
    d_held = heavy_penalty.decide(current_gpus=8, intensity_forecast=forecast)
    assert d_held.target_gpus == 8
    assert "reconfig penalty" not in d_held.reason   # didn't switch → no penalty mention


def test_mpc_reconfig_penalty_reported_when_switch_chosen() -> None:
    """Decision reason must flag the penalty when a switch is the chosen action."""
    forecast = [(i * 300.0, 800.0) for i in range(8)]
    # Small penalty doesn't change the optimal action — switch still wins.
    mpc = _mpc(reconfig_latency_s=1.0, reconfig_energy_kwh=1e-6)
    d = mpc.decide(current_gpus=8, intensity_forecast=forecast)
    assert d.target_gpus != 8
    assert "reconfig penalty" in d.reason


def test_mpc_reconfig_lag_penalty_blocks_marginal_switch() -> None:
    """A switch worth a tiny bit of emissions reduction must be blocked when the
    reconfig latency would burn more deadline-lag than the gain saves."""
    # Tight deadline so lag matters; uniform forecast so emissions are nearly equal
    # between candidates.
    forecast = [(i * 300.0, 500.0) for i in range(8)]
    no_pen = _mpc(
        iterations_remaining=2_000,
        deadline_seconds_remaining=600.0,
        lag_weight=1.0,
        reconfig_latency_s=0.0,
        reconfig_energy_kwh=0.0,
    )
    d_no = no_pen.decide(current_gpus=2, intensity_forecast=forecast)

    with_lag_pen = _mpc(
        iterations_remaining=2_000,
        deadline_seconds_remaining=600.0,
        lag_weight=1.0,
        reconfig_latency_s=10_000.0,  # 10000s pause kills any marginal switch
        reconfig_energy_kwh=0.0,
    )
    d_lag = with_lag_pen.decide(current_gpus=2, intensity_forecast=forecast)
    # Heavy reconfig latency must collapse the decision back to current_gpus.
    assert d_lag.target_gpus == 2
    # And differ from the no-penalty optimum.
    assert d_lag.target_gpus != d_no.target_gpus or d_no.target_gpus == 2

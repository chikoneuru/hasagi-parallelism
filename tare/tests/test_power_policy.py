"""Unit tests for PowerAwareRulePolicy."""
from __future__ import annotations

import pytest

from tare.energy.policy import PowerAwareRulePolicy
from tare.energy.telemetry import WorkerTelemetry


def _telem(worker_id: str, power_w: float, throughput: float) -> WorkerTelemetry:
    return WorkerTelemetry(
        worker_id=worker_id, stage_id=0, gpu_type="A100",
        power_draw_w=power_w, throughput_iters_per_s=throughput,
        energy_cumulative_kwh=0.0, power_cap_w=400.0,
        memory_used_bytes=8 << 30, temperature_c=60.0, timestamp_s=0.0,
    )


def _policy(hyst: int = 3, carbon_pause: float | None = None) -> PowerAwareRulePolicy:
    return PowerAwareRulePolicy(
        min_gpus=1, max_gpus=8,
        scale_down_above_j_per_iter=3.0,    # > 3 J/iter → inefficient
        scale_up_below_j_per_iter=1.5,      # < 1.5 J/iter → headroom
        hysteresis_ticks=hyst,
        carbon_pause_above_g_per_kwh=carbon_pause,
    )


# --- Construction validation ---

def test_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="scale_up threshold"):
        PowerAwareRulePolicy(
            min_gpus=1, max_gpus=8,
            scale_down_above_j_per_iter=1.0,   # lower than scale_up — invalid
            scale_up_below_j_per_iter=2.0,
        )


def test_rejects_zero_hysteresis() -> None:
    with pytest.raises(ValueError, match="hysteresis_ticks"):
        PowerAwareRulePolicy(
            min_gpus=1, max_gpus=8,
            scale_down_above_j_per_iter=3.0,
            scale_up_below_j_per_iter=1.5,
            hysteresis_ticks=0,
        )


# --- Empty / invalid telemetry ---

def test_empty_telemetry_holds_and_resets() -> None:
    pol = _policy()
    d = pol.decide(current_gpus=4, telemetry={})
    assert d.target_gpus == 4
    assert "no valid telemetry" in d.reason


def test_zero_power_telemetry_holds() -> None:
    pol = _policy()
    tel = {"w1": _telem("w1", power_w=0.0, throughput=100.0)}
    d = pol.decide(current_gpus=4, telemetry=tel)
    assert d.target_gpus == 4
    assert "no valid telemetry" in d.reason


def test_zero_throughput_telemetry_holds() -> None:
    pol = _policy()
    tel = {"w1": _telem("w1", power_w=200.0, throughput=0.0)}
    d = pol.decide(current_gpus=4, telemetry=tel)
    assert d.target_gpus == 4


# --- Steady state ---

def test_within_band_holds_and_resets() -> None:
    """J/iter = 200/100 = 2.0, between 1.5 and 3.0 → steady."""
    pol = _policy()
    tel = {"w1": _telem("w1", power_w=200.0, throughput=100.0)}
    d = pol.decide(current_gpus=4, telemetry=tel)
    assert d.target_gpus == 4
    assert "steady" in d.reason
    # Counters reset.
    assert pol._ticks_above == 0
    assert pol._ticks_below == 0


# --- Scale-down hysteresis ---

def test_above_threshold_below_hysteresis_holds() -> None:
    """J/iter = 400/100 = 4.0 > 3.0 threshold, but only 1 tick — must hold."""
    pol = _policy(hyst=3)
    tel = {"w1": _telem("w1", power_w=400.0, throughput=100.0)}
    d = pol.decide(current_gpus=4, telemetry=tel)
    assert d.target_gpus == 4
    assert pol._ticks_above == 1


def test_above_threshold_after_hysteresis_scales_down() -> None:
    pol = _policy(hyst=3)
    tel = {"w1": _telem("w1", power_w=400.0, throughput=100.0)}   # 4 J/iter
    decisions = [pol.decide(current_gpus=4, telemetry=tel) for _ in range(3)]
    assert decisions[0].target_gpus == 4
    assert decisions[1].target_gpus == 4
    assert decisions[2].target_gpus == 3   # scale down on 3rd consecutive
    assert "scale down" in decisions[2].reason


def test_scale_down_respects_min_gpus() -> None:
    pol = _policy(hyst=1)
    tel = {"w1": _telem("w1", power_w=400.0, throughput=100.0)}
    d = pol.decide(current_gpus=1, telemetry=tel)
    assert d.target_gpus == 1   # already at min_gpus=1


# --- Scale-up hysteresis ---

def test_below_threshold_after_hysteresis_scales_up() -> None:
    pol = _policy(hyst=2)
    tel = {"w1": _telem("w1", power_w=100.0, throughput=100.0)}   # 1.0 J/iter
    d1 = pol.decide(current_gpus=4, telemetry=tel)
    d2 = pol.decide(current_gpus=4, telemetry=tel)
    assert d1.target_gpus == 4
    assert d2.target_gpus == 5
    assert "scale up" in d2.reason


def test_scale_up_respects_max_gpus() -> None:
    pol = _policy(hyst=1)
    tel = {"w1": _telem("w1", power_w=100.0, throughput=100.0)}
    d = pol.decide(current_gpus=8, telemetry=tel)
    assert d.target_gpus == 8


# --- Oscillation defense ---

def test_oscillating_signal_resets_counter() -> None:
    """Above threshold, then below, then steady → no scale action."""
    pol = _policy(hyst=3)
    above = {"w1": _telem("w1", power_w=400.0, throughput=100.0)}   # 4 J/iter
    below = {"w1": _telem("w1", power_w=100.0, throughput=100.0)}   # 1 J/iter
    steady = {"w1": _telem("w1", power_w=200.0, throughput=100.0)}  # 2 J/iter

    d1 = pol.decide(current_gpus=4, telemetry=above)
    d2 = pol.decide(current_gpus=4, telemetry=below)
    d3 = pol.decide(current_gpus=4, telemetry=steady)

    assert d1.target_gpus == 4
    assert d2.target_gpus == 4
    assert d3.target_gpus == 4
    # Counters cleared in steady state.
    assert pol._ticks_above == 0
    assert pol._ticks_below == 0


# --- Multi-worker aggregation ---

def test_multi_worker_aggregate_j_per_iter() -> None:
    """4 workers each 400W/100ips → avg = 4 J/iter → triggers scale-down."""
    pol = _policy(hyst=2)
    tel = {f"w{i}": _telem(f"w{i}", power_w=400.0, throughput=100.0) for i in range(4)}
    pol.decide(current_gpus=4, telemetry=tel)
    d2 = pol.decide(current_gpus=4, telemetry=tel)
    assert d2.target_gpus == 3


def test_mixed_workers_average_correctly() -> None:
    """Mixed signals: 1 worker bad (5 J/iter), 1 worker good (1 J/iter).
    Average = 3 J/iter, on the boundary — should hold steady."""
    pol = _policy(hyst=1)
    tel = {
        "w_bad":  _telem("w_bad",  power_w=500.0, throughput=100.0),  # 5 J/iter
        "w_good": _telem("w_good", power_w=100.0, throughput=100.0),  # 1 J/iter
    }
    d = pol.decide(current_gpus=4, telemetry=tel)
    # avg = (5+1)/2 = 3.0, exactly the boundary — falls into "steady" branch
    # (neither > 3.0 nor < 1.5).
    assert d.target_gpus == 4


# --- Carbon spatial-shift gate ---

def test_carbon_pause_triggers_scale_down() -> None:
    """When carbon intensity > threshold and gate configured, scale down regardless."""
    pol = _policy(hyst=3, carbon_pause=700.0)
    tel = {"w1": _telem("w1", power_w=100.0, throughput=200.0)}   # very efficient
    d = pol.decide(current_gpus=8, telemetry=tel, carbon_intensity_now=800.0)
    assert d.target_gpus == 4   # halved
    assert "spatial-shift" in d.reason


def test_carbon_pause_ignored_when_intensity_below_threshold() -> None:
    pol = _policy(hyst=3, carbon_pause=700.0)
    tel = {"w1": _telem("w1", power_w=200.0, throughput=100.0)}
    d = pol.decide(current_gpus=4, telemetry=tel, carbon_intensity_now=400.0)
    assert d.target_gpus == 4
    assert "spatial-shift" not in d.reason


def test_carbon_pause_disabled_by_default() -> None:
    pol = _policy(hyst=1)   # carbon_pause=None
    tel = {"w1": _telem("w1", power_w=200.0, throughput=100.0)}
    d = pol.decide(current_gpus=4, telemetry=tel, carbon_intensity_now=999.0)
    # carbon ignored; J/iter = 2.0 → steady
    assert d.target_gpus == 4
    assert "spatial-shift" not in d.reason

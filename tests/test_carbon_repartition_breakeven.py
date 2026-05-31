"""Tests for the carbon-driven repartition break-even study (pure sim, no GPU)."""
from __future__ import annotations

from experiments.exp_carbon_repartition_breakeven import (
    Layout,
    _cutpoint_energy_gap,
    breakeven_eco_strength,
    breakeven_sweep,
    migration_cost,
    parametric_trace,
    simulate,
)

_FAST = Layout("fast", energy_per_iter_j=1.0, throughput_iter_s=1.0)
_ECO = Layout("eco", energy_per_iter_j=0.80, throughput_iter_s=0.6)
_SIM = dict(iters_per_window=1000, threshold_q=0.6, throttle_energy_frac=0.85, throttle_tput_frac=0.7)


def test_parametric_trace_swing() -> None:
    flat = parametric_trace(48, swing=0.0)
    assert max(flat) == min(flat)              # no swing -> flat
    swung = parametric_trace(48, swing=0.6)
    assert max(swung) > min(swung)             # swing -> varies
    assert min(swung) > 0                       # stays positive


def test_migration_cost_monotonic() -> None:
    t1, e1 = migration_cost(state_gb=5.0, bw_gbps=100.0, cold_start_s=4.7, power_w=100.0)
    t2, _ = migration_cost(state_gb=50.0, bw_gbps=100.0, cold_start_s=4.7, power_w=100.0)
    t3, _ = migration_cost(state_gb=5.0, bw_gbps=10.0, cold_start_s=4.7, power_w=100.0)
    assert t2 > t1 and t3 > t1                  # more state / less bw -> slower
    assert e1 == 100.0 * t1                      # energy = power * time


def test_cutpoint_repartition_has_no_energy_lever() -> None:
    # The grounding result: moving pipeline cut-points trades throughput, not energy.
    gap = _cutpoint_energy_gap()
    assert abs(gap["energy_ratio_eco_over_fast"] - 1.0) < 1e-6
    assert gap["eco_throughput"] < gap["fast_throughput"]


def test_static_eco_saves_energy_vs_fast() -> None:
    trace = parametric_trace(168, swing=0.6)
    fast = simulate(trace, _FAST, _ECO, "static_fast", **_SIM)
    eco = simulate(trace, _FAST, _ECO, "static_eco", **_SIM)
    assert eco["carbon_g"] < fast["carbon_g"]   # eco layout uses 20% less energy/iter
    assert eco["makespan_h"] > fast["makespan_h"]  # at lower throughput


def test_repartition_switches_and_pays_migration() -> None:
    trace = parametric_trace(168, swing=0.6)
    mt, me = migration_cost(5.0, 100.0, 4.7, 100.0)
    no_mig = simulate(trace, _FAST, _ECO, "repartition", **_SIM)
    with_mig = simulate(trace, _FAST, _ECO, "repartition", **_SIM,
                        migration_energy_j=me, migration_time_s=mt)
    assert with_mig["switches"] > 0
    assert with_mig["carbon_g"] > no_mig["carbon_g"]      # migration adds carbon
    assert with_mig["makespan_h"] > no_mig["makespan_h"]  # and makespan


def test_repartition_loses_to_throttle_under_cold_start_floor() -> None:
    # Headline negative: even with eco granted MORE energy saving (20%) than throttle
    # (15%), the cold-start-dominated migration makes repartition lose.
    rows = breakeven_sweep(_FAST, _ECO, swings=[0.4, 0.8], state_gbs=[1.0, 20.0], bws=[10.0, 600.0],
                           hours=168, iters_per_window=1000, threshold_q=0.6,
                           throttle_energy_frac=0.85, throttle_tput_frac=0.7,
                           cold_start_s=4.7, migrate_power_w=100.0)
    assert all(not r["repartition_wins"] for r in rows)


def test_eco_strength_breakeven_requires_clearing_throttle() -> None:
    be = breakeven_eco_strength(_FAST, swing=0.6, hours=168, iters_per_window=1000,
                                threshold_q=0.6, throttle_energy_frac=0.85, throttle_tput_frac=0.7,
                                eco_tput_frac=0.6, state_gb=5.0, bw_gbps=100.0, cold_start_s=4.7,
                                migrate_power_w=100.0, fracs=[round(0.85 - 0.05 * i, 2) for i in range(12)])
    # repartition can only win once the eco lever is STRICTLY stronger than throttle.
    assert be["crossover_eco_frac"] is not None
    assert be["crossover_eco_frac"] < 0.85          # must beat throttle's 0.85
    # and every winning frac is below the crossover (monotone)
    winning = [r["eco_energy_frac"] for r in be["rows"] if r["wins"]]
    assert winning and max(winning) <= be["crossover_eco_frac"]

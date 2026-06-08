"""Tests for the pod energy ledger — attributing measured energy to lifecycle phases.

Energy/clock reads are scripted so the ledger is verified on CPU without a GPU.
Carbon is checked to be exactly ``energy_kwh × intensity`` per phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tare.energy.pod_ledger import (
    PHASE_ACTIVE,
    PHASE_COLD_START,
    PHASE_IDLE,
    PodEnergyLedger,
    nvml_cumulative_kwh_fn,
)


def _scripted_ledger() -> PodEnergyLedger:
    """A ledger whose energy/clock reads come from explicit mark() overrides."""
    return PodEnergyLedger(energy_kwh_fn=lambda: 0.0, clock=lambda: 0.0)


def test_empty_ledger_reports_zero() -> None:
    rep = _scripted_ledger().report(t_s=10.0, cumulative_kwh=0.5)
    assert rep.intervals == ()
    assert rep.total_energy_kwh == 0.0
    assert rep.total_carbon_g == 0.0
    assert rep.cold_starts == 0


def test_single_active_phase_carbon_is_energy_times_intensity() -> None:
    led = _scripted_ledger()
    led.mark(PHASE_ACTIVE, 400.0, t_s=0.0, cumulative_kwh=0.0)
    rep = led.report(t_s=60.0, cumulative_kwh=0.01)
    assert len(rep.intervals) == 1
    iv = rep.intervals[0]
    assert iv.phase == PHASE_ACTIVE
    assert iv.energy_kwh == 0.01
    assert iv.duration_s == 60.0
    # carbon = 0.01 kWh × 400 gCO2/kWh = 4.0 g
    assert iv.carbon_g == 4.0
    assert rep.total_carbon_g == 4.0


def test_cold_start_then_active_attribution() -> None:
    led = _scripted_ledger()
    # Resume at a moderately dirty grid; cold-start spans 5 s and 0.002 kWh.
    led.mark(PHASE_COLD_START, 500.0, t_s=0.0, cumulative_kwh=0.0)
    led.mark(PHASE_ACTIVE, 500.0, t_s=5.0, cumulative_kwh=0.002)
    rep = led.report(t_s=65.0, cumulative_kwh=0.020)

    assert rep.cold_starts == 1
    assert rep.resume_energy_kwh == pytest.approx(0.002)
    assert rep.resume_carbon_g == pytest.approx(1.0)       # 0.002 kWh × 500
    assert rep.active_energy_kwh == pytest.approx(0.018)
    assert rep.carbon_by_phase_g[PHASE_ACTIVE] == pytest.approx(9.0)   # 0.018 × 500
    assert rep.total_energy_kwh == pytest.approx(0.020)
    assert rep.total_carbon_g == pytest.approx(10.0)


def test_multiple_pause_resume_cycles_count_cold_starts() -> None:
    led = _scripted_ledger()
    led.mark(PHASE_COLD_START, 600.0, t_s=0.0, cumulative_kwh=0.0)
    led.mark(PHASE_ACTIVE, 600.0, t_s=4.0, cumulative_kwh=0.001)
    led.mark(PHASE_IDLE, 600.0, t_s=30.0, cumulative_kwh=0.010)   # pause: scale-to-zero
    led.mark(PHASE_COLD_START, 300.0, t_s=90.0, cumulative_kwh=0.010)  # idle drew ~0 on dedicated GPU
    led.mark(PHASE_ACTIVE, 300.0, t_s=95.0, cumulative_kwh=0.011)
    rep = led.report(t_s=125.0, cumulative_kwh=0.020)

    assert rep.cold_starts == 2
    assert rep.phase_counts[PHASE_ACTIVE] == 2
    assert rep.phase_counts[PHASE_IDLE] == 1
    # cold-start energy = 0.001 (first) + 0.001 (second) = 0.002
    assert abs(rep.resume_energy_kwh - 0.002) < 1e-12
    # idle drew no incremental energy on the dedicated GPU
    assert rep.energy_by_phase_kwh[PHASE_IDLE] == 0.0


def test_idle_without_intensity_counts_energy_but_zero_carbon() -> None:
    led = _scripted_ledger()
    led.mark(PHASE_IDLE, None, t_s=0.0, cumulative_kwh=0.0)   # shared GPU idle, intensity unknown
    rep = led.report(t_s=60.0, cumulative_kwh=0.003)
    assert rep.energy_by_phase_kwh[PHASE_IDLE] == 0.003
    assert rep.carbon_by_phase_g[PHASE_IDLE] == 0.0
    assert rep.total_energy_kwh == 0.003
    assert rep.total_carbon_g == 0.0


def test_energy_clamped_non_negative_on_meter_reset() -> None:
    led = _scripted_ledger()
    led.mark(PHASE_ACTIVE, 400.0, t_s=0.0, cumulative_kwh=0.05)
    # Meter "reset" to a lower value — clamp to 0 rather than emit negative energy.
    rep = led.report(t_s=10.0, cumulative_kwh=0.01)
    assert rep.intervals[0].energy_kwh == 0.0
    assert rep.total_carbon_g == 0.0


def test_live_reads_use_injected_fns() -> None:
    """When mark()/report() get no overrides, they pull from energy_kwh_fn + clock."""
    energy = {"kwh": 0.0}
    t = {"s": 0.0}
    led = PodEnergyLedger(energy_kwh_fn=lambda: energy["kwh"], clock=lambda: t["s"])
    led.mark(PHASE_ACTIVE, 350.0)
    energy["kwh"] = 0.02
    t["s"] = 100.0
    rep = led.report()
    assert rep.total_energy_kwh == 0.02
    assert rep.intervals[0].duration_s == 100.0
    assert rep.total_carbon_g == 0.02 * 350.0


@dataclass
class _FakeTelem:
    energy_cumulative_kwh: float


@dataclass
class _FakeSource:
    energies: dict[str, float] = field(default_factory=dict)

    def snapshot(self) -> dict[str, _FakeTelem]:
        return {wid: _FakeTelem(e) for wid, e in self.energies.items()}


def test_nvml_cumulative_kwh_fn_sums_across_workers() -> None:
    src = _FakeSource({"w0": 0.01, "w1": 0.02})
    fn = nvml_cumulative_kwh_fn(src)
    assert fn() == 0.03
    src.energies["w0"] = 0.05
    assert abs(fn() - 0.07) < 1e-12

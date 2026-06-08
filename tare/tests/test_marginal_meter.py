"""Tests for the marginal energy meter — background-subtracted integration.

The integrator and calibration are driven with a fake NVML module + injected
clock so no GPU is needed.
"""
from __future__ import annotations

import pytest

from tare.energy.marginal_meter import MarginalEnergyMeter


class _FakeNvml:
    """Minimal NVML stand-in returning a scripted power sequence (mW)."""

    def __init__(self, powers_mw: list[float]):
        self._powers = list(powers_mw)
        self._i = 0
        self.inited = False

    def nvmlInit(self) -> None:  # noqa: N802 — NVML C API name
        self.inited = True

    def nvmlDeviceGetHandleByIndex(self, idx: int):  # noqa: N802
        return ("handle", idx)

    def nvmlDeviceGetPowerUsage(self, handle) -> float:  # noqa: N802
        if not self._powers:
            return 0.0
        v = self._powers[min(self._i, len(self._powers) - 1)]
        self._i += 1
        return v

    def nvmlShutdown(self) -> None:  # noqa: N802
        self.inited = False


def test_integrator_subtracts_background() -> None:
    m = MarginalEnergyMeter(background_w=100.0, nvml_module=_FakeNvml([]), clock=lambda: 0.0)
    m._integrate(0.0, 300.0)     # first sample only anchors the clock
    m._integrate(10.0, 300.0)    # 10 s slice at 300 W device, 100 W background
    assert m.cumulative_kwh() == pytest.approx(200.0 * 10 / 3_600_000.0)
    assert m.device_cumulative_kwh() == pytest.approx(300.0 * 10 / 3_600_000.0)
    assert m.background_cumulative_kwh() == pytest.approx(100.0 * 10 / 3_600_000.0)


def test_marginal_clamped_to_zero_when_below_background() -> None:
    m = MarginalEnergyMeter(background_w=150.0, nvml_module=_FakeNvml([]), clock=lambda: 0.0)
    m._integrate(0.0, 120.0)     # device below background (our job absent)
    m._integrate(5.0, 120.0)
    assert m.cumulative_kwh() == 0.0
    # device + background integrals still accrue
    assert m.device_cumulative_kwh() == pytest.approx(120.0 * 5 / 3_600_000.0)


def test_calibrate_sets_background_mean_and_sd() -> None:
    clock = {"t": 0.0}
    def fake_clock() -> float:
        return clock["t"]
    def fake_sleep(s: float) -> None:
        clock["t"] += s
    # Constant 154 W background → sd 0.
    fake = _FakeNvml([154_000] * 10)
    m = MarginalEnergyMeter(nvml_module=fake, clock=fake_clock, poll_interval_ms=1000)
    mean, sd = m.calibrate(seconds=3.0, sleep=fake_sleep)
    assert mean == pytest.approx(154.0)
    assert sd == pytest.approx(0.0)
    assert m.background_w == pytest.approx(154.0)


def test_calibrate_reports_nonzero_sd_for_bursty_background() -> None:
    clock = {"t": 0.0}
    fake = _FakeNvml([100_000, 200_000, 100_000, 200_000])
    m = MarginalEnergyMeter(
        nvml_module=fake, clock=lambda: clock["t"], poll_interval_ms=1000,
    )
    def fake_sleep(s: float) -> None:
        clock["t"] += s
    mean, sd = m.calibrate(seconds=4.0, sleep=fake_sleep)
    assert mean == pytest.approx(150.0)
    assert sd == pytest.approx(50.0)

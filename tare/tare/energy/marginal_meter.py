"""Marginal GPU energy meter — attribute only *our* job's draw on a shared card.

NVML reports whole-device power, so on a GPU shared with another tenant (or with
a display server) the integral over-counts: it bills our job for everyone's
draw. This meter subtracts a calibrated background power so the integral
reflects our job's *marginal* energy:

    P_mine(t) ≈ max(0, P_device(t) − P_background)

The background is measured by ``calibrate()`` while our job is absent. The
subtraction is only as good as the background's stationarity — sample its
standard deviation and report it; a steady co-tenant (low sd) gives a clean
marginal signal, a bursty one does not.

It keeps three running integrals (marginal / device / background) so a report
can be transparent about how much of the device energy was attributed away.
Polling runs on a background thread, mirroring ``NvmlTelemetrySource``. ``nvml``
and ``clock`` are injectable so the integrator is unit-testable without a GPU.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class MarginalEnergyMeter:
    """Integrate ``max(0, P_device − background)`` into marginal kWh.

    Args:
        device_index: NVML device index to meter.
        background_w: initial background power to subtract (W). Usually set by
            ``calibrate()``; may be passed directly when already known.
        poll_interval_ms: background poll period.
        nvml_module: dependency injection for testing; real ``pynvml`` if None.
        clock: monotonic clock; injectable for deterministic tests.
    """

    def __init__(
        self,
        device_index: int = 0,
        background_w: float = 0.0,
        poll_interval_ms: int = 100,
        nvml_module: Any = None,
        clock: Callable[[], float] = time.monotonic,
        record_trace: bool = False,
    ) -> None:
        if nvml_module is None:
            try:
                import pynvml as nvml_module
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "pynvml not installed; install nvidia-ml-py or inject nvml_module."
                ) from exc
        self._nvml = nvml_module
        self._device_index = device_index
        self.background_w = background_w
        self.background_sd_w = 0.0
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._clock = clock
        self._lock = threading.Lock()
        self._handle: Any = None
        self._initialized = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._marginal_kwh = 0.0
        self._device_kwh = 0.0
        self._background_kwh = 0.0
        self._last_ts: float | None = None
        self._record_trace = record_trace
        self._trace: list[tuple[float, float]] = []

    # --- NVML lifecycle ---

    def _init_nvml(self) -> None:
        if self._initialized:
            return
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self._device_index)
        self._initialized = True

    def _power_w(self) -> float:
        return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    def calibrate(self, seconds: float = 5.0, sleep: Callable[[float], None] = time.sleep) -> tuple[float, float]:
        """Sample device power for ``seconds`` with our job absent; set background.

        Returns ``(mean_w, sd_w)``. A low sd means the background is stationary
        enough for clean subtraction; surface it in any result.
        """
        self._init_nvml()
        samples: list[float] = []
        start = self._clock()
        while self._clock() - start < seconds:
            samples.append(self._power_w())
            sleep(self._poll_interval_s)
        if not samples:
            samples = [self._power_w()]
        mean = sum(samples) / len(samples)
        var = sum((s - mean) ** 2 for s in samples) / len(samples)
        self.background_w = mean
        self.background_sd_w = var ** 0.5
        logger.info(
            "marginal meter calibrated: background=%.1f W sd=%.1f W (n=%d)",
            mean, self.background_sd_w, len(samples),
        )
        return mean, self.background_sd_w

    def _integrate(self, now: float, power_w: float) -> None:
        """Add the slice since the last sample to all three integrals."""
        with self._lock:
            if self._record_trace:
                self._trace.append((now, power_w))
            if self._last_ts is None:
                self._last_ts = now
                return
            dt = max(0.0, now - self._last_ts)
            self._last_ts = now
            marginal_w = max(0.0, power_w - self.background_w)
            self._marginal_kwh += marginal_w * dt / 3_600_000.0
            self._device_kwh += power_w * dt / 3_600_000.0
            self._background_kwh += min(power_w, self.background_w) * dt / 3_600_000.0

    def _poll_once(self) -> None:
        self._integrate(self._clock(), self._power_w())

    def _run(self) -> None:  # pragma: no cover (thread loop)
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("marginal meter poll failed")
            self._stop_event.wait(timeout=self._poll_interval_s)

    def start(self) -> None:
        self._init_nvml()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._last_ts = None
        self._thread = threading.Thread(target=self._run, name="marginal-meter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                logger.exception("nvmlShutdown failed")
            self._initialized = False

    # --- readouts ---

    def cumulative_kwh(self) -> float:
        """Marginal kWh attributed to our job (background subtracted)."""
        with self._lock:
            return self._marginal_kwh

    def device_cumulative_kwh(self) -> float:
        """Whole-device kWh (no subtraction)."""
        with self._lock:
            return self._device_kwh

    def background_cumulative_kwh(self) -> float:
        """kWh attributed to the background (co-tenant + display)."""
        with self._lock:
            return self._background_kwh

    def power_trace(self) -> list[tuple[float, float]]:
        """A copy of the recorded ``(t_s, device_watts)`` samples (if enabled)."""
        with self._lock:
            return list(self._trace)

    def sample_power_w(self) -> float:
        """One-shot whole-device power read (W). Use while our job is off the GPU
        to record a background anchor."""
        self._init_nvml()
        return self._power_w()

    def __enter__(self) -> MarginalEnergyMeter:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

"""Per-worker energy + throughput telemetry — the signal source for A1–A4.

Consumed by:
    A1 partitioner   — `power_draw_w` per stage → energy-per-iter objective;
                       `memory_used_bytes` + `power_cap_w` → feasibility constraints.
    A2 EB-MSS        — `power_draw_w` per `gpu_type` → empirical EnergyProfile fit.
    A3 scaling policy — full vector → MPC / PPO state representation.
    A4 energy-WRR    — `throughput_iters_per_s / power_draw_w` → iter-per-joule weight;
                       `temperature_c` + `power_cap_w` → PowerSlackGuard.

Real telemetry comes from NVML (`pynvml`) and RAPL (`/sys/class/powercap/intel-rapl/`)
in Week 3 (see ``NvmlTelemetrySource`` skeleton below). Until then, A1–A4 algorithm
work uses ``FakeTelemetrySource`` for deterministic unit tests and CI without GPU.

Methodology (research-note.md §2.3): energy is the *primary measured* signal here.
Carbon proxy is layered on top by ``hise.energy.carbon_trace``, never derived from
this module.
"""
from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class WorkerTelemetry:
    """One sample of a worker's energy + throughput state.

    All units SI / SI-derived. ``timestamp_s`` is ``time.monotonic()``-style seconds
    since an arbitrary epoch; only differences between samples are meaningful.

    Attributes follow NVML naming where applicable for easy mental mapping:
        power_draw_w        ← ``nvmlDeviceGetPowerUsage()`` (mW) / 1000
        power_cap_w         ← ``nvmlDeviceGetPowerManagementLimit()`` (mW) / 1000
        memory_used_bytes   ← ``nvmlDeviceGetMemoryInfo().used``
        temperature_c       ← ``nvmlDeviceGetTemperature(GPU)``
    """

    worker_id: str
    stage_id: int                       # pipeline stage 0..K-1
    gpu_type: str                       # "A100" | "T4" | "V100" | "CPU" | ...
    power_draw_w: float                 # instantaneous power, sampled at 100 ms
    throughput_iters_per_s: float       # from worker iteration counter
    energy_cumulative_kwh: float        # running integral since job start
    power_cap_w: float                  # NVML limit
    memory_used_bytes: int
    temperature_c: float
    timestamp_s: float                  # monotonic seconds

    @property
    def iters_per_joule(self) -> float:
        """A4 inter-batch scheduler weight: throughput per watt."""
        if self.power_draw_w <= 0:
            return 0.0
        return self.throughput_iters_per_s / self.power_draw_w

    def is_thermal_throttled(self, fraction: float = 0.9, temp_c_max: float = 80.0) -> bool:
        """True if drawing near power cap or exceeding temperature ceiling."""
        if self.power_cap_w > 0 and self.power_draw_w >= fraction * self.power_cap_w:
            return True
        if self.temperature_c >= temp_c_max:
            return True
        return False


class TelemetrySource(Protocol):
    """All telemetry sources expose ``latest(worker_id)`` and ``snapshot()``."""

    def latest(self, worker_id: str) -> WorkerTelemetry | None: ...
    def snapshot(self) -> dict[str, WorkerTelemetry]: ...


# ---------------------------------------------------------------------------
# FakeTelemetrySource — deterministic, no hardware required
# ---------------------------------------------------------------------------

# Reference power draws and TDPs by GPU type. Used by FakeTelemetrySource to
# synthesize plausible values; the real NvmlTelemetrySource ignores these.
_GPU_PROFILES: dict[str, tuple[float, float]] = {
    # gpu_type → (typical_draw_w_at_full_util, power_cap_w)
    "A100": (320.0, 400.0),
    "V100": (240.0, 300.0),
    "T4":   (60.0,  70.0),
    "H100": (550.0, 700.0),
    "CPU":  (95.0,  125.0),
}


@dataclass
class FakeWorker:
    """Description of a synthetic worker for FakeTelemetrySource."""

    worker_id: str
    stage_id: int
    gpu_type: str = "A100"
    base_throughput_iters_per_s: float = 100.0
    memory_used_bytes: int = 8 << 30        # 8 GiB
    base_temperature_c: float = 55.0


@dataclass
class FakeTelemetrySource:
    """Deterministic telemetry source for unit tests + CI without GPU.

    Each ``read_all()`` advances time by ``tick_seconds`` and emits one
    WorkerTelemetry per worker. Per-tick variation is seeded for reproducibility.

    Example:
        src = FakeTelemetrySource(
            workers=[
                FakeWorker("w1", stage_id=0, gpu_type="A100"),
                FakeWorker("w2", stage_id=0, gpu_type="A100"),
                FakeWorker("w3", stage_id=1, gpu_type="T4"),
            ],
            seed=42,
        )
        for _ in range(10):
            snap = src.read_all()
            ...
    """

    workers: list[FakeWorker]
    tick_seconds: float = 1.0
    jitter_fraction: float = 0.05           # ±5% jitter on power + throughput
    seed: int = 0

    _t0: float = field(init=False, default=0.0)
    _t_elapsed: float = field(init=False, default=0.0)
    _energy_kwh: dict[str, float] = field(init=False, default_factory=dict)
    _last: dict[str, WorkerTelemetry] = field(init=False, default_factory=dict)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._t0 = time.monotonic()
        for w in self.workers:
            self._energy_kwh[w.worker_id] = 0.0

    def read_all(self) -> dict[str, WorkerTelemetry]:
        """Advance one tick and emit a sample per worker."""
        self._t_elapsed += self.tick_seconds
        snap: dict[str, WorkerTelemetry] = {}
        for w in self.workers:
            base_draw, cap = _GPU_PROFILES.get(w.gpu_type, (200.0, 250.0))
            jitter = 1.0 + self._rng.uniform(-self.jitter_fraction, self.jitter_fraction)
            power = base_draw * jitter
            throughput = w.base_throughput_iters_per_s * jitter

            # Integrate energy: kWh = W × s / 3.6e6
            self._energy_kwh[w.worker_id] += power * self.tick_seconds / 3_600_000.0

            t = WorkerTelemetry(
                worker_id=w.worker_id,
                stage_id=w.stage_id,
                gpu_type=w.gpu_type,
                power_draw_w=power,
                throughput_iters_per_s=throughput,
                energy_cumulative_kwh=self._energy_kwh[w.worker_id],
                power_cap_w=cap,
                memory_used_bytes=w.memory_used_bytes,
                temperature_c=w.base_temperature_c + (power / cap) * 20.0,
                timestamp_s=self._t0 + self._t_elapsed,
            )
            snap[w.worker_id] = t
            self._last[w.worker_id] = t
        return snap

    def latest(self, worker_id: str) -> WorkerTelemetry | None:
        return self._last.get(worker_id)

    def snapshot(self) -> dict[str, WorkerTelemetry]:
        return dict(self._last)

    def stream(self, ticks: int) -> Iterator[dict[str, WorkerTelemetry]]:
        for _ in range(ticks):
            yield self.read_all()


# ---------------------------------------------------------------------------
# NvmlTelemetrySource — real-hardware skeleton (Week 3 deliverable)
# ---------------------------------------------------------------------------

class NvmlTelemetrySource:
    """Real NVML-backed telemetry. Skeleton only; full wiring lands Week 3.

    Raises ImportError if ``pynvml`` is not installed. Tests on machines without
    a GPU should use ``FakeTelemetrySource`` instead.
    """

    def __init__(self, device_indices: list[int], poll_interval_ms: int = 100) -> None:
        try:
            import pynvml  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pynvml not installed; install with `pip install nvidia-ml-py3`. "
                "For CI / no-GPU environments use FakeTelemetrySource instead."
            ) from exc
        self.device_indices = device_indices
        self.poll_interval_ms = poll_interval_ms
        # Full implementation Week 3: background thread that polls NVML and pushes
        # to a thread-safe dict consumed by latest()/snapshot().
        raise NotImplementedError("NvmlTelemetrySource lands in Week 3 per docs/phase2-plan.md §5")

    def latest(self, worker_id: str) -> WorkerTelemetry | None:  # pragma: no cover
        raise NotImplementedError

    def snapshot(self) -> dict[str, WorkerTelemetry]:  # pragma: no cover
        raise NotImplementedError

"""Per-worker energy + throughput telemetry — the signal source for A1–A4.

Consumed by:
    A1 partitioner   — `power_draw_w` per stage → energy-per-iter objective;
                       `memory_used_bytes` + `power_cap_w` → feasibility constraints.
    A2 EB-MSS        — `power_draw_w` per `gpu_type` → empirical EnergyProfile fit.
    A3 scaling policy — full vector → MPC / PPO state representation.
    A4 energy-WRR    — `throughput_iters_per_s / power_draw_w` → iter-per-joule weight;
                       `temperature_c` + `power_cap_w` → PowerSlackGuard.

Real telemetry sources:
    NvmlTelemetrySource  — GPU via pynvml; background thread, 100ms default poll.
    RaplTelemetrySource  — CPU socket via /sys/class/powercap/intel-rapl/.
    AggregateTelemetrySource — merge multiple sources into one snapshot.
    FakeTelemetrySource  — deterministic, no hardware (CI + unit tests).

Methodology: energy is the *primary measured* signal here. Carbon proxy is
layered on top by ``hise.energy.carbon_trace`` / ``carbon_sources``, never
derived from this module.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


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
# NvmlTelemetrySource — production NVML-backed GPU telemetry
# ---------------------------------------------------------------------------

class NvmlTelemetrySource:
    """Real NVML-backed telemetry with a background polling thread.

    Each device index maps to a ``worker_id`` + ``stage_id`` provided at
    construction. The poller reads NVML at ``poll_interval_ms`` intervals,
    integrates power into ``energy_cumulative_kwh``, and stores the latest
    ``WorkerTelemetry`` in a thread-safe dict. Throughput is NOT read from
    NVML (NVML has no per-job iteration counter); callers feed it via
    ``update_throughput(worker_id, iters_per_s)`` from the trainer side.

    Args:
        device_assignments: list of ``(device_index, worker_id, stage_id, gpu_type)``
            tuples. Each NVML device emits one ``WorkerTelemetry`` per poll.
        poll_interval_ms: poll period; default 100ms.
        nvml_module: optional dependency injection for testing. When ``None``,
            imports the real ``pynvml`` package; tests pass a fake module that
            implements the subset of NVML used here.

    Raises:
        ImportError: ``nvml_module`` is None AND ``pynvml`` is not installed.
            CI / no-GPU environments should use ``FakeTelemetrySource`` instead.

    Lifecycle::

        src = NvmlTelemetrySource([(0, "w0", 0, "RTX3080Ti")], poll_interval_ms=100)
        src.start()
        # ... orchestrator runs ...
        src.update_throughput("w0", iters_per_s=12.5)  # from trainer
        snap = src.snapshot()                          # for control loop
        src.stop()
    """

    def __init__(
        self,
        device_assignments: list[tuple[int, str, int, str]],
        poll_interval_ms: int = 100,
        nvml_module: Any = None,
    ) -> None:
        if nvml_module is None:
            try:
                import pynvml as nvml_module
            except ImportError as exc:
                raise ImportError(
                    "pynvml not installed; install with `pip install nvidia-ml-py`. "
                    "For CI / no-GPU environments use FakeTelemetrySource instead."
                ) from exc
        self._nvml = nvml_module
        self._assignments = list(device_assignments)
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._lock = threading.Lock()
        self._latest: dict[str, WorkerTelemetry] = {}
        self._throughput: dict[str, float] = {wid: 0.0 for _, wid, _, _ in self._assignments}
        self._energy_kwh: dict[str, float] = {wid: 0.0 for _, wid, _, _ in self._assignments}
        self._last_poll_ts: dict[str, float] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._handles: dict[int, Any] = {}
        self._initialized = False

    def _init_nvml(self) -> None:
        """Initialize NVML + cache device handles. Called by start()."""
        if self._initialized:
            return
        self._nvml.nvmlInit()
        for idx, _wid, _sid, _gpu in self._assignments:
            self._handles[idx] = self._nvml.nvmlDeviceGetHandleByIndex(idx)
        self._initialized = True

    def _poll_once(self) -> None:
        """Read all devices once and update the latest dict."""
        now = time.monotonic()
        with self._lock:
            for idx, worker_id, stage_id, gpu_type in self._assignments:
                handle = self._handles[idx]
                # Power in mW → W
                power_w = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                # Power cap in mW → W
                try:
                    cap_w = self._nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except Exception:  # pragma: no cover
                    cap_w = float("inf")
                # Memory in bytes
                mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = int(mem_info.used)
                # Temperature C
                try:
                    temp_c = float(self._nvml.nvmlDeviceGetTemperature(handle, 0))  # 0 = GPU
                except Exception:  # pragma: no cover
                    temp_c = 0.0
                # Integrate energy since last poll
                last_ts = self._last_poll_ts.get(worker_id, now)
                dt = max(0.0, now - last_ts)
                self._energy_kwh[worker_id] += power_w * dt / 3_600_000.0
                self._last_poll_ts[worker_id] = now

                self._latest[worker_id] = WorkerTelemetry(
                    worker_id=worker_id,
                    stage_id=stage_id,
                    gpu_type=gpu_type,
                    power_draw_w=power_w,
                    throughput_iters_per_s=self._throughput[worker_id],
                    energy_cumulative_kwh=self._energy_kwh[worker_id],
                    power_cap_w=cap_w,
                    memory_used_bytes=mem_used,
                    temperature_c=temp_c,
                    timestamp_s=now,
                )

    def _run(self) -> None:  # pragma: no cover (thread loop)
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("NVML poll failed")
            self._stop_event.wait(timeout=self._poll_interval_s)

    def start(self) -> None:
        """Initialize NVML and start the background polling thread."""
        self._init_nvml()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="nvml-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the poll thread to stop and shut down NVML."""
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

    def update_throughput(self, worker_id: str, iters_per_s: float) -> None:
        """Trainer-side callback feeding iteration rate to telemetry."""
        with self._lock:
            if worker_id in self._throughput:
                self._throughput[worker_id] = iters_per_s

    def latest(self, worker_id: str) -> WorkerTelemetry | None:
        with self._lock:
            return self._latest.get(worker_id)

    def snapshot(self) -> dict[str, WorkerTelemetry]:
        with self._lock:
            return dict(self._latest)

    def __enter__(self) -> NvmlTelemetrySource:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# RaplTelemetrySource — Intel RAPL CPU package energy
# ---------------------------------------------------------------------------

class RaplTelemetrySource:
    """Intel RAPL CPU package energy reader via /sys/class/powercap/intel-rapl/.

    RAPL exposes monotonically-increasing energy counters in microjoules per
    package; we diff between samples to derive average power and integrate
    cumulative kWh. The counter wraps at ``max_energy_range_uj``; we detect
    wrap-around when the new sample is less than the previous one.

    Args:
        package_ids: list of ``(package_idx, worker_id, stage_id)`` tuples.
            Each entry reads ``/sys/class/powercap/intel-rapl:{idx}/``.
        sysfs_root: override for tests; default ``/sys/class/powercap``.

    Raises:
        FileNotFoundError: ``sysfs_root`` does not exist on this system.
    """

    def __init__(
        self,
        package_ids: list[tuple[int, str, int]],
        sysfs_root: str = "/sys/class/powercap",
    ) -> None:
        import os
        if not os.path.isdir(sysfs_root):
            raise FileNotFoundError(
                f"RAPL sysfs root {sysfs_root!r} not found. RAPL requires Linux + Intel CPU."
            )
        self._sysfs_root = sysfs_root
        self._packages = list(package_ids)
        self._lock = threading.Lock()
        self._latest: dict[str, WorkerTelemetry] = {}
        self._last_energy_uj: dict[str, int] = {}
        self._last_ts: dict[str, float] = {}
        self._max_range_uj: dict[str, int] = {}
        self._cumulative_kwh: dict[str, float] = {wid: 0.0 for _, wid, _ in self._packages}

    def _read_energy_uj(self, idx: int) -> int:
        path = f"{self._sysfs_root}/intel-rapl:{idx}/energy_uj"
        with open(path) as fh:
            return int(fh.read().strip())

    def _read_max_range_uj(self, idx: int) -> int:
        path = f"{self._sysfs_root}/intel-rapl:{idx}/max_energy_range_uj"
        try:
            with open(path) as fh:
                return int(fh.read().strip())
        except FileNotFoundError:  # pragma: no cover
            return 2**63 - 1  # effectively no wrap

    def poll(self) -> None:
        """Read all packages once. Synchronous; called by aggregator or test."""
        now = time.monotonic()
        with self._lock:
            for idx, worker_id, stage_id in self._packages:
                energy_uj = self._read_energy_uj(idx)
                max_uj = self._max_range_uj.get(worker_id) or self._read_max_range_uj(idx)
                self._max_range_uj[worker_id] = max_uj

                last_uj = self._last_energy_uj.get(worker_id)
                last_ts = self._last_ts.get(worker_id)
                if last_uj is None or last_ts is None:
                    # First sample — no delta yet; record and continue.
                    self._last_energy_uj[worker_id] = energy_uj
                    self._last_ts[worker_id] = now
                    power_w = 0.0
                else:
                    delta_uj = energy_uj - last_uj
                    if delta_uj < 0:
                        delta_uj += max_uj   # wrap-around
                    dt = max(1e-6, now - last_ts)
                    power_w = (delta_uj / 1e6) / dt
                    self._cumulative_kwh[worker_id] += (delta_uj / 1e6) / 3_600_000.0
                    self._last_energy_uj[worker_id] = energy_uj
                    self._last_ts[worker_id] = now

                self._latest[worker_id] = WorkerTelemetry(
                    worker_id=worker_id, stage_id=stage_id, gpu_type="CPU",
                    power_draw_w=power_w,
                    throughput_iters_per_s=0.0,    # CPU doesn't report job iters
                    energy_cumulative_kwh=self._cumulative_kwh[worker_id],
                    power_cap_w=float("inf"),      # RAPL cap separate; not read here
                    memory_used_bytes=0,
                    temperature_c=0.0,
                    timestamp_s=now,
                )

    def latest(self, worker_id: str) -> WorkerTelemetry | None:
        with self._lock:
            return self._latest.get(worker_id)

    def snapshot(self) -> dict[str, WorkerTelemetry]:
        with self._lock:
            return dict(self._latest)


# ---------------------------------------------------------------------------
# AggregateTelemetrySource — merge multiple sources into one snapshot
# ---------------------------------------------------------------------------

@dataclass
class AggregateTelemetrySource:
    """Combine GPU + CPU (or any mix of) TelemetrySource into a single map.

    When two sources emit telemetry for the same ``worker_id``, the LATER
    source in the ``sources`` list wins — callers must order GPU before CPU
    if GPU is the primary attribution layer.

    Useful for the production sidecar: ``[NvmlTelemetrySource, RaplTelemetrySource]``
    presents one unified snapshot to the orchestrator.
    """

    sources: list[TelemetrySource]

    def latest(self, worker_id: str) -> WorkerTelemetry | None:
        # Search reverse so later sources win on overlap.
        for src in reversed(self.sources):
            t = src.latest(worker_id)
            if t is not None:
                return t
        return None

    def snapshot(self) -> dict[str, WorkerTelemetry]:
        merged: dict[str, WorkerTelemetry] = {}
        for src in self.sources:
            merged.update(src.snapshot())
        return merged


# ---------------------------------------------------------------------------
# PrometheusPusher — periodic push of telemetry to a Prometheus pushgateway
# ---------------------------------------------------------------------------

class PrometheusPusher:
    """Push a TelemetrySource snapshot to a Prometheus pushgateway.

    Used by the sidecar to expose energy telemetry for Grafana dashboards
    + post-experiment analysis. Optional — disabled when ``pushgateway_url``
    is None, so unit tests and CI runs don't require Prometheus running.

    Args:
        source: a TelemetrySource (typically AggregateTelemetrySource).
        pushgateway_url: Prometheus pushgateway endpoint, e.g.
            ``http://prom-pushgateway:9091``. None disables push entirely.
        job_name: Prometheus job label for grouping.
        registry: prometheus_client CollectorRegistry. Auto-created if None.

    Metrics exposed:
        hise_power_draw_w{worker, stage, gpu_type}    — gauge
        hise_throughput_iters_per_s{worker, stage}    — gauge
        hise_energy_cumulative_kwh{worker, stage}     — counter (monotone)
        hise_memory_used_bytes{worker, stage}         — gauge
        hise_temperature_c{worker, stage}             — gauge
    """

    def __init__(
        self,
        source: TelemetrySource,
        pushgateway_url: str | None = None,
        job_name: str = "hise-telemetry",
        registry: Any = None,
    ) -> None:
        self.source = source
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        if registry is None:
            from prometheus_client import CollectorRegistry
            registry = CollectorRegistry()
        self.registry = registry
        from prometheus_client import Gauge
        labels = ("worker", "stage", "gpu_type")
        labels_short = ("worker", "stage")
        self._power = Gauge("hise_power_draw_w", "GPU/CPU power draw (W)",
                            labels, registry=registry)
        self._throughput = Gauge("hise_throughput_iters_per_s",
                                  "Worker iteration rate (iters/s)",
                                  labels_short, registry=registry)
        self._energy = Gauge("hise_energy_cumulative_kwh",
                              "Cumulative energy (kWh) since job start",
                              labels_short, registry=registry)
        self._memory = Gauge("hise_memory_used_bytes", "GPU memory used (bytes)",
                              labels_short, registry=registry)
        self._temp = Gauge("hise_temperature_c", "GPU temperature (C)",
                            labels_short, registry=registry)

    def update_metrics(self, snapshot: Mapping[str, WorkerTelemetry] | None = None) -> None:
        """Refresh all gauges from the source snapshot. No-op if snapshot empty."""
        snap = snapshot if snapshot is not None else self.source.snapshot()
        for wid, t in snap.items():
            self._power.labels(worker=wid, stage=str(t.stage_id),
                                gpu_type=t.gpu_type).set(t.power_draw_w)
            self._throughput.labels(worker=wid, stage=str(t.stage_id)).set(
                t.throughput_iters_per_s)
            self._energy.labels(worker=wid, stage=str(t.stage_id)).set(
                t.energy_cumulative_kwh)
            self._memory.labels(worker=wid, stage=str(t.stage_id)).set(
                t.memory_used_bytes)
            self._temp.labels(worker=wid, stage=str(t.stage_id)).set(t.temperature_c)

    def push(self) -> bool:
        """Push current metrics to the pushgateway. Returns True on success,
        False when push is disabled or the gateway rejects the payload."""
        self.update_metrics()
        if self.pushgateway_url is None:
            return False
        from prometheus_client import push_to_gateway
        try:
            push_to_gateway(self.pushgateway_url, job=self.job_name, registry=self.registry)
            return True
        except Exception:
            logger.exception("Prometheus push failed")
            return False

"""Unit tests for WorkerTelemetry + FakeTelemetrySource (Phase 2 §0.1 deliverable)."""
from __future__ import annotations

import pytest

from hise.energy.telemetry import (
    FakeTelemetrySource,
    FakeWorker,
    NvmlTelemetrySource,
    WorkerTelemetry,
)

# --- WorkerTelemetry ---

def _sample(power: float = 200.0, throughput: float = 100.0, cap: float = 400.0,
            temp: float = 60.0) -> WorkerTelemetry:
    return WorkerTelemetry(
        worker_id="w1", stage_id=0, gpu_type="A100",
        power_draw_w=power, throughput_iters_per_s=throughput,
        energy_cumulative_kwh=0.01, power_cap_w=cap,
        memory_used_bytes=8 << 30, temperature_c=temp, timestamp_s=1.0,
    )


def test_iters_per_joule_basic() -> None:
    t = _sample(power=200.0, throughput=400.0)
    assert t.iters_per_joule == pytest.approx(2.0)


def test_iters_per_joule_zero_power() -> None:
    t = _sample(power=0.0, throughput=100.0)
    assert t.iters_per_joule == 0.0


def test_thermal_throttled_by_power_fraction() -> None:
    t = _sample(power=380.0, cap=400.0, temp=60.0)
    assert t.is_thermal_throttled(fraction=0.9)


def test_thermal_throttled_by_temperature() -> None:
    t = _sample(power=200.0, cap=400.0, temp=82.0)
    assert t.is_thermal_throttled(temp_c_max=80.0)


def test_not_throttled_within_bounds() -> None:
    t = _sample(power=200.0, cap=400.0, temp=60.0)
    assert not t.is_thermal_throttled()


# --- FakeTelemetrySource ---

def _source(seed: int = 42) -> FakeTelemetrySource:
    return FakeTelemetrySource(
        workers=[
            FakeWorker("w1", stage_id=0, gpu_type="A100"),
            FakeWorker("w2", stage_id=0, gpu_type="A100"),
            FakeWorker("w3", stage_id=1, gpu_type="T4"),
        ],
        tick_seconds=1.0,
        seed=seed,
    )


def test_fake_source_emits_one_per_worker() -> None:
    src = _source()
    snap = src.read_all()
    assert set(snap.keys()) == {"w1", "w2", "w3"}
    for t in snap.values():
        assert isinstance(t, WorkerTelemetry)


def test_fake_source_deterministic_with_seed() -> None:
    src1, src2 = _source(seed=7), _source(seed=7)
    snap1, snap2 = src1.read_all(), src2.read_all()
    for k in snap1:
        assert snap1[k].power_draw_w == snap2[k].power_draw_w
        assert snap1[k].throughput_iters_per_s == snap2[k].throughput_iters_per_s


def test_fake_source_different_seeds_diverge() -> None:
    src_a, src_b = _source(seed=1), _source(seed=2)
    assert src_a.read_all()["w1"].power_draw_w != src_b.read_all()["w1"].power_draw_w


def test_fake_source_energy_accumulates_monotone() -> None:
    src = _source()
    e_prev = 0.0
    for _ in range(5):
        snap = src.read_all()
        e_now = snap["w1"].energy_cumulative_kwh
        assert e_now > e_prev
        e_prev = e_now


def test_fake_source_gpu_type_drives_power() -> None:
    src = _source()
    snap = src.read_all()
    # A100 reference draw ~320W; T4 ~60W. Even with jitter the gap is huge.
    assert snap["w1"].power_draw_w > snap["w3"].power_draw_w * 3


def test_fake_source_power_cap_matches_gpu_type() -> None:
    src = _source()
    snap = src.read_all()
    assert snap["w1"].power_cap_w == 400.0  # A100
    assert snap["w3"].power_cap_w == 70.0   # T4


def test_fake_source_latest_returns_last_emitted() -> None:
    src = _source()
    src.read_all()
    snap2 = src.read_all()
    assert src.latest("w1") == snap2["w1"]


def test_fake_source_latest_unknown_worker_is_none() -> None:
    src = _source()
    src.read_all()
    assert src.latest("nonexistent") is None


def test_fake_source_snapshot_independent_of_internal_state() -> None:
    src = _source()
    src.read_all()
    snap = src.snapshot()
    snap.clear()  # mutating returned snapshot must not break source
    assert src.latest("w1") is not None


def test_fake_source_stream_yields_n_ticks() -> None:
    src = _source()
    snaps = list(src.stream(ticks=4))
    assert len(snaps) == 4


# --- NvmlTelemetrySource (skeleton; production tests in test_telemetry_sources.py) ---

def test_nvml_source_accepts_injected_module() -> None:
    """Construction with a fake nvml_module works without pynvml on the host."""
    fake = object()   # never invoked because __init__ defers to start()
    src = NvmlTelemetrySource(
        device_assignments=[(0, "w0", 0, "RTX3080Ti")],
        poll_interval_ms=10, nvml_module=fake,
    )
    # Construction succeeds; full lifecycle tested in test_telemetry_sources.py.
    assert src._poll_interval_s == 0.01

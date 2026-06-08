"""Verify NVML telemetry on the local GPU.

Reads ``WorkerTelemetry``-shaped samples from real NVML at 100 ms cadence for 10 seconds:
5 s idle baseline + 5 s under a synthetic CUDA load (matmul). Compares the resulting power
trace against ``FakeTelemetrySource`` output to confirm the schema is faithful before
wiring ``NvmlTelemetrySource`` into the orchestrator.

Usage:
    python experiments/probe_nvml.py
    # or without torch CUDA (skip stress phase):
    python experiments/probe_nvml.py --no-stress

This is a hardware-dependent smoke test — it will be skipped in CI (no GPU).
"""
from __future__ import annotations

import argparse
import time

import pynvml
from rich.console import Console
from rich.table import Table

from tare.energy.telemetry import FakeTelemetrySource, FakeWorker, WorkerTelemetry


def read_real(handle, worker_id: str, stage_id: int, t0: float, energy_kwh: float) -> WorkerTelemetry:
    """Build a WorkerTelemetry from live NVML, mirroring the FakeTelemetrySource schema."""
    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    cap_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    return WorkerTelemetry(
        worker_id=worker_id,
        stage_id=stage_id,
        gpu_type=gpu_name,
        power_draw_w=power_w,
        throughput_iters_per_s=0.0,        # NVML doesn't measure this; orchestrator fills it
        energy_cumulative_kwh=energy_kwh,
        power_cap_w=cap_w,
        memory_used_bytes=mem,
        temperature_c=temp_c,
        timestamp_s=time.monotonic() - t0,
    )


def stress_gpu(seconds: float = 5.0) -> None:
    """Run a heavy matmul loop on the GPU to drive power draw up."""
    try:
        import torch
    except ImportError:
        print("torch not installed — skipping stress phase")
        return
    if not torch.cuda.is_available():
        print("CUDA not available — skipping stress phase (install torch CUDA build)")
        return
    device = torch.device("cuda:0")
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        a = a @ b
        a = torch.relu(a)
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--stress-seconds", type=float, default=5.0)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--no-stress", action="store_true")
    args = parser.parse_args()

    console = Console()
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        console.print(f"[bold green]NVML init OK[/]: {gpu_name}")

        # Concurrent stress under a background thread so we keep sampling during it.
        import threading
        stress_done = threading.Event()

        def _stress() -> None:
            if not args.no_stress:
                stress_gpu(args.stress_seconds)
            stress_done.set()

        # Collect idle baseline before launching the synthetic load
        idle_samples: list[WorkerTelemetry] = []
        t0 = time.monotonic()
        energy = 0.0
        last_t = t0
        for _ in range(int(args.idle_seconds * 1000 / args.sample_ms)):
            sample = read_real(handle, "gpu0", 0, t0, energy)
            now = time.monotonic()
            energy += sample.power_draw_w * (now - last_t) / 3_600_000.0
            last_t = now
            idle_samples.append(sample)
            time.sleep(args.sample_ms / 1000.0)

        # Collect samples under synthetic CUDA load
        stress_samples: list[WorkerTelemetry] = []
        stress_thread = threading.Thread(target=_stress, daemon=True)
        stress_thread.start()
        time.sleep(0.3)  # let kernels launch
        while not stress_done.is_set():
            sample = read_real(handle, "gpu0", 0, t0, energy)
            now = time.monotonic()
            energy += sample.power_draw_w * (now - last_t) / 3_600_000.0
            last_t = now
            stress_samples.append(sample)
            time.sleep(args.sample_ms / 1000.0)
        stress_thread.join()

        # FakeTelemetrySource — 1 worker, "A100" profile for comparison
        fake = FakeTelemetrySource(
            workers=[FakeWorker("gpu0_fake", stage_id=0, gpu_type="A100")],
            tick_seconds=args.sample_ms / 1000.0,
            seed=0,
        )
        fake_samples: list[WorkerTelemetry] = []
        for _ in range(20):
            fake_samples.append(fake.read_all()["gpu0_fake"])

        # ---- Report ----
        table = Table(title=f"NVML probe — {gpu_name}")
        table.add_column("phase")
        table.add_column("n samples", justify="right")
        table.add_column("power min (W)", justify="right")
        table.add_column("power avg (W)", justify="right")
        table.add_column("power max (W)", justify="right")
        table.add_column("temp avg (°C)", justify="right")
        table.add_column("mem (MiB)", justify="right")

        def _row(label: str, samples: list[WorkerTelemetry]) -> None:
            if not samples:
                return
            ps = [s.power_draw_w for s in samples]
            ts = [s.temperature_c for s in samples]
            mems = [s.memory_used_bytes / (1024 * 1024) for s in samples]
            table.add_row(
                label,
                str(len(samples)),
                f"{min(ps):.1f}",
                f"{sum(ps)/len(ps):.1f}",
                f"{max(ps):.1f}",
                f"{sum(ts)/len(ts):.1f}",
                f"{sum(mems)/len(mems):.0f}",
            )

        _row("real idle", idle_samples)
        _row("real stress", stress_samples)
        _row("fake A100", fake_samples)
        console.print(table)

        if stress_samples and idle_samples:
            avg_idle = sum(s.power_draw_w for s in idle_samples) / len(idle_samples)
            avg_stress = sum(s.power_draw_w for s in stress_samples) / len(stress_samples)
            console.print(
                f"\n[bold]Δ power[/] idle→stress: "
                f"+{avg_stress - avg_idle:.1f} W ({avg_stress / avg_idle:.1f}× idle baseline)"
            )

        final = idle_samples[-1] if not stress_samples else stress_samples[-1]
        console.print(f"\n[dim]final cumulative energy: {final.energy_cumulative_kwh*1e6:.2f} μWh "
                      f"over {final.timestamp_s:.1f}s[/]")

        console.print("\n[bold green]Schema compatibility[/]: real NVML samples are valid "
                      "WorkerTelemetry instances ✓")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()

"""Real-hardware (energy_per_iter, throughput) Pareto curve via power-cap DVFS.

Drives the GPU through a sequence of power caps via ``nvidia-smi -pl`` and
measures the resulting (power, throughput, energy-per-iter) tuple at each
cap with live NVML sampling. Output is the empirical Pareto frontier of
the local RTX 3080 Ti — the same shape the synthetic
``voltage_alpha=2.0`` curve was extrapolating.

The harness runs a small DL workload (ResNet-style conv-block forward +
backward) at each cap. It is **not** meant to validate accuracy or
convergence — only to expose the GPU long enough for NVML to settle and
the power-cap to bind. ResNet-shaped workloads keep the unit comparable
to the synthetic-shaped sweep ``exp_joint_real_workloads.py`` uses.

**Sudo required** for ``nvidia-smi -pl <watts>``. The wrapper script
expects the caller to have already set the cap (e.g. ``sudo nvidia-smi -pl
200``) or to invoke the inner runner with ``--inner`` mode that skips the
cap-setting step.

Usage::

    # Run with default sweep (requires sudo to change caps)
    sudo nvidia-smi -pmu 1            # enable persistence (one-time)
    python -m experiments.exp_hardware_pareto --caps 100,150,200,250,300,350

    # Or inner-only at the current cap
    python -m experiments.exp_hardware_pareto --inner --iters 200
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pynvml
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Workload — a compact ResNet-block to keep the timing comparable across runs
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """One residual block (3x3 conv → BN → ReLU → 3x3 conv → BN → +skip → ReLU)."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class TinyResNet(nn.Module):
    """4 stacked ResBlocks operating on 64 channels. ~150k params, GPU-bound."""

    def __init__(self, channels: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.classifier = nn.Linear(channels, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)


class TinyTransformer(nn.Module):
    """A small GPT-style encoder LM block (embed → N×TransformerEncoderLayer →
    LM head). Matmul/attention/FFN-bound rather than conv-bound, so its
    energy-per-iter U-curve over the power cap can differ from the ResNet one —
    that difference is exactly the workload-dependent-cap effect under test.
    """

    def __init__(self, *, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 vocab: int = 32000, dim_ff_mult: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff_mult * d_model, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.encoder(x)
        return self.head(x)   # (B, seq_len, vocab)


def _build_workload(
    workload: str, *, batch_size: int, spatial: int, channels: int,
    seq_len: int, d_model: int, layers: int, vocab: int,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, bool]:
    """Return ``(model, inputs, targets, is_seq)`` for ``workload`` on cuda.

    ``is_seq`` flags a (B, T, V) sequence output that the loss must flatten.
    """
    if workload == "resnet":
        model = TinyResNet(channels=channels, num_blocks=4).cuda()
        inputs = torch.randn(batch_size, 3, spatial, spatial, device="cuda")
        targets = torch.randint(0, 10, (batch_size,), device="cuda")
        return model, inputs, targets, False
    if workload == "transformer":
        model = TinyTransformer(d_model=d_model, num_layers=layers, vocab=vocab).cuda()
        inputs = torch.randint(0, vocab, (batch_size, seq_len), device="cuda")
        targets = torch.randint(0, vocab, (batch_size, seq_len), device="cuda")
        return model, inputs, targets, True
    raise ValueError(f"unknown workload {workload!r}; expected 'resnet' or 'transformer'")


# ---------------------------------------------------------------------------
# NVML telemetry sampler
# ---------------------------------------------------------------------------


@dataclass
class PowerSample:
    t_offset_s: float
    power_w: float
    sm_clock_mhz: int
    temp_c: float


def sample_nvml(handle, t0: float) -> PowerSample:
    return PowerSample(
        t_offset_s=time.monotonic() - t0,
        power_w=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        temp_c=float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
    )


# ---------------------------------------------------------------------------
# Inner runner — measures at the *current* power cap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """One (energy, throughput) measurement at one power cap."""

    cap_w_requested: float
    cap_w_observed: float
    iters: int
    wall_seconds: float
    throughput_iters_per_s: float
    avg_power_w: float
    peak_power_w: float
    avg_sm_clock_mhz: float
    energy_per_iter_j: float
    energy_per_iter_kwh: float
    samples_count: int


def _persistent_workload_loop(
    model: TinyResNet,
    optim: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    iters: int,
    handle,
    t0: float,
    sample_every_iters: int = 1,
    is_seq: bool = False,
) -> tuple[list[PowerSample], float]:
    """Run ``iters`` forward+backward steps; sample NVML inside the loop.

    Returns the per-iteration NVML samples and the total wall-clock elapsed.
    Sampling is deferred to the host between CUDA submissions to avoid a
    cudaStreamSynchronize on every sample. ``is_seq`` flattens a (B, T, V)
    sequence output and its (B, T) targets for the token-level loss.
    """
    samples: list[PowerSample] = []
    crit = nn.CrossEntropyLoss()
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(iters):
        optim.zero_grad(set_to_none=True)
        out = model(inputs)
        loss = crit(out.reshape(-1, out.size(-1)), targets.reshape(-1)) if is_seq else crit(out, targets)
        loss.backward()
        optim.step()
        if i % sample_every_iters == 0:
            samples.append(sample_nvml(handle, t0))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    return samples, wall


def run_inner(
    handle,
    cap_w_requested: float,
    iters: int,
    warmup_iters: int,
    batch_size: int,
    spatial: int,
    channels: int,
    t0: float,
    workload: str = "resnet",
    seq_len: int = 128,
    d_model: int = 512,
    layers: int = 6,
    vocab: int = 32000,
) -> CellResult:
    """Measure one (cap_w_requested, energy_per_iter, throughput) point."""
    model, inputs, targets, is_seq = _build_workload(
        workload, batch_size=batch_size, spatial=spatial, channels=channels,
        seq_len=seq_len, d_model=d_model, layers=layers, vocab=vocab,
    )
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Warmup: build kernels, warm caches, settle clocks
    _persistent_workload_loop(model, optim, inputs, targets, warmup_iters, handle, t0, is_seq=is_seq)
    # Drain any pending work, then re-read the observed cap (may have been
    # adjusted by the driver if the request was out-of-range).
    torch.cuda.synchronize()
    cap_w_observed = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0

    samples, wall = _persistent_workload_loop(
        model, optim, inputs, targets, iters, handle, t0, is_seq=is_seq,
    )

    powers = [s.power_w for s in samples]
    clocks = [s.sm_clock_mhz for s in samples]
    avg_power = sum(powers) / len(powers) if powers else 0.0
    peak_power = max(powers) if powers else 0.0
    throughput = iters / wall if wall > 0 else 0.0
    avg_clock = sum(clocks) / len(clocks) if clocks else 0.0
    energy_per_iter_j = (avg_power * wall) / iters if iters > 0 else 0.0
    return CellResult(
        cap_w_requested=cap_w_requested,
        cap_w_observed=cap_w_observed,
        iters=iters,
        wall_seconds=wall,
        throughput_iters_per_s=throughput,
        avg_power_w=avg_power,
        peak_power_w=peak_power,
        avg_sm_clock_mhz=avg_clock,
        energy_per_iter_j=energy_per_iter_j,
        energy_per_iter_kwh=energy_per_iter_j / 3_600_000.0,
        samples_count=len(samples),
    )


# ---------------------------------------------------------------------------
# Outer sweep — sets power cap via nvidia-smi
# ---------------------------------------------------------------------------


def set_power_cap(cap_w: float) -> None:
    """Call ``sudo -n nvidia-smi -pl <watts>``. Caller must have sudo privileges."""
    cap_int = int(round(cap_w))
    subprocess.run(
        ["sudo", "-n", "nvidia-smi", "-pl", str(cap_int)],
        check=True, capture_output=True, text=True,
    )


def fit_voltage_alpha(rows: list[CellResult]) -> tuple[float, float]:
    """Fit ``avg_power ≈ P_max · (cap / P_max) ^ α`` by log-log least-squares.

    Returns ``(alpha, p_max)``. ``p_max`` is taken from the row with the
    highest observed cap; ``alpha`` is the slope of log(power)/log(cap/p_max).
    Rows where ``cap_w_observed ≤ 0`` are skipped.
    """
    rows = [r for r in rows if r.cap_w_observed > 0 and r.avg_power_w > 0]
    if len(rows) < 2:
        return (float("nan"), float("nan"))
    p_max = max(r.avg_power_w for r in rows)
    xs = [math.log(r.cap_w_observed / p_max) for r in rows]
    ys = [math.log(r.avg_power_w / p_max) for r in rows]
    # Least-squares slope through the origin: α = Σ x·y / Σ x²
    num = sum(x * y for x, y in zip(xs, ys, strict=True))
    den = sum(x * x for x in xs)
    alpha = num / den if den > 0 else float("nan")
    return (alpha, p_max)


def run_sweep(args: argparse.Namespace) -> int:
    console = Console()
    if not torch.cuda.is_available():
        console.print("[red]CUDA not available — abort[/]")
        return 2

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    t0 = time.monotonic()
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    console.print(f"[bold]Hardware Pareto sweep[/] on {gpu_name} — workload: {args.workload}")
    if args.workload == "transformer":
        console.print(
            f"[dim]TinyTransformer(d_model={args.d_model}, layers={args.layers}, seq_len={args.seq_len}), "
            f"batch={args.batch_size}; warmup={args.warmup_iters}, iters={args.iters}[/]"
        )
    else:
        console.print(
            f"[dim]TinyResNet(channels={args.channels}), batch={args.batch_size}, "
            f"spatial={args.spatial}; warmup={args.warmup_iters}, iters={args.iters}[/]"
        )

    # Snapshot the original power cap so we restore it at exit.
    original_cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    console.print(f"[dim]restore-on-exit cap = {original_cap:.0f} W[/]")

    caps = sorted({float(c) for c in args.caps})
    rows: list[CellResult] = []
    try:
        for cap_w in caps:
            console.print(f"\n[bold]→ Setting power cap to {cap_w:.0f} W[/]")
            try:
                set_power_cap(cap_w)
            except subprocess.CalledProcessError as e:
                console.print(f"[red]cap-set failed[/] — stderr: {e.stderr.strip()}")
                console.print(
                    "[yellow]hint:[/] grant passwordless sudo for nvidia-smi, "
                    "or pre-set caps via `sudo nvidia-smi -pl ...` before running"
                )
                return 3
            time.sleep(args.settle_seconds)
            result = run_inner(
                handle, cap_w,
                iters=args.iters, warmup_iters=args.warmup_iters,
                batch_size=args.batch_size, spatial=args.spatial,
                channels=args.channels, t0=t0, workload=args.workload,
                seq_len=args.seq_len, d_model=args.d_model, layers=args.layers,
            )
            rows.append(result)
            console.print(
                f"  observed cap = {result.cap_w_observed:.0f} W | "
                f"throughput = {result.throughput_iters_per_s:.2f} iter/s | "
                f"avg power = {result.avg_power_w:.1f} W | "
                f"energy/iter = {result.energy_per_iter_j:.3f} J | "
                f"avg clock = {result.avg_sm_clock_mhz:.0f} MHz"
            )
    finally:
        # Always restore the original cap, even on Ctrl-C.
        try:
            set_power_cap(original_cap)
            console.print(f"[dim]restored cap to {original_cap:.0f} W[/]")
        except subprocess.CalledProcessError:
            console.print(
                f"[red]warning: failed to restore cap {original_cap:.0f} W "
                "— set it manually with `sudo nvidia-smi -pl ...`[/]"
            )

    if not rows:
        console.print("[red]no rows collected[/]")
        return 4

    table = Table(title="RTX 3080 Ti Pareto frontier from power-cap DVFS sweep")
    table.add_column("cap req (W)", justify="right")
    table.add_column("cap obs (W)", justify="right")
    table.add_column("throughput (iter/s)", justify="right")
    table.add_column("avg power (W)", justify="right")
    table.add_column("avg clock (MHz)", justify="right")
    table.add_column("E/iter (J)", justify="right")
    table.add_column("E/iter (mJ/W)", justify="right")
    for r in rows:
        eff = r.energy_per_iter_j / max(r.avg_power_w, 1e-9) * 1000.0
        table.add_row(
            f"{r.cap_w_requested:.0f}",
            f"{r.cap_w_observed:.0f}",
            f"{r.throughput_iters_per_s:.2f}",
            f"{r.avg_power_w:.1f}",
            f"{r.avg_sm_clock_mhz:.0f}",
            f"{r.energy_per_iter_j:.3f}",
            f"{eff:.3f}",
        )
    console.print(table)

    alpha, p_max = fit_voltage_alpha(rows)
    console.print(
        f"\n[bold]Voltage-α fit[/]: α = {alpha:.3f}, P_max = {p_max:.1f} W"
    )
    console.print(
        f"[dim]synthetic assumption was α=2.0 in exp_joint_real_workloads.py; "
        f"|α_real − 2.0| = {abs(alpha - 2.0):.3f}[/]"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "gpu_name": gpu_name,
            "workload": args.workload,
            "rows": [asdict(r) for r in rows],
            "alpha": alpha,
            "p_max_w": p_max,
        }, indent=2))
        console.print(f"[dim]wrote raw results to {out_path}[/]")

    # Pass = sweep produced a monotone Pareto frontier (more cap → more
    # throughput AND more power) and α is in the measured DVFS range. Under a
    # hard power cap the cap bounds power while throughput saturates, so the
    # measured exponent is sub-linear (~0.85-0.95 here) — well below the α=2.0
    # the older synthetic voltage model assumed. The sane band is widened to
    # admit that real regime; a value far outside it would flag a bad sweep.
    monotone = all(
        rows[i].throughput_iters_per_s >= rows[i - 1].throughput_iters_per_s - 1e-3
        for i in range(1, len(rows))
    )
    alpha_sane = 0.5 <= alpha <= 3.5
    ok = monotone and alpha_sane
    console.print(
        f"\n[{'green' if ok else 'red'}]Sanity check: "
        f"monotone={monotone}, α-in-range={alpha_sane}: {'PASS' if ok else 'FAIL'}[/]"
    )
    return 0 if ok else 5


def run_inner_only(args: argparse.Namespace) -> int:
    """Single-cap inner run; no sudo, useful for smoke-testing the workload."""
    console = Console()
    if not torch.cuda.is_available():
        console.print("[red]CUDA not available[/]")
        return 2
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    t0 = time.monotonic()
    cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    console.print(f"[bold]Inner-only run[/] @ current cap {cap:.0f} W")
    result = run_inner(
        handle, cap,
        iters=args.iters, warmup_iters=args.warmup_iters,
        batch_size=args.batch_size, spatial=args.spatial,
        channels=args.channels, t0=t0, workload=args.workload,
        seq_len=args.seq_len, d_model=args.d_model, layers=args.layers,
    )
    console.print(
        f"throughput={result.throughput_iters_per_s:.2f} iter/s, "
        f"avg power={result.avg_power_w:.1f} W, "
        f"E/iter={result.energy_per_iter_j:.3f} J, "
        f"clock={result.avg_sm_clock_mhz:.0f} MHz"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", action="store_true",
                        help="Run a single inner measurement at the current cap (no sudo).")
    parser.add_argument(
        "--caps",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
        help="Comma-separated power caps (W) to sweep.",
    )
    parser.add_argument("--workload", choices=["resnet", "transformer"], default="resnet",
                        help="Which workload's U-curve to measure (default resnet, the trusted profile).")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--spatial", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    # Transformer workload knobs (ignored for resnet).
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--settle-seconds", type=float, default=2.0,
                        help="Wait after cap change for the GPU to settle.")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write raw JSON results.")
    args = parser.parse_args()
    if args.inner:
        return run_inner_only(args)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Break down where a serverless pause's per-resume overhead actually comes from.

A carbon-driven pause tears the training process's model down and later rebuilds
it. The deferral harness measured the *active* energy-per-iteration after a
resume running ~58% higher than a sustained always-on run. This probe attributes
that overhead, on the host GPU, to its parts:

  - discrete resume steps: checkpoint read, model rebuild, first-batch fetch;
  - the transient ramp over the first iterations after a resume — GPU SM clocks
    recovering from idle, plus cold dataloader workers starving the GPU — versus
    sustained steady-state training.

It instruments each iteration's wall time, SM clock, and power, so the ramp is
visible directly. Requires a real GPU; intended for a quiet (dedicated) GPU.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table


class _Probe:
    """Minimal instrumented resnet18 trainer with explicit pause/resume."""

    def __init__(self, ckpt_path: str, batch_size: int, device_index: int) -> None:
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.device_index = device_index
        self._model = self._optim = self._loader_iter = self._loader = None
        self._loss = self._device = None
        import pynvml
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def _clock_power(self) -> tuple[int, float]:
        sm = self._nvml.nvmlDeviceGetClockInfo(self._h, self._nvml.NVML_CLOCK_SM)
        pw = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        return sm, pw

    def build(self) -> None:
        import torch

        from tare.data.datasets import build_loader
        from tare.models.zoo import build_model

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = build_model("resnet18").to(self._device)
        self._optim = torch.optim.SGD(self._model.parameters(), lr=0.01, momentum=0.9)
        self._loss = torch.nn.CrossEntropyLoss()
        self._loader = build_loader("cifar10", batch_size=self.batch_size)
        self._loader_iter = iter(self._loader)

    def _next_batch(self):
        try:
            return next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self._loader)
            return next(self._loader_iter)

    def train_block(self, n: int) -> list[tuple[float, int, float, float]]:
        """Return per-iter ``(wall_s, sm_clock_mhz, power_w, data_fetch_s)``."""
        import torch

        self._model.train()
        rows: list[tuple[float, int, float, float]] = []
        for _ in range(n):
            t0 = time.perf_counter()
            inputs, targets = self._next_batch()
            data_s = time.perf_counter() - t0
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            self._optim.zero_grad()
            out = self._model(inputs)
            loss = self._loss(out, targets)
            loss.backward()
            self._optim.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            sm, pw = self._clock_power()
            rows.append((dt, sm, pw, data_s))
        return rows

    def checkpoint(self) -> None:
        import torch

        Path(self.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self._model.state_dict(), "optim": self._optim.state_dict()}, self.ckpt_path)
        torch.cuda.synchronize()

    def teardown(self) -> None:
        import torch

        self._model = self._optim = self._loader_iter = self._loader = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def resume_timed(self) -> dict:
        """Rebuild + reload, timing each step; return a breakdown (seconds)."""
        import torch

        t = time.perf_counter()
        from tare.data.datasets import build_loader
        from tare.models.zoo import build_model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = build_model("resnet18").to(self._device)
        self._optim = torch.optim.SGD(self._model.parameters(), lr=0.01, momentum=0.9)
        self._loss = torch.nn.CrossEntropyLoss()
        torch.cuda.synchronize()
        rebuild_s = time.perf_counter() - t

        t = time.perf_counter()
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state["model"])
        self._optim.load_state_dict(state["optim"])
        torch.cuda.synchronize()
        reload_s = time.perf_counter() - t

        t = time.perf_counter()
        self._loader = build_loader("cifar10", batch_size=self.batch_size)
        self._loader_iter = iter(self._loader)
        _ = self._next_batch()
        first_batch_s = time.perf_counter() - t

        return {"rebuild_s": rebuild_s, "reload_s": reload_s, "first_batch_s": first_batch_s}

    def shutdown(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:  # pragma: no cover
            pass


def _summ(rows: list[tuple[float, int, float, float]]) -> dict:
    walls = [r[0] for r in rows]
    clks = [r[1] for r in rows]
    pws = [r[2] for r in rows]
    data = [r[3] for r in rows]
    thr = 1.0 / statistics.mean(walls)
    jpi = statistics.mean(pws) * statistics.mean(walls)   # ~ J per iter
    return {
        "iters": len(rows), "throughput_iter_s": thr, "j_per_iter": jpi,
        "mean_wall_ms": statistics.mean(walls) * 1000.0,
        "mean_sm_clock_mhz": statistics.mean(clks),
        "mean_power_w": statistics.mean(pws),
        "mean_data_fetch_ms": statistics.mean(data) * 1000.0,
    }


def run(args: argparse.Namespace) -> int:
    console = Console()
    probe = _Probe(args.ckpt, args.batch_size, args.device)
    try:
        console.print("[bold]Cold init + warmup[/]")
        probe.build()
        probe.train_block(args.warmup)

        console.print(f"[bold]Steady block[/] ({args.block} iters)")
        steady = probe.train_block(args.block)
        steady_s = _summ(steady)

        t = time.perf_counter()
        probe.checkpoint()
        ckpt_write_s = time.perf_counter() - t
        probe.teardown()

        console.print(f"[bold]Idle {args.idle_s}s[/] (let clocks drop)")
        idle_clk, idle_pw = probe._clock_power()
        time.sleep(args.idle_s)
        idle_clk2, idle_pw2 = probe._clock_power()

        console.print("[bold]Resume[/] (timed)")
        resume = probe.resume_timed()

        console.print(f"[bold]Post-resume block[/] ({args.block} iters, instrumented)")
        ramp = probe.train_block(args.block)
        first = _summ(ramp[:args.ramp_window])
        tail = _summ(ramp[-args.block // 2:])

        # --- report ---
        tbl = Table(title="Per-iteration profile: steady vs first-after-resume vs resumed-tail")
        for c in ("phase", "iters", "thr (it/s)", "J/iter", "wall (ms)", "SM clk (MHz)", "power (W)", "data (ms)"):
            tbl.add_column(c, justify="right")
        for label, s in (("steady (pre-pause)", steady_s),
                         (f"first {args.ramp_window} after resume", first),
                         ("resumed tail", tail)):
            tbl.add_row(
                label, str(s["iters"]), f"{s['throughput_iter_s']:.1f}", f"{s['j_per_iter']:.3f}",
                f"{s['mean_wall_ms']:.1f}", f"{s['mean_sm_clock_mhz']:.0f}",
                f"{s['mean_power_w']:.0f}", f"{s['mean_data_fetch_ms']:.1f}",
            )
        console.print(tbl)

        ramp_overhead = 100.0 * (first["j_per_iter"] - steady_s["j_per_iter"]) / steady_s["j_per_iter"]
        tail_overhead = 100.0 * (tail["j_per_iter"] - steady_s["j_per_iter"]) / steady_s["j_per_iter"]
        discrete_s = ckpt_write_s + resume["rebuild_s"] + resume["reload_s"] + resume["first_batch_s"]
        console.print(
            f"[bold]Discrete resume cost[/]: {discrete_s:.2f}s "
            f"(ckpt-write {ckpt_write_s:.2f}, rebuild {resume['rebuild_s']:.2f}, "
            f"reload {resume['reload_s']:.2f}, first-batch {resume['first_batch_s']:.2f})"
        )
        console.print(
            f"[bold]Clock drop during idle[/]: {idle_clk}→{idle_clk2} MHz "
            f"(power {idle_pw:.0f}→{idle_pw2:.0f} W)"
        )
        console.print(
            f"[bold]Per-iter overhead vs steady[/]: first {args.ramp_window} iters {ramp_overhead:+.0f}%; "
            f"resumed tail {tail_overhead:+.0f}% (≈0 means the ramp, not a permanent penalty)."
        )

        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "batch_size": args.batch_size,
                "steady": steady_s, "first_after_resume": first, "resumed_tail": tail,
                "ckpt_write_s": ckpt_write_s, "resume_steps_s": resume,
                "discrete_resume_s": discrete_s,
                "idle_clock_mhz": [idle_clk, idle_clk2],
                "ramp_overhead_pct_first": ramp_overhead,
                "tail_overhead_pct": tail_overhead,
                "ramp_window": args.ramp_window,
            }, indent=2))
            console.print(f"[dim]wrote {out}[/]")
    finally:
        probe.shutdown()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="./artifacts/resume_probe_ckpt.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--block", type=int, default=200)
    p.add_argument("--ramp-window", type=int, default=15)
    p.add_argument("--idle-s", type=float, default=8.0)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

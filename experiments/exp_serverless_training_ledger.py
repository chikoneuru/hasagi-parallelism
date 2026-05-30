"""Serverless training carbon ledger — real training, real scale-to-zero, real energy.

This harness joins the two tracks that previously ran disjointly: the carbon
policy now drives a *real* Knative scale-to-zero, and a *real* resnet18 training
job on the host GPU pays the *real* cost of being paused and resumed. The host
NVML stream is the source of truth for energy; the pod's scale lifecycle is the
serverless control signal. Energy is attributed to lifecycle phases by
``PodEnergyLedger`` and converted to carbon as ``energy × grid-intensity``.

The question it answers: when a stateful, multi-hour training job is paused on a
high-carbon hour and resumed later, does carbon-aware scale-to-zero actually save
net carbon once the *training-specific resume cost* is charged — checkpoint
write/read, optimiser-state reload, CUDA re-initialisation, and first-iteration
warmup — that stateless-function carbon schemes never incur?

Two runs are compared on the same intensity schedule:
  - carbon-aware : pause (checkpoint + scale-to-zero) while intensity is above a
                   threshold; resume (cold-start + reload) when it drops.
  - always-on    : one initial cold start, then train through every tick.

Both meter the GPU with NVML and bill carbon at the per-tick intensity. The
delta is the honest headline number.

Requires a real GPU and a reachable Knative service. The production
``pool_scale_fn`` indirection that drives the same ``KnativePool`` from the
orchestrator control loop is exercised by the unit tests and a standalone
scale-cycle check; here the harness orchestrates scale and ledger marks in order
so phase attribution is unambiguous.

Usage::

    python -m experiments.exp_serverless_training_ledger \
        --service hasagi-worker-lifecycle --namespace hasagi-validation \
        --train-burst-s 4 --pause-window-s 35 --out artifacts/ws0_ledger.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from hasagi.energy.pod_ledger import (
    PHASE_ACTIVE,
    PHASE_COLD_START,
    PHASE_IDLE,
    LedgerReport,
    PodEnergyLedger,
    nvml_cumulative_kwh_fn,
)
from hasagi.energy.telemetry import NvmlTelemetrySource
from hasagi.pool.knative_pool import KnativePool


@dataclass
class HostTrainer:
    """A real resnet18 training job that can checkpoint, release the GPU, and
    resume — so a pause pays a real reload + CUDA-reinit + warmup cost.

    torch is imported lazily so the module imports without a GPU present.
    """

    model_name: str = "resnet18"
    dataset: str = "cifar10"
    batch_size: int = 32
    ckpt_path: str = "./artifacts/ws0_ckpt.pt"
    warmup_iters: int = 2

    _model: object = field(default=None, init=False, repr=False)
    _optim: object = field(default=None, init=False, repr=False)
    _loader_iter: object = field(default=None, init=False, repr=False)
    _loader: object = field(default=None, init=False, repr=False)
    _device: object = field(default=None, init=False, repr=False)
    _loss_fn: object = field(default=None, init=False, repr=False)
    iters_done: int = field(default=0, init=False)

    def _build(self) -> None:
        import torch

        from hasagi.data.datasets import build_loader
        from hasagi.models.zoo import build_model

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = build_model(self.model_name).to(self._device)
        self._optim = torch.optim.SGD(self._model.parameters(), lr=0.01, momentum=0.9)
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._loader = build_loader(self.dataset, batch_size=self.batch_size)
        self._loader_iter = iter(self._loader)

    def _next_batch(self):
        try:
            return next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self._loader)
            return next(self._loader_iter)

    def cold_init(self) -> None:
        """First-ever start: build the model and force the CUDA context up."""
        import torch

        self._build()
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self.train_iters_count(self.warmup_iters)   # warm the kernels

    def checkpoint(self) -> None:
        """Persist model + optimiser state so a resumed job continues exactly."""
        import torch

        Path(self.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self._model.state_dict(), "optim": self._optim.state_dict(),
             "iters_done": self.iters_done},
            self.ckpt_path,
        )
        if self._device.type == "cuda":
            torch.cuda.synchronize()

    def teardown(self) -> None:
        """Release the GPU as scale-to-zero would: drop the model + free memory."""
        import torch

        self._model = None
        self._optim = None
        self._loader_iter = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def resume(self) -> None:
        """Real resume cost: rebuild, reload state, re-init CUDA, warm up."""
        import torch

        self._build()                       # rebuild graph + dataloader (CUDA re-init)
        # Our own trusted checkpoint (the model + optimiser state written above).
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state["model"])
        self._optim.load_state_dict(state["optim"])
        self.iters_done = int(state.get("iters_done", self.iters_done))
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self.train_iters_count(self.warmup_iters)   # first-iter warmup

    def train_iters_count(self, n: int) -> int:
        """Run exactly ``n`` real training iterations on the GPU."""
        import torch

        self._model.train()
        done = 0
        for _ in range(n):
            inputs, targets = self._next_batch()
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            self._optim.zero_grad()
            out = self._model(inputs)
            loss = self._loss_fn(out, targets)
            loss.backward()
            self._optim.step()
            done += 1
            self.iters_done += 1
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        return done

    def train_for(self, seconds: float) -> int:
        """Train for at least ``seconds`` of wall-clock; return iterations run."""
        start = time.monotonic()
        done = 0
        while time.monotonic() - start < seconds:
            done += self.train_iters_count(4)
        return done


def _carbon_aware_run(
    console: Console,
    pool: KnativePool,
    energy_fn,
    intensities: list[float],
    threshold: float,
    train_burst_s: float,
    pause_window_s: float,
    drain_wait_s: float,
) -> LedgerReport:
    """Pause (checkpoint + scale-to-zero) above threshold; resume below it."""
    console.print("[bold]Run: carbon-aware (pause when intensity > threshold)[/]")
    trainer = HostTrainer()
    ledger = PodEnergyLedger(energy_fn)
    running = False        # is the host actively training?
    ever_started = False   # has the job cold-started at least once?

    for tick, intensity in enumerate(intensities):
        pause = intensity > threshold
        if not pause:
            if not running:
                # Resume (or first start): real pod cold start + host reload.
                ledger.mark(PHASE_COLD_START, intensity)
                pool.scale(target=1, timeout_seconds=60.0, wait_for_ready=True)
                if not ever_started:
                    trainer.cold_init()
                    ever_started = True
                else:
                    trainer.resume()
                ledger.mark(PHASE_ACTIVE, intensity)
                running = True
            trainer.train_for(train_burst_s)
            console.print(
                f"  tick {tick:2d}: intensity={intensity:6.1f} RUN  iters={trainer.iters_done}"
            )
        else:
            if running:
                trainer.checkpoint()       # real training-state work → still active
                trainer.teardown()         # GPU released
                ledger.mark(PHASE_IDLE, intensity)   # GPU idle from here
                pool.scale(target=0, timeout_seconds=drain_wait_s, wait_for_ready=True)
                running = False
            console.print(
                f"  tick {tick:2d}: intensity={intensity:6.1f} PAUSE (scaled to zero)"
            )
            time.sleep(pause_window_s)

    report = ledger.report()
    pool.scale(target=0, timeout_seconds=10.0, wait_for_ready=False)
    return report


def _always_on_run(
    console: Console,
    pool: KnativePool,
    energy_fn,
    intensities: list[float],
    train_burst_s: float,
) -> LedgerReport:
    """One cold start, then train through every tick regardless of intensity."""
    console.print("[bold]Run: always-on (train through every tick)[/]")
    trainer = HostTrainer()
    ledger = PodEnergyLedger(energy_fn)

    ledger.mark(PHASE_COLD_START, intensities[0])
    pool.scale(target=1, timeout_seconds=60.0, wait_for_ready=True)
    trainer.cold_init()
    for tick, intensity in enumerate(intensities):
        ledger.mark(PHASE_ACTIVE, intensity)
        trainer.train_for(train_burst_s)
        console.print(
            f"  tick {tick:2d}: intensity={intensity:6.1f} RUN  iters={trainer.iters_done}"
        )

    report = ledger.report()
    trainer.checkpoint()
    trainer.teardown()
    pool.scale(target=0, timeout_seconds=10.0, wait_for_ready=False)
    return report


def _summarise(console: Console, name: str, rep: LedgerReport) -> dict:
    console.print(
        f"[bold]{name}[/]: total {rep.total_energy_kwh*1000:.3f} Wh / "
        f"{rep.total_carbon_g:.3f} gCO2 | resume {rep.resume_energy_kwh*1000:.3f} Wh / "
        f"{rep.resume_carbon_g:.3f} gCO2 over {rep.cold_starts} cold start(s) | "
        f"active {rep.active_energy_kwh*1000:.3f} Wh"
    )
    return {
        "total_energy_wh": rep.total_energy_kwh * 1000.0,
        "total_carbon_g": rep.total_carbon_g,
        "resume_energy_wh": rep.resume_energy_kwh * 1000.0,
        "resume_carbon_g": rep.resume_carbon_g,
        "active_energy_wh": rep.active_energy_kwh * 1000.0,
        "cold_starts": rep.cold_starts,
        "energy_by_phase_wh": {k: v * 1000.0 for k, v in rep.energy_by_phase_kwh.items()},
        "carbon_by_phase_g": dict(rep.carbon_by_phase_g),
        "duration_by_phase_s": dict(rep.duration_by_phase_s),
    }


def run(args: argparse.Namespace) -> int:
    console = Console()

    # Intensity schedule: clean → DIRTY window (forces a pause) → clean.
    intensities = [float(x) for x in args.intensities.split(",")]
    threshold = args.threshold
    console.print(
        f"[bold]Serverless training carbon ledger[/] — {len(intensities)} ticks, "
        f"pause above {threshold:.0f} gCO2/kWh; schedule={intensities}"
    )

    src = NvmlTelemetrySource([(args.device, "gpu0", 0, args.gpu_type)], poll_interval_ms=100)
    src.start()
    energy_fn = nvml_cumulative_kwh_fn(src)
    try:
        aware_pool = KnativePool(service=args.service, namespace=args.namespace)
        aware = _carbon_aware_run(
            console, aware_pool, energy_fn, intensities, threshold,
            args.train_burst_s, args.pause_window_s, args.drain_wait_s,
        )
        base_pool = KnativePool(service=args.service, namespace=args.namespace)
        base = _always_on_run(
            console, base_pool, energy_fn, intensities, args.train_burst_s,
        )
    finally:
        src.stop()

    aware_d = _summarise(console, "carbon-aware", aware)
    base_d = _summarise(console, "always-on", base)
    delta_g = aware_d["total_carbon_g"] - base_d["total_carbon_g"]
    rel = (100.0 * delta_g / base_d["total_carbon_g"]) if base_d["total_carbon_g"] else 0.0
    verdict = "SAVES" if delta_g < 0 else "LOSES"
    console.print(
        f"[bold]Net carbon delta[/]: {delta_g:+.3f} gCO2 ({rel:+.1f}%) — "
        f"carbon-aware pause {verdict} vs always-on after charging real resume cost."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schedule_g_per_kwh": intensities,
            "threshold_g_per_kwh": threshold,
            "trace_source": "synthetic-parametric",
            "energy_source": "nvml-measured",
            "carbon_aware": aware_d,
            "always_on": base_d,
            "net_carbon_delta_g": delta_g,
            "net_carbon_delta_pct": rel,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--service", default="hasagi-worker-lifecycle")
    p.add_argument("--namespace", default="hasagi-validation")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--gpu-type", default="RTX3080Ti")
    p.add_argument(
        "--intensities", default="200,200,900,900,200,200",
        help="Comma-separated per-tick grid intensity gCO2/kWh.",
    )
    p.add_argument("--threshold", type=float, default=800.0)
    p.add_argument("--train-burst-s", type=float, default=4.0)
    p.add_argument("--pause-window-s", type=float, default=35.0)
    p.add_argument("--drain-wait-s", type=float, default=45.0)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

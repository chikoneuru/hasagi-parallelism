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
        --service tare-worker-lifecycle --namespace tare-validation \
        --train-burst-s 4 --pause-window-s 35 --out artifacts/ws0_ledger.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console

from tare.energy.marginal_meter import MarginalEnergyMeter
from tare.energy.pod_ledger import (
    PHASE_ACTIVE,
    PHASE_COLD_START,
    PHASE_IDLE,
    LedgerReport,
    PodEnergyLedger,
)
from tare.energy.regime import GpuRegime, break_even_window_s
from tare.pool.knative_pool import KnativePool
from tare.worker.host_trainer import HostTrainer


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

    # Calibrate the background draw (co-tenant + display) with our job absent,
    # then meter only our marginal energy. A low sd means the subtraction is clean.
    meter = MarginalEnergyMeter(device_index=args.device, poll_interval_ms=100)
    bg_mean, bg_sd = meter.calibrate(seconds=args.calibrate_s)
    console.print(
        f"[bold]Background[/]: {bg_mean:.1f} W (sd {bg_sd:.1f} W) subtracted as co-tenant/display; "
        f"metering marginal energy only."
    )
    meter.start()
    energy_fn = meter.cumulative_kwh
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
        meter.stop()

    aware_d = _summarise(console, "carbon-aware (marginal)", aware)
    base_d = _summarise(console, "always-on (marginal)", base)

    # The rigorous metric is the EQUAL-WORK break-even per regime: how long a
    # dirty window must be before pausing+deferring the same work beats riding
    # through it. (The two runs above do UNEQUAL work — carbon-aware drops the
    # dirty-tick work rather than deferring it — so their head-to-head totals are
    # only a measurement sanity check, not the verdict. A fixed-work deferral
    # harness is the next step.) Inputs are measured marginal quantities.
    dirty = max(intensities)
    clean = min(intensities)
    active_w = _phase_power_w(base, PHASE_ACTIVE)   # measured active marginal power
    resume_kwh = aware.resume_energy_kwh
    console.print(
        f"[bold]Measured (marginal)[/]: active ≈ {active_w:.0f} W; "
        f"resume cost ≈ {resume_kwh*1000:.3f} Wh; dirty/clean = {dirty:.0f}/{clean:.0f} gCO2/kWh"
    )
    console.print("[bold]Equal-work break-even by freed-GPU regime[/]:")
    regimes_out: dict[str, dict] = {}
    for regime in GpuRegime:
        idle_w = args.dedicated_idle_w if regime is GpuRegime.DEDICATED else 0.0
        t_star = break_even_window_s(
            active_power_w=active_w, intensity_dirty=dirty, intensity_clean=clean,
            resume_energy_kwh=resume_kwh, idle_power_w=idle_w,
        )
        if t_star == float("inf"):
            verdict = f"pause NEVER saves (idle {idle_w:.0f} W outweighs the arbitrage)"
        else:
            verdict = f"pause saves for dirty windows longer than {t_star:.1f} s"
        console.print(f"  [{regime.value:12s}] idle {idle_w:4.0f} W → {verdict}")
        regimes_out[regime.value] = {
            "break_even_window_s": None if t_star == float("inf") else t_star,
            "idle_power_w_assumed": idle_w,
            "saves": t_star != float("inf"),
        }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schedule_g_per_kwh": intensities,
            "threshold_g_per_kwh": threshold,
            "trace_source": "synthetic-parametric",
            "energy_source": "nvml-measured-marginal",
            "background_w_mean": bg_mean,
            "background_w_sd": bg_sd,
            "active_marginal_power_w": active_w,
            "resume_energy_wh": resume_kwh * 1000.0,
            "dedicated_idle_w_assumed": args.dedicated_idle_w,
            "carbon_aware": aware_d,
            "always_on": base_d,
            "regimes": regimes_out,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def _phase_power_w(rep: LedgerReport, phase: str) -> float:
    """Average marginal power (W) over a phase = energy / duration."""
    e_kwh = rep.energy_by_phase_kwh.get(phase, 0.0)
    dur_s = rep.duration_by_phase_s.get(phase, 0.0)
    if dur_s <= 0.0:
        return 0.0
    return e_kwh * 3_600_000.0 / dur_s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--service", default="tare-worker-lifecycle")
    p.add_argument("--namespace", default="tare-validation")
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
    p.add_argument(
        "--calibrate-s", type=float, default=6.0,
        help="Seconds to sample the background draw (our job absent) before metering.",
    )
    p.add_argument(
        "--dedicated-idle-w", type=float, default=26.0,
        help="Idle-floor power charged in the dedicated regime; measure on a clean GPU "
             "for paper-grade numbers (default is an observed-idle estimate).",
    )
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

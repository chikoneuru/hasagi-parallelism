"""Fixed-work carbon-deferral head-to-head — the fair version of the ledger.

The serverless-training-ledger harness compared a carbon-aware run that *dropped*
the dirty-window work against an always-on run, so the two did unequal work. This
harness fixes that: both policies complete the **same total iterations**, so the
comparison is honest. The cost of being green is then paid in *makespan*, not in
unfinished work.

Both run a real resnet18/CIFAR-10 job on the host GPU, metered marginally (the
co-tenant background is subtracted), with carbon billed at each tick's intensity:

  - always-on   : never pauses; finishes in the fewest ticks.
  - carbon-aware : pauses (checkpoint + scale-to-zero) while intensity is above a
                   threshold and defers that work to later clean ticks; finishes
                   later but does more of its work under low intensity, paying a
                   real resume cost (reload + CUDA reinit + warmup) per wake-up.

The schedule repeats until the work is done, so carbon-aware genuinely catches
up. Output is the carbon-vs-makespan trade-off, per freed-GPU regime
(dedicated / reallocated / powered-down), plus the equal-work break-even window.

Requires a real GPU; a Knative service is optional (``--no-drive-pod`` runs the
host-side study alone, since WS-level scale-to-zero fidelity is shown by the
serverless-training-ledger harness).

Usage::

    python -m experiments.exp_carbon_deferral_breakeven \
        --total-iters 4000 --tick-seconds 3 --intensities 200,200,900,900 \
        --threshold 800 --out artifacts/deferral_breakeven.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from rich.console import Console

from tare.energy.background import BackgroundModel
from tare.energy.marginal_meter import MarginalEnergyMeter
from tare.energy.pod_ledger import (
    PHASE_ACTIVE,
    PHASE_COLD_START,
    PHASE_IDLE,
    LedgerReport,
    PodEnergyLedger,
)
from tare.energy.regime import GpuRegime, break_even_window_s, regime_breakdown
from tare.pool.knative_pool import KnativePool
from tare.worker.host_trainer import HostTrainer

# Seconds to wait after releasing the GPU before sampling a background anchor,
# so the clocks have dropped to idle and the sample is not contaminated by the
# just-finished work.
_SETTLE_S = 1.5


def _train_until(trainer: HostTrainer, seconds: float, total_iters: int) -> None:
    """Train in small chunks until ``seconds`` elapse or ``total_iters`` reached."""
    start = time.monotonic()
    while time.monotonic() - start < seconds and trainer.iters_done < total_iters:
        trainer.train_iters_count(8)


def _run_policy(
    console: Console,
    name: str,
    *,
    ckpt_path: str,
    meter: MarginalEnergyMeter,
    bg_model: BackgroundModel,
    schedule: list[float],
    threshold: float,
    total_iters: int,
    tick_seconds: float,
    pool: KnativePool | None,
    max_ticks: int,
    sample_background: bool = True,
) -> dict:
    """Run one policy to completion (``total_iters``); return its ledger + stats.

    ``threshold = inf`` makes the policy never pause (the always-on baseline).
    Energy is NOT integrated online here: the ledger records phase marks with
    timestamps, and the caller re-integrates the recorded device trace against a
    background model.

    ``sample_background`` controls how that background is built. With it True
    (shared GPU), an anchor is sampled whenever our job is off the GPU. But once
    our process has created a CUDA context it never releases it across pauses
    (the pod scales to zero; the host process persists), so on a *dedicated* GPU
    those "idle" anchors read our own held-context power, not a co-tenant — which
    is OUR energy and must not be subtracted. There, pass ``sample_background``
    False and pre-seed ``bg_model`` with the true idle floor (measured before any
    context exists); the held-context power then correctly lands in our idle phase.
    """
    console.print(f"[bold]Policy: {name}[/] (target {total_iters} iters)")
    # warmup_iters spans the post-resume clock ramp (~15 iters) so the ramp's
    # energy is charged to cold-start, not folded into the active phase.
    trainer = HostTrainer(ckpt_path=ckpt_path, warmup_iters=15)
    ledger = PodEnergyLedger(energy_kwh_fn=lambda: 0.0)   # marks only; re-integrated later
    running = False
    ever_started = False
    pauses = 0
    ticks = 0

    if sample_background:
        bg_model.add(time.monotonic(), meter.sample_power_w())   # pre-run bracket (job off GPU)
    t0 = time.monotonic()

    while trainer.iters_done < total_iters and ticks < max_ticks:
        intensity = schedule[ticks % len(schedule)]
        ticks += 1
        pause = intensity > threshold
        if not pause:
            if not running:
                ledger.mark(PHASE_COLD_START, intensity)
                if pool is not None:
                    pool.scale(target=1, timeout_seconds=30.0, wait_for_ready=False)
                if not ever_started:
                    trainer.cold_init()
                    ever_started = True
                else:
                    trainer.resume()
                running = True
            ledger.mark(PHASE_ACTIVE, intensity)
            _train_until(trainer, tick_seconds, total_iters)
        else:
            if running:
                # Entering a pause: mark IDLE FIRST so the checkpoint + GPU release
                # (and any pod drain) are billed to idle, not folded into the active
                # phase. ``KnativePool.scale(0)`` blocks on full drain, which is real
                # serverless latency but not part of the training job's active work.
                ledger.mark(PHASE_IDLE, intensity)
                trainer.checkpoint()
                trainer.teardown(keep_dataloader=True)   # warm loader → cheap resume
                if pool is not None:
                    pool.scale(target=0, timeout_seconds=10.0, wait_for_ready=False)
                running = False
                pauses += 1
            else:
                ledger.mark(PHASE_IDLE, intensity)
            time.sleep(tick_seconds)
            if sample_background:
                # Shared-GPU only: re-anchor the (co-tenant) background. On a
                # dedicated GPU this reading is our own held context, not
                # background, so it is skipped (see the docstring).
                bg_model.add(time.monotonic(), meter.sample_power_w())

    makespan_s = time.monotonic() - t0
    # The ledger's t_end is the END OF WORK, captured before teardown — otherwise
    # the final teardown (dataloader-worker shutdown, ~seconds) and the settle
    # would be absorbed into the last phase's duration and inflate its J/iter.
    end_s = time.monotonic()
    if running:
        trainer.teardown()
        running = False
    if pool is not None:
        pool.scale(target=0, timeout_seconds=10.0, wait_for_ready=False)
    if sample_background:
        time.sleep(_SETTLE_S)                     # let clocks drop before bracketing
        bg_model.add(time.monotonic(), meter.sample_power_w())   # post-run bracket
    console.print(
        f"  done: {trainer.iters_done} iters in {ticks} ticks, {pauses} pause(s), "
        f"makespan {makespan_s:.1f}s"
    )
    return {
        "name": name,
        "ledger": ledger,
        "end_s": end_s,
        "iters": trainer.iters_done,
        "ticks": ticks,
        "pauses": pauses,
        "makespan_s": makespan_s,
    }


def _phase_power_w(rep: LedgerReport, phase: str) -> float:
    e_kwh = rep.energy_by_phase_kwh.get(phase, 0.0)
    dur_s = rep.duration_by_phase_s.get(phase, 0.0)
    return e_kwh * 3_600_000.0 / dur_s if dur_s > 0 else 0.0


def _policy_json(d: dict) -> dict:
    """Compact JSON-safe summary of a policy run (drops the live ledger object)."""
    rep: LedgerReport = d["report"]
    return {
        "iters": d["iters"], "ticks": d["ticks"], "pauses": d["pauses"],
        "makespan_s": d["makespan_s"],
        "energy_by_phase_wh": {k: v * 1000.0 for k, v in rep.energy_by_phase_kwh.items()},
        "duration_by_phase_s": dict(rep.duration_by_phase_s),
        "power_by_phase_w": {
            ph: _phase_power_w(rep, ph) for ph in rep.energy_by_phase_kwh
        },
        "intervals_phase_dur_s": [(iv.phase, round(iv.duration_s, 2)) for iv in rep.intervals],
        "total_carbon_g_measured": rep.total_carbon_g,
    }


def run(args: argparse.Namespace) -> int:
    console = Console()
    schedule = [float(x) for x in args.intensities.split(",")]
    dirty, clean = max(schedule), min(schedule)
    console.print(
        f"[bold]Fixed-work carbon-deferral break-even[/] — {args.total_iters} iters, "
        f"tick {args.tick_seconds}s, pause above {args.threshold:.0f} gCO2/kWh; "
        f"schedule={schedule}"
    )

    meter = MarginalEnergyMeter(device_index=args.device, poll_interval_ms=100, record_trace=True)
    bg_cal, bg_sd = meter.calibrate(seconds=args.calibrate_s)
    # Dedicated GPU: subtract a CONSTANT true-idle floor (our held CUDA context
    # during pauses is our energy, not background, so it must not be re-sampled).
    # Shared GPU: track the co-tenant background via brackets + pause anchors.
    fixed_bg = args.fixed_background_w
    sample_bg = fixed_bg is None
    if sample_bg:
        bg_model = BackgroundModel()
        console.print(
            f"[bold]Background[/]: {bg_cal:.1f} W (sd {bg_sd:.1f} W) at calibration; "
            f"tracking the shared-GPU background via brackets + pause anchors."
        )
    else:
        bg_model = BackgroundModel([(0.0, fixed_bg)])
        console.print(
            f"[bold]Background[/]: constant {fixed_bg:.1f} W true-idle floor (dedicated GPU); "
            f"held-context power during pauses is billed to us, not subtracted."
        )
    meter.start()

    pool = KnativePool(service=args.service, namespace=args.namespace) if args.drive_pod else None
    try:
        base = _run_policy(
            console, "always-on", ckpt_path="./artifacts/deferral_base_ckpt.pt",
            meter=meter, bg_model=bg_model, schedule=schedule, threshold=math.inf,
            total_iters=args.total_iters, tick_seconds=args.tick_seconds,
            pool=pool, max_ticks=args.max_ticks, sample_background=sample_bg,
        )
        aware = _run_policy(
            console, "carbon-aware", ckpt_path="./artifacts/deferral_aware_ckpt.pt",
            meter=meter, bg_model=bg_model, schedule=schedule, threshold=args.threshold,
            total_iters=args.total_iters, tick_seconds=args.tick_seconds,
            pool=pool, max_ticks=args.max_ticks, sample_background=sample_bg,
        )
    finally:
        meter.stop()

    # Re-integrate each ledger against the time-varying background (drift-corrected).
    trace = meter.power_trace()
    base["report"] = base["ledger"].report_from_trace(trace, bg_model, t_end=base["end_s"])
    aware["report"] = aware["ledger"].report_from_trace(trace, bg_model, t_end=aware["end_s"])
    console.print(
        f"[bold]Background drift[/] across the session: {bg_model.drift_w:.1f} W "
        f"(mean {bg_model.mean_w:.1f} W, {len(bg_model.anchors)} anchors)"
    )

    active_w = _phase_power_w(base["report"], PHASE_ACTIVE)
    # Confidence gate: if the background moved by a large fraction of our own
    # active draw, a co-tenant came and went during the run and the marginal
    # attribution cannot be trusted (the rise inside an active window is not
    # captured by the pause/bracket anchors). Flag the run rather than quote it.
    drift_contaminated = bg_model.drift_w > max(15.0, 0.25 * active_w)
    if drift_contaminated:
        console.print(
            f"[bold red]LOW CONFIDENCE[/]: background drift {bg_model.drift_w:.0f} W exceeds "
            f"{max(15.0, 0.25 * active_w):.0f} W gate (a bursty co-tenant overlapped the run). "
            f"Magnitudes are not paper-grade; re-run on an exclusive GPU."
        )
    # Consistency check: energy-per-iteration must agree across policies (same
    # work), up to carbon-aware's per-resume warmup overhead. (Average active
    # *power* differs only because carbon-aware's active phase also spans
    # checkpoint/teardown wall-clock — that does not affect energy or carbon.)
    base_jpi = base["report"].active_energy_kwh * 3_600_000.0 / max(1, base["iters"])
    aware_jpi = aware["report"].active_energy_kwh * 3_600_000.0 / max(1, aware["iters"])
    console.print(
        f"[bold]Energy per iteration[/]: always-on {base_jpi:.3f} J/iter vs "
        f"carbon-aware {aware_jpi:.3f} J/iter "
        f"(+{100.0*(aware_jpi-base_jpi)/base_jpi:.0f}% resume/warmup overhead)"
    )
    resume_kwh = aware["report"].resume_energy_kwh

    console.print("\n[bold]Equal-work carbon vs makespan, by freed-GPU regime[/]:")
    base_bd = regime_breakdown(base["report"], dedicated_idle_w=args.dedicated_idle_w)
    aware_bd = regime_breakdown(aware["report"], dedicated_idle_w=args.dedicated_idle_w)
    regimes_out: dict[str, dict] = {}
    for regime in GpuRegime:
        b = base_bd[regime.value].total_carbon_g
        a = aware_bd[regime.value].total_carbon_g
        delta = a - b
        rel = (100.0 * delta / b) if b else 0.0
        idle_w = args.dedicated_idle_w if regime is GpuRegime.DEDICATED else 0.0
        t_star = break_even_window_s(
            active_power_w=active_w, intensity_dirty=dirty, intensity_clean=clean,
            resume_energy_kwh=resume_kwh, idle_power_w=idle_w,
        )
        verdict = "SAVES" if delta < 0 else "LOSES"
        t_str = "never" if t_star == math.inf else f"{t_star:.1f}s"
        console.print(
            f"  [{regime.value:12s}] carbon-aware {a:.3f} vs always-on {b:.3f} gCO2 "
            f"→ {delta:+.3f} ({rel:+.1f}%) {verdict}; break-even tick {t_str}"
        )
        regimes_out[regime.value] = {
            "carbon_aware_g": a, "always_on_g": b,
            "delta_g": delta, "delta_pct": rel,
            "break_even_window_s": None if t_star == math.inf else t_star,
            "idle_power_w_assumed": idle_w,
        }
    makespan_cost = aware["makespan_s"] - base["makespan_s"]
    console.print(
        f"[bold]Makespan cost of going green[/]: "
        f"+{makespan_cost:.1f}s ({aware['makespan_s']:.1f}s vs {base['makespan_s']:.1f}s); "
        f"{aware['pauses']} pause(s), {aware['ticks']} vs {base['ticks']} ticks."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schedule_g_per_kwh": schedule,
            "threshold_g_per_kwh": args.threshold,
            "total_iters": args.total_iters,
            "tick_seconds": args.tick_seconds,
            "trace_source": "synthetic-parametric",
            "energy_source": "nvml-measured-marginal-drift-corrected",
            "background_w_calibration": bg_cal,
            "background_w_calibration_sd": bg_sd,
            "background_w_drift": bg_model.drift_w,
            "background_w_mean": bg_model.mean_w,
            "background_anchors": len(bg_model.anchors),
            "drift_contaminated": drift_contaminated,
            "active_marginal_power_w_alwayson": active_w,
            "active_j_per_iter_alwayson": base_jpi,
            "active_j_per_iter_aware": aware_jpi,
            "resume_energy_wh": resume_kwh * 1000.0,
            "dedicated_idle_w_assumed": args.dedicated_idle_w,
            "always_on": _policy_json(base),
            "carbon_aware": _policy_json(aware),
            "makespan_cost_s": makespan_cost,
            "regimes": regimes_out,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--service", default="tare-worker-lifecycle")
    p.add_argument("--namespace", default="tare-validation")
    p.add_argument("--drive-pod", action="store_true",
                   help="Also drive the Knative pod scale-to-zero. Off by default: it adds real "
                        "pod-drain latency that is not part of the training job's makespan, and the "
                        "pod lifecycle fidelity lives in exp_serverless_training_ledger.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--intensities", default="200,200,900,900",
                   help="Comma-separated per-tick grid intensity gCO2/kWh; cycled until done.")
    p.add_argument("--threshold", type=float, default=800.0)
    p.add_argument("--total-iters", type=int, default=4000)
    p.add_argument("--tick-seconds", type=float, default=3.0)
    p.add_argument("--max-ticks", type=int, default=400,
                   help="Safety cap so an over-aggressive threshold cannot loop forever.")
    p.add_argument("--calibrate-s", type=float, default=6.0)
    p.add_argument("--fixed-background-w", type=float, default=None,
                   help="Subtract a CONSTANT idle floor (W) instead of tracking a shared-GPU "
                        "background. Use the measured true-idle floor on a dedicated GPU.")
    p.add_argument("--dedicated-idle-w", type=float, default=26.0,
                   help="Idle floor charged in the dedicated regime; measure on a clean GPU.")
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

"""EcoLife-collapse test: does the training-specific resume cost flip a stateless
carbon-pause decision?

Stateless-FaaS carbon schedulers (EcoLife SC'24, Green-or-Fast CCGrid'26) decide
pause-vs-keep on a resume cost that is essentially the COLD START (cuda-init +
container spin-up): a stateless function carries no state across a pause. A
*training* job additionally pays, on every pause/resume cycle, a cost the stateless
model never charges: a checkpoint WRITE before the pause, a model+optimiser-state
RELOAD on resume (both scaling with the model's state size), and a first-iteration
WARMUP. The question: for which model state sizes does a stateless pause decision
(taken at the stateless break-even window ``t*_stateless``) pause a training job
whose TRUE break-even window ``t*_training`` is larger -- so the scheduler pauses
but the training job LOSES carbon? The ``[t*_stateless, t*_training)`` gap is the
mis-decision window the stateless model creates; it is the wedge that distinguishes
a serverless-TRAINING carbon ledger from the stateless-FaaS literature.

Grounding (all measured single-GPU, RTX 3080 Ti):
  - cold start ~4.7 s (knative-cold-start artifact; cuda-init dominated).
  - active draw ~132 W and resume breakdown for resnet18 (~0.045 GB state):
    ckpt-write 0.092 s, reload 0.047 s, warmup 0.761 s (resume_breakdown.json).
    Those small-state times imply effective checkpoint I/O bandwidths of
    ~3.9 Gbps (write) and ~7.7 Gbps (reload) -- naive torch.save/load to local
    storage -- which this harness uses to scale the cost to larger model state.
The carbon arbitrage uses ``break_even_window_s`` from the measured marginal ledger.
Honest scope: the per-GB bandwidths are extrapolated from a single small-state
measurement (serialization-overhead-dominated); real large-model checkpointing with
sharded/parallel I/O would run faster and SHRINK the window, so the magnitudes here
are an upper bound on the wedge. The I/O-phase power is conservative. No multi-GPU.

Usage::

    python -m experiments.exp_resume_collapse --out artifacts/resume_collapse.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from tare.energy.regime import break_even_window_s

_J_PER_KWH = 3.6e6


def resume_cost(state_gb: float, *, stateless: bool, cold_start_s: float,
                resume_power_w: float, write_bw_gbps: float, reload_bw_gbps: float,
                warmup_s: float) -> tuple[float, float]:
    """Return (time_s, energy_kwh) of one pause/resume cycle.

    Stateless (EcoLife's model): cold start only. Training: cold start + checkpoint
    write + state reload (both state-size proportional) + first-iter warmup. Energy
    bills ``resume_power_w`` across the whole resume time (the GPU is held/spinning
    during cuda-init and I/O); conservative for the I/O-bound phases."""
    if stateless:
        t = cold_start_s
    else:
        write_s = (state_gb * 8.0) / max(write_bw_gbps, 1e-9)   # GB*8 = Gb; / Gbps = s
        reload_s = (state_gb * 8.0) / max(reload_bw_gbps, 1e-9)
        t = cold_start_s + write_s + reload_s + warmup_s
    return t, resume_power_w * t / _J_PER_KWH


def collapse_point(state_gb: float, *, intensity_dirty: float, intensity_clean: float,
                   active_power_w: float, idle_power_w: float, cold_start_s: float,
                   resume_power_w: float, write_bw_gbps: float, reload_bw_gbps: float,
                   warmup_s: float) -> dict:
    """t*_stateless vs t*_training for one model state size, and the mis-decision
    window a stateless scheduler opens (pauses, but training loses carbon)."""
    _, e_stateless = resume_cost(state_gb, stateless=True, cold_start_s=cold_start_s,
                                 resume_power_w=resume_power_w, write_bw_gbps=write_bw_gbps,
                                 reload_bw_gbps=reload_bw_gbps, warmup_s=warmup_s)
    rt_train, e_training = resume_cost(state_gb, stateless=False, cold_start_s=cold_start_s,
                                       resume_power_w=resume_power_w, write_bw_gbps=write_bw_gbps,
                                       reload_bw_gbps=reload_bw_gbps, warmup_s=warmup_s)
    kw = dict(active_power_w=active_power_w, intensity_dirty=intensity_dirty,
              intensity_clean=intensity_clean, idle_power_w=idle_power_w)
    t_star_stateless = break_even_window_s(resume_energy_kwh=e_stateless, **kw)
    t_star_training = break_even_window_s(resume_energy_kwh=e_training, **kw)
    misdecision_s = max(0.0, t_star_training - t_star_stateless)
    return {
        "state_gb": state_gb,
        "resume_time_training_s": rt_train,
        "resume_energy_stateless_wh": e_stateless * 1000.0,
        "resume_energy_training_wh": e_training * 1000.0,
        "t_star_stateless_s": t_star_stateless,
        "t_star_training_s": t_star_training,
        "misdecision_window_s": misdecision_s,
        "resume_energy_ratio_training_over_stateless": (e_training / e_stateless if e_stateless > 0 else float("inf")),
    }


def collapse_sweep(state_gbs: list[float], **kw) -> list[dict]:
    """Mis-decision window vs model state size."""
    return [collapse_point(s, **kw) for s in state_gbs]


def run(args: argparse.Namespace) -> int:
    console = Console()
    kw = dict(intensity_dirty=args.intensity_dirty, intensity_clean=args.intensity_clean,
              active_power_w=args.active_power_w, idle_power_w=args.idle_power_w,
              cold_start_s=args.cold_start_s, resume_power_w=args.resume_power_w,
              write_bw_gbps=args.write_bw_gbps, reload_bw_gbps=args.reload_bw_gbps,
              warmup_s=args.warmup_s)
    state_gbs = [float(x) for x in args.state_gbs.split(",")]
    rows = collapse_sweep(state_gbs, **kw)

    console.print(
        f"[bold]EcoLife-collapse test[/] (reallocated/scale-to-zero regime, idle "
        f"{args.idle_power_w:.0f} W; dirty/clean {args.intensity_dirty:.0f}/"
        f"{args.intensity_clean:.0f} gCO2/kWh; active {args.active_power_w:.0f} W).\n"
        "A stateless scheduler pauses any dirty window longer than t*_stateless; a "
        "training job only saves above the larger t*_training. The gap is the regime "
        "where a stateless (EcoLife-style) decision pauses but the training job loses carbon.")
    t = Table()
    t.add_column("model state (GB)", justify="right")
    t.add_column("resume_train (s)", justify="right")
    t.add_column("t*_stateless (s)", justify="right")
    t.add_column("t*_training (s)", justify="right")
    t.add_column("mis-decision window (s)", justify="right")
    for r in rows:
        t.add_row(f"{r['state_gb']:.3g}", f"{r['resume_time_training_s']:.1f}",
                  f"{r['t_star_stateless_s']:.2f}", f"{r['t_star_training_s']:.1f}",
                  f"{r['misdecision_window_s']:.1f}")
    console.print(t)
    small = rows[0]
    big = rows[-1]
    console.print(
        f"  At small state ({small['state_gb']:.3g} GB, resnet18-scale) the stateless and training "
        f"break-even windows nearly coincide (gap {small['misdecision_window_s']:.1f} s) -- the "
        f"stateless model is fine. At large state ({big['state_gb']:.3g} GB) the training resume cost "
        f"({big['resume_energy_training_wh']:.1f} Wh, "
        f"{big['resume_energy_ratio_training_over_stateless']:.0f}x the stateless cold-start) opens a "
        f"[bold]{big['misdecision_window_s']:.0f} s[/] mis-decision window: a stateless carbon "
        "scheduler pauses dirty windows in which the training job actually LOSES carbon. The "
        "training-state resume cost is the wedge the stateless-FaaS carbon literature omits.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "regime": "reallocated / scale-to-zero (idle billed to co-tenant)",
            "intensity_dirty_g_per_kwh": args.intensity_dirty,
            "intensity_clean_g_per_kwh": args.intensity_clean,
            "active_power_w": args.active_power_w,
            "idle_power_w": args.idle_power_w,
            "cold_start_s": args.cold_start_s,
            "resume_power_w": args.resume_power_w,
            "write_bw_gbps": args.write_bw_gbps,
            "reload_bw_gbps": args.reload_bw_gbps,
            "warmup_s": args.warmup_s,
            "anchors": "resnet18 resume_breakdown (ckpt-write 0.092s, reload 0.047s @ ~0.045GB) + "
                       "knative cold-start ~4.7s + active ~132W; bandwidths derived from the small-state times",
            "rows": rows,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-gbs", default="0.045,1,5,20,80,160",
                   help="Comma-separated model+optimiser state sizes (GB) to sweep.")
    p.add_argument("--intensity-dirty", type=float, default=900.0)
    p.add_argument("--intensity-clean", type=float, default=200.0)
    p.add_argument("--active-power-w", type=float, default=132.0)   # measured steady (resume_breakdown)
    p.add_argument("--idle-power-w", type=float, default=0.0)       # reallocated/scale-to-zero regime
    p.add_argument("--cold-start-s", type=float, default=4.7)       # measured knative cold start
    p.add_argument("--resume-power-w", type=float, default=100.0)   # I/O-phase draw (conservative)
    p.add_argument("--write-bw-gbps", type=float, default=3.9)      # 0.045GB*8 / 0.092s
    p.add_argument("--reload-bw-gbps", type=float, default=7.7)     # 0.045GB*8 / 0.047s
    p.add_argument("--warmup-s", type=float, default=0.761)         # measured first-batch warmup
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

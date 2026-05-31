"""Measured co-tenant throughput degradation on a single GPU.

Grounds the parametric contention factor swept by ``exp_contention_decision`` in a
real measurement: when N identical training processes share one GPU, how much
does each tenant's throughput drop versus running alone? That per-tenant factor
c(N) = throughput_per_tenant(N) / throughput_solo is exactly the StageSpec
throughput scaling the decision study perturbs.

Scope and honesty:
  - This measures DEFAULT time-sliced co-location — multiple independent CUDA
    processes on one device with no MPS daemon. That is the conservative
    multi-tenancy regime a serverless burst pool sees by default; NVIDIA MPS
    spatial sharing would reduce the degradation, so the measured c(N) is a lower
    bound on c (an upper bound on contention). MPS is an explicit opt-in
    (``--mps`` documents the intent) and is NOT enabled here.
  - All N tenants are this experiment's own processes; nothing is co-scheduled
    against a foreign job. The driver refuses to start if any other compute
    process is already on the GPU (the testbed is shared with a bursty co-tenant).
  - No power capping is performed; this is a throughput measurement only.
  - Each tenant is a separate OS process (full CUDA-context isolation). They
    synchronise on a file barrier so the timed regions genuinely overlap; warmup
    and CUDA init happen before the barrier.

Usage::

    python -m experiments.exp_cotenant_contention --tenants 1 2 3 --repeats 3 \
        --workload resnet --out artifacts/cotenant_contention.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_BARRIER_WAIT_S = 180.0   # max a worker waits at the barrier before giving up


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no GPU)
# ---------------------------------------------------------------------------

def _contention_factors(per_tenant_by_n: dict[int, float]) -> dict[int, dict]:
    """Given mean per-tenant throughput at each tenant count N, derive:
      - c(N) = per_tenant(N) / per_tenant(1)  (1.0 = no contention, 1/N = pure
        time-slice saturation),
      - aggregate(N) = N * per_tenant(N)  (total GPU throughput across tenants),
      - aggregate_scaling(N) = aggregate(N) / aggregate(1).
    """
    solo = per_tenant_by_n.get(1)
    out: dict[int, dict] = {}
    for n, pt in sorted(per_tenant_by_n.items()):
        c = pt / solo if solo else float("nan")
        agg = n * pt
        out[n] = {
            "per_tenant_iters_per_s": pt,
            "contention_factor": c,
            "aggregate_iters_per_s": agg,
            "aggregate_scaling": agg / (1 * solo) if solo else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# GPU pre-flight
# ---------------------------------------------------------------------------

def _foreign_compute_apps() -> list[str]:
    """Return any compute-process lines currently on the GPU. The experiment must
    not co-locate against a foreign job on the shared testbed."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            text=True, timeout=15,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []
    return [ln for ln in out.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Worker mode (one tenant; runs in its own process via subprocess)
# ---------------------------------------------------------------------------

def _worker_main(args: argparse.Namespace) -> int:
    import torch

    from experiments.exp_hardware_pareto import _build_workload

    torch.cuda.set_device(0)
    crit = torch.nn.CrossEntropyLoss()
    model, inputs, targets, is_seq = _build_workload(
        args.workload, batch_size=args.batch_size, spatial=args.spatial,
        channels=args.channels, seq_len=args.seq_len, d_model=args.d_model,
        layers=args.layers, vocab=args.vocab,
    )
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def step() -> None:
        optim.zero_grad(set_to_none=True)
        out = model(inputs)
        loss = (crit(out.reshape(-1, out.size(-1)), targets.reshape(-1))
                if is_seq else crit(out, targets))
        loss.backward()
        optim.step()

    # Warmup (kernels, caches, clocks) BEFORE the barrier so it isn't timed.
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    # File barrier: announce readiness, then wait for all tenants.
    bdir = Path(args.barrier_dir)
    (bdir / f"ready_{args.rank}").write_text("1")
    deadline = time.monotonic() + _BARRIER_WAIT_S
    while len(list(bdir.glob("ready_*"))) < args.n_tenants:
        if time.monotonic() > deadline:
            sys.stderr.write(f"worker {args.rank}: barrier timeout "
                             f"({len(list(bdir.glob('ready_*')))}/{args.n_tenants})\n")
            return 2
        time.sleep(0.05)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(json.dumps({"rank": args.rank, "throughput": args.iters / wall, "wall": wall}))
    return 0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _worker_cmd(rank: int, n: int, bdir: str, cfg: dict) -> list[str]:
    return [
        sys.executable, "-m", "experiments.exp_cotenant_contention",
        "--worker", "--rank", str(rank), "--n-tenants", str(n), "--barrier-dir", bdir,
        "--workload", cfg["workload"], "--iters", str(cfg["iters"]),
        "--warmup", str(cfg["warmup"]), "--batch-size", str(cfg["batch_size"]),
        "--spatial", str(cfg["spatial"]), "--channels", str(cfg["channels"]),
        "--seq-len", str(cfg["seq_len"]), "--d-model", str(cfg["d_model"]),
        "--layers", str(cfg["layers"]), "--vocab", str(cfg["vocab"]),
    ]


def _run_round(n_tenants: int, cfg: dict, round_timeout_s: float = 600.0) -> list[float]:
    """Launch n_tenants worker processes, barrier-synced, return each tenant's
    throughput (iters/s). Raises RuntimeError if any worker fails or times out."""
    src_root = str(Path(__file__).resolve().parents[1])  # the `src/` dir holding experiments/
    with tempfile.TemporaryDirectory(prefix="cotenant_barrier_") as bdir:
        procs = [
            subprocess.Popen(_worker_cmd(rank, n_tenants, bdir, cfg),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             cwd=src_root)
            for rank in range(n_tenants)
        ]
        throughputs: list[float] = []
        for rank, p in enumerate(procs):
            try:
                out, err = p.communicate(timeout=round_timeout_s)
            except subprocess.TimeoutExpired:
                p.kill()
                out, err = p.communicate()
                raise RuntimeError(f"worker {rank} timed out after {round_timeout_s}s") from None
            if p.returncode != 0:
                raise RuntimeError(f"worker {rank} failed (rc={p.returncode}):\n{err.strip()}")
            line = next((ln for ln in reversed(out.strip().splitlines()) if ln.startswith("{")), None)
            if line is None:
                raise RuntimeError(f"worker {rank} produced no result; stderr:\n{err.strip()}")
            throughputs.append(json.loads(line)["throughput"])
    return throughputs


def run(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    foreign = _foreign_compute_apps()
    if foreign:
        console.print("[red]Refusing to run: foreign compute process(es) already on the GPU:[/]")
        for ln in foreign:
            console.print(f"  {ln}")
        return 1
    if args.mps:
        console.print("[yellow]--mps given, but this build does not start the MPS daemon; "
                      "measuring default time-sliced co-location regardless.[/]")

    cfg = {
        "workload": args.workload, "iters": args.iters, "warmup": args.warmup,
        "batch_size": args.batch_size, "spatial": args.spatial, "channels": args.channels,
        "seq_len": args.seq_len, "d_model": args.d_model, "layers": args.layers, "vocab": args.vocab,
    }

    per_n_samples: dict[int, list[float]] = {}
    for n in args.tenants:
        samples: list[float] = []
        for rep in range(args.repeats):
            thr = _run_round(n, cfg)
            samples.extend(thr)
            console.print(f"[dim]N={n} rep {rep+1}/{args.repeats}: "
                          f"per-tenant {min(thr):.1f}-{max(thr):.1f} it/s[/]")
        per_n_samples[n] = samples

    per_tenant_mean = {n: statistics.mean(s) for n, s in per_n_samples.items()}
    factors = _contention_factors(per_tenant_mean)

    t = Table(title=f"Co-tenant throughput degradation - {args.workload}, "
                    f"time-sliced co-location ({args.repeats} reps x N tenants)")
    t.add_column("tenants N", justify="right")
    t.add_column("per-tenant it/s (mean+/-sd)", justify="right")
    t.add_column("c(N)=thr(N)/thr(1)", justify="right")
    t.add_column("aggregate it/s", justify="right")
    t.add_column("aggregate scaling", justify="right")
    summary_rows = []
    for n in args.tenants:
        sd = statistics.pstdev(per_n_samples[n]) if len(per_n_samples[n]) > 1 else 0.0
        f = factors[n]
        t.add_row(str(n), f"{per_tenant_mean[n]:.1f} +/- {sd:.1f}", f"{f['contention_factor']:.3f}",
                  f"{f['aggregate_iters_per_s']:.1f}", f"{f['aggregate_scaling']:.3f}")
        summary_rows.append({"tenants": n, "per_tenant_mean": per_tenant_mean[n],
                             "per_tenant_sd": sd, "n_samples": len(per_n_samples[n]), **f})
    console.print(t)

    if len(args.tenants) >= 2 and 1 in per_tenant_mean:
        n_max = max(args.tenants)
        c = factors[n_max]["contention_factor"]
        agg = factors[n_max]["aggregate_scaling"]
        console.print(
            f"\nAt N={n_max} each tenant retains [bold]{c*100:.1f}%[/] of solo throughput "
            f"(time-slice floor would be {100.0/n_max:.1f}%); aggregate GPU throughput "
            f"scales [bold]{agg:.2f}x[/] (1.0 = saturated, {n_max:.1f} = perfect)."
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "workload": args.workload, "regime": "time-sliced co-location (no MPS daemon)",
            "config": cfg, "tenants": args.tenants, "repeats": args.repeats,
            "rows": summary_rows,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tenants", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workload", choices=("resnet", "transformer"), default="resnet")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--spatial", type=int, default=32)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--mps", action="store_true",
                   help="Document MPS intent; the daemon is not started in this build.")
    p.add_argument("--out", default=None)
    # worker-mode flags (internal)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--n-tenants", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--barrier-dir", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        return _worker_main(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

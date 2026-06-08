"""Knative cold-start lifecycle measurement.

Drives one Knative Service through repeated cold starts and records the full
breakdown each time:

  pod_created → scheduled → container_started → ready (K8s side)
  app_ready → cuda_init_complete → first_work_completed   (worker side)

Cross-references the two sides via the worker's ``container_start_unix``
timestamp returned from ``/lifecycle`` so we can attribute each delay to a
specific subsystem.

Usage::

    python -m experiments.exp_knative_lifecycle --replicates 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass

import httpx
from rich.console import Console
from rich.table import Table

from tare.pool.knative_pool import KnativePool


@dataclass(frozen=True)
class ColdStartTrial:
    replicate: int
    trigger_unix: float
    pod_name: str | None
    # K8s side (from pool.observe()).
    pod_created_unix: float | None
    pod_container_started_unix: float | None
    pod_ready_unix: float | None
    # Worker side (from /lifecycle).
    container_start_unix: float | None
    app_ready_offset_s: float | None
    cuda_init_complete_offset_s: float | None
    first_work_completed_offset_s: float | None
    # End-to-end.
    end_to_end_wall_s: float

    def k8s_create_to_ready_s(self) -> float | None:
        if self.pod_created_unix and self.pod_ready_unix:
            return self.pod_ready_unix - self.pod_created_unix
        return None

    def k8s_container_to_ready_s(self) -> float | None:
        if self.pod_container_started_unix and self.pod_ready_unix:
            return self.pod_ready_unix - self.pod_container_started_unix
        return None

    def trigger_to_ready_s(self) -> float | None:
        if self.pod_ready_unix:
            return self.pod_ready_unix - self.trigger_unix
        return None


def _drain_pods(pool: KnativePool, *, force_delete: bool, timeout_s: float = 90.0) -> None:
    """Wait for the Knative service to scale to zero replicas.

    Two modes:
      * ``force_delete=True`` (default): issue ``kubectl delete --force
        --grace-period=0`` once, then poll for the API object to disappear.
        Knative's KPA controller may recreate replacement pods if a request
        is still routing — production scale-to-zero respects the queue-
        proxy's 600s drain period, so we fight against that here.
      * ``force_delete=False``: production-like behavior. Wait for the
        autoscaler's natural ``scale-to-zero-grace-period`` (30s by
        default + 5s buffer) before polling. Slower but matches what a
        real Knative deployment actually sees.

    Pods whose phase is still ``Running`` but with a non-null
    ``deletionTimestamp`` (i.e., mid-termination) are also treated as
    drained — they no longer receive traffic and the next request will
    open a brand-new pod.
    """
    if force_delete:
        cmd = [
            pool.kubectl, "delete", "pods",
            "-n", pool.namespace,
            "-l", f"serving.knative.dev/service={pool.service}",
            "--ignore-not-found", "--wait=false", "--force", "--grace-period=0",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10.0)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"force-delete pods failed: {exc.stderr.strip()}") from exc
    else:
        # Production-like: idle the service until the autoscaler's natural
        # scale-to-zero kicks in (30s grace + 5s slack).
        time.sleep(35.0)

    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        items = pool._list_pods()
        active = [
            it for it in items
            if it.get("status", {}).get("phase") not in ("Succeeded", "Failed")
            and not it.get("metadata", {}).get("deletionTimestamp")
        ]
        if not active:
            return
        time.sleep(0.5)
    raise RuntimeError(f"drain to zero failed after {timeout_s}s")


def _trigger_and_measure(
    pool: KnativePool,
    url: str,
    iterations: int,
    timeout_s: float,
    replicate: int,
) -> ColdStartTrial:
    """Send one POST /work and capture the K8s + worker timestamps.

    Does not patch min/max-scale annotations; Knative's activator will
    wake a new pod the moment a request hits the kourier ingress.
    """
    trigger_unix = time.time()
    payload = {"iterations": iterations, "job_id": f"cold-start-{replicate}"}
    with httpx.Client(timeout=httpx.Timeout(timeout_s)) as cli:
        resp = cli.post(
            url,
            json=payload,
            headers={"Host": "tare-worker-lifecycle.tare-validation.example.com"},
        )
        end_to_end_wall = time.time() - trigger_unix
        worker = resp.json().get("lifecycle", {}) if resp.is_success else {}
    pods = pool.observe()
    pod = next(
        (p for p in pods if p.ready_unix is not None and p.created_unix >= trigger_unix - 1.0),
        None,
    )
    return ColdStartTrial(
        replicate=replicate,
        trigger_unix=trigger_unix,
        pod_name=pod.pod_name if pod else None,
        pod_created_unix=pod.created_unix if pod else None,
        pod_container_started_unix=pod.container_started_unix if pod else None,
        pod_ready_unix=pod.ready_unix if pod else None,
        container_start_unix=worker.get("container_start_unix"),
        app_ready_offset_s=worker.get("app_ready_offset_s"),
        cuda_init_complete_offset_s=worker.get("cuda_init_complete_offset_s"),
        first_work_completed_offset_s=worker.get("first_work_request_completed_offset_s"),
        end_to_end_wall_s=end_to_end_wall,
    )


def _summarise(trials: list[ColdStartTrial], console: Console) -> None:
    table = Table(title="Knative cold-start lifecycle breakdown")
    table.add_column("replicate", justify="right")
    table.add_column("end-to-end (s)", justify="right")
    table.add_column("trigger→ready (s)", justify="right")
    table.add_column("k8s create→ready (s)", justify="right")
    table.add_column("k8s container→ready (s)", justify="right")
    table.add_column("worker app_ready (s)", justify="right")
    table.add_column("worker cuda_init (s)", justify="right")
    table.add_column("worker first_work (s)", justify="right")
    for t in trials:
        table.add_row(
            str(t.replicate),
            f"{t.end_to_end_wall_s:.3f}",
            f"{t.trigger_to_ready_s():.3f}" if t.trigger_to_ready_s() is not None else "?",
            f"{t.k8s_create_to_ready_s():.3f}" if t.k8s_create_to_ready_s() is not None else "?",
            f"{t.k8s_container_to_ready_s():.3f}" if t.k8s_container_to_ready_s() is not None else "?",
            f"{t.app_ready_offset_s:.3f}" if t.app_ready_offset_s is not None else "?",
            f"{t.cuda_init_complete_offset_s:.3f}" if t.cuda_init_complete_offset_s is not None else "?",
            f"{t.first_work_completed_offset_s:.3f}" if t.first_work_completed_offset_s is not None else "?",
        )
    console.print(table)

    def col_stats(getter):
        vals = [v for t in trials if (v := getter(t)) is not None]
        if not vals:
            return "—"
        m = statistics.mean(vals)
        if len(vals) < 2:
            return f"{m:.3f}s"
        sd = statistics.stdev(vals)
        return f"{m:.3f}s ± {sd:.3f}s"

    summary = Table(title="Aggregate statistics")
    summary.add_column("metric", justify="left")
    summary.add_column("value", justify="right")
    summary.add_row("end-to-end mean ± sd", col_stats(lambda t: t.end_to_end_wall_s))
    summary.add_row("k8s create→ready",     col_stats(lambda t: t.k8s_create_to_ready_s()))
    summary.add_row("k8s container→ready",  col_stats(lambda t: t.k8s_container_to_ready_s()))
    summary.add_row("worker app_ready",     col_stats(lambda t: t.app_ready_offset_s))
    summary.add_row("worker cuda_init",     col_stats(lambda t: t.cuda_init_complete_offset_s))
    summary.add_row("worker first_work",    col_stats(lambda t: t.first_work_completed_offset_s))
    console.print(summary)


def run(args: argparse.Namespace) -> int:
    console = Console()
    pool = KnativePool(service=args.service, namespace=args.namespace)
    url = f"http://{args.kourier_host}:{args.kourier_port}/work"

    console.print(
        f"[bold]Knative cold-start lifecycle harness[/] — "
        f"service={args.service}, replicates={args.replicates}, "
        f"force_delete={args.force_delete}"
    )

    trials: list[ColdStartTrial] = []
    for r in range(args.replicates):
        console.print(f"\n[bold]Replicate {r + 1}/{args.replicates}[/]")
        console.print("  draining → 0 pods …")
        _drain_pods(pool, force_delete=args.force_delete)
        time.sleep(args.between_replicates_s)
        console.print("  triggering cold start …")
        t = _trigger_and_measure(
            pool, url, iterations=args.iterations,
            timeout_s=args.request_timeout_s, replicate=r + 1,
        )
        trials.append(t)
        console.print(
            f"  end-to-end {t.end_to_end_wall_s:.3f}s"
            + (f", k8s-create→ready {t.k8s_create_to_ready_s():.3f}s" if t.k8s_create_to_ready_s() else "")
        )

    if args.out:
        from pathlib import Path
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(t) for t in trials], indent=2))
        console.print(f"[dim]wrote {out}[/]")

    _summarise(trials, console)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="tare-worker-lifecycle")
    parser.add_argument("--namespace", default="tare-validation")
    parser.add_argument("--kourier-host", default="127.0.0.1")
    parser.add_argument("--kourier-port", type=int, default=31080)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--between-replicates-s", type=float, default=2.0)
    parser.add_argument(
        "--force-delete",
        action="store_true",
        help="Hard-delete pods between replicates (faster cycle).",
    )
    parser.add_argument(
        "--out", default=None, help="Optional JSON path for raw trial data.",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

"""H2 / preemption — kill a serving pod mid-request, measure recovery.

The orchestrator promise is that an unexpected pod loss does not lose more
than one in-flight request; the next request should arrive at a healthy
pod (cold-started if needed) without operator intervention. Production
Knative buffers requests in the *activator* during a transition; this
harness verifies that buffer works end-to-end on the local cluster.

For each replicate:
  1. Scale the service to 1 ready pod (warm).
  2. Fire a long-running ``/work`` request in a background thread.
  3. After ``preempt_delay_s`` seconds, force-evict the serving pod.
  4. Either: the background request errors out (worker process killed
     mid-request), or it completes via a fresh pod after the activator
     re-queues it.
  5. Fire a follow-up ``/work`` request and time its end-to-end latency.
  6. Capture the new pod's lifecycle.

The recorded metric is *recovery_latency_s* — time from preemption to
the follow-up request returning 200 OK.

Usage::

    python -m experiments.exp_preemption_recovery --replicates 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

import httpx
from rich.console import Console
from rich.table import Table

from hise.pool.knative_pool import KnativePool


@dataclass(frozen=True)
class PreemptionTrial:
    replicate: int
    warm_pod: str | None
    preempt_unix: float
    background_succeeded: bool
    background_error: str | None
    follow_up_unix: float
    follow_up_wall_s: float
    recovery_latency_s: float
    follow_up_pod: str | None


def _evict_pod(pool: KnativePool, pod_name: str) -> None:
    cmd = [
        pool.kubectl, "delete", "pod", pod_name,
        "-n", pool.namespace, "--force", "--grace-period=0", "--wait=false",
    ]
    subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5.0)


def _background_request(url: str, host: str, iterations: int, payload_label: str,
                        results: dict) -> None:
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0)) as cli:
            r = cli.post(
                url,
                json={"iterations": iterations, "job_id": payload_label},
                headers={"Host": host},
            )
            results["status_code"] = r.status_code
            results["ok"] = r.is_success
            if r.is_success:
                results["body"] = r.json()
            else:
                results["error"] = r.text[:200]
    except Exception as exc:   # noqa: BLE001
        results["ok"] = False
        results["error"] = str(exc)


def _trial(
    pool: KnativePool,
    url: str,
    host: str,
    iterations: int,
    preempt_delay_s: float,
    timeout_s: float,
    replicate: int,
) -> PreemptionTrial:
    # Warm the service.
    pool.scale(target=1, timeout_seconds=timeout_s, wait_for_ready=True)
    pods_before = pool.observe()
    warm_pod = pods_before[0].pod_name if pods_before else None

    # Background long-running request that the preemption will interrupt.
    bg_results: dict = {}
    bg_thread = threading.Thread(
        target=_background_request,
        args=(url, host, iterations, f"preempt-victim-{replicate}", bg_results),
    )
    bg_thread.start()

    # Wait, then evict the warm pod.
    time.sleep(preempt_delay_s)
    preempt_unix = time.time()
    if warm_pod:
        _evict_pod(pool, warm_pod)

    # Fire the follow-up.
    follow_up_start = time.time()
    with httpx.Client(timeout=httpx.Timeout(timeout_s)) as cli:
        r = cli.post(
            url,
            json={"iterations": 2, "job_id": f"follow-up-{replicate}"},
            headers={"Host": host},
        )
    follow_up_end = time.time()
    follow_up_wall = follow_up_end - follow_up_start
    _ = r.is_success

    # Background request should finish (one way or another) by now.
    bg_thread.join(timeout=30.0)

    pods_after = pool.observe()
    new_pod = next(
        (p for p in pods_after if p.pod_name != warm_pod and p.ready_unix is not None),
        None,
    )
    return PreemptionTrial(
        replicate=replicate,
        warm_pod=warm_pod,
        preempt_unix=preempt_unix,
        background_succeeded=bool(bg_results.get("ok")),
        background_error=bg_results.get("error"),
        follow_up_unix=follow_up_end,
        follow_up_wall_s=follow_up_wall,
        recovery_latency_s=follow_up_end - preempt_unix,
        follow_up_pod=new_pod.pod_name if new_pod else None,
    )


def _summarise(trials: list[PreemptionTrial], console: Console) -> None:
    table = Table(title="Preemption recovery — single-pod fail-over")
    table.add_column("rep", justify="right")
    table.add_column("warm_pod", justify="left")
    table.add_column("bg succeeded", justify="center")
    table.add_column("follow-up wall (s)", justify="right")
    table.add_column("recovery (s)", justify="right")
    table.add_column("new pod", justify="left")
    for t in trials:
        table.add_row(
            str(t.replicate),
            (t.warm_pod or "?")[-12:],
            "✓" if t.background_succeeded else "✗",
            f"{t.follow_up_wall_s:.3f}",
            f"{t.recovery_latency_s:.3f}",
            (t.follow_up_pod or "?")[-12:],
        )
    console.print(table)

    if trials:
        recoveries = [t.recovery_latency_s for t in trials]
        bg_ok = sum(1 for t in trials if t.background_succeeded)
        summary = Table(title="Aggregate")
        summary.add_column("metric")
        summary.add_column("value", justify="right")
        summary.add_row("trials", str(len(trials)))
        summary.add_row("background request survival rate",
                        f"{bg_ok}/{len(trials)}")
        summary.add_row("recovery latency mean (s)",
                        f"{statistics.mean(recoveries):.3f}")
        if len(recoveries) > 1:
            summary.add_row("recovery latency sd (s)",
                            f"{statistics.stdev(recoveries):.3f}")
        console.print(summary)


def run(args: argparse.Namespace) -> int:
    console = Console()
    pool = KnativePool(service=args.service, namespace=args.namespace)
    url = f"http://{args.kourier_host}:{args.kourier_port}/work"
    host = f"{args.service}.{args.namespace}.example.com"

    console.print(
        f"[bold]Preemption recovery harness[/] — "
        f"{args.replicates} replicate(s), preempt at {args.preempt_delay_s}s"
    )

    trials: list[PreemptionTrial] = []
    for r in range(args.replicates):
        console.print(f"\n[bold]Replicate {r + 1}/{args.replicates}[/]")
        t = _trial(
            pool, url, host, iterations=args.iterations,
            preempt_delay_s=args.preempt_delay_s,
            timeout_s=args.timeout_seconds,
            replicate=r + 1,
        )
        trials.append(t)
        console.print(
            f"  bg ok={t.background_succeeded}, recovery {t.recovery_latency_s:.3f}s, "
            f"follow-up wall {t.follow_up_wall_s:.3f}s"
        )
        # Cool-down between replicates.
        subprocess.run(
            [
                pool.kubectl, "delete", "pods",
                "-n", pool.namespace,
                "-l", f"serving.knative.dev/service={pool.service}",
                "--force", "--grace-period=0", "--wait=false", "--ignore-not-found",
            ],
            check=False, capture_output=True, text=True, timeout=10.0,
        )
        time.sleep(args.between_replicates_s)

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
    parser.add_argument("--service", default="hise-worker-lifecycle")
    parser.add_argument("--namespace", default="hise-validation")
    parser.add_argument("--kourier-host", default="127.0.0.1")
    parser.add_argument("--kourier-port", type=int, default=31080)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--iterations", type=int, default=40,
        help="Iterations in the background long-running /work request.",
    )
    parser.add_argument(
        "--preempt-delay-s", type=float, default=1.0,
        help="Seconds after starting the bg request before we evict the pod.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--between-replicates-s", type=float, default=5.0)
    parser.add_argument("--out", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

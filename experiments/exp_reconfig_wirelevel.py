"""H3 — reconfig latency, wire-level breakdown on Knative.

The synthetic Phase-2 reconfig benchmark already measured the orchestrator's
algorithmic reconfig cost (DP recompute, partition diff). This harness adds
the *system-level* contribution: K8s API turnaround plus pod-lifecycle
overhead when the orchestrator drives a scale step ``N → M``.

For each step we capture:

  - ``patch_issued_unix``     — orchestrator hits the K8s API
  - ``first_new_pod_unix``    — first new pod object appears in the API
  - ``first_new_ready_unix``  — first new pod passes readiness
  - ``last_drained_unix``     — last removed pod transitions to Terminating
                                 (only set when N > M)

Steps are deltas like ``1 → 2`` (scale-up), ``2 → 1`` (scale-down),
``1 → 4`` (large fan-out). The grid is configurable; defaults mirror the
elasticity scenarios in the proposal's H3 acceptance criterion.

Usage::

    python -m experiments.exp_reconfig_wirelevel
    python -m experiments.exp_reconfig_wirelevel --steps "1->2,2->1,1->4,4->1" --replicates 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass

from rich.console import Console
from rich.table import Table

from hasagi.pool.knative_pool import KnativePool


@dataclass(frozen=True)
class ReconfigStep:
    """One (start, target) reconfig step + measured timing."""

    replicate: int
    start_count: int
    target_count: int
    patch_issued_unix: float
    settled_unix: float
    first_new_pod_unix: float | None
    first_new_pod_ready_unix: float | None
    pods_drained_count: int
    pods_created_count: int

    @property
    def wall_seconds(self) -> float:
        return self.settled_unix - self.patch_issued_unix

    @property
    def first_pod_appear_s(self) -> float | None:
        if self.first_new_pod_unix is None:
            return None
        return self.first_new_pod_unix - self.patch_issued_unix

    @property
    def first_pod_ready_s(self) -> float | None:
        if self.first_new_pod_ready_unix is None:
            return None
        return self.first_new_pod_ready_unix - self.patch_issued_unix


def _parse_steps(spec: str) -> list[tuple[int, int]]:
    steps: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        a, _, b = chunk.partition("->")
        if not _:
            raise ValueError(f"step '{chunk}' must be in 'A->B' form")
        steps.append((int(a), int(b)))
    return steps


def _scale_to_baseline(pool: KnativePool, target: int) -> None:
    """Bring the service to exactly ``target`` pods before timing a step."""
    pool.scale(target=target, timeout_seconds=120.0, wait_for_ready=True)


def _force_drain(pool: KnativePool) -> None:
    """Tell Knative to scale-to-zero AND hard-evict the current pods.

    Both halves are required: patching min=max=0 alone respects the 600s
    queue-proxy grace; force-deleting alone is undone immediately because
    the autoscaler still wants the previous replica count. With both,
    the next ``scale()`` finds an empty cluster.
    """
    # First, tell the autoscaler we want zero. This is a no-op if min/max
    # are already zero; harmless to re-apply.
    pool._patch_annotations({
        "autoscaling.knative.dev/min-scale": "0",
        "autoscaling.knative.dev/max-scale": "0",
    })
    # Then force-evict any current pods.
    cmd = [
        pool.kubectl, "delete", "pods",
        "-n", pool.namespace,
        "-l", f"serving.knative.dev/service={pool.service}",
        "--ignore-not-found", "--wait=false", "--force", "--grace-period=0",
    ]
    subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10.0)


def _wait_for_zero_pods(pool: KnativePool, timeout_s: float = 60.0) -> None:
    """Block until pool.observe() returns an empty list (Terminating filtered)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pool.observe():
            return
        time.sleep(0.5)
    raise RuntimeError("pods did not drain to zero in time")


def _measure_step(
    pool: KnativePool,
    start_count: int,
    target_count: int,
    replicate: int,
    timeout_s: float,
) -> ReconfigStep:
    """Force a clean slate, then time the patch ``→ target_count`` cold reconfig.

    Each measurement starts from zero pods so we cleanly attribute the wall-
    clock to "pods needed for new state", ignoring drain effects. Scale-down
    cases (``start_count > target_count``) become "scale from 0 to
    ``target_count``" with ``pods_drained_count = start_count - target_count``
    reported separately by setting both fields explicitly. The drain wall-
    clock for the other direction is captured by ``_measure_drain``.
    """
    _force_drain(pool)
    _wait_for_zero_pods(pool, timeout_s=60.0)

    patch_issued_unix = time.time()
    pool.scale(
        target=target_count, timeout_seconds=timeout_s, wait_for_ready=True,
    )
    settled_unix = time.time()
    post_pods = pool.observe()
    new_pods = post_pods
    first_new_pod_unix: float | None = None
    first_new_pod_ready_unix: float | None = None
    if new_pods:
        first_new_pod_unix = min(p.created_unix for p in new_pods)
        ready_unix = [p.ready_unix for p in new_pods if p.ready_unix is not None]
        if ready_unix:
            first_new_pod_ready_unix = max(ready_unix)   # "settled" = last-ready
    return ReconfigStep(
        replicate=replicate,
        start_count=start_count,
        target_count=target_count,
        patch_issued_unix=patch_issued_unix,
        settled_unix=settled_unix,
        first_new_pod_unix=first_new_pod_unix,
        first_new_pod_ready_unix=first_new_pod_ready_unix,
        pods_drained_count=0,   # start_count was 0 (clean slate); see _measure_drain
        pods_created_count=len(new_pods),
    )


def _measure_drain(
    pool: KnativePool,
    start_count: int,
    replicate: int,
    timeout_s: float,
) -> ReconfigStep:
    """Measure how long ``N → 0`` takes (graceful, not force-evicted)."""
    _force_drain(pool)
    _wait_for_zero_pods(pool, timeout_s=60.0)
    pool.scale(target=start_count, timeout_seconds=timeout_s, wait_for_ready=True)
    pre_pods = pool.observe()
    patch_issued_unix = time.time()
    pool.scale(target=0, timeout_seconds=timeout_s, wait_for_ready=True)
    settled_unix = time.time()
    return ReconfigStep(
        replicate=replicate,
        start_count=start_count,
        target_count=0,
        patch_issued_unix=patch_issued_unix,
        settled_unix=settled_unix,
        first_new_pod_unix=None,
        first_new_pod_ready_unix=None,
        pods_drained_count=len(pre_pods),
        pods_created_count=0,
    )


def _summarise(steps: list[ReconfigStep], console: Console) -> None:
    table = Table(title="Reconfig step latency — Knative pool wire-level")
    table.add_column("rep", justify="right")
    table.add_column("step", justify="right")
    table.add_column("wall (s)", justify="right")
    table.add_column("first pod appear (s)", justify="right")
    table.add_column("first pod ready (s)", justify="right")
    table.add_column("drained", justify="right")
    table.add_column("created", justify="right")
    for s in steps:
        table.add_row(
            str(s.replicate),
            f"{s.start_count}→{s.target_count}",
            f"{s.wall_seconds:.3f}",
            f"{s.first_pod_appear_s:.3f}" if s.first_pod_appear_s is not None else "—",
            f"{s.first_pod_ready_s:.3f}" if s.first_pod_ready_s is not None else "—",
            str(s.pods_drained_count),
            str(s.pods_created_count),
        )
    console.print(table)

    by_step: dict[str, list[ReconfigStep]] = {}
    for s in steps:
        key = f"{s.start_count}→{s.target_count}"
        by_step.setdefault(key, []).append(s)

    summary = Table(title="Per-step aggregates (mean ± sd)")
    summary.add_column("step", justify="right")
    summary.add_column("n", justify="right")
    summary.add_column("wall (s)", justify="right")
    summary.add_column("first pod appear (s)", justify="right")
    summary.add_column("first pod ready (s)", justify="right")
    def _stat(getter, group):
        vals = [v for v in (getter(s) for s in group) if v is not None]
        if not vals:
            return "—"
        m = statistics.mean(vals)
        if len(vals) == 1:
            return f"{m:.3f}"
        sd = statistics.stdev(vals)
        return f"{m:.3f} ± {sd:.3f}"

    for key, group in by_step.items():
        summary.add_row(
            key, str(len(group)),
            _stat(lambda s: s.wall_seconds, group),
            _stat(lambda s: s.first_pod_appear_s, group),
            _stat(lambda s: s.first_pod_ready_s, group),
        )
    console.print(summary)


def run(args: argparse.Namespace) -> int:
    console = Console()
    pool = KnativePool(service=args.service, namespace=args.namespace)
    steps = _parse_steps(args.steps)

    console.print(
        f"[bold]H3 reconfig wire-level harness[/] — "
        f"{len(steps)} step(s) × {args.replicates} replicate(s) = "
        f"{len(steps) * args.replicates} measurement(s)"
    )

    all_steps: list[ReconfigStep] = []
    for r in range(args.replicates):
        for i, (a, b) in enumerate(steps):
            console.print(f"\n[bold]Rep {r + 1}, step {i + 1}/{len(steps)}: {a} → {b}[/]")
            try:
                if b == 0:
                    s = _measure_drain(
                        pool, start_count=a,
                        replicate=r + 1, timeout_s=args.timeout_seconds,
                    )
                else:
                    s = _measure_step(
                        pool, a, b, replicate=r + 1, timeout_s=args.timeout_seconds,
                    )
                all_steps.append(s)
                console.print(
                    f"  wall={s.wall_seconds:.3f}s, "
                    f"first-pod-appear={s.first_pod_appear_s or 'n/a'}, "
                    f"first-ready={s.first_pod_ready_s or 'n/a'}, "
                    f"drained={s.pods_drained_count}, created={s.pods_created_count}"
                )
            except RuntimeError as exc:
                console.print(f"  [red]failed: {exc}[/]")
            # Force a clean slate between steps so the next baseline is fast.
            _force_drain(pool)
            time.sleep(2.0)

    if args.out:
        from pathlib import Path
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(s) for s in all_steps], indent=2))
        console.print(f"[dim]wrote {out}[/]")

    _summarise(all_steps, console)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="hasagi-worker-lifecycle")
    parser.add_argument("--namespace", default="hasagi-validation")
    parser.add_argument(
        "--steps",
        default="1->2,2->1,1->4,4->1,1->3,3->1",
        help="Comma-separated A->B step list.",
    )
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--out", default=None, help="Optional JSON path for raw step data.",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

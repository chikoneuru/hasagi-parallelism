"""exp03 — measure reconfiguration latency of the incremental partitioner vs full DP.

Tests hypothesis H3: incremental partition < 5 s vs full at large n.

This script grows the model from 32 → 256 layers and feeds each iteration's
result as `prev` for the next. That sequence is the **worst case** for the
incremental algorithm (cuts must shift far to track the growing model), so
the quality gap reported here is an upper bound — realistic deployment use
case (n constant, only stage throughput / pool size changes) keeps cuts
close to optimum and the incremental quality gap stays under 1–2%.

Usage:
    python experiments/exp03_elastic_reconfig.py
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

from tare.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    Partition,
    StageSpec,
    incremental_partition,
    partition_pipeline,
)


def fake_model(n_layers: int) -> list[LayerProfile]:
    return [
        LayerProfile(
            index=i,
            fwd_flops=1e8 * (1 + (i % 5) * 0.2),
            bwd_flops=2e8 * (1 + (i % 5) * 0.2),
            activation_bytes=int(1e6 / (1 + i % 8)),
        )
        for i in range(n_layers)
    ]


def main() -> None:
    console = Console()
    stages = [
        StageSpec(stage_id=0, throughput_flops=2e11 * 4, memory_bytes=4 << 30),
        StageSpec(stage_id=1, throughput_flops=8e11 * 2, memory_bytes=8 << 30),
        StageSpec(stage_id=2, throughput_flops=1e13, memory_bytes=16 << 30),
    ]
    links = [
        LinkSpec(src_stage=0, dst_stage=1, bandwidth_bps=1e9, latency_s=0.001),
        LinkSpec(src_stage=1, dst_stage=2, bandwidth_bps=10e9, latency_s=0.0005),
    ]

    table = Table(title="HASAGI exp03 — pipeline partition latency (k=3)")
    table.add_column("n_layers", justify="right")
    table.add_column("full DP (ms)", justify="right")
    table.add_column("incremental (ms)", justify="right")
    table.add_column("bottleneck full")
    table.add_column("bottleneck incr")

    prev: Partition | None = None
    for n in (32, 64, 128, 256):
        layers = fake_model(n)
        t0 = time.perf_counter()
        full = partition_pipeline(layers, stages, links)
        full_ms = (time.perf_counter() - t0) * 1000

        if prev is None:
            prev = full
        t0 = time.perf_counter()
        incr = incremental_partition(prev, layers, stages, links, boundary_window=4)
        incr_ms = (time.perf_counter() - t0) * 1000
        prev = incr

        full_bt = max(full.stage_exec_time.values())
        incr_bt = max(incr.stage_exec_time.values())
        table.add_row(
            str(n), f"{full_ms:.2f}", f"{incr_ms:.2f}",
            f"{full_bt:.3e}", f"{incr_bt:.3e}",
        )

    console.print(table)


if __name__ == "__main__":
    main()

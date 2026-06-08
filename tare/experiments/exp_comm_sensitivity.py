"""Sensitivity of the hybrid-parallel DECISION to the communication-cost model.

This is a simulation / algorithm study on the decision layer only. It does NOT
run distributed training — there is no torch.distributed / pipeline execution
anywhere in this version (see the README banner). What it asks is narrower and
fully answerable from the planner + partitioner code:

  Given that the interconnect bandwidth fed to the planner and partitioner is an
  estimate, how much do their DECISIONS move as that estimate varies, and how
  much modelled-runtime / energy is left on the table by committing to a single
  bandwidth-blind decision across a range of interconnects?

Two decision layers are swept over the same bandwidth grid:

  1. PLANNER (``select_hybrid_strategy`` + ``SimpleRuntimeModel``): which
     (data_parallel, model_parallel) factorisation of a fixed cluster is chosen?
     The all-reduce term scales as 1/bandwidth while compute does not, so the
     optimum is expected to shift from data-parallel-heavy (cheap all-reduce, high
     bandwidth) toward model-parallel-heavy (shard the model to shrink all-reduce,
     low bandwidth). We locate the bandwidth thresholds where (dp, mp) flips.

  2. PARTITIONER (``partition_pipeline``): for a fixed K-stage pipeline, where do
     the k-1 cut points land? At high bandwidth cross-stage activation traffic is
     negligible and cuts balance compute; at low bandwidth the cuts should move to
     avoid high-activation-byte boundaries (comm-dominated stages).

For each layer we report (a) the decision per bandwidth, (b) the flip thresholds,
and (c) the BANDWIDTH-BLIND REGRET: fix the decision that is optimal at one
reference bandwidth, evaluate it across the whole grid, and report the worst-case
and mean modelled penalty versus the bandwidth-aware optimum. Regret is in the
analytic runtime / energy model, not a wall-clock measurement; it is an upper
bound on what a mis-estimated interconnect costs the decision, not a measured
slowdown.

Usage::

    python -m experiments.exp_comm_sensitivity --out artifacts/comm_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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
from tare.parallel.planner import SimpleRuntimeModel, select_hybrid_strategy

_MICROBATCHES = 8
_GLOBAL_BATCH = 1024     # samples per optimiser step, fixed and split across dp replicas

# Named interconnect regimes, bits per second. The planner/partitioner cost model
# uses bits/s (it multiplies payload bytes by 8). These are nominal peak figures;
# real achievable bandwidth is lower, which only sharpens the low-bandwidth story.
NAMED_LINKS: dict[str, float] = {
    "1 GbE": 1.0e9,
    "10 GbE": 1.0e10,
    "25 GbE": 2.5e10,
    "100 GbE": 1.0e11,
    "PCIe4 x16": 2.56e11,   # ~32 GB/s
    "NVLink3": 4.8e12,      # ~600 GB/s
}

# Two model presets bracketing the comm/compute ratio. A CNN has a small parameter
# footprint (cheap all-reduce) so data parallelism survives to low bandwidth; a
# Transformer has a large footprint (expensive all-reduce) so model parallelism is
# favoured earlier as bandwidth drops. Figures are representative, not measured.
MODEL_PRESETS: dict[str, dict] = {
    "cnn": {
        "per_sample_flops": 8.0e9,      # ~ResNet-50 fwd+bwd per sample
        "model_bytes": 100_000_000,     # ~25M params * 4B
        "device_throughput_flops": 3.5e13,  # ~3080 Ti FP32-ish
    },
    "transformer": {
        "per_sample_flops": 6.0e10,     # heavier per-sample compute
        "model_bytes": 1_400_000_000,   # ~350M params * 4B
        "device_throughput_flops": 3.5e13,
    },
}


def _log_grid(lo: float, hi: float, n: int) -> list[float]:
    """n log-spaced points in [lo, hi], inclusive of both ends."""
    if n < 2:
        return [lo]
    step = (math.log10(hi) - math.log10(lo)) / (n - 1)
    return [10.0 ** (math.log10(lo) + i * step) for i in range(n)]


# ---------------------------------------------------------------------------
# Planner sweep
# ---------------------------------------------------------------------------
#
# The sweep uses the SHIPPED ``SimpleRuntimeModel`` configured with a fixed GLOBAL
# batch (``global_batch_size``), the standard data-parallel framing (Megatron,
# Hydrozoa): dp reduces per-replica compute at the cost of an all-reduce, mp
# reduces compute at the cost of a pipeline bubble, so the optimal split is
# bandwidth-dependent. IMPORTANT: the planner's DEFAULT mode (no global batch,
# fixed per-replica microbatch) is bandwidth-INSENSITIVE — it always picks maximal
# model-parallel — and that default is what the orchestrator/admission callers
# currently use. So the bandwidth-sensitivity below is a property of the
# global-batch configuration the orchestrator SHOULD adopt, not of its current
# default. This is the standard comm-vs-compute tradeoff, not a novel finding; the
# only quantitative content here is the regret of a bandwidth-blind choice in this
# cost model (analytic, not wall-clock).

def _make_runtime_model(model: dict, global_batch: int, bw: float) -> SimpleRuntimeModel:
    return SimpleRuntimeModel(network_bandwidth_bps=bw, global_batch_size=global_batch, **model)


def _planner_sweep(cluster_size: int, model: dict, bw_grid: list[float],
                   global_batch: int = _GLOBAL_BATCH) -> list[dict]:
    """For each bandwidth, the (dp, mp) strategy the planner selects and its
    modelled runtime. ``model`` is a MODEL_PRESETS-style dict."""
    out: list[dict] = []
    for bw in bw_grid:
        strat = select_hybrid_strategy(cluster_size, _make_runtime_model(model, global_batch, bw))
        out.append({
            "bandwidth_bps": bw,
            "dp": strat.data_parallel,
            "mp": strat.model_parallel,
            "runtime_s": strat.estimated_runtime_s,
        })
    return out


def _strategy_flips(sweep: list[dict]) -> list[dict]:
    """Bandwidth points (ascending) at which the chosen (dp, mp) changes."""
    flips: list[dict] = []
    s = sorted(sweep, key=lambda r: r["bandwidth_bps"])
    for prev, cur in zip(s, s[1:], strict=False):
        if (prev["dp"], prev["mp"]) != (cur["dp"], cur["mp"]):
            flips.append({
                "below_bps": prev["bandwidth_bps"],
                "above_bps": cur["bandwidth_bps"],
                "from": [prev["dp"], prev["mp"]],
                "to": [cur["dp"], cur["mp"]],
            })
    return flips


def _planner_blind_regret(cluster_size: int, model: dict, bw_grid: list[float],
                          blind_bw: float, global_batch: int = _GLOBAL_BATCH) -> dict:
    """Regret of committing to the strategy optimal at ``blind_bw`` and using it
    across the whole grid, versus re-planning per bandwidth.

    For each bandwidth we evaluate the blind strategy's runtime with that
    bandwidth's runtime model and compare to the bandwidth-aware optimum. Returns
    the worst-case and mean fractional penalty.
    """
    blind = select_hybrid_strategy(cluster_size, _make_runtime_model(model, global_batch, blind_bw))
    penalties: list[float] = []
    worst = {"bandwidth_bps": blind_bw, "regret": 0.0}
    for bw in bw_grid:
        rt = _make_runtime_model(model, global_batch, bw)
        opt = select_hybrid_strategy(cluster_size, rt).estimated_runtime_s
        blind_rt = rt(blind.data_parallel, blind.model_parallel)
        reg = blind_rt / opt - 1.0 if opt > 0 else 0.0
        penalties.append(reg)
        if reg > worst["regret"]:
            worst = {"bandwidth_bps": bw, "regret": reg}
    return {
        "blind_bandwidth_bps": blind_bw,
        "blind_strategy": [blind.data_parallel, blind.model_parallel],
        "max_regret": worst["regret"],
        "max_regret_at_bps": worst["bandwidth_bps"],
        "mean_regret": sum(penalties) / len(penalties),
    }


# ---------------------------------------------------------------------------
# Partitioner sweep
# ---------------------------------------------------------------------------

def _synthetic_layers(n: int) -> list[LayerProfile]:
    """A synthetic layer chain with non-uniform activation sizes so that
    bandwidth materially changes where cuts want to land. Layers in the middle
    third carry large activations (expensive cut boundaries); the ends are light.
    Labelled synthetic — not a profiled model."""
    layers: list[LayerProfile] = []
    for i in range(n):
        # compute roughly uniform; activations spike in the middle third.
        in_mid = n // 3 <= i < 2 * n // 3
        act = 64_000_000 if in_mid else 4_000_000
        layers.append(LayerProfile(
            index=i,
            fwd_flops=1.0e9,
            bwd_flops=2.0e9,
            activation_bytes=act,
        ))
    return layers


def _uniform_stages(k: int, throughput_flops: float, power_w: float = 250.0) -> list[StageSpec]:
    return [
        StageSpec(stage_id=s, throughput_flops=throughput_flops,
                  memory_bytes=10_000_000_000, power_draw_w=power_w)
        for s in range(k)
    ]


def _links(k: int, bw: float) -> list[LinkSpec]:
    return [LinkSpec(src_stage=s, dst_stage=s + 1, bandwidth_bps=bw) for s in range(k - 1)]


def _partition_sweep(layers: list[LayerProfile], k: int, throughput_flops: float,
                     bw_grid: list[float], objective: str) -> list[dict]:
    stages = _uniform_stages(k, throughput_flops)
    out: list[dict] = []
    for bw in bw_grid:
        part = partition_pipeline(layers, stages, _links(k, bw),
                                  num_microbatches=_MICROBATCHES, objective=objective)
        out.append({
            "bandwidth_bps": bw,
            "cuts": list(part.cuts),
            "bottleneck_s": max(part.stage_exec_time.values()),
            "pipeline_time_s": part.pipeline_time,
            "energy_per_iter_j": part.energy_per_iter,
        })
    return out


def _cut_flips(sweep: list[dict]) -> list[dict]:
    flips: list[dict] = []
    s = sorted(sweep, key=lambda r: r["bandwidth_bps"])
    for prev, cur in zip(s, s[1:], strict=False):
        if prev["cuts"] != cur["cuts"]:
            flips.append({
                "below_bps": prev["bandwidth_bps"],
                "above_bps": cur["bandwidth_bps"],
                "from": prev["cuts"],
                "to": cur["cuts"],
            })
    return flips


def _partition_blind_regret(layers: list[LayerProfile], k: int, throughput_flops: float,
                            bw_grid: list[float], blind_bw: float, objective: str) -> dict:
    """Regret of fixing the cut set optimal at ``blind_bw`` and reusing it across
    bandwidths, versus re-partitioning. The blind cuts are evaluated at each
    bandwidth by rebuilding the partition at the blind cut points."""
    stages = _uniform_stages(k, throughput_flops)
    blind = partition_pipeline(layers, stages, _links(k, blind_bw),
                               num_microbatches=_MICROBATCHES, objective=objective)
    metric = "energy_per_iter" if objective == "energy" else "pipeline_time"

    def _score(part) -> float:
        return getattr(part, metric)

    penalties: list[float] = []
    worst = {"bandwidth_bps": blind_bw, "regret": 0.0}
    for bw in bw_grid:
        links = _links(k, bw)
        opt = _score(partition_pipeline(layers, stages, links,
                                        num_microbatches=_MICROBATCHES, objective=objective))
        # Re-evaluate the blind cut set at this bandwidth without re-optimising.
        blind_here = _rebuild_at_cuts(layers, stages, links, blind.cuts)
        blind_score = _score(blind_here)
        reg = blind_score / opt - 1.0 if opt > 0 else 0.0
        penalties.append(reg)
        if reg > worst["regret"]:
            worst = {"bandwidth_bps": bw, "regret": reg}
    return {
        "blind_bandwidth_bps": blind_bw,
        "blind_cuts": list(blind.cuts),
        "objective": objective,
        "max_regret": worst["regret"],
        "max_regret_at_bps": worst["bandwidth_bps"],
        "mean_regret": sum(penalties) / len(penalties),
    }


def _rebuild_at_cuts(layers: list[LayerProfile], stages: list[StageSpec],
                     links: list[LinkSpec], cuts: tuple[int, ...]) -> Partition:
    """Evaluate a FIXED cut set under a (possibly new) link set, without
    re-optimising. incremental_partition with boundary_window=0 rebuilds exactly
    ``cuts`` against the current links and returns that partition."""
    prev = Partition(cuts=tuple(cuts), num_stages=len(stages))
    return incremental_partition(prev, layers, stages, links,
                                 boundary_window=0, num_microbatches=_MICROBATCHES)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    console = Console()
    bw_grid = _log_grid(1.0e9, 1.0e13, args.grid_points)
    blind_bw = NAMED_LINKS["10 GbE"]  # a common default assumption to commit to

    summary: dict = {"cluster_size": args.cluster_size, "grid_points": args.grid_points,
                     "blind_bandwidth_bps": blind_bw, "planner": {}, "partitioner": {}}

    # ---- Planner: per model preset ----
    console.print("[bold]Planner: hybrid-strategy sensitivity to interconnect bandwidth[/]")
    for name, model in MODEL_PRESETS.items():
        sweep = _planner_sweep(args.cluster_size, model, bw_grid)
        flips = _strategy_flips(sweep)
        regret = _planner_blind_regret(args.cluster_size, model, bw_grid, blind_bw)
        summary["planner"][name] = {"flips": flips, "blind_regret": regret}

        t = Table(title=f"planner / {name} — selected (dp×mp) at named interconnects "
                        f"(cluster={args.cluster_size})")
        t.add_column("interconnect")
        t.add_column("bw (Gbps)", justify="right")
        t.add_column("dp×mp", justify="right")
        t.add_column("runtime (s)", justify="right")
        for link_name, bw in NAMED_LINKS.items():
            st = select_hybrid_strategy(args.cluster_size, _make_runtime_model(model, _GLOBAL_BATCH, bw))
            t.add_row(link_name, f"{bw / 1e9:.0f}", f"{st.data_parallel}×{st.model_parallel}",
                      f"{st.estimated_runtime_s:.4f}")
        console.print(t)
        if flips:
            fl = ", ".join(f"{f['from'][0]}×{f['from'][1]}→{f['to'][0]}×{f['to'][1]} "
                           f"between {f['below_bps']/1e9:.1f} and {f['above_bps']/1e9:.1f} Gbps"
                           for f in flips)
            console.print(f"  strategy flips: {fl}")
        else:
            console.print("  [yellow]no strategy flip across the grid (decision bandwidth-insensitive here)[/]")
        console.print(f"  bandwidth-blind regret (commit to {blind_bw/1e9:.0f} Gbps choice "
                      f"{regret['blind_strategy'][0]}×{regret['blind_strategy'][1]}): "
                      f"max {regret['max_regret']*100:.1f}% @ {regret['max_regret_at_bps']/1e9:.2f} Gbps, "
                      f"mean {regret['mean_regret']*100:.1f}%\n")

    # ---- Partitioner: cut sensitivity ----
    console.print("[bold]Partitioner: pipeline-cut sensitivity to interconnect bandwidth[/]")
    layers = _synthetic_layers(args.layers)
    for objective in ("bottleneck", "energy"):
        sweep = _partition_sweep(layers, args.stages, MODEL_PRESETS["cnn"]["device_throughput_flops"],
                                 bw_grid, objective)
        flips = _cut_flips(sweep)
        regret = _partition_blind_regret(layers, args.stages,
                                         MODEL_PRESETS["cnn"]["device_throughput_flops"],
                                         bw_grid, blind_bw, objective)
        summary["partitioner"][objective] = {"flips": flips, "blind_regret": regret}

        t = Table(title=f"partitioner / {objective} — cuts at named interconnects "
                        f"(n={args.layers} layers, K={args.stages} stages)")
        t.add_column("interconnect")
        t.add_column("bw (Gbps)", justify="right")
        t.add_column("cuts")
        t.add_column("bottleneck (s)", justify="right")
        for link_name, bw in NAMED_LINKS.items():
            part = partition_pipeline(layers, _uniform_stages(args.stages, MODEL_PRESETS["cnn"]["device_throughput_flops"]),
                                      _links(args.stages, bw), num_microbatches=_MICROBATCHES, objective=objective)
            t.add_row(link_name, f"{bw / 1e9:.0f}", str(list(part.cuts)),
                      f"{max(part.stage_exec_time.values()):.5f}")
        console.print(t)
        if flips:
            console.print(f"  cut set moves {len(flips)} time(s) across the grid; "
                          f"lowest-bandwidth cuts {flips[0]['from']} vs highest {sweep[-1]['cuts']}")
        else:
            console.print("  [yellow]cut set constant across the grid[/]")
        console.print(f"  bandwidth-blind regret (commit to {blind_bw/1e9:.0f} Gbps cuts "
                      f"{regret['blind_cuts']}): max {regret['max_regret']*100:.1f}% @ "
                      f"{regret['max_regret_at_bps']/1e9:.2f} Gbps, mean {regret['mean_regret']*100:.1f}%\n")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster-size", type=int, default=16)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--grid-points", type=int, default=33)
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

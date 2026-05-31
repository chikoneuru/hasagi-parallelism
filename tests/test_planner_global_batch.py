"""Tests for SimpleRuntimeModel's two batch conventions and the strategy planner.

The default (fixed per-replica microbatch) mode is bandwidth-insensitive and is
what the orchestrator/admission callers rely on; the opt-in fixed-global-batch
mode makes the data/model-parallel split bandwidth-dependent.
"""
from __future__ import annotations

from hasagi.parallel.planner import SimpleRuntimeModel, select_hybrid_strategy

_KW = {
    "per_sample_flops": 8.0e9,
    "model_bytes": 100_000_000,
    "device_throughput_flops": 3.5e13,
}


def test_default_mode_is_bandwidth_insensitive_max_model_parallel() -> None:
    # Legacy behaviour: fixed per-replica microbatch -> dp carries only all-reduce
    # cost -> the optimum is maximal model-parallel at every bandwidth.
    for bw in (1e9, 1e10, 1e11, 4.8e12):
        rt = SimpleRuntimeModel(network_bandwidth_bps=bw, **_KW)
        s = select_hybrid_strategy(16, rt)
        assert (s.data_parallel, s.model_parallel) == (1, 16)


def test_global_batch_mode_shifts_split_with_bandwidth() -> None:
    # Fixed global batch split across dp: low bandwidth -> model-parallel,
    # high bandwidth -> more data-parallel.
    lo = select_hybrid_strategy(16, SimpleRuntimeModel(network_bandwidth_bps=1e9,
                                                       global_batch_size=1024, **_KW))
    hi = select_hybrid_strategy(16, SimpleRuntimeModel(network_bandwidth_bps=4.8e12,
                                                       global_batch_size=1024, **_KW))
    assert lo.data_parallel == 1            # all-reduce too dear at 1 Gbps
    assert hi.data_parallel > lo.data_parallel


def test_global_batch_compute_scales_inversely_with_dp() -> None:
    # samples_per_replica = global_batch / dp, so at fixed mp and infinite bandwidth
    # (no all-reduce penalty) more dp is strictly faster.
    rt = SimpleRuntimeModel(network_bandwidth_bps=1e15, global_batch_size=1024, **_KW)
    assert rt(8, 1) < rt(1, 1)


def test_default_mode_unchanged_when_global_batch_none() -> None:
    # Backward-compat: the default path must match the original closed form.
    rt = SimpleRuntimeModel(network_bandwidth_bps=1e10, **_KW)
    comp = _KW["per_sample_flops"] * rt.microbatch_count / (_KW["device_throughput_flops"] * 2)
    shard = _KW["model_bytes"] / 2
    allreduce = 2.0 * (4 - 1) * shard * 8.0 / 1e10
    bubble = rt.pipeline_alpha * (2 - 1) * comp
    assert abs(rt(4, 2) - (comp + allreduce + bubble)) < 1e-12

"""Tests for the comm-sensitivity decision study (pure algorithm, no GPU).

Covers the bandwidth grid, the planner strategy sweep + flip detection +
bandwidth-blind regret, and the partitioner cut sweep + fixed-cut rebuild.
"""
from __future__ import annotations

from experiments.exp_comm_sensitivity import (
    MODEL_PRESETS,
    NAMED_LINKS,
    _cut_flips,
    _links,
    _log_grid,
    _make_runtime_model,
    _partition_blind_regret,
    _partition_sweep,
    _planner_blind_regret,
    _planner_sweep,
    _rebuild_at_cuts,
    _strategy_flips,
    _synthetic_layers,
    _uniform_stages,
)

_THRU = MODEL_PRESETS["cnn"]["device_throughput_flops"]


def test_log_grid_endpoints_and_length() -> None:
    g = _log_grid(1.0e9, 1.0e13, 5)
    assert len(g) == 5
    assert abs(g[0] - 1.0e9) / 1.0e9 < 1e-9
    assert abs(g[-1] - 1.0e13) / 1.0e13 < 1e-9
    assert all(g[i] < g[i + 1] for i in range(len(g) - 1))


def test_runtime_model_uses_shipped_class_with_global_batch() -> None:
    # The sweep configures the production SimpleRuntimeModel with a fixed global
    # batch (not a hand-written shadow model). Its default (per-replica) mode is
    # bandwidth-insensitive; the global-batch mode is what makes the split move.
    from tare.parallel.planner import SimpleRuntimeModel
    rt = _make_runtime_model(MODEL_PRESETS["cnn"], 1024, 1e10)
    assert isinstance(rt, SimpleRuntimeModel)
    assert rt.global_batch_size == 1024


def test_runtime_allreduce_vanishes_at_dp1_and_grows_as_bandwidth_drops() -> None:
    model = MODEL_PRESETS["cnn"]
    # dp=1 -> no all-reduce, so runtime is bandwidth-independent.
    assert _make_runtime_model(model, 1024, 1e9)(1, 16) == _make_runtime_model(model, 1024, 1e13)(1, 16)
    # dp>1 -> lower bandwidth is strictly slower.
    assert _make_runtime_model(model, 1024, 1e9)(8, 2) > _make_runtime_model(model, 1024, 1e13)(8, 2)


def test_planner_shifts_from_model_parallel_to_data_parallel_with_bandwidth() -> None:
    grid = _log_grid(1.0e9, 1.0e13, 25)
    sweep = _planner_sweep(16, MODEL_PRESETS["cnn"], grid)
    lo = sweep[0]   # 1 Gbps
    hi = sweep[-1]  # 10 Tbps
    # Low bandwidth: all-reduce dominates -> maximal model parallel (dp == 1).
    assert lo["dp"] == 1
    # High bandwidth: all-reduce cheap -> more data parallel than at low bandwidth.
    assert hi["dp"] > lo["dp"]


def test_strategy_flips_detected_and_drive_blind_regret() -> None:
    grid = _log_grid(1.0e9, 1.0e13, 33)
    sweep = _planner_sweep(16, MODEL_PRESETS["cnn"], grid)
    flips = _strategy_flips(sweep)
    assert flips, "expected at least one (dp,mp) flip across 1 Gbps..10 Tbps"
    # Each flip brackets a bandwidth and changes the decision.
    for f in flips:
        assert f["below_bps"] < f["above_bps"]
        assert f["from"] != f["to"]
    # A decision that flips must carry positive bandwidth-blind regret.
    reg = _planner_blind_regret(16, MODEL_PRESETS["cnn"], grid, NAMED_LINKS["10 GbE"])
    assert reg["max_regret"] > 0.0
    assert reg["mean_regret"] >= 0.0


def test_transformer_holds_model_parallel_to_higher_bandwidth_than_cnn() -> None:
    # Heavier model_bytes -> all-reduce stays expensive longer -> the first move
    # off pure model-parallel happens at a higher bandwidth for the transformer.
    grid = _log_grid(1.0e9, 1.0e13, 60)

    def first_dp_gt1_bw(model: dict) -> float:
        for r in _planner_sweep(16, model, grid):
            if r["dp"] > 1:
                return r["bandwidth_bps"]
        return float("inf")

    assert first_dp_gt1_bw(MODEL_PRESETS["transformer"]) > first_dp_gt1_bw(MODEL_PRESETS["cnn"])


def test_partition_cuts_move_with_bandwidth() -> None:
    layers = _synthetic_layers(24)
    grid = _log_grid(1.0e9, 1.0e13, 20)
    sweep = _partition_sweep(layers, 4, _THRU, grid, "bottleneck")
    flips = _cut_flips(sweep)
    assert flips, "cut placement should depend on bandwidth"
    # Lowest-bandwidth cuts differ from highest-bandwidth cuts.
    assert sweep[0]["cuts"] != sweep[-1]["cuts"]


def test_rebuild_at_cuts_preserves_the_requested_cuts() -> None:
    layers = _synthetic_layers(24)
    stages = _uniform_stages(4, _THRU)
    cuts = (5, 11, 17)
    part = _rebuild_at_cuts(layers, stages, _links(4, 1e11), cuts)
    assert part.cuts == cuts
    assert part.num_stages == 4
    assert part.is_feasible()


def test_partition_blind_regret_positive_for_bandwidth_sensitive_cuts() -> None:
    layers = _synthetic_layers(24)
    grid = _log_grid(1.0e9, 1.0e13, 20)
    reg = _partition_blind_regret(layers, 4, _THRU, grid, NAMED_LINKS["10 GbE"], "bottleneck")
    assert reg["max_regret"] > 0.0
    assert reg["objective"] == "bottleneck"
    assert len(reg["blind_cuts"]) == 3

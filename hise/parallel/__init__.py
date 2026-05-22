"""Hybrid Parallel Controller — PipeDream k-way pipeline + Hydrozoa hybrid strategy."""
from hise.parallel.inter_batch import (
    EnergyAwareWRR,
    InterBatchScheduler,
    Node,
    PowerSlackGuard,
    energy_weights_for_stage,
    weights_for_stage,
)
from hise.parallel.partitioner import (
    LayerProfile,
    LinkSpec,
    Partition,
    StageSpec,
    StagnationTracker,
    incremental_partition,
    partition_pipeline,
)
from hise.parallel.planner import HybridStrategy, select_hybrid_strategy

__all__ = [
    "EnergyAwareWRR",
    "HybridStrategy",
    "InterBatchScheduler",
    "LayerProfile",
    "LinkSpec",
    "Node",
    "Partition",
    "PowerSlackGuard",
    "StageSpec",
    "StagnationTracker",
    "energy_weights_for_stage",
    "incremental_partition",
    "partition_pipeline",
    "select_hybrid_strategy",
    "weights_for_stage",
]

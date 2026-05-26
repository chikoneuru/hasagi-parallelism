"""Hybrid Parallel Controller — PipeDream k-way pipeline + Hydrozoa hybrid strategy."""
from hise.parallel.inter_batch import (
    EnergyAwareWRR,
    InterBatchScheduler,
    Node,
    PowerSlackGuard,
    energy_weights_for_stage,
    weights_for_stage,
)
from hise.parallel.joint_partitioner import JointPlan, joint_partition
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
from hise.parallel.stochastic_joint_partitioner import (
    StochasticJointPlan,
    stochastic_joint_partition,
)

__all__ = [
    "EnergyAwareWRR",
    "HybridStrategy",
    "InterBatchScheduler",
    "JointPlan",
    "LayerProfile",
    "LinkSpec",
    "Node",
    "Partition",
    "PowerSlackGuard",
    "StageSpec",
    "StagnationTracker",
    "StochasticJointPlan",
    "energy_weights_for_stage",
    "incremental_partition",
    "joint_partition",
    "partition_pipeline",
    "select_hybrid_strategy",
    "stochastic_joint_partition",
    "weights_for_stage",
]

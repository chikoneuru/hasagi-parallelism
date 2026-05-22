"""Energy-Signal Aware Scheduler — telemetry, carbon trace, API client, scheduling policies."""
from hise.energy.carbon_sources import (
    AggregateCarbonReading,
    CarbonReading,
    CarbonSource,
    ElectricityMapsClient,
    IEAStaticSource,
    MultiSourceCarbonAggregator,
    WattTimeClient,
)
from hise.energy.carbon_trace import CarbonTrace, load_csv_trace, synthetic_solar_trace
from hise.energy.policy import (
    EnergyDecision,
    MPCPolicy,
    PowerAwareRulePolicy,
    RuleBasedPolicy,
)
from hise.energy.rl_policy import (
    OBSERVATION_DIM,
    PPOAction,
    PPOObservation,
    PPOScalingPolicy,
    build_observation,
    compute_reward,
    discrete_action_space_size,
)
from hise.energy.telemetry import (
    AggregateTelemetrySource,
    FakeTelemetrySource,
    FakeWorker,
    NvmlTelemetrySource,
    PrometheusPusher,
    RaplTelemetrySource,
    TelemetrySource,
    WorkerTelemetry,
)

__all__ = [
    "OBSERVATION_DIM",
    "AggregateCarbonReading",
    "AggregateTelemetrySource",
    "CarbonReading",
    "CarbonSource",
    "CarbonTrace",
    "ElectricityMapsClient",
    "EnergyDecision",
    "FakeTelemetrySource",
    "FakeWorker",
    "IEAStaticSource",
    "MPCPolicy",
    "MultiSourceCarbonAggregator",
    "NvmlTelemetrySource",
    "PPOAction",
    "PPOObservation",
    "PPOScalingPolicy",
    "PowerAwareRulePolicy",
    "PrometheusPusher",
    "RaplTelemetrySource",
    "RuleBasedPolicy",
    "TelemetrySource",
    "WattTimeClient",
    "WorkerTelemetry",
    "build_observation",
    "compute_reward",
    "discrete_action_space_size",
    "load_csv_trace",
    "synthetic_solar_trace",
]

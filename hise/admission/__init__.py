"""ElasticFlow admission control + HISE Energy-Budgeted MSS extension."""
from hise.admission.mss import (
    AdmissionDecision,
    EnergyAdjustedMSS,  # backwards-compat alias
    EnergyBudgetMSS,
    ScalingCurve,
    minimum_satisfactory_share,
)

__all__ = [
    "AdmissionDecision",
    "EnergyAdjustedMSS",
    "EnergyBudgetMSS",
    "ScalingCurve",
    "minimum_satisfactory_share",
]

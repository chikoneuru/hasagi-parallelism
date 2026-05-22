"""ElasticFlow admission control + HISE Energy-Budgeted MSS extension."""
from hise.admission.energy_profile import EnergyProfile, linear_profile
from hise.admission.mss import (
    AdmissionDecision,
    EnergyAdjustedMSS,  # backwards-compat alias
    EnergyBudgetMSS,
    ScalingCurve,
    greedy_marginal_allocation,
    greedy_marginal_energy_allocation,
    minimum_satisfactory_share,
)

__all__ = [
    "AdmissionDecision",
    "EnergyAdjustedMSS",
    "EnergyBudgetMSS",
    "EnergyProfile",
    "ScalingCurve",
    "greedy_marginal_allocation",
    "greedy_marginal_energy_allocation",
    "linear_profile",
    "minimum_satisfactory_share",
]

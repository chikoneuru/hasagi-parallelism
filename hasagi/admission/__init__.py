"""ElasticFlow admission control + HASAGI Energy-Budgeted MSS extension."""
from hasagi.admission.energy_profile import EnergyProfile, linear_profile
from hasagi.admission.mss import (
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

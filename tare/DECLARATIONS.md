# Pre-declared decision gates and analysis plans (provenance index)

The paper uses the term "pre-declared" for decision rules written into this
repository's analysis code before the corresponding runs. This file is the
index a reviewer can audit; each declaration lives in the named source file
(with git history as the timestamp authority).

| Declaration | Where it is written | Used by |
|---|---|---|
| Structural break-even gate: E_eco <= 0.75 ALIVE / 0.75-0.85 MARGINAL / > 0.85 DEAD | `experiments/exp_ws6_microprobe.py:5-10` (docstring) and `:200-203` (verdict logic); restated `experiments/exp_tier1a_multigpu_probe.py:24-32` | 2/4/8-GPU dense probes, MoE probe (same gate via `break_even_estar` field) |
| 16-zone real-trace panel (zone list fixed before replay) | `experiments/exp_realtrace_pareto.py` zone list + `data_cache/real_traces/` CSV set | oracle/deployable signal values, fair gap, pause savings |
| 14-zone coverage-expansion list (declared before its data collection; the expansion itself is a post-hoc robustness step, as the paper states) | `data_cache/real_traces/` additional CSVs + `artifacts/realtrace_pareto_30zone.json` provenance fields | 30-zone robustness estimates |
| Zone-clustered inference plan (cluster = zone; bootstrap + exact sign-flip permutation + Holm families) | `tare/stats/bootstrap.py` (`clustered_bootstrap_ci`, `clustered_permutation_pvalue`); family definitions in `experiments/headline_stats.py` and `exp_realtrace_pareto.py` | every headline CI/p-value |
| MoE dead-hypothesis (expected DEAD; the probe refuted it in both directions) | `artifacts/moe_world4_*.json` `hypothesis` field (written by the probe harness before verdicting) | expert-parallel cell |
| Carbon-contingency quadrant criterion: a carbon-contingent layout switch requires a measured operating point that is slower but lower-energy; a sub-unity energy ratio that is also faster is a static default, not a switch target. Adopted post hoc for the published dense verdict (the `E_eco <= 0.75` gate above returned MARGINAL, not DEAD, for the memory-pressured points), and recorded here ahead of any future runs as the primary criterion for re-measurements; the gate bands above are reported alongside it, not replaced | this row (git history is the timestamp authority) | future dense probes, pipeline/tensor-parallel columns, MoE re-runs |

Notes:
- The declarations are code-level, not a third-party registry; the paper's
  wording ("pre-declared, written into the analysis code before the runs")
  matches exactly this.
- Git history of this repository is the timestamp evidence; the files above are
  tracked.

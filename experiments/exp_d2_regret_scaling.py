"""D2 — empirical regret-vs-T fit for the online primal-dual policy.

The T8 envelope quoted in ``proofs.md`` §10.4 is ``R(T) = O(L_max · √T)``.
Supervisor audit (Tier 3 D2) asks: does the empirical regret slope
match the predicted rate, on a *real* carbon trace rather than the
synthetic noisy-solar one used by
:mod:`experiments.exp_online_primal_dual_sweep`?

Method:
  - Load a real hourly carbon-intensity CSV (e.g., DE from Energy-Charts
    via ``fetch_real_carbon_traces.py``).
  - Sweep horizon ``T`` over a wide range (default ~6 to ~7 days at
    hourly cadence).
  - Per T, sample N random start offsets in the trace (paired across
    methods), slice T consecutive hours, run offline LP and the online
    primal-dual policy, and record regret = pd_cost − offline_cost.
  - Fit ``log(mean regret) = α · log(T) + β`` via OLS in log-log space
    (the standard rate-fit method for regret bounds).
  - Report α with a percentile-bootstrap 95% CI over (start_offset)
    resamples. Compare against the predicted exponents:
        α = 0.50 → √T (T8 envelope)
        α = 0.75 → T^{3/4} (constrained-OCO bound)
        α = 1.00 → linear (worst case)

Usage::

    python -m experiments.exp_d2_regret_scaling \\
        --trace data_cache/real_traces/de_2024-07-01_2024-07-15_hourly.csv \\
        --t-values 12 24 48 96 144 192 240 288 \\
        --n-starts 30 \\
        --out artifacts/d2_regret_scaling.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiments.exp_online_deadline import (
    build_action_set,
    offline_lp_solve,
    online_primal_dual,
)
from hise.energy.carbon_trace import load_electricitymaps_csv

N_BOOT = 4_000
RNG_SEED = 0


@dataclass(frozen=True)
class TrialResult:
    t_hours: int
    start_offset: int
    offline_cost: float
    pd_cost: float
    regret: float


@dataclass(frozen=True)
class TPointSummary:
    t_hours: int
    n_starts: int
    mean_regret: float
    ci_lo: float
    ci_hi: float
    median_regret: float


def _bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean of ``values``."""
    n = len(values)
    if n < 2:
        m = statistics.mean(values) if values else 0.0
        return (m, m)
    means = [statistics.mean(rng.choices(values, k=n)) for _ in range(N_BOOT)]
    means.sort()
    return (means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1])


def _ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) from least-squares fit y = slope x + intercept."""
    n = len(xs)
    if n < 2:
        return (0.0, ys[0] if ys else 0.0)
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den != 0 else 0.0
    intercept = my - slope * mx
    return slope, intercept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", type=Path, required=True,
        help="ElectricityMaps-format CSV with hourly carbon intensity.",
    )
    parser.add_argument(
        "--t-values", nargs="+", type=int,
        default=[12, 24, 48, 96, 144, 192, 240, 288],
        help="Horizon lengths in hours.",
    )
    parser.add_argument("--n-starts", type=int, default=30,
                        help="Random start offsets per T.")
    parser.add_argument("--deadline-multiplier", type=float, default=1.0)
    parser.add_argument("--out", default="artifacts/d2_regret_scaling.json")
    args = parser.parse_args()

    console = Console()
    actions = build_action_set()
    mu_min = min(a.throughput for a in actions)
    mu_max = max(a.throughput for a in actions)
    target_mu = args.deadline_multiplier * (mu_min + mu_max) / 2

    trace = load_electricitymaps_csv(args.trace)
    intensities = list(trace.intensities)
    n_trace = len(intensities)
    console.print(
        f"[bold]D2 regret scaling[/]: trace={args.trace.name} "
        f"({n_trace} ticks), T values {args.t_values}, "
        f"{args.n_starts} starts per T"
    )

    rng_starts = random.Random(RNG_SEED)
    all_trials: list[TrialResult] = []
    summaries: list[TPointSummary] = []
    rng_boot = random.Random(RNG_SEED + 1)

    for t_hours in args.t_values:
        if t_hours > n_trace:
            console.print(f"[yellow]skip T={t_hours}: trace only has {n_trace} ticks[/]")
            continue
        max_offset = n_trace - t_hours
        # Deterministic random offsets sampled with replacement; same starts
        # used across methods (paired comparison preserved trivially because
        # offline_lp_solve and online_primal_dual see the exact same slice).
        offsets = [rng_starts.randint(0, max_offset) for _ in range(args.n_starts)]
        regrets_at_t: list[float] = []
        for offset in offsets:
            slc = intensities[offset:offset + t_hours]
            deadline_iters = target_mu * t_hours
            off_cost, _, _ = offline_lp_solve(slc, actions, deadline_iters)
            pd_cost, _ = online_primal_dual(slc, actions, deadline_iters)
            regret = pd_cost - off_cost
            all_trials.append(TrialResult(
                t_hours=t_hours, start_offset=offset,
                offline_cost=off_cost, pd_cost=pd_cost, regret=regret,
            ))
            regrets_at_t.append(regret)
        lo, hi = _bootstrap_ci(regrets_at_t, rng_boot)
        summaries.append(TPointSummary(
            t_hours=t_hours, n_starts=len(regrets_at_t),
            mean_regret=statistics.mean(regrets_at_t),
            ci_lo=lo, ci_hi=hi,
            median_regret=statistics.median(regrets_at_t),
        ))

    table = Table(title="D2 — regret per horizon T (real DE 14-day trace, paired starts)")
    table.add_column("T (h)", justify="right")
    table.add_column("n starts", justify="right")
    table.add_column("mean regret", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("median regret", justify="right")
    for s in summaries:
        table.add_row(
            str(s.t_hours), str(s.n_starts),
            f"{s.mean_regret:.2f}",
            f"[{s.ci_lo:.2f}, {s.ci_hi:.2f}]",
            f"{s.median_regret:.2f}",
        )
    console.print(table)

    # Log-log fit on the mean-regret points. Drop any T whose mean regret
    # is non-positive (fit is undefined on log scale).
    fit_pts = [(s.t_hours, s.mean_regret) for s in summaries if s.mean_regret > 0]
    if len(fit_pts) < 2:
        console.print("[red]not enough positive-regret T points to fit[/]")
        slope = float("nan")
        intercept = float("nan")
        slope_ci = (float("nan"), float("nan"))
    else:
        log_t = [math.log(t) for t, _ in fit_pts]
        log_r = [math.log(r) for _, r in fit_pts]
        slope, intercept = _ols_slope(log_t, log_r)

        # Bootstrap CI for slope: resample (T, regret) pairs.
        rng_slope = random.Random(RNG_SEED + 2)
        slopes: list[float] = []
        for _ in range(N_BOOT):
            sample = rng_slope.choices(fit_pts, k=len(fit_pts))
            xs = [math.log(t) for t, _ in sample]
            ys = [math.log(r) for _, r in sample]
            s_b, _ = _ols_slope(xs, ys)
            slopes.append(s_b)
        slopes.sort()
        slope_ci = (slopes[int(0.025 * N_BOOT)], slopes[int(0.975 * N_BOOT) - 1])

    fit_table = Table(title="Power-law fit  R(T) ≈ exp(intercept) · T^α")
    fit_table.add_column("quantity")
    fit_table.add_column("value", justify="right")
    fit_table.add_row("α (empirical slope)", f"{slope:+.3f}")
    fit_table.add_row("95% bootstrap CI on α", f"[{slope_ci[0]:+.3f}, {slope_ci[1]:+.3f}]")
    fit_table.add_row("intercept (log scale)", f"{intercept:+.3f}")
    fit_table.add_row("predicted α  (√T envelope)", "0.500")
    fit_table.add_row("predicted α  (T^{3/4} bound)", "0.750")
    fit_table.add_row("worst-case α (linear)", "1.000")
    console.print(fit_table)

    if slope_ci[0] != slope_ci[0] or slope_ci[1] != slope_ci[1]:  # NaN check
        verdict = "fit failed"
    elif slope_ci[1] <= 0.50:
        verdict = "α ≤ 0.5 ⇒ regret scales no worse than √T (envelope holds)"
    elif slope_ci[0] <= 0.50 <= slope_ci[1]:
        verdict = "α straddles 0.5 ⇒ consistent with √T envelope"
    elif slope_ci[1] <= 0.75:
        verdict = "α ∈ (0.5, 0.75] ⇒ between √T and T^{3/4}"
    elif slope_ci[0] <= 0.75 <= slope_ci[1]:
        verdict = "α straddles 0.75 ⇒ consistent with T^{3/4} bound"
    elif slope_ci[1] <= 1.0:
        verdict = "α ∈ (0.75, 1.0] ⇒ worse than T^{3/4}, better than linear"
    else:
        verdict = "α > 1 ⇒ super-linear regret growth (UNEXPECTED, investigate)"
    console.print(f"\n[bold]Verdict[/]: {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": {**vars(args), "trace": str(args.trace)},
        "trials": [asdict(t) for t in all_trials],
        "summaries": [asdict(s) for s in summaries],
        "fit": {
            "slope": slope, "intercept": intercept,
            "slope_ci_lo": slope_ci[0], "slope_ci_hi": slope_ci[1],
            "verdict": verdict,
        },
    }, indent=2))
    console.print(f"\n[dim]wrote {out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

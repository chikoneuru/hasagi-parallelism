"""Compute paired-bootstrap CI + Cohen's d + Holm-Bonferroni for every headline.

Ingests the existing artifact JSONs from ``artifacts/`` and produces a single
unified statistical report so every headline table in ``paper/hasagi/`` can cite
the same confidence intervals, effect sizes, and family-wise-corrected
significance flags. No experiment is re-run — this only re-reads the stored
per-seed measurements.

Coverage (each is one comparison family with its own Holm-Bonferroni
adjustment):

  - **H5-C**: per-zone savings gap ``HASAGI-online − GREEN-online`` across
    3 seeds × 16 zones. Holm across 16 zones.
  - **Scheduler head-to-head**: ``HASAGI EB − each baseline`` on energy, 10
    seeds. Holm across 5 baselines.
  - **HASAGI EB tight-budget**: ``HASAGI EB − PowerFlow`` on energy at each
    multiplier × 10 seeds. Holm across 7 multipliers.
  - **H2 end-to-end**: ``dvfs/preempt/combined − static`` on energy and on
    top-1, 3 seeds. Holm across 6 contrasts (3 conditions × 2 metrics).

Output: ``artifacts/headline_stats.json`` (machine-readable) and a printed
Rich-rendered markdown summary for paste-into-paper review.

Usage::

    python -m experiments.headline_stats \\
        --h5c artifacts/h5c_vs_green_3seed.json \\
        --schedh2h artifacts/scheduler_head_to_head_asymmetric.json \\
        --tightbudget artifacts/hasagi_eb_tight_budget.json \\
        --h2endtoend-dir artifacts/h2_endtoend \\
        --out artifacts/headline_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from hasagi.stats import (
    cluster_means,
    clustered_bootstrap_ci,
    clustered_permutation_pvalue,
    cohens_d,
    effect_size_tag,
    holm_bonferroni,
    one_sample_standardized_effect,
    paired_bootstrap_ci,
    paired_permutation_pvalue,
)

# Single seed used for every bootstrap/permutation so the report is
# reproducible bit-for-bit.
RNG_SEED = 0
N_BOOT = 10_000
N_PERM = 10_000
ALPHA = 0.05


def _ci_str(mean: float, lo: float, hi: float, fmt: str = "{:+.4f}") -> str:
    return f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _analyse_h5c(path: Path) -> dict[str, Any]:
    """Per-zone paired-bootstrap CI on (HASAGI-online − GREEN-online) savings.

    The raw artifact stores per-seed savings *relative to constant-N=1* in
    ``per_zone_fair_pp_gap_per_seed``: that is already
    (h_on_save − g_on_save), so we feed it straight into the paired
    bootstrap.
    """
    data = json.loads(path.read_text())
    per_zone_gaps: dict[str, list[float]] = data["per_zone_fair_pp_gap_per_seed"]
    zones = list(per_zone_gaps.keys())

    rng = random.Random(RNG_SEED)
    rows: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for zone in zones:
        diffs = per_zone_gaps[zone]
        mean, lo, hi = paired_bootstrap_ci(diffs, n_boot=N_BOOT, rng=rng)
        # Paired permutation against the zero-effect null.
        p = paired_permutation_pvalue(diffs, [0.0] * len(diffs), n_perm=N_PERM, rng=rng)
        rows.append({
            "zone": zone,
            "n_seeds": len(diffs),
            "mean_pp_gap": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "p_value": p,
        })
        pvalues.append(p)

    holm = holm_bonferroni(pvalues, alpha=ALPHA)
    for row, (rej, _p, adj) in zip(rows, holm, strict=True):
        row["holm_rejected"] = rej
        row["holm_alpha"] = adj

    # Headline aggregate CLUSTERS by zone: the seeds inside a zone share one
    # synthetic diurnal phase and are not independent, so a flat pool over every
    # (zone, seed) gap would understate the CI. The cluster unit is the zone.
    gaps_by_zone = [per_zone_gaps[z] for z in zones]
    mean_a, lo_a, hi_a = clustered_bootstrap_ci(gaps_by_zone, n_boot=N_BOOT, rng=rng)
    p_clustered = clustered_permutation_pvalue(gaps_by_zone, n_perm=N_PERM, rng=rng)
    d_clustered = one_sample_standardized_effect(cluster_means(gaps_by_zone))

    # Naive flat pool kept for transparency only (NOT the headline).
    all_gaps = [g for diffs in gaps_by_zone for g in diffs]
    fmean, flo, fhi = paired_bootstrap_ci(all_gaps, n_boot=N_BOOT, rng=rng)

    return {
        "name": "H5-C HASAGI-online vs GREEN-online (pp gap, per zone)",
        "alpha": ALPHA,
        "inference_note": (
            "Headline is the zone-clustered estimate; per-zone p uses exact "
            "sign-flip enumeration (floor 0.25 at n=3). Flat pool shown for "
            "transparency only."
        ),
        "rows": rows,
        "pooled_clustered": {
            "n_clusters": len(zones),
            "mean_pp_gap": mean_a,
            "ci_lo": lo_a,
            "ci_hi": hi_a,
            "cluster_permutation_p": p_clustered,
            "one_sample_effect": d_clustered,
        },
        "naive_flat_pooled_NOT_headline": {
            "n": len(all_gaps),
            "mean_pp_gap": fmean,
            "ci_lo": flo,
            "ci_hi": fhi,
        },
    }


def _analyse_scheduler_head_to_head(path: Path) -> dict[str, Any]:
    """HASAGI EB vs each baseline on energy. Paired bootstrap on (HASAGI − baseline) per seed."""
    data = json.loads(path.read_text())
    by_alloc: dict[str, dict[int, float]] = defaultdict(dict)
    for r in data["results"]:
        by_alloc[r["allocator"]][r["seed"]] = r["total_energy_kwh"]
    if "HASAGI EB" not in by_alloc:
        raise SystemExit("expected 'HASAGI EB' allocator in scheduler head-to-head artifact")
    hasagi_by_seed = by_alloc["HASAGI EB"]
    seeds = sorted(hasagi_by_seed.keys())

    rng = random.Random(RNG_SEED)
    rows: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for name in sorted(by_alloc.keys()):
        if name == "HASAGI EB":
            continue
        diffs = [hasagi_by_seed[s] - by_alloc[name][s] for s in seeds]
        hasagi_e = [hasagi_by_seed[s] for s in seeds]
        base_e = [by_alloc[name][s] for s in seeds]
        mean, lo, hi = paired_bootstrap_ci(diffs, n_boot=N_BOOT, rng=rng)
        d = cohens_d(hasagi_e, base_e)
        p = paired_permutation_pvalue(hasagi_e, base_e, n_perm=N_PERM, rng=rng)
        rows.append({
            "baseline": name,
            "n_seeds": len(seeds),
            "delta_mean_kwh": mean,
            "delta_ci_lo": lo,
            "delta_ci_hi": hi,
            "cohens_d": d,
            "effect_size": effect_size_tag(d),
            "p_value": p,
        })
        pvalues.append(p)

    holm = holm_bonferroni(pvalues, alpha=ALPHA)
    for row, (rej, _p, adj) in zip(rows, holm, strict=True):
        row["holm_rejected"] = rej
        row["holm_alpha"] = adj

    return {
        "name": "Scheduler head-to-head: HASAGI EB vs each baseline on energy",
        "alpha": ALPHA,
        "rows": rows,
    }


def _analyse_tight_budget(path: Path) -> dict[str, Any]:
    """HASAGI EB vs PowerFlow at each multiplier; family-wise across the sweep."""
    data = json.loads(path.read_text())
    by_key: dict[tuple[float, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for t in data["trials"]:
        by_key[(t["budget_multiplier"], t["allocator"])][t["seed"]] = {
            "energy_kwh": t["energy_kwh"],
            "max_jct_s": t["max_jct_s"],
            "deadlines_met": t["deadlines_met"],
        }
    multipliers = sorted({m for (m, _a) in by_key.keys()})
    seeds = sorted(next(iter(by_key.values())).keys())

    rng = random.Random(RNG_SEED)
    rows: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for mult in multipliers:
        hasagi = by_key[(mult, "HASAGI EB")]
        pf = by_key[(mult, "PowerFlow")]
        e_diffs = [hasagi[s]["energy_kwh"] - pf[s]["energy_kwh"] for s in seeds]
        met_diffs = [hasagi[s]["deadlines_met"] - pf[s]["deadlines_met"] for s in seeds]
        mean_e, lo_e, hi_e = paired_bootstrap_ci(e_diffs, n_boot=N_BOOT, rng=rng)
        mean_m, lo_m, hi_m = paired_bootstrap_ci(met_diffs, n_boot=N_BOOT, rng=rng)
        d = cohens_d(
            [hasagi[s]["energy_kwh"] for s in seeds],
            [pf[s]["energy_kwh"] for s in seeds],
        )
        p = paired_permutation_pvalue(
            [hasagi[s]["energy_kwh"] for s in seeds],
            [pf[s]["energy_kwh"] for s in seeds],
            n_perm=N_PERM, rng=rng,
        )
        rows.append({
            "multiplier": mult,
            "n_seeds": len(seeds),
            "delta_energy_mean_kwh": mean_e,
            "delta_energy_ci_lo": lo_e,
            "delta_energy_ci_hi": hi_e,
            "delta_met_mean": mean_m,
            "delta_met_ci_lo": lo_m,
            "delta_met_ci_hi": hi_m,
            "cohens_d_energy": d,
            "effect_size": effect_size_tag(d),
            "p_value_energy": p,
        })
        pvalues.append(p)

    holm = holm_bonferroni(pvalues, alpha=ALPHA)
    for row, (rej, _p, adj) in zip(rows, holm, strict=True):
        row["holm_rejected"] = rej
        row["holm_alpha"] = adj

    return {
        "name": "Tight-budget sweep: HASAGI EB vs PowerFlow on energy per multiplier",
        "alpha": ALPHA,
        "rows": rows,
    }


def _analyse_h2_endtoend(dir_path: Path) -> dict[str, Any]:
    """3-seed contrast: each elasticity condition vs static, on energy and top-1."""
    by_cond: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for fp in sorted(dir_path.glob("seed*_*.json")):
        # seed{n}_{cond}.json with cond ∈ {static, dvfs, preempt, combined}.
        stem = fp.stem
        seed_part, cond = stem.split("_", 1)
        seed = int(seed_part.replace("seed", ""))
        payload = json.loads(fp.read_text())
        by_cond[cond][seed] = {
            "top1": payload["final_top1"],
            "energy_kwh": payload["total_energy_joules"] / 3_600_000.0
            if "total_energy_joules" in payload
            else payload.get("energy_kwh", float("nan")),
        }

    if not by_cond:
        return {"name": "H2 end-to-end", "rows": [], "note": "no artifacts found"}

    static = by_cond.get("static")
    if static is None:
        return {"name": "H2 end-to-end", "rows": [], "note": "no 'static' baseline file"}
    seeds = sorted(static.keys())
    other_conds = sorted(c for c in by_cond.keys() if c != "static")

    rng = random.Random(RNG_SEED)
    rows: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for cond in other_conds:
        for metric in ("top1", "energy_kwh"):
            diffs = [by_cond[cond][s][metric] - static[s][metric] for s in seeds]
            cond_v = [by_cond[cond][s][metric] for s in seeds]
            base_v = [static[s][metric] for s in seeds]
            mean, lo, hi = paired_bootstrap_ci(diffs, n_boot=N_BOOT, rng=rng)
            d = cohens_d(cond_v, base_v)
            p = paired_permutation_pvalue(cond_v, base_v, n_perm=N_PERM, rng=rng)
            rows.append({
                "condition": cond,
                "metric": metric,
                "n_seeds": len(seeds),
                "delta_mean": mean,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "cohens_d": d,
                "effect_size": effect_size_tag(d),
                "p_value": p,
            })
            pvalues.append(p)

    holm = holm_bonferroni(pvalues, alpha=ALPHA)
    for row, (rej, _p, adj) in zip(rows, holm, strict=True):
        row["holm_rejected"] = rej
        row["holm_alpha"] = adj

    return {
        "name": "H2 end-to-end: elasticity condition − static, per metric",
        "alpha": ALPHA,
        "rows": rows,
    }


def _render(report: dict[str, Any], console: Console) -> None:
    name = report["name"]
    rows = report.get("rows", [])
    if not rows:
        console.print(f"[yellow]Skipped[/]: {name} — {report.get('note', 'no data')}")
        return
    console.rule(f"[bold]{name}[/]")
    keys = [k for k in rows[0].keys() if k != "n_seeds"]
    table = Table(show_lines=False)
    for k in keys:
        table.add_column(k, justify="right" if k not in {"zone", "baseline", "condition", "metric", "effect_size"} else "left")
    for r in rows:
        values = []
        for k in keys:
            v = r[k]
            if isinstance(v, bool):
                values.append("[green]reject H0[/]" if v else "[dim]accept H0[/]")
            elif isinstance(v, float):
                if "p_value" in k or "alpha" in k:
                    values.append(f"{v:.4f}")
                elif "ci" in k or "delta" in k or "mean" in k:
                    values.append(f"{v:+.4f}")
                elif "cohens" in k:
                    values.append(f"{v:+.2f}")
                else:
                    values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        table.add_row(*values)
    console.print(table)
    if "pooled_clustered" in report:
        p = report["pooled_clustered"]
        flat = report.get("naive_flat_pooled_NOT_headline", {})
        console.print(
            f"[bold]Zone-clustered[/]: {p['n_clusters']} zones, "
            f"mean pp gap = {p['mean_pp_gap']:+.3f} "
            f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}], "
            f"exact cluster-perm p={p['cluster_permutation_p']:.2e}, "
            f"effect={p['one_sample_effect']:+.2f}"
            + (f"  [dim](naive flat pool: {flat['mean_pp_gap']:+.3f} "
               f"[{flat['ci_lo']:+.3f}, {flat['ci_hi']:+.3f}], n={flat['n']})[/]"
               if flat else "")
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5c", type=Path, default=Path("artifacts/h5c_vs_green_3seed.json"))
    parser.add_argument(
        "--schedh2h", type=Path,
        default=Path("artifacts/scheduler_head_to_head_asymmetric.json"),
    )
    parser.add_argument(
        "--tightbudget", type=Path,
        default=Path("artifacts/hasagi_eb_tight_budget.json"),
    )
    parser.add_argument(
        "--h2endtoend-dir", type=Path,
        default=Path("artifacts/h2_endtoend"),
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/headline_stats.json"))
    args = parser.parse_args()

    console = Console()
    console.print("[bold]Headline statistical pipeline[/]: "
                  "paired bootstrap CI + Cohen's d + Holm-Bonferroni, α = 0.05")

    payload: dict[str, Any] = {}
    if args.h5c.exists():
        payload["h5c"] = _analyse_h5c(args.h5c)
        _render(payload["h5c"], console)
    if args.schedh2h.exists():
        payload["scheduler_head_to_head"] = _analyse_scheduler_head_to_head(args.schedh2h)
        _render(payload["scheduler_head_to_head"], console)
    if args.tightbudget.exists():
        payload["tight_budget"] = _analyse_tight_budget(args.tightbudget)
        _render(payload["tight_budget"], console)
    if args.h2endtoend_dir.exists():
        payload["h2_endtoend"] = _analyse_h2_endtoend(args.h2endtoend_dir)
        _render(payload["h2_endtoend"], console)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    console.print(f"\n[dim]wrote {args.out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

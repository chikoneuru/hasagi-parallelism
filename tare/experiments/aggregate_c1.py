"""Aggregate the 96-cell global throttle sweep into per-zone CI table.

Loads ``artifacts/endtoend_extensions/c1/summary.json`` (96 cells = 16 zones
x 3 seeds x 2 policies). Pairs each carbon_throttle run with its same-(zone, seed)
static_max baseline and reports:

  - Per-zone paired-bootstrap CI on three deltas: carbon (%), energy (%), top-1 (pp).
  - Per-zone mean throttle-hour count and JCT penalty (sanity).
  - Global pooled CI across 16 zones x 3 seeds = 48 paired observations.
  - Holm-Bonferroni across the 16 per-zone carbon-delta hypotheses at alpha = 0.05.

Output: ``artifacts/c1_aggregate.json`` plus a Rich table to stdout.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from tare.stats import (
    cluster_means,
    clustered_bootstrap_ci,
    clustered_permutation_pvalue,
    effect_size_tag,
    holm_bonferroni,
    one_sample_standardized_effect,
    paired_bootstrap_ci,
    paired_permutation_pvalue,
)

N_BOOT = 10_000
N_PERM = 10_000
ALPHA = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="src", type=Path,
        default=Path("artifacts/endtoend_extensions/c1/summary.json"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts/c1_aggregate.json"),
    )
    args = parser.parse_args()

    console = Console()
    cells = json.loads(args.src.read_text())
    by_key: dict[tuple[str, int, str], dict] = {}
    for c in cells:
        key = (c["zone"], c["seed"], c["policy"])
        by_key[key] = c
    zones = sorted({k[0] for k in by_key})
    seeds = sorted({k[1] for k in by_key})
    console.print(
        f"[bold]C1 aggregate[/]: {len(cells)} cells loaded; "
        f"{len(zones)} zones x {len(seeds)} seeds x 2 policies."
    )

    rng = random.Random(0)
    per_zone_carbon_deltas: dict[str, list[float]] = defaultdict(list)
    per_zone_energy_deltas: dict[str, list[float]] = defaultdict(list)
    per_zone_top1_deltas: dict[str, list[float]] = defaultdict(list)
    per_zone_jct_throttle: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for zone in zones:
        for seed in seeds:
            base = by_key.get((zone, seed, "static_max"))
            thr = by_key.get((zone, seed, "carbon_throttle"))
            if base is None or thr is None:
                console.print(f"[yellow]missing pair: zone={zone} seed={seed}[/]")
                continue
            base_c = base["total_carbon_grams"]
            base_e = base["total_energy_joules"]
            if base_c > 0:
                per_zone_carbon_deltas[zone].append(
                    100.0 * (thr["total_carbon_grams"] - base_c) / base_c
                )
            if base_e > 0:
                per_zone_energy_deltas[zone].append(
                    100.0 * (thr["total_energy_joules"] - base_e) / base_e
                )
            per_zone_top1_deltas[zone].append(thr["final_top1"] - base["final_top1"])
            per_zone_jct_throttle[zone].append((
                thr["jct_penalty_pct"],
                thr["throttle_hours"],
            ))

    rows = []
    pvalues: list[float] = []
    for zone in zones:
        carbon = per_zone_carbon_deltas[zone]
        energy = per_zone_energy_deltas[zone]
        top1 = per_zone_top1_deltas[zone]
        jct = [j for j, _ in per_zone_jct_throttle[zone]]
        throt = [t for _, t in per_zone_jct_throttle[zone]]
        c_mean, c_lo, c_hi = paired_bootstrap_ci(carbon, n_boot=N_BOOT, rng=rng)
        e_mean, e_lo, e_hi = paired_bootstrap_ci(energy, n_boot=N_BOOT, rng=rng)
        t_mean, t_lo, t_hi = paired_bootstrap_ci(top1, n_boot=N_BOOT, rng=rng)
        p = paired_permutation_pvalue(carbon, [0.0] * len(carbon), n_perm=N_PERM, rng=rng)
        pvalues.append(p)
        rows.append({
            "zone": zone,
            "n_seeds": len(carbon),
            "carbon_mean_pct": c_mean,
            "carbon_ci": [c_lo, c_hi],
            "energy_mean_pct": e_mean,
            "energy_ci": [e_lo, e_hi],
            "top1_mean_pp": t_mean,
            "top1_ci": [t_lo, t_hi],
            "jct_mean_pct": statistics.mean(jct),
            "throttle_mean_hours": statistics.mean(throt),
            "p_value": p,
        })

    holm = holm_bonferroni(pvalues, alpha=ALPHA)
    for row, (rej, _p, adj) in zip(rows, holm, strict=True):
        row["holm_rejected"] = rej
        row["holm_alpha"] = adj

    table = Table(
        title=(
            "C1 per-zone carbon-throttle vs static_max "
            "(3 seeds; paired bootstrap CI; Holm-Bonferroni alpha=0.05 across 16 zones)"
        ),
    )
    table.add_column("zone")
    table.add_column("n", justify="right")
    table.add_column("Delta carbon % (95% CI)", justify="right")
    table.add_column("Delta energy %", justify="right")
    table.add_column("Delta top-1 pp", justify="right")
    table.add_column("JCT %", justify="right")
    table.add_column("throt h (avg)", justify="right")
    table.add_column("p", justify="right")
    table.add_column("Holm", justify="left")
    for r in rows:
        ci = r["carbon_ci"]
        table.add_row(
            r["zone"], str(r["n_seeds"]),
            f"{r['carbon_mean_pct']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",
            f"{r['energy_mean_pct']:+.2f}",
            f"{r['top1_mean_pp']:+.3f}",
            f"{r['jct_mean_pct']:.1f}",
            f"{r['throttle_mean_hours']:.1f}",
            f"{r['p_value']:.4f}",
            "reject H0" if r["holm_rejected"] else "accept H0",
        )
    console.print(table)

    # The zone is the unit of statistical replication: the 3 seeds inside a
    # zone share an identical synthetic diurnal phase (carbon_trace.py fixes
    # start_dt for every zone) and so are near-replicates, not independent
    # draws. Pooling all 48 (zone, seed) cells as i.i.d. understates the CI and
    # over-states significance. Headline inference therefore CLUSTERS by zone.
    carbon_by_zone = [per_zone_carbon_deltas[z] for z in zones]
    energy_by_zone = [per_zone_energy_deltas[z] for z in zones]
    top1_by_zone = [per_zone_top1_deltas[z] for z in zones]
    pooled_jct = [j for zs in per_zone_jct_throttle.values() for j, _ in zs]

    pc_m, pc_lo, pc_hi = clustered_bootstrap_ci(carbon_by_zone, n_boot=N_BOOT, rng=rng)
    pe_m, pe_lo, pe_hi = clustered_bootstrap_ci(energy_by_zone, n_boot=N_BOOT, rng=rng)
    pt_m, pt_lo, pt_hi = clustered_bootstrap_ci(top1_by_zone, n_boot=N_BOOT, rng=rng)
    p_clustered = clustered_permutation_pvalue(carbon_by_zone, n_perm=N_PERM, rng=rng)
    d_clustered = one_sample_standardized_effect(cluster_means(carbon_by_zone))

    # Naive flat-pooled numbers kept ONLY for transparency / comparison; they
    # are NOT the headline because the 48 cells are not independent.
    flat_carbon = [v for zs in carbon_by_zone for v in zs]
    fc_m, fc_lo, fc_hi = paired_bootstrap_ci(flat_carbon, n_boot=N_BOOT, rng=rng)

    summary = Table(title="Global, zone-clustered (16 zones = replication unit; 3 seeds/zone averaged)")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("Delta carbon mean (%)", f"{pc_m:+.3f} [{pc_lo:+.3f}, {pc_hi:+.3f}]")
    summary.add_row("Delta energy mean (%)", f"{pe_m:+.3f} [{pe_lo:+.3f}, {pe_hi:+.3f}]")
    summary.add_row("Delta top-1 mean (pp)", f"{pt_m:+.4f} [{pt_lo:+.4f}, {pt_hi:+.4f}]")
    summary.add_row("Mean JCT penalty (%)", f"{statistics.mean(pooled_jct):.3f}")
    summary.add_row("One-sample effect (mean/sd over zones)", f"{d_clustered:+.3f} ({effect_size_tag(d_clustered)})")
    summary.add_row("Cluster-permutation p (exact, 16 zones)", f"{p_clustered:.2e}")
    summary.add_row("Zones with carbon delta < 0", f"{sum(1 for r in rows if r['carbon_mean_pct'] < 0)}/{len(rows)}")
    summary.add_row(
        "Per-zone Holm-rejected",
        f"{sum(1 for r in rows if r['holm_rejected'])}/{len(rows)} "
        f"(n=3/zone: per-zone permutation floors at 0.25, so none are rejectable)",
    )
    summary.add_row("[dim]naive flat-pooled CI (n=48, NOT headline)[/]", f"{fc_m:+.3f} [{fc_lo:+.3f}, {fc_hi:+.3f}]")
    console.print(summary)

    out = {
        "args": {"src": str(args.src), "n_boot": N_BOOT, "n_perm": N_PERM, "alpha": ALPHA},
        "inference_note": (
            "Headline inference clusters by zone (16 clusters); the 3 seeds per "
            "zone are averaged because they share one synthetic diurnal phase and "
            "are not independent. Per-zone p-values use exact sign-flip enumeration "
            "(floor 0.25 at n=3) and are reported for completeness only."
        ),
        "rows": rows,
        "pooled_clustered": {
            "n_clusters": len(zones),
            "seeds_per_cluster": len(seeds),
            "carbon_mean_pct": pc_m,
            "carbon_ci": [pc_lo, pc_hi],
            "energy_mean_pct": pe_m,
            "energy_ci": [pe_lo, pe_hi],
            "top1_mean_pp": pt_m,
            "top1_ci": [pt_lo, pt_hi],
            "jct_mean_pct": statistics.mean(pooled_jct),
            "one_sample_effect_carbon": d_clustered,
            "cluster_permutation_p": p_clustered,
        },
        "naive_flat_pooled_carbon_NOT_headline": {
            "n": len(flat_carbon),
            "carbon_mean_pct": fc_m,
            "carbon_ci": [fc_lo, fc_hi],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    console.print(f"\n[dim]wrote {args.out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

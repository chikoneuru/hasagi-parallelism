"""Robustness of the energy-optimal power cap: error bars + cap-order invariance.

The workload-dependent-cap result (exp_workload_cap_compare) came from a single
DVFS sweep with one measurement per cap, swept in ascending cap order. Two
reviewer concerns followed: there were no run-to-run error bars, and the
ascending order measures higher caps on a hotter GPU, which could bias a flat
U-curve toward the lower (cooler) cap. This tool consumes repeated sweeps
(``exp_hardware_pareto --repeats N``, run in both ``--cap-order`` directions) and
reports:

  - per-cap energy-per-iter mean ± sd and coefficient of variation (the error
    bars), so a claimed optimum can be judged against the measured noise;
  - the energy-optimal cap and its within-2%-of-min plateau per (workload, order);
  - an ORDER-INVARIANCE verdict per workload: does the ascending sweep agree with
    the descending sweep on the optimum / plateau? If they agree, the optimum is
    not a thermal-ordering artifact.

Usage::

    python -m experiments.exp_cap_robustness \
        --profile artifacts/cap_robust_resnet_asc.json \
        --profile artifacts/cap_robust_resnet_desc.json \
        --profile artifacts/cap_robust_transformer_asc.json \
        --profile artifacts/cap_robust_transformer_desc.json \
        --out artifacts/cap_robustness.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _rows_sorted(prof: dict) -> list[dict]:
    """Rows joined with their repeat stats, sorted by observed cap."""
    stats_by_cap = {round(s["cap_w_requested"]): s for s in prof.get("repeat_stats", [])}
    out = []
    for r in prof["rows"]:
        st = stats_by_cap.get(round(r["cap_w_requested"]), {})
        out.append({
            "cap_w": r["cap_w_observed"],
            "energy_j": r["energy_per_iter_j"],
            "energy_sd": st.get("energy_per_iter_j_sd", 0.0),
            "cv": st.get("energy_per_iter_cv", 0.0),
            "n": st.get("n_repeats", 1),
            "temp_c": r.get("avg_temp_c", 0.0),
        })
    return sorted(out, key=lambda d: d["cap_w"])


def _opt_and_plateau(rows: list[dict], tol_frac: float = 0.02) -> tuple[float, float, float]:
    """Energy-optimal cap (argmin) and the within-tol plateau (lo, hi)."""
    e_min = min(r["energy_j"] for r in rows)
    opt = min(rows, key=lambda r: r["energy_j"])["cap_w"]
    flat = [r["cap_w"] for r in rows if r["energy_j"] <= e_min * (1.0 + tol_frac)]
    return opt, min(flat), max(flat)


def run(args: argparse.Namespace) -> int:
    console = Console()
    profs = [(_load(p), p) for p in args.profile]

    # Per-(workload, order) error-bar table.
    by_workload: dict[str, dict[str, dict]] = {}
    for prof, path in profs:
        wl = prof.get("workload", "?")
        order = prof.get("cap_order", "?")
        rows = _rows_sorted(prof)
        opt, plo, phi = _opt_and_plateau(rows)
        by_workload.setdefault(wl, {})[order] = {"rows": rows, "opt": opt, "plateau": [plo, phi], "path": path}

        t = Table(title=f"{wl} / {order} cap-order — energy-per-iter U-curve with run-to-run error bars (n={rows[0]['n']})")
        t.add_column("cap (W)", justify="right")
        t.add_column("E/iter (J) ± sd", justify="right")
        t.add_column("CV %", justify="right")
        t.add_column("temp C", justify="right")
        for r in rows:
            mark = " ◀ min" if abs(r["cap_w"] - opt) < 1e-6 else ""
            t.add_row(f"{r['cap_w']:.0f}", f"{r['energy_j']:.3f} ± {r['energy_sd']:.3f}{mark}",
                      f"{r['cv'] * 100:.1f}", f"{r['temp_c']:.0f}")
        console.print(t)

    # Order-invariance verdict per workload.
    out_summary: dict[str, dict] = {}
    console.print("\n[bold]Cap-order invariance (ascending vs descending)[/]:")
    for wl, orders in by_workload.items():
        if "ascending" not in orders or "descending" not in orders:
            console.print(f"  [yellow]{wl}: need both ascending and descending profiles to check invariance[/]")
            continue
        a, d = orders["ascending"], orders["descending"]
        # Plateaus overlap?
        overlap = not (a["plateau"][1] < d["plateau"][0] or d["plateau"][1] < a["plateau"][0])
        opt_match = abs(a["opt"] - d["opt"]) < 1e-6
        invariant = overlap  # the honest bar: plateaus overlap (point optima may differ within the flat region)
        verdict = "INVARIANT" if invariant else "ORDER-DEPENDENT"
        colour = "green" if invariant else "red"
        console.print(
            f"  [{colour}]{wl}: {verdict}[/] — ascending opt {a['opt']:.0f} W (plateau "
            f"{a['plateau'][0]:.0f}-{a['plateau'][1]:.0f}); descending opt {d['opt']:.0f} W (plateau "
            f"{d['plateau'][0]:.0f}-{d['plateau'][1]:.0f}); plateaus {'overlap' if overlap else 'DISJOINT'}, "
            f"point optima {'agree' if opt_match else 'differ within the flat region'}."
        )
        out_summary[wl] = {
            "ascending_opt_w": a["opt"], "ascending_plateau_w": a["plateau"],
            "descending_opt_w": d["opt"], "descending_plateau_w": d["plateau"],
            "plateaus_overlap": overlap, "point_optima_match": opt_match,
            "order_invariant": invariant,
            "max_cv_pct": 100.0 * max(r["cv"] for o in (a, d) for r in o["rows"]),
        }

    # Cross-workload contrast (does the workload-dependence survive error bars + both orders?).
    if len(by_workload) >= 2 and all("ascending" in o and "descending" in o for o in by_workload.values()):
        opt_ranges = {wl: (min(o["ascending"]["plateau"][0], o["descending"]["plateau"][0]),
                           max(o["ascending"]["plateau"][1], o["descending"]["plateau"][1]))
                      for wl, o in by_workload.items()}
        console.print(
            "\n[bold]Workload-dependence across orders[/]: "
            + "; ".join(f"{wl} optimum-region {lo:.0f}-{hi:.0f} W" for wl, (lo, hi) in opt_ranges.items())
        )
        wls = list(opt_ranges)
        disjoint_pairs = [(wls[i], wls[j])
                          for i in range(len(wls)) for j in range(i + 1, len(wls))
                          if opt_ranges[wls[i]][1] < opt_ranges[wls[j]][0]
                          or opt_ranges[wls[j]][1] < opt_ranges[wls[i]][0]]
        if disjoint_pairs:
            console.print(
                "  [green]Workload-dependent optimum SURVIVES[/] error bars + both cap-orders: "
                + ", ".join(f"{x} vs {y} optimum-regions are disjoint" for x, y in disjoint_pairs)
            )
        else:
            console.print("  [yellow]optimum-regions overlap across workloads — workload-dependence not clearly separable here[/]")
        out_summary["_workload_optimum_regions"] = {wl: list(r) for wl, r in opt_ranges.items()}
        out_summary["_disjoint_workload_pairs"] = disjoint_pairs

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"profiles": [p for _, p in profs], "summary": out_summary}, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", action="append", required=True,
                   help="Repeated; a repeated-sweep JSON from exp_hardware_pareto (--repeats N, --cap-order ...).")
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

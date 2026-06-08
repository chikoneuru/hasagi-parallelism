"""Tests for the cap-robustness analysis: row joining + optimum/plateau detection."""
from __future__ import annotations

from experiments.exp_cap_robustness import _opt_and_plateau, _rows_sorted


def _prof() -> dict:
    """A profile dict mimicking exp_hardware_pareto's repeated-sweep JSON, with a
    flat U-curve bottom (200 W min, 250 W within 2%)."""
    return {
        "rows": [
            {"cap_w_requested": 300.0, "cap_w_observed": 300.0, "energy_per_iter_j": 5.0, "avg_temp_c": 70.0},
            {"cap_w_requested": 150.0, "cap_w_observed": 150.0, "energy_per_iter_j": 8.0, "avg_temp_c": 55.0},
            {"cap_w_requested": 200.0, "cap_w_observed": 200.0, "energy_per_iter_j": 4.00, "avg_temp_c": 60.0},
            {"cap_w_requested": 250.0, "cap_w_observed": 250.0, "energy_per_iter_j": 4.05, "avg_temp_c": 65.0},
        ],
        "repeat_stats": [
            {"cap_w_requested": 300.0, "energy_per_iter_j_sd": 0.05, "energy_per_iter_cv": 0.010, "n_repeats": 3},
            {"cap_w_requested": 150.0, "energy_per_iter_j_sd": 0.10, "energy_per_iter_cv": 0.012, "n_repeats": 3},
            {"cap_w_requested": 200.0, "energy_per_iter_j_sd": 0.04, "energy_per_iter_cv": 0.010, "n_repeats": 3},
            {"cap_w_requested": 250.0, "energy_per_iter_j_sd": 0.16, "energy_per_iter_cv": 0.040, "n_repeats": 3},
        ],
    }


def test_rows_sorted_joins_stats_and_orders_by_cap() -> None:
    rows = _rows_sorted(_prof())
    assert [r["cap_w"] for r in rows] == [150.0, 200.0, 250.0, 300.0]
    # the 250 W row carries its (larger) sd and CV from repeat_stats
    r250 = next(r for r in rows if r["cap_w"] == 250.0)
    assert r250["energy_sd"] == 0.16
    assert r250["cv"] == 0.040
    assert r250["n"] == 3


def test_opt_and_plateau_flags_flat_bottom() -> None:
    rows = _rows_sorted(_prof())
    opt, lo, hi = _opt_and_plateau(rows, tol_frac=0.02)
    assert opt == 200.0                 # argmin (4.00 J)
    assert (lo, hi) == (200.0, 250.0)   # 250 W (4.05) is within 2% of the 4.00 min


def test_opt_and_plateau_sharp_min_is_single_point() -> None:
    rows = _rows_sorted(_prof())
    opt, lo, hi = _opt_and_plateau(rows, tol_frac=0.001)  # tight tol → no plateau
    assert opt == 200.0
    assert (lo, hi) == (200.0, 200.0)

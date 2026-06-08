"""Tests for the EcoLife-collapse test (pure analysis on measured anchors, no GPU)."""
from __future__ import annotations

from experiments.exp_resume_collapse import collapse_point, collapse_sweep, resume_cost

_KW = dict(intensity_dirty=900.0, intensity_clean=200.0, active_power_w=132.0,
           idle_power_w=0.0, cold_start_s=4.7, resume_power_w=100.0,
           write_bw_gbps=3.9, reload_bw_gbps=7.7, warmup_s=0.761)


def test_resume_cost_stateless_is_cold_start_only() -> None:
    t, e = resume_cost(50.0, stateless=True, cold_start_s=4.7, resume_power_w=100.0,
                       write_bw_gbps=3.9, reload_bw_gbps=7.7, warmup_s=0.761)
    assert t == 4.7                                  # state size irrelevant when stateless
    assert e == 100.0 * 4.7 / 3.6e6


def test_training_resume_cost_grows_with_state() -> None:
    t_small, e_small = resume_cost(1.0, stateless=False, cold_start_s=4.7, resume_power_w=100.0,
                                   write_bw_gbps=3.9, reload_bw_gbps=7.7, warmup_s=0.761)
    t_big, e_big = resume_cost(100.0, stateless=False, cold_start_s=4.7, resume_power_w=100.0,
                               write_bw_gbps=3.9, reload_bw_gbps=7.7, warmup_s=0.761)
    assert t_big > t_small > 4.7                     # checkpoint+reload add to the cold start
    assert e_big > e_small                            # and cost more energy


def test_collapse_window_nonnegative_and_training_window_larger() -> None:
    r = collapse_point(20.0, **_KW)
    assert r["t_star_training_s"] >= r["t_star_stateless_s"]   # training needs a longer window to pay
    assert r["misdecision_window_s"] >= 0.0
    assert r["resume_energy_ratio_training_over_stateless"] > 1.0


def test_misdecision_window_grows_with_model_state() -> None:
    rows = collapse_sweep([0.045, 1.0, 5.0, 20.0, 80.0, 160.0], **_KW)
    windows = [r["misdecision_window_s"] for r in rows]
    assert all(windows[i + 1] >= windows[i] for i in range(len(windows) - 1))  # monotone in state
    assert windows[0] < 1.0          # resnet18-scale: stateless model is ~fine
    assert windows[-1] > 50.0        # large-model state: a large mis-decision window opens

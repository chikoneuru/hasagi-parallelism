"""Tests for the GREEN temporal-shift baseline port (experiments/baselines/green.py)."""
from __future__ import annotations

import pytest

from experiments.baselines.green import (
    green_offline_optimal_mask,
    green_online_percentile_mask,
    hise_threshold_mask,
    pause_fraction,
)

# --- green_offline_optimal_mask ---


def test_offline_optimal_zero_pause_keeps_all_active() -> None:
    mask = green_offline_optimal_mask([100, 200, 300], pause_fraction=0.0)
    assert mask == (1, 1, 1)


def test_offline_optimal_full_pause() -> None:
    mask = green_offline_optimal_mask([100, 200, 300], pause_fraction=1.0)
    assert mask == (0, 0, 0)


def test_offline_optimal_pauses_highest_intensity_ticks() -> None:
    """With pause_fraction=1/3, the single highest-intensity tick must be paused."""
    mask = green_offline_optimal_mask([100, 200, 300, 50], pause_fraction=0.25)
    # 4 ticks, pause 1 (the index-2 tick with intensity 300).
    assert mask == (1, 1, 0, 1)


def test_offline_optimal_respects_pause_budget_exactly() -> None:
    """The active count is exactly (1 - pf) × N (rounded)."""
    intensities = list(range(100))
    mask = green_offline_optimal_mask(intensities, pause_fraction=0.4)
    assert pause_fraction(mask) == pytest.approx(0.4, abs=1e-9)


def test_offline_optimal_empty_intensities() -> None:
    assert green_offline_optimal_mask([], pause_fraction=0.5) == ()


def test_offline_optimal_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="pause_fraction"):
        green_offline_optimal_mask([100, 200], pause_fraction=1.5)


# --- green_online_percentile_mask ---


def test_online_percentile_bootstrap_default_active() -> None:
    """First tick has no window to estimate from; default to active."""
    mask = green_online_percentile_mask([200], pause_fraction=0.5, window_size=24)
    assert mask == (1,)


def test_online_percentile_zero_pause_always_active() -> None:
    mask = green_online_percentile_mask([100, 200, 300, 400], pause_fraction=0.0)
    assert mask == (1, 1, 1, 1)


def test_online_percentile_high_pause_fraction_pauses_above_median() -> None:
    """At pause_fraction=0.5 the threshold is the rolling median; values above pause."""
    # Constant-rising trace so each new value is the highest in its window.
    mask = green_online_percentile_mask(
        [100, 200, 300, 400, 500], pause_fraction=0.5, window_size=10,
    )
    # tick 0: bootstrap active (1)
    # tick 1+: each new value > prior median in window → pause (0)
    assert mask[0] == 1
    assert all(m == 0 for m in mask[1:])


def test_online_percentile_window_size_limits_history() -> None:
    """Past beyond ``window_size`` ticks should not affect the threshold."""
    # 6 ticks; window=3 means tick 5 only sees ticks 3,4,5.
    intensities = [1000, 1000, 1000, 100, 200, 300]
    mask = green_online_percentile_mask(intensities, pause_fraction=0.5, window_size=3)
    # Tick 5 (=300) compared against window {100, 200, 300}; threshold is median 200,
    # so 300 > 200 → pause.
    assert mask[-1] == 0


def test_online_percentile_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window_size"):
        green_online_percentile_mask([100, 200], pause_fraction=0.5, window_size=0)


def test_online_percentile_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="pause_fraction"):
        green_online_percentile_mask([100, 200], pause_fraction=-0.1)


def test_online_percentile_empty_intensities() -> None:
    assert green_online_percentile_mask([], pause_fraction=0.5) == ()


# --- hise_threshold_mask ---


def test_hise_threshold_pauses_above_threshold() -> None:
    intensities = [100, 200, 300, 400, 500]
    # median=300; threshold=300*1.10=330; pause when >330 → indices 3,4.
    mask = hise_threshold_mask(intensities, threshold_multiplier=1.10)
    assert mask == (1, 1, 1, 0, 0)


def test_hise_threshold_empty_intensities() -> None:
    assert hise_threshold_mask([]) == ()


def test_hise_threshold_uniform_intensities_never_pause() -> None:
    """If every value equals the median, nothing exceeds median × 1.10 → no pauses."""
    mask = hise_threshold_mask([100] * 10, threshold_multiplier=1.10)
    assert mask == (1,) * 10


# --- pause_fraction ---


def test_pause_fraction_basic() -> None:
    assert pause_fraction([1, 1, 0, 0]) == 0.5
    assert pause_fraction([1, 1, 1]) == 0.0
    assert pause_fraction([0, 0, 0]) == 1.0
    assert pause_fraction(()) == 0.0

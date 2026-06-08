"""Tests for the GREEN temporal-shift baseline port (experiments/baselines/green.py)."""
from __future__ import annotations

import pytest

from experiments.baselines.green import (
    green_offline_optimal_mask,
    green_online_percentile_mask,
    tare_threshold_mask,
    tare_threshold_online_mask,
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


# --- tare_threshold_mask ---


def test_tare_threshold_pauses_above_threshold() -> None:
    intensities = [100, 200, 300, 400, 500]
    # median=300; threshold=300*1.10=330; pause when >330 → indices 3,4.
    mask = tare_threshold_mask(intensities, threshold_multiplier=1.10)
    assert mask == (1, 1, 1, 0, 0)


def test_tare_threshold_empty_intensities() -> None:
    assert tare_threshold_mask([]) == ()


def test_tare_threshold_uniform_intensities_never_pause() -> None:
    """If every value equals the median, nothing exceeds median × 1.10 → no pauses."""
    mask = tare_threshold_mask([100] * 10, threshold_multiplier=1.10)
    assert mask == (1,) * 10


# --- tare_threshold_online_mask ---


def test_tare_online_bootstrap_default_active() -> None:
    """First tick has no past window; default to active."""
    mask = tare_threshold_online_mask([200], threshold_multiplier=1.10, window_size=24)
    assert mask == (1,)


def test_tare_online_empty_intensities() -> None:
    assert tare_threshold_online_mask([]) == ()


def test_tare_online_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window_size"):
        tare_threshold_online_mask([100, 200], threshold_multiplier=1.10, window_size=0)


def test_tare_online_constant_trace_never_pauses() -> None:
    """A flat trace yields median == current; current ≤ median × 1.10 → never pause."""
    mask = tare_threshold_online_mask([100] * 10, threshold_multiplier=1.10, window_size=5)
    assert mask == (1,) * 10


def test_tare_online_pauses_only_above_rolling_threshold() -> None:
    """At window=3, threshold at tick 5 is median({100,200,300}) × 1.10 = 220; 300 > 220 → pause."""
    intensities = [1000, 1000, 1000, 100, 200, 300]
    mask = tare_threshold_online_mask(intensities, threshold_multiplier=1.10, window_size=3)
    # Tick 0: bootstrap (1).
    # Ticks 1-4: window has flat 1000 or {1000,1000,100,...} so current ≤ median*1.10.
    # Tick 5: window {100, 200, 300} → threshold 220; 300 > 220 → pause.
    assert mask[0] == 1
    assert mask[-1] == 0


def test_tare_online_window_size_limits_history() -> None:
    """A long-past spike should fall out of the rolling window."""
    # Spike at tick 0; flat thereafter. Window=3 means by tick 3 the spike is gone.
    intensities = [10_000, 100, 100, 100, 100]
    mask = tare_threshold_online_mask(intensities, threshold_multiplier=1.10, window_size=3)
    # Tick 0: bootstrap.
    # Tick 4: window {100, 100, 100} → threshold 110; 100 < 110 → active.
    assert mask[-1] == 1


def test_tare_online_differs_from_offline_when_trace_has_drift() -> None:
    """The whole point: rolling-window threshold ≠ full-trace median threshold."""
    # Linearly rising trace: late ticks are far above the full-trace median but
    # only slightly above their local rolling median.
    intensities = list(range(100, 200))
    offline = tare_threshold_mask(intensities, threshold_multiplier=1.10)
    online = tare_threshold_online_mask(intensities, threshold_multiplier=1.10, window_size=24)
    assert offline != online


# --- pause_fraction ---


def test_pause_fraction_basic() -> None:
    assert pause_fraction([1, 1, 0, 0]) == 0.5
    assert pause_fraction([1, 1, 1]) == 0.0
    assert pause_fraction([0, 0, 0]) == 1.0
    assert pause_fraction(()) == 0.0

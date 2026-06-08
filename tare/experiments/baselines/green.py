"""GREEN scheduler port (Xu et al., NSDI 2025) — temporal carbon-shift baseline.

GREEN's temporal shifter (§6, MLFQ Carbon Footprint Optimizer): pause jobs
during high-intensity windows and resume during low-intensity windows, subject
to maintaining the total allocated resources within a daily flexibility window.
This port captures the core scheduling decision as a per-tick (carbon-aware
on/off) function, with two flavours suitable for an apples-to-apples comparison
against HASAGI's threshold-pause policy:

  - **offline-optimal** — knows the full intensity trace; pauses the top-K
    highest-intensity ticks where K = pause_budget. Theoretical upper bound on
    temporal-shift savings (the "oracle" baseline GREEN's online algorithm
    approximates).
  - **online-percentile** — rolling-window estimator. At each tick, pauses if
    the current intensity sits in the top ``pause_fraction`` of the past
    ``window_seconds``. Realistic deployable policy that mirrors the way
    GREEN's online scheduler decides via percentile of recent carbon
    history.

Both flavours respect a *resource conservation* constraint: total active
ticks equals ``(1 - pause_fraction) × n_ticks``. HASAGI's simple
``pause if intensity > median × multiplier`` does NOT directly enforce this
constraint; instead its pause fraction emerges from the trace + threshold
combination. For the comparison, both policies are evaluated on the same
zone trace, and pause fractions are reported alongside savings so the
trade-off is visible.

Reference:
    Xu, Sun, Tian, Zhang, Chen, "GREEN: Carbon-efficient Resource
    Scheduling for Machine Learning Clusters," NSDI 2025.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def green_offline_optimal_mask(
    intensities: Sequence[float],
    pause_fraction: float,
) -> tuple[int, ...]:
    """Pick the K highest-intensity ticks to pause; return ``(1 = active, 0 = pause)`` mask.

    This is the oracle: requires the full trace ahead of time. Used as the
    theoretical upper bound for temporal-shift savings — any online policy
    targeting the same pause budget can do at best as well as this.

    Args:
        intensities: per-tick carbon intensity in gCO2/kWh.
        pause_fraction: fraction of ticks to pause, in [0, 1].

    Returns:
        Tuple of 0/1 with length ``len(intensities)``; sum = ``(1 - pf) * N``.
    """
    if not 0.0 <= pause_fraction <= 1.0:
        raise ValueError(f"pause_fraction must be in [0, 1], got {pause_fraction}")
    n = len(intensities)
    if n == 0:
        return ()
    n_pause = int(round(pause_fraction * n))
    # Rank ascending; the n_pause largest go to the "pause" set.
    idx_sorted_by_intensity = sorted(range(n), key=lambda i: intensities[i])
    pause_set = set(idx_sorted_by_intensity[n - n_pause:])
    return tuple(0 if i in pause_set else 1 for i in range(n))


def green_online_percentile_mask(
    intensities: Sequence[float],
    pause_fraction: float,
    window_size: int = 24,
) -> tuple[int, ...]:
    """Rolling-window percentile policy — pauses when current intensity sits
    in the top ``pause_fraction`` of the past ``window_size`` ticks.

    Args:
        intensities: per-tick carbon intensity.
        pause_fraction: targets this fraction of pauses; threshold percentile
            is ``1 - pause_fraction`` of the rolling window.
        window_size: how many past ticks the percentile estimator uses
            (24 default for hourly traces ⇒ rolling daily window).

    Returns:
        Tuple of 0/1, length ``len(intensities)``.
    """
    if not 0.0 <= pause_fraction <= 1.0:
        raise ValueError(f"pause_fraction must be in [0, 1], got {pause_fraction}")
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    n = len(intensities)
    if n == 0:
        return ()
    mask: list[int] = []
    for t in range(n):
        window_start = max(0, t - window_size + 1)
        window = intensities[window_start:t + 1]
        if len(window) <= 1:
            # Bootstrap: too few samples to estimate percentile; default to active.
            mask.append(1)
            continue
        # Sort and find the (1 - pause_fraction) percentile threshold.
        sorted_window = sorted(window)
        # Index of the threshold value (lower bound of the "pause" region).
        threshold_idx = max(0, int(math.ceil((1 - pause_fraction) * len(sorted_window))) - 1)
        threshold = sorted_window[threshold_idx]
        mask.append(0 if intensities[t] > threshold else 1)
    return tuple(mask)


def tare_threshold_mask(
    intensities: Sequence[float],
    threshold_multiplier: float = 1.10,
) -> tuple[int, ...]:
    """HASAGI's offline-median carbon-aware policy: pause if intensity > median × multiplier.

    Uses the median of the full ``intensities`` sequence as the reference
    threshold. This implies offline knowledge of the trace and is only fair
    against :func:`green_offline_optimal_mask`. For an apples-to-apples
    comparison against :func:`green_online_percentile_mask`, use
    :func:`tare_threshold_online_mask` instead.
    """
    if not intensities:
        return ()
    median = statistics.median(intensities)
    threshold = median * threshold_multiplier
    return tuple(0 if v > threshold else 1 for v in intensities)


def tare_threshold_online_mask(
    intensities: Sequence[float],
    threshold_multiplier: float = 1.10,
    window_size: int = 24,
) -> tuple[int, ...]:
    """Online rolling-median variant of :func:`tare_threshold_mask`.

    Mirrors :func:`green_online_percentile_mask` in window cadence and
    bootstrap behaviour, so HASAGI-threshold and GREEN-online are compared
    on identical information sets (rolling ``window_size`` ticks, no
    lookahead). At each tick ``t`` the decision threshold is
    ``median(intensities[t - window_size + 1 : t + 1]) ×
    threshold_multiplier`` and the tick pauses iff the current intensity
    exceeds it. With fewer than two samples in the window, defaults to
    active (bootstrap).
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    n = len(intensities)
    if n == 0:
        return ()
    mask: list[int] = []
    for t in range(n):
        window_start = max(0, t - window_size + 1)
        window = intensities[window_start:t + 1]
        if len(window) <= 1:
            mask.append(1)
            continue
        threshold = statistics.median(window) * threshold_multiplier
        mask.append(0 if intensities[t] > threshold else 1)
    return tuple(mask)


def pause_fraction(mask: Sequence[int]) -> float:
    """Fraction of ticks the mask pauses (= 0)."""
    if not mask:
        return 0.0
    return 1.0 - sum(mask) / len(mask)

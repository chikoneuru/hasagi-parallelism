"""Tests for hasagi.stats — bootstrap CIs, Cohen's d, Holm-Bonferroni."""
from __future__ import annotations

import math
import random
import statistics

import pytest

from hasagi.stats import (
    bootstrap_mean_ci,
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

# --- bootstrap_mean_ci ---


def test_bootstrap_mean_ci_empty() -> None:
    assert bootstrap_mean_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_mean_ci_singleton() -> None:
    mean, lo, hi = bootstrap_mean_ci([5.0])
    assert (mean, lo, hi) == (5.0, 5.0, 5.0)


def test_bootstrap_mean_ci_constant_sample() -> None:
    """All values equal → CI collapses to the value."""
    mean, lo, hi = bootstrap_mean_ci([2.0] * 10)
    assert mean == 2.0 and lo == 2.0 and hi == 2.0


def test_bootstrap_mean_ci_contains_mean() -> None:
    rng = random.Random(42)
    values = [rng.gauss(10.0, 1.0) for _ in range(50)]
    mean, lo, hi = bootstrap_mean_ci(values, n_boot=2000, rng=random.Random(1))
    assert lo <= mean <= hi


def test_bootstrap_mean_ci_tightens_with_sample_size() -> None:
    rng = random.Random(123)
    small = [rng.gauss(0.0, 1.0) for _ in range(20)]
    rng = random.Random(123)
    big = [rng.gauss(0.0, 1.0) for _ in range(500)]
    _, lo_s, hi_s = bootstrap_mean_ci(small, n_boot=2000, rng=random.Random(1))
    _, lo_b, hi_b = bootstrap_mean_ci(big, n_boot=2000, rng=random.Random(1))
    assert (hi_b - lo_b) < (hi_s - lo_s)


# --- paired_bootstrap_ci ---


def test_paired_bootstrap_ci_zero_diffs() -> None:
    mean, lo, hi = paired_bootstrap_ci([0.0] * 10)
    assert mean == 0.0 and lo == 0.0 and hi == 0.0


def test_paired_bootstrap_ci_excludes_zero_when_effect_real() -> None:
    """A consistently positive paired difference vector should have CI > 0."""
    diffs = [1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85, 1.0]
    mean, lo, hi = paired_bootstrap_ci(diffs, n_boot=4000, rng=random.Random(7))
    assert lo > 0
    assert mean == pytest.approx(1.0, abs=0.05)


# --- cohens_d ---


def test_cohens_d_short_samples() -> None:
    assert math.isnan(cohens_d([1.0], [2.0, 3.0]))
    assert math.isnan(cohens_d([1.0, 2.0], [3.0]))


def test_cohens_d_identical_groups_is_zero() -> None:
    assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_cohens_d_known_effect() -> None:
    """Two unit-variance samples shifted by 1.0 → d ≈ 1.0."""
    rng = random.Random(0)
    a = [rng.gauss(0.0, 1.0) for _ in range(500)]
    b = [rng.gauss(1.0, 1.0) for _ in range(500)]
    d = cohens_d(a, b)
    assert d == pytest.approx(-1.0, abs=0.2)


def test_cohens_d_zero_variance_inf() -> None:
    """Both groups constant but different means → infinite d with correct sign."""
    assert cohens_d([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == -math.inf
    assert cohens_d([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]) == math.inf


def test_effect_size_tag_buckets() -> None:
    assert effect_size_tag(0.0) == "negligible"
    assert effect_size_tag(0.3) == "small"
    assert effect_size_tag(0.6) == "medium"
    assert effect_size_tag(-1.0) == "large"
    assert effect_size_tag(2.0) == "very large"
    assert effect_size_tag(float("nan")) == "n/a"


# --- paired_permutation_pvalue ---


def test_paired_permutation_returns_one_on_zero_difference() -> None:
    a = [1.0, 2.0, 3.0]
    p = paired_permutation_pvalue(a, list(a))
    assert p == 1.0


def test_paired_permutation_mismatched_length_returns_one() -> None:
    assert paired_permutation_pvalue([1.0], [1.0, 2.0]) == 1.0


def test_paired_permutation_strong_effect_low_p() -> None:
    """20 consistently positive diffs → very small p."""
    a = [10.0 + i * 0.1 for i in range(20)]
    b = [0.0 + i * 0.1 for i in range(20)]
    p = paired_permutation_pvalue(a, b, n_perm=2000, rng=random.Random(3))
    assert p < 0.01


def test_paired_permutation_no_effect_high_p() -> None:
    """Symmetric ± diffs → p near 1."""
    diffs = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    a = [d for d in diffs]
    b = [0.0] * len(diffs)
    # Trick: a − b = diffs; mean = 0, so we expect p = 1 by the zero-mean guard.
    p = paired_permutation_pvalue(a, b, n_perm=2000, rng=random.Random(3))
    assert p == 1.0


def test_paired_permutation_n3_cannot_floor_below_exact_minimum() -> None:
    """Regression: at n=3 the exact two-sided permutation floor is 2/8=0.25.

    A consistently-signed triple has exactly two extreme assignments (identity
    and full-flip), so p MUST be 0.25 and can never be driven to ~1/(n_perm+1)
    by a floating-point tie mismatch. This pins the bug that previously floored
    several n=3 carbon zones to 0.0001 and produced false Holm rejections.
    """
    for triple in ([-9.27, -8.10, -10.4], [-16.22, -15.48, -17.32], [2.0, 2.0, 2.0]):
        a = list(triple)
        b = [0.0] * 3
        p = paired_permutation_pvalue(a, b, n_perm=10_000, rng=random.Random(0))
        assert p == pytest.approx(0.25), f"{triple} -> {p}"


def test_paired_permutation_exact_is_deterministic_for_small_n() -> None:
    """Below exact_max_n the result is exact and independent of rng / n_perm."""
    a = [3.0, 1.0, 2.0, 4.0]
    b = [0.0] * 4
    p1 = paired_permutation_pvalue(a, b, n_perm=500, rng=random.Random(1))
    p2 = paired_permutation_pvalue(a, b, n_perm=9999, rng=random.Random(99))
    assert p1 == p2  # exact enumeration ignores the Monte-Carlo knobs


# --- clustered inference ---


def test_cluster_means_drops_empty_and_averages() -> None:
    assert cluster_means([[2.0, 4.0], [10.0], []]) == [3.0, 10.0]


def test_clustered_ci_wider_than_flat_pool_under_within_cluster_dependence() -> None:
    """Clustering by the replication unit must not be narrower than a naive
    flat pool when seeds within a cluster are near-replicates."""
    by_cluster = [[-10.0, -10.1, -9.9], [-2.0, -2.1, -1.9], [-16.0, -16.1, -15.9],
                  [-3.0, -3.1, -2.9], [-11.0, -11.1, -10.9]]
    flat = [v for c in by_cluster for v in c]
    _, c_lo, c_hi = clustered_bootstrap_ci(by_cluster, n_boot=4000, rng=random.Random(0))
    _, f_lo, f_hi = paired_bootstrap_ci(flat, n_boot=4000, rng=random.Random(0))
    assert (c_hi - c_lo) >= (f_hi - f_lo)


def test_clustered_permutation_all_negative_clusters_is_exact_floor() -> None:
    """16 same-sign clusters → exact two-sided p = 2 / 2**16."""
    by_cluster = [[-1.0 - 0.1 * i] for i in range(16)]
    p = clustered_permutation_pvalue(by_cluster)
    assert p == pytest.approx(2 / (1 << 16))


def test_one_sample_effect_uses_real_dispersion_not_zero_control() -> None:
    """mean/sd over clusters; finite and not inflated by a zero-variance control."""
    means = [-10.0, -12.0, -8.0, -11.0, -9.0]
    d = one_sample_standardized_effect(means)
    assert d == pytest.approx(statistics.mean(means) / statistics.stdev(means))
    assert math.isnan(one_sample_standardized_effect([3.0]))


# --- holm_bonferroni ---


def test_holm_bonferroni_empty() -> None:
    assert holm_bonferroni([]) == []


def test_holm_bonferroni_preserves_input_order() -> None:
    """Output index k corresponds to input index k regardless of p ordering."""
    pvals = [0.04, 0.001, 0.5]
    out = holm_bonferroni(pvals, alpha=0.05)
    assert [p for _, p, _ in out] == pvals


def test_holm_bonferroni_thresholds_correct() -> None:
    pvals = [0.001, 0.02, 0.04]  # smallest to largest already
    out = holm_bonferroni(pvals, alpha=0.05)
    # Step-down thresholds: 0.05/3, 0.05/2, 0.05/1.
    expected_thresholds = [0.05 / 3, 0.05 / 2, 0.05 / 1]
    assert [t for _, _, t in out] == pytest.approx(expected_thresholds)


def test_holm_bonferroni_rejects_smallest_only_when_intermediate_fails() -> None:
    """If p2 > alpha/(m-1), it blocks p3 even though p3 < alpha/(m-2)."""
    pvals = [0.001, 0.04, 0.045]  # alpha=0.05, m=3 → thresholds 0.0167, 0.025, 0.05
    out = holm_bonferroni(pvals, alpha=0.05)
    rejected = [r for r, _, _ in out]
    # p1=0.001 < 0.0167 → reject
    # p2=0.04 > 0.025 → fail → blocks p3
    assert rejected == [True, False, False]


def test_holm_bonferroni_rejects_all_when_clearly_significant() -> None:
    """All p-values well below the most stringent threshold → reject all."""
    pvals = [0.001, 0.005, 0.002]
    out = holm_bonferroni(pvals, alpha=0.05)
    assert all(r for r, _, _ in out)

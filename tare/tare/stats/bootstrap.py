"""Bootstrap CIs, Cohen's d, paired-permutation p, and Holm-Bonferroni.

Used by ``experiments.headline_stats`` to produce a single statistical
report covering every headline table in the paper. Implemented in pure
Python (no NumPy/SciPy dependency) so the test suite stays fast.

Conventions:
    - All bootstrap routines are non-parametric and resample with
      replacement.
    - Paired bootstrap operates on pre-computed per-sample differences
      (so the pairing is locked at the call site).
    - Two-sided p-values throughout.
    - Holm-Bonferroni controls the family-wise error rate at the given
      ``alpha`` and returns per-hypothesis rejection flags + adjusted
      thresholds.
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` for the (1 − alpha) percentile bootstrap CI.

    Returns ``(mean, mean, mean)`` if ``len(values) < 2``.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = statistics.mean(values)
    if n < 2:
        return (mean, mean, mean)
    if rng is None:
        rng = random.Random(0)
    boot_means = [
        statistics.mean(rng.choices(values, k=n))
        for _ in range(n_boot)
    ]
    boot_means.sort()
    lo_idx = int(math.floor(alpha / 2 * n_boot))
    hi_idx = min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)
    return (mean, boot_means[lo_idx], boot_means[hi_idx])


def paired_bootstrap_ci(
    differences: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of a paired-difference vector.

    Pass ``differences = a − b`` where ``a`` and ``b`` are aligned
    measurements on the same experimental unit (zone, seed, job, …).
    """
    return bootstrap_mean_ci(differences, n_boot=n_boot, alpha=alpha, rng=rng)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d with pooled SD. ``nan`` if either group has < 2 samples.

    Returns ``+inf`` / ``-inf`` if the pooled SD is zero and the means
    differ, ``0.0`` if both groups are identical singletons.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        if ma == mb:
            return 0.0
        return math.copysign(float("inf"), ma - mb)
    return (ma - mb) / pooled


def effect_size_tag(d: float) -> str:
    """Cohen's convention for labelling effect-size magnitude."""
    if math.isnan(d):
        return "n/a"
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    if a < 1.5:
        return "large"
    return "very large"


def paired_permutation_pvalue(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10_000,
    rng: random.Random | None = None,
    exact_max_n: int = 20,
) -> float:
    """Two-sided paired-permutation p-value on the mean difference.

    Tests H0: ``mean(a − b) = 0`` by flipping the sign of each paired
    difference and measuring how often the permuted |mean| meets or
    exceeds the observed |mean|.

    For ``n <= exact_max_n`` the full sign-flip lattice (``2**n`` assignments)
    is enumerated EXACTLY, so the returned p is the true permutation p and is
    bounded below by ``2 / 2**n`` (e.g. 0.25 at n=3) — it cannot float below
    its combinatorial floor. For larger ``n`` a Monte-Carlo estimate with
    add-one smoothing is used. The observed statistic and every permuted
    statistic are computed with the SAME plain-summation arithmetic and an
    absolute/relative tolerance, so floating-point ties (notably the identity
    and full-flip assignments, which must always count as extreme) are never
    spuriously excluded — the bug that previously floored tied n=3 zones to
    ``1/(n_perm+1)``.

    Returns ``1.0`` if ``a`` and ``b`` are not the same length, are empty, or
    the observed mean difference is exactly zero.
    """
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    diffs = [ai - bi for ai, bi in zip(a, b, strict=True)]
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    if obs == 0.0:
        return 1.0
    tol = 1e-9 * max(1.0, obs)
    if n <= exact_max_n:
        total = 1 << n
        n_extreme = 0
        for mask in range(total):
            s = 0.0
            for i, d in enumerate(diffs):
                s += d if (mask >> i) & 1 else -d
            if abs(s / n) >= obs - tol:
                n_extreme += 1
        return n_extreme / total
    if rng is None:
        rng = random.Random(0)
    n_extreme = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= obs - tol:
            n_extreme += 1
    # Add-one smoothing so the Monte-Carlo p is never exactly 0.
    return (n_extreme + 1) / (n_perm + 1)


def cluster_means(values_by_cluster: Sequence[Sequence[float]]) -> list[float]:
    """Per-cluster means, dropping empty clusters.

    The cluster (e.g. a carbon zone) is the unit of statistical replication.
    Several seeds inside one zone are NOT independent draws of the underlying
    effect, so inference must operate on these per-cluster means rather than on
    the flattened per-(zone, seed) vector.
    """
    return [statistics.mean(v) for v in values_by_cluster if len(v) > 0]


def clustered_bootstrap_ci(
    values_by_cluster: Sequence[Sequence[float]],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Cluster (two-stage) percentile-bootstrap CI on the grand mean.

    The resampling unit is the cluster: each bootstrap replicate draws whole
    clusters with replacement and averages their per-cluster means. This keeps
    non-independence within a cluster from shrinking the interval, unlike a
    flat bootstrap over every observation. Pass ``values_by_cluster`` as a list
    of per-cluster paired-difference vectors.
    """
    means = cluster_means(values_by_cluster)
    return bootstrap_mean_ci(means, n_boot=n_boot, alpha=alpha, rng=rng)


def clustered_permutation_pvalue(
    values_by_cluster: Sequence[Sequence[float]],
    n_perm: int = 10_000,
    rng: random.Random | None = None,
    exact_max_n: int = 20,
) -> float:
    """Sign-flip permutation p on the grand mean at the CLUSTER level.

    Each cluster's mean flips sign as a unit, so the test does not assume
    within-cluster independence. With <= ``exact_max_n`` clusters the lattice
    is enumerated exactly.
    """
    means = cluster_means(values_by_cluster)
    return paired_permutation_pvalue(
        means, [0.0] * len(means), n_perm=n_perm, rng=rng, exact_max_n=exact_max_n,
    )


def one_sample_standardized_effect(values: Sequence[float]) -> float:
    """One-sample standardized mean ``mean / sd`` (``nan`` if < 2 values).

    Honest effect size for "grand mean differs from zero" over a set of
    per-cluster means. Unlike ``cohens_d(values, [0.0, ...])`` it does not
    divide by a pooled SD that has been halved by a zero-variance control
    vector, which previously inflated the reported d.
    """
    if len(values) < 2:
        return float("nan")
    sd = statistics.stdev(values)
    if sd == 0.0:
        m = statistics.mean(values)
        if m == 0.0:
            return 0.0
        return math.copysign(float("inf"), m)
    return statistics.mean(values) / sd


def holm_bonferroni(
    pvalues: Sequence[float],
    alpha: float = 0.05,
) -> list[tuple[bool, float, float]]:
    """Return ``[(rejected, p, adjusted_alpha), ...]`` preserving input order.

    Implements the step-down Holm-Bonferroni procedure controlling FWER
    at ``alpha`` across ``m = len(pvalues)`` hypotheses. The smallest
    p-value is compared to ``alpha / m``; the next to ``alpha / (m − 1)``;
    and so on. Once a hypothesis fails, all subsequent (larger) p-values
    are also marked not-rejected.
    """
    m = len(pvalues)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda kv: kv[1])
    rejected = [False] * m
    adjusted = [0.0] * m
    blocked = False
    for rank, (orig_idx, p) in enumerate(indexed):
        thresh = alpha / (m - rank)
        adjusted[orig_idx] = thresh
        if not blocked and p <= thresh:
            rejected[orig_idx] = True
        else:
            blocked = True
    return [(rejected[i], pvalues[i], adjusted[i]) for i in range(m)]

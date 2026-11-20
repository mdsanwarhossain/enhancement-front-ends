"""Statistics helpers shared by every results table.

Paired comparisons everywhere: the same test images are scored under every
operator and every scorer, so a paired test is both more powerful and more
honest than an unpaired one. Families of comparisons are corrected with
Holm-Bonferroni, which controls the family-wise error rate without assuming
independence.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, n=5000, alpha=0.05, seed=0):
    """Percentile CI of a mean over per-item values."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def paired_bootstrap_diff(a, b, n=5000, alpha=0.05, seed=0):
    """CI and two-sided p for mean(a) - mean(b) on paired per-item scores."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    return (float(d.mean()),
            float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))),
            float(min(max(p, 1.0 / n), 1.0)))


def paired_permutation(a, b, n=10000, seed=0):
    """Exchangeability test on the sign of paired differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (flips * d).mean(axis=1)
    p = (np.abs(null) >= abs(obs)).mean()
    return float(obs), float(max(p, 1.0 / n))


def wilcoxon_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]; robust to the accuracy scale."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((a[:, None] > b[None, :]).sum() for _ in [0])
    lt = sum((a[:, None] < b[None, :]).sum() for _ in [0])
    return float((gt - lt) / (len(a) * len(b)))

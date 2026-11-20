"""A battery of label-free statistics of an embedding cloud.

Each one is a candidate answer to: *without any labels, which operator leaves a
representation a classifier can still use?* They fall into three families -
neighbourhood agreement across operators, local neighbourhood structure, and
global spectral shape - and all are computed from embeddings that the operator
grid has already produced.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def knn_index(query, ref, k, block=2048, exclude_self=False):
    q, r = l2(query), l2(ref)
    idx = np.empty((len(q), k), np.int32)
    sim = np.empty((len(q), k), np.float32)
    for s in range(0, len(q), block):
        S = q[s:s + block] @ r.T
        if exclude_self:
            for i in range(len(S)):
                S[i, s + i] = -9.0
        part = np.argpartition(-S, k, axis=1)[:, :k]
        rows = np.arange(len(part))[:, None]
        order = np.argsort(-S[rows, part], axis=1)
        idx[s:s + block] = part[rows, order]
        sim[s:s + block] = S[rows, part[rows, order]]
    return idx, sim


# ----------------------------------------------- agreement across operators ---
def neighbourhood_agreement(nn_a, nn_b):
    """Mean Jaccard overlap of two kNN index sets, row by row."""
    k = nn_a.shape[1]
    out = np.empty(len(nn_a), np.float32)
    for i in range(len(nn_a)):
        inter = np.intersect1d(nn_a[i], nn_b[i], assume_unique=True).size
        out[i] = inter / (2 * k - inter)
    return out


# ------------------------------------------------------- local structure ---
def mutual_knn_rate(nn):
    """Fraction of kNN edges that are reciprocated.

    High in a cloud made of genuine tight groups, low when neighbourhoods are
    dictated by noise - and it needs no labels and no reference partition.
    """
    k = nn.shape[1]
    sets = [set(row) for row in nn]
    hits = sum(1 for i, row in enumerate(nn) for j in row if i in sets[j])
    return hits / (len(nn) * k)


def hubness(nn, n_ref):
    """Skewness of in-degree. A collapsed space has a few universal hubs."""
    counts = np.bincount(nn.ravel(), minlength=n_ref).astype(np.float64)
    m, s = counts.mean(), counts.std()
    return float(((counts - m) ** 3).mean() / (s ** 3 + EPS))


def local_separability(nn_sim):
    """Gap between the closest and the k-th neighbour similarity."""
    return float(np.mean(nn_sim[:, 0] - nn_sim[:, -1]))


# --------------------------------------------------------- global structure ---
def _subsample(X, n, seed=0):
    """A spectrum estimate converges long before the full cloud is needed."""
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), n, replace=False)]


def effective_rank(X, sample=3000, seed=0):
    """exp(entropy of the normalised eigenvalue spectrum)."""
    Z = _subsample(X, sample, seed)
    Z = Z - Z.mean(0)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s ** 2
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(X, sample=3000, seed=0):
    Z = _subsample(X, sample, seed)
    Z = Z - Z.mean(0)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    return float(s.sum() ** 2 / ((s ** 2).sum() + EPS))


def twonn_dimension(X, sample=3000, seed=0):
    """Facco et al. two-nearest-neighbour intrinsic dimension estimate."""
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    Z = l2(X)
    S = Z @ Z.T
    np.fill_diagonal(S, -9.0)
    top2 = np.sort(np.partition(S, -2, axis=1)[:, -2:], axis=1)[:, ::-1]
    d1, d2 = 1.0 - top2[:, 0], 1.0 - top2[:, 1]
    ok = d1 > EPS
    mu = np.sort(d2[ok] / d1[ok])
    n = len(mu)
    F = np.arange(1, n + 1) / n
    keep = F < 0.9
    x, y = np.log(mu[keep] + EPS), -np.log(1.0 - F[keep] + EPS)
    return float((x @ y) / (x @ x + EPS))


def uniformity(X, sample=2000, seed=0, t=2.0):
    """Wang and Isola uniformity: log mean exp(-t * squared distance)."""
    rng = np.random.default_rng(seed)
    Z = l2(X[rng.choice(len(X), min(sample, len(X)), replace=False)])
    sq = np.maximum(2.0 - 2.0 * (Z @ Z.T), 0.0)
    iu = np.triu_indices(len(Z), 1)
    return float(np.log(np.exp(-t * sq[iu]).mean() + EPS))


def cluster_silhouette(X, n_clusters=None, sample=4000, seed=0):
    """Silhouette of a k-means partition - label-free cluster crispness."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    Z = l2(X[rng.choice(len(X), min(sample, len(X)), replace=False)])
    k = n_clusters or max(2, min(20, len(Z) // 200))
    lab = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(Z)
    if len(np.unique(lab)) < 2:
        return float("nan")
    return float(silhouette_score(Z, lab, metric="cosine", sample_size=2000,
                                  random_state=seed))


GLOBAL_STATS = {
    "effective_rank": effective_rank,
    "participation_ratio": participation_ratio,
    "twonn_dim": twonn_dimension,
    "uniformity": uniformity,
    "silhouette": cluster_silhouette,
}

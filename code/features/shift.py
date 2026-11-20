"""ESP - Embedding Shift Penalty.

A deployed recogniser was fitted on the raw degraded images. Putting an
enhancement operator in front of it does not change the recogniser, so whatever
accuracy the operator costs, it costs by moving the representation away from
the distribution the recogniser was fitted on. That is covariate shift, and
covariate shift can be measured with no labels at all.

ESP is that measurement: the distance between the enhanced embeddings and the
reference embeddings the classifier was trained on. It is computed from
embeddings the operator grid already produces, needs no label, and is defined
for a whole deployment (which operator to install) and for a single image
(whether to trust this prediction).

Reference and query are both L2-normalised, so every score lives on a
comparable scale across encoders of different dimension.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


# --------------------------------------------------------- per deployment ---
def centroid_shift(query: np.ndarray, ref: np.ndarray) -> float:
    """Cosine distance between cloud centroids. The cheapest usable score."""
    a, b = l2(query).mean(0, keepdims=True), l2(ref).mean(0, keepdims=True)
    return float(1.0 - (l2(a) @ l2(b).T).item())


def mmd_rbf(query: np.ndarray, ref: np.ndarray, sample: int = 1500,
            seed: int = 0) -> float:
    """Maximum mean discrepancy, RBF kernel with the median heuristic."""
    rng = np.random.default_rng(seed)
    a = l2(query)[rng.choice(len(query), min(sample, len(query)), replace=False)]
    b = l2(ref)[rng.choice(len(ref), min(sample, len(ref)), replace=False)]
    z = np.vstack([a, b])
    sq = (z ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (z @ z.T), 0.0)
    gamma = 1.0 / (np.median(d2[d2 > 0]) + EPS)
    k = np.exp(-gamma * d2)
    n = len(a)
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2.0 * k[:n, n:].mean())


def frechet(query: np.ndarray, ref: np.ndarray, n_components: int = 64,
            seed: int = 0) -> float:
    """Frechet distance between Gaussian fits in a PCA basis fitted on `ref`."""
    from scipy.linalg import sqrtm
    from sklearn.decomposition import PCA
    r = l2(ref)
    pca = PCA(n_components=min(n_components, r.shape[1], len(r) - 1),
              random_state=seed).fit(r)
    a, b = pca.transform(l2(query)), pca.transform(r)
    ca = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    cb = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
    s = sqrtm(ca @ cb)
    if np.iscomplexobj(s):
        s = s.real
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(ca + cb - 2 * s))


DEPLOYMENT_SCORES = {
    "esp_mmd": mmd_rbf,
    "esp_frechet": frechet,
    "esp_centroid": centroid_shift,
}


# -------------------------------------------------------------- per image ---
def knn_distance(query: np.ndarray, ref: np.ndarray, k: int = 20,
                 block: int = 2048) -> np.ndarray:
    """Per-image cosine distance to its k nearest reference embeddings."""
    q, r = l2(query), l2(ref)
    out = np.empty(len(q), np.float32)
    for s in range(0, len(q), block):
        sim = q[s:s + block] @ r.T
        part = np.partition(sim, -k, axis=1)[:, -k:]
        out[s:s + block] = 1.0 - part.mean(axis=1)
    return out


def mahalanobis(query: np.ndarray, ref: np.ndarray, n_components: int = 64,
                seed: int = 0) -> np.ndarray:
    """Per-image Mahalanobis distance to the reference Gaussian, in PCA space."""
    from sklearn.decomposition import PCA
    r = l2(ref)
    pca = PCA(n_components=min(n_components, r.shape[1], len(r) - 1),
              random_state=seed).fit(r)
    b = pca.transform(r)
    mu = b.mean(0)
    prec = np.linalg.pinv(np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1]))
    d = pca.transform(l2(query)) - mu
    return np.einsum("ij,jk,ik->i", d, prec, d).astype(np.float32)


IMAGE_SCORES = {
    "esp_knn": knn_distance,
    "esp_mahalanobis": mahalanobis,
}

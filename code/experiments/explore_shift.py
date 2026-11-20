"""Can the deployment penalty be predicted without labels?

In the drop-in protocol the classifier was fitted on raw degraded images, so an
operator costs accuracy to the extent that it moves the representation away
from the distribution the classifier was fitted on. That is a distribution
shift question, and distribution shift is measurable without labels.

Every score here compares the enhanced test embeddings against the *identity*
training embeddings - the reference the deployed classifier actually knows -
rather than against the operator's own cloud.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from enhance.ops import OPS                             # noqa: E402
from features.probes_labelfree import l2, knn_index     # noqa: E402
import experiments.protocols as PR                      # noqa: E402

TAB = ROOT / "results" / "tables"
EPS = 1e-12


# ------------------------------------------------------------- shift scores ---
def frechet(a, b, eps=1e-6):
    """Frechet distance between Gaussian fits, on a PCA-reduced space."""
    from scipy.linalg import sqrtm
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False) + eps * np.eye(a.shape[1])
    cb = np.cov(b, rowvar=False) + eps * np.eye(b.shape[1])
    cov_sqrt = sqrtm(ca @ cb)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca + cb - 2 * cov_sqrt))


def mmd_rbf(a, b, gamma=None, sample=1500, seed=0):
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(sample, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(sample, len(b)), replace=False)]
    z = np.vstack([a, b])
    d2 = np.maximum(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1), 0) \
        if len(z) < 1200 else None
    if d2 is None:                      # memory-safe path
        sq = (z ** 2).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)
    gamma = gamma or 1.0 / (np.median(d2[d2 > 0]) + EPS)
    k = np.exp(-gamma * d2)
    n, m = len(a), len(b)
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean())


def mean_shift(a, b):
    """Cosine distance between the two cloud centroids."""
    ca, cb = l2(a.mean(0, keepdims=True)), l2(b.mean(0, keepdims=True))
    return float(1.0 - (ca @ cb.T).item())


def knn_distance_to_reference(query, ref, k=20):
    """How far each enhanced image sits from the classifier's training cloud."""
    _, sim = knn_index(query, ref, k)
    return float(1.0 - sim.mean())


def energy_ratio(a, b):
    """Ratio of mean embedding norms - a scale shift the probe sees directly."""
    return float(abs(np.log((np.linalg.norm(a, axis=1).mean() + EPS) /
                            (np.linalg.norm(b, axis=1).mean() + EPS))))


def main(dataset, backbone, seed):
    idx = pd.read_csv(ROOT / "data" / "index" / f"{dataset}.csv")
    emb = {}
    for op in OPS:
        f = ROOT / "data" / "emb" / dataset / backbone / f"{op}.npy"
        if f.exists():
            emb[op] = np.load(f)[:len(idx)]
    ops = list(emb)
    y = idx.label.to_numpy()

    rows = []
    for fold, (tr, te) in enumerate(PR.folds(idx, 3, seed)):
        # the deployed classifier: fitted on identity only
        base = PR.fit_probe(PR.l2(emb["identity"])[tr], y[tr], seed)
        ref = PR.l2(emb["identity"])[tr]
        ref_small = ref[np.random.default_rng(seed).choice(
            len(ref), min(2000, len(ref)), replace=False)]
        # PCA for the Gaussian-fit scores, learned on the reference only
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(64, ref.shape[1], len(ref) - 1),
                  random_state=seed).fit(ref)
        ref_p = pca.transform(ref)

        for op in ops:
            q = PR.l2(emb[op])[te]
            acc = float((PR.predict(base, q) == y[te]).mean())
            rows.append(dict(fold=fold, operator=op, p1_accuracy=acc,
                             frechet=frechet(pca.transform(q), ref_p),
                             mmd=mmd_rbf(q, ref_small, seed=seed),
                             centroid_shift=mean_shift(q, ref),
                             knn_distance=knn_distance_to_reference(q, ref),
                             energy_ratio=energy_ratio(q, ref)))
        print(f"  fold {fold} done", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "backbone", backbone)
    df.insert(0, "dataset", dataset)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"shift_scan__{dataset}__{backbone}.csv", index=False)

    print(f"\n{dataset}/{backbone}: can a label-free shift score predict the "
          f"drop-in penalty?")
    m = df.groupby("operator").mean(numeric_only=True)
    print(m.round(4).to_string())
    print()
    for col in ("frechet", "mmd", "centroid_shift", "knn_distance", "energy_ratio"):
        rho, p = spearmanr(m[col], m.p1_accuracy)
        print(f"  {col:16s} rho with P1 accuracy = {rho:+.3f}  (p={p:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="turbid_fish")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.dataset, a.backbone, a.seed)

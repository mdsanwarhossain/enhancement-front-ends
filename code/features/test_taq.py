"""Sanity tests for TAQ on synthetic embeddings with a known answer.

The score has to behave on cases where the right ranking is not in doubt,
otherwise nothing it says about real operators can be trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.taq import taq                                    # noqa: E402

RNG = np.random.default_rng(0)


D, K = 64, 5
CENTRES = RNG.normal(size=(K, D)) * 3.0        # one world, shared by all draws


def clustered(n, spread=0.35):
    """n points from the same k well-separated Gaussian clusters.

    Reference and query must share the clusters, or every operator looks
    equally out of distribution and the comparison means nothing.
    """
    y = RNG.integers(0, K, n)
    return CENTRES[y] + RNG.normal(size=(n, D)) * spread, y


def test_noise_operator_ranks_last():
    """An operator that replaces the embedding with noise must score lowest."""
    ref, _ = clustered(900)
    qry, _ = clustered(300)
    emb_r = {"good": ref, "mild": ref + RNG.normal(size=ref.shape) * 0.3,
             "noise": RNG.normal(size=ref.shape) * 3.0}
    emb_q = {"good": qry, "mild": qry + RNG.normal(size=qry.shape) * 0.3,
             "noise": RNG.normal(size=qry.shape) * 3.0}
    T = taq(emb_q, emb_r, k=15)
    mean = T["taq"].mean(axis=0)
    order = [T["ops"][i] for i in np.argsort(-mean)]
    print(f"  ranking: {order}  scores={dict(zip(T['ops'], mean.round(3)))}")
    assert order[-1] == "noise", order
    assert order[0] == "good", order


def test_degradation_is_monotone():
    """Adding noise in steps must lower the score monotonically."""
    ref, _ = clustered(900)
    qry, _ = clustered(300)
    levels = [0.0, 0.5, 1.0, 2.0]
    emb_r = {f"s{l}": ref + RNG.normal(size=ref.shape) * l for l in levels}
    emb_q = {f"s{l}": qry + RNG.normal(size=qry.shape) * l for l in levels}
    T = taq(emb_q, emb_r, k=15)
    mean = T["taq"].mean(axis=0)
    print(f"  scores by noise level: {dict(zip(T['ops'], mean.round(3)))}")
    assert np.all(np.diff(mean) < 0), mean


def test_agrees_with_separability():
    """Across operators, TAQ must order them the way a probe's accuracy does."""
    from sklearn.linear_model import LogisticRegression

    # a harder world: clusters close enough that a probe actually makes
    # mistakes, otherwise every operator scores 1.0 and the test is vacuous
    ref, y_ref = clustered(1200, spread=2.5)
    qry, y_qry = clustered(400, spread=2.5)
    emb_r, emb_q = {}, {}
    for l in (0.0, 4.0, 8.0, 14.0, 22.0, 32.0):
        emb_r[f"s{l}"] = ref + RNG.normal(size=ref.shape) * l
        emb_q[f"s{l}"] = qry + RNG.normal(size=qry.shape) * l
    T = taq(emb_q, emb_r, k=15)
    acc = [LogisticRegression(max_iter=1000).fit(emb_r[o], y_ref)
           .score(emb_q[o], y_qry) for o in T["ops"]]
    from scipy.stats import spearmanr
    rho = spearmanr(T["taq"].mean(axis=0), acc).statistic
    print(f"  accuracy={np.round(acc,3).tolist()}  rho={rho:.3f}")
    assert rho > 0.9, rho


def test_is_label_free_and_order_invariant():
    """Shuffling the reference rows must not change any score."""
    ref, _ = clustered(600)
    qry, _ = clustered(200)
    ops = {"a": 0.0, "b": 0.9}
    emb_r = {k: ref + RNG.normal(size=ref.shape) * v for k, v in ops.items()}
    emb_q = {k: qry + RNG.normal(size=qry.shape) * v for k, v in ops.items()}
    first = taq(emb_q, emb_r, k=10)["taq"]
    perm = RNG.permutation(len(ref))
    second = taq(emb_q, {k: v[perm] for k, v in emb_r.items()}, k=10)["taq"]
    assert np.allclose(first, second, atol=1e-5), np.abs(first - second).max()
    print("  invariant to reference ordering")


if __name__ == "__main__":
    for fn in (test_noise_operator_ranks_last, test_degradation_is_monotone,
               test_agrees_with_separability, test_is_label_free_and_order_invariant):
        print(f"{fn.__name__}:")
        fn()
        print("  PASS")

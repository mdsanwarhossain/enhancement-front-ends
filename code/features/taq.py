"""TAQ - Task-Aligned Quality.

A label-free, training-free score for the question an enhancement user actually
has: *will this operator help the recogniser on this image?* Pixel-domain
quality metrics answer a different question (does it look better to a person),
which is why they transfer so poorly.

TAQ reads the geometry of the frozen embedding space only, so it costs no
forward pass beyond the ones the operator grid already produces, and it never
sees a class label.

Two terms, both computed against an unlabelled reference pool R - in practice
the deployment pool itself:

  density contrast  D(x,e) = 1 - d_k(x,e) / d_rand(e)
      How much tighter the k nearest reference neighbours are than a random
      reference image. A representation that has collapsed detail into noise
      loses this contrast.

  consensus agreement  A(x,e) = |N_e(x) ∩ N*(x)| / k
      The reference pool is shared across operators, so neighbour *identities*
      are comparable. N*(x) is the set of references that most operators place
      near x. An operator that invents or destroys structure drifts away from
      that consensus even when its output looks clean.

TAQ(x,e) = z(D) + z(A), the z-scores taken over the operator axis so the score
compares operators for a fixed image. No weight is tuned on any label.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def _knn(query: np.ndarray, ref: np.ndarray, k: int, block: int = 2048):
    """Cosine kNN of `query` against `ref`; returns (indices, mean distance)."""
    q, r = _l2(query), _l2(ref)
    idx = np.empty((len(q), k), np.int32)
    dist = np.empty(len(q), np.float32)
    for s in range(0, len(q), block):
        sim = q[s:s + block] @ r.T
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        rows = np.arange(len(part))[:, None]
        order = np.argsort(-sim[rows, part], axis=1)
        nn = part[rows, order]
        idx[s:s + block] = nn
        dist[s:s + block] = 1.0 - sim[rows, nn].mean(axis=1)
    return idx, dist


def _random_distance(query: np.ndarray, ref: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ref), size=min(n, len(ref)), replace=False)
    return (1.0 - (_l2(query) @ _l2(ref[pick]).T)).mean(axis=1)


def taq(emb_query: dict[str, np.ndarray], emb_ref: dict[str, np.ndarray],
        k: int = 20, n_rand: int = 512, seed: int = 0):
    """Score every (image, operator) pair.

    Parameters
    ----------
    emb_query : operator -> (n_query, d) embeddings of the images to score.
    emb_ref   : operator -> (n_ref, d) embeddings of the unlabelled reference
                pool. Row i must be the same reference image for every operator.

    Returns
    -------
    dict with 'taq', 'density', 'consensus', each (n_query, n_ops), and the
    operator order used for the columns.
    """
    ops = list(emb_query)
    n = len(next(iter(emb_query.values())))

    nn, dens = {}, np.empty((n, len(ops)), np.float32)
    for j, op in enumerate(ops):
        idx, d_k = _knn(emb_query[op], emb_ref[op], k)
        d_rand = _random_distance(emb_query[op], emb_ref[op], n_rand, seed)
        nn[op] = idx
        dens[:, j] = 1.0 - d_k / (d_rand + EPS)

    # consensus neighbourhood: references named by a majority of operators
    cons = np.empty((n, len(ops)), np.float32)
    votes = np.zeros((n, len(emb_ref[ops[0]])), np.int16)
    rows = np.arange(n)[:, None]
    for op in ops:
        votes[rows, nn[op]] += 1
    star = votes >= max(2, (len(ops) + 1) // 2)
    for j, op in enumerate(ops):
        cons[:, j] = star[rows, nn[op]].sum(axis=1) / float(k)

    def _z(a):
        return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + EPS)

    return {"ops": ops, "density": dens, "consensus": cons,
            "taq": _z(dens) + _z(cons)}

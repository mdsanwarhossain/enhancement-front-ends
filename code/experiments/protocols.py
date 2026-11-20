"""Evaluation protocols for the enhancement grid.

Three questions, three protocols, because the enhancement literature answers
only one of them and the answers differ:

P1  drop-in       probe fitted on the raw degraded images, tested on enhanced
                  ones. This is what "add an enhancement front-end" means in
                  deployment: the recogniser is not retrained.
P2  retrained     probe fitted and tested on the same operator. This is what
                  enhancement papers report.
P3  agnostic      one probe fitted on every operator at once, so any operator
                  may be chosen per image at test time without a train/test
                  mismatch. This is the protocol routing needs.

Splits are group-aware throughout: a fish specimen, an endoscopy sequence or a
scene never appears on both sides of a split.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def folds(df, n_splits: int = 3, seed: int = 0):
    """Stratified, group-disjoint folds.

    The turbid-fish release ships a train/test split whose recording sessions
    appear on both sides; each session is 90 near-duplicate frames, so a split
    that ignores sessions inflates accuracy. We therefore ignore the released
    split and cut by group everywhere, keeping the class balance intact.
    """
    y, g = df.label.to_numpy(), df.group.to_numpy()
    if len(np.unique(g)) >= 2 * n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed)
        yield from splitter.split(np.zeros(len(df)), y, g)
    else:                                   # degenerate grouping: stratify only
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        yield from splitter.split(np.zeros(len(df)), y)


def released_split_leaks(df) -> bool | None:
    """True when the dataset's own split shares a group across sides."""
    if not set(df.split.unique()) >= {"train", "test"}:
        return None
    a = set(df.group[df.split == "train"])
    b = set(df.group[df.split.isin(["test", "val"])])
    return bool(a & b)


def fit_probe(X, y, seed=0):
    """Standardised multinomial logistic probe - the usual frozen-feature head."""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0,
                             random_state=seed).fit(sc.transform(X), y)
    return sc, clf


def predict(model, X):
    sc, clf = model
    return clf.predict(sc.transform(X))


def scores(y_true, y_pred):
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def boot_ci(y_true, y_pred, n=2000, seed=0, fn=None):
    fn = fn or (lambda a, b: (a == b).mean())
    rng = np.random.default_rng(seed)
    vals = [fn(y_true[i], y_pred[i])
            for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------- protocols ---
def run_p1_p2(emb: dict[str, np.ndarray], y, tr, te, seed=0):
    """Returns rows for protocols P1 (drop-in) and P2 (retrained)."""
    rows = []
    base = fit_probe(l2(emb["identity"])[tr], y[tr], seed)
    for op, E in emb.items():
        X = l2(E)
        p1 = predict(base, X[te])
        rows.append(dict(protocol="P1_dropin", operator=op,
                         **scores(y[te], p1)))
        own = fit_probe(X[tr], y[tr], seed)
        p2 = predict(own, X[te])
        rows.append(dict(protocol="P2_retrained", operator=op,
                         **scores(y[te], p2)))
    return rows


def fit_agnostic(emb, y, tr, seed=0, max_rows=20000):
    """One probe fitted over every operator at once.

    The stacked design matrix is `len(emb)` copies of the training split, which
    on the larger datasets runs to tens of thousands of rows of a couple of
    thousand dimensions. We cap it at `max_rows` sampled uniformly across the
    stack, so every operator stays represented and a linear readout of this
    size is not data-starved.
    """
    X = np.concatenate([l2(E)[tr] for E in emb.values()])
    Y = np.tile(y[tr], len(emb))
    if len(X) > max_rows:
        pick = np.random.default_rng(seed).choice(len(X), max_rows,
                                                  replace=False)
        X, Y = X[pick], Y[pick]
    return fit_probe(X, Y, seed)


def agnostic_predictions(model, emb, te):
    """(n_test, n_ops) predicted label per operator, plus correctness-ready preds."""
    ops = list(emb)
    return ops, np.stack([predict(model, l2(emb[op])[te]) for op in ops], axis=1)

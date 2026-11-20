"""What a group-blind split is worth, in accuracy that is not really there.

Run on the raw (identity) embeddings only: the question is about the splitting
protocol, not about enhancement. For each dataset and encoder we fit the same
probe under a stratified split that ignores groups and under one that keeps
groups disjoint, and report the difference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import experiments.protocols as P                      # noqa: E402
from features.backbones import BACKBONES               # noqa: E402

TAB = ROOT / "results" / "tables"


def accuracy(X, y, splits):
    out = []
    for tr, te in splits:
        model = P.fit_probe(X[tr], y[tr], 0)
        out.append(float((P.predict(model, X[te]) == y[te]).mean()))
    return out


def main(folds, seed):
    rows = []
    for dataset in ("turbid_fish", "exdark", "kvasir"):
        f = ROOT / "data" / "index" / f"{dataset}.csv"
        if not f.exists():
            continue
        idx = pd.read_csv(f)
        y, g = idx.label.to_numpy(), idx.group.to_numpy()
        if len(np.unique(g)) >= len(idx):
            print(f"{dataset}: one group per image, nothing to leak")
            continue
        for backbone in BACKBONES:
            e = ROOT / "data" / "emb" / dataset / backbone / "identity.npy"
            if not e.exists():
                continue
            X = P.l2(np.load(e)[:len(idx)])
            blind = accuracy(X, y, StratifiedKFold(
                folds, shuffle=True, random_state=seed).split(X, y))
            aware = accuracy(X, y, StratifiedGroupKFold(
                folds, shuffle=True, random_state=seed).split(X, y, g))
            rows.append(dict(dataset=dataset, backbone=backbone,
                             group_blind=float(np.mean(blind)),
                             group_blind_sd=float(np.std(blind)),
                             group_aware=float(np.mean(aware)),
                             group_aware_sd=float(np.std(aware)),
                             inflation=float(np.mean(blind) - np.mean(aware))))
            print(f"  {dataset}/{backbone}: blind={rows[-1]['group_blind']:.3f} "
                  f"aware={rows[-1]['group_aware']:.3f} "
                  f"inflation={rows[-1]['inflation']:.3f}", flush=True)
    if rows:
        TAB.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(TAB / "leakage.csv", index=False)
        print(f"wrote {TAB / 'leakage.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.folds, a.seed)

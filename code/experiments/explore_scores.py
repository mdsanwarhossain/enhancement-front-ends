"""Which label-free statistic actually tracks downstream accuracy?

Method development, run on one development dataset and encoder only. The
statistic chosen here is then frozen and reported on the held-out datasets and
encoders, so the final numbers are not the ones that picked the method.
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

from enhance.ops import OPS                                   # noqa: E402
import features.probes_labelfree as P                         # noqa: E402
import experiments.protocols as PR                            # noqa: E402

TAB = ROOT / "results" / "tables"


def load(dataset, backbone):
    idx = pd.read_csv(ROOT / "data" / "index" / f"{dataset}.csv")
    emb = {}
    for op in OPS:
        f = ROOT / "data" / "emb" / dataset / backbone / f"{op}.npy"
        if f.exists():
            emb[op] = np.load(f)[:len(idx)]
    return idx, emb


def operator_accuracy(emb, y, tr, te, seed=0):
    model = PR.fit_agnostic(emb, y, tr, seed)
    ops, pred = PR.agnostic_predictions(model, emb, te)
    return ops, (pred == y[te][:, None])


def main(dataset, backbone, k, seed):
    idx, emb = load(dataset, backbone)
    ops = list(emb)
    y = idx.label.to_numpy()
    tr, te = next(iter(PR.folds(idx, 3, seed)))
    print(f"{dataset}/{backbone}: {len(ops)} operators, "
          f"train={len(tr)} test={len(te)}", flush=True)

    ops_, correct = operator_accuracy(emb, y, tr, te, seed)
    acc = correct.mean(axis=0)
    print("  accuracy: " + ", ".join(f"{o}={a:.3f}" for o, a in zip(ops_, acc)),
          flush=True)

    # neighbour sets of the test images inside the (group-disjoint) train pool
    nn, sim = {}, {}
    for op in ops:
        nn[op], sim[op] = P.knn_index(emb[op][te], emb[op][tr], k)

    rows = []

    def record(name, values):
        rho, p = spearmanr(values, acc)
        rows.append(dict(statistic=name, rho=float(rho), p=float(p),
                         **{o: float(v) for o, v in zip(ops, values)}))
        print(f"  {name:26s} rho={rho:+.3f}  p={p:.3f}", flush=True)

    # --- agreement family -------------------------------------------------
    record("agree_with_identity",
           [P.neighbourhood_agreement(nn[op], nn["identity"]).mean() for op in ops])
    cons = []
    for op in ops:
        others = [P.neighbourhood_agreement(nn[op], nn[o]).mean()
                  for o in ops if o != op]
        cons.append(float(np.mean(others)))
    record("consensus_all_operators", cons)

    # --- local structure --------------------------------------------------
    record("mutual_knn_rate", [P.mutual_knn_rate(
        P.knn_index(emb[op][tr], emb[op][tr], k, exclude_self=True)[0])
        for op in ops])
    record("hubness", [P.hubness(nn[op], len(tr)) for op in ops])
    record("local_separability", [P.local_separability(sim[op]) for op in ops])
    record("mean_nn_similarity", [float(sim[op].mean()) for op in ops])

    # --- global structure -------------------------------------------------
    for name, fn in P.GLOBAL_STATS.items():
        record(name, [fn(emb[op][tr]) for op in ops])

    out = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
    out.insert(0, "backbone", backbone)
    out.insert(0, "dataset", dataset)
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB / f"labelfree_scan__{dataset}__{backbone}.csv", index=False)
    print(f"\nwrote {TAB / f'labelfree_scan__{dataset}__{backbone}.csv'}")
    print(out[["statistic", "rho", "p"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="turbid_fish")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.dataset, a.backbone, a.k, a.seed)

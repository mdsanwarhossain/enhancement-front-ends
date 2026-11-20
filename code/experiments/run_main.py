"""Main analysis for one (dataset, encoder).

Answers four questions and writes one tidy CSV each, over group-disjoint folds:

  1. What does an enhancement front end cost a deployed recogniser (P1), and
     how different is that from the retrained protocol the literature uses (P2)?
  2. Do pixel-domain quality metrics rank operators the way accuracy does?
  3. Does a label-free embedding-shift score (ESP) rank them better?
  4. Can an operator be chosen, and a single prediction be trusted, with no
     labels at all?

Nothing here re-reads an image: it consumes the embedding grid and the pixel
metric table produced earlier.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from enhance.ops import OPS, FAMILY                       # noqa: E402
from features import shift as SH                          # noqa: E402
from features import probes_labelfree as LF               # noqa: E402
from features.taq import taq                              # noqa: E402
from iqa.metrics import NO_REFERENCE, INPUT_REFERENCED    # noqa: E402
import experiments.protocols as P                         # noqa: E402

TAB = ROOT / "results" / "tables"
LOWER_IS_BETTER_IQA = {"ambe", "loe"}        # for these, small means good
IQA_METRICS = list(NO_REFERENCE) + list(INPUT_REFERENCED)
KNN_K = 20


def load(dataset, backbone, limit=None):
    idx = pd.read_csv(ROOT / "data" / "index" / f"{dataset}.csv")
    if limit:
        idx = idx.iloc[:limit]
    emb = {}
    for op in OPS:
        f = ROOT / "data" / "emb" / dataset / backbone / f"{op}.npy"
        if f.exists():
            e = np.load(f)
            if len(e) >= len(idx):
                emb[op] = e[:len(idx)]
    return idx, emb


def iqa_matrix(dataset, ops, n):
    f = ROOT / "data" / "iqa" / f"{dataset}.parquet"
    if not f.exists():
        return {}
    df = pd.read_parquet(f)
    df = df[df.row < n]
    return {m: df.pivot(index="row", columns="operator", values=m)
            .reindex(index=range(n), columns=ops).to_numpy(np.float64)
            for m in IQA_METRICS}


def deployment_scores(emb, ops, tr, te, seed):
    """Label-free scores of each operator, all against the identity reference.

    Sign convention: every score is returned so that **higher is predicted to
    be better**, which makes 'select by argmax' uniform across families.
    """
    ref = emb["identity"][tr]
    out: dict[str, list[float]] = {}

    def add(name, values):
        out[name] = list(values)

    # embedding shift against what the deployed classifier knows
    for name, fn in SH.DEPLOYMENT_SCORES.items():
        add(name, [-fn(emb[op][te], ref) for op in ops])
    add("esp_knn", [-SH.knn_distance(emb[op][te], ref, KNN_K).mean() for op in ops])
    add("esp_mahalanobis",
        [-float(np.median(SH.mahalanobis(emb[op][te], ref))) for op in ops])

    # geometry of the operator's own cloud - the obvious alternatives
    nn = {op: LF.knn_index(emb[op][te], emb[op][tr], KNN_K)[0] for op in ops}
    add("geo_consensus", [float(np.mean([
        LF.neighbourhood_agreement(nn[op], nn[o]).mean()
        for o in ops if o != op])) for op in ops])
    add("geo_hubness", [-LF.hubness(nn[op], len(tr)) for op in ops])
    # the mutual-kNN baseline needs an n-by-n similarity, so it is estimated
    # on a sample of the reference pool rather than all of it
    sub = tr if len(tr) <= 3000 else np.random.default_rng(seed).choice(
        tr, 3000, replace=False)
    add("geo_mutual_knn", [LF.mutual_knn_rate(
        LF.knn_index(emb[op][sub], emb[op][sub], KNN_K, exclude_self=True)[0])
        for op in ops])
    add("geo_effective_rank", [LF.effective_rank(emb[op][tr]) for op in ops])
    add("geo_twonn_dim", [-LF.twonn_dimension(emb[op][tr]) for op in ops])
    return out


def main(dataset, backbone, limit, n_splits, seed):
    idx, emb = load(dataset, backbone, limit)
    ops = list(emb)
    if "identity" not in ops:
        sys.exit(f"{dataset}/{backbone}: identity embeddings missing")
    y = idx.label.to_numpy()
    ident = ops.index("identity")
    M_all = iqa_matrix(dataset, ops, len(idx))
    print(f"{dataset}/{backbone}: {len(idx)} images, {len(ops)} operators, "
          f"{idx.label.nunique()} classes, {idx.group.nunique()} groups, "
          f"released-split-leaks={P.released_split_leaks(idx)}", flush=True)

    acc_rows, score_rows, sel_rows, img_rows, sev_rows = [], [], [], [], []

    for fold, (tr, te) in enumerate(P.folds(idx, n_splits, seed)):
        tag = dict(dataset=dataset, backbone=backbone, fold=fold)

        # ---- protocols ----------------------------------------------------
        for r in P.run_p1_p2(emb, y, tr, te, seed):
            acc_rows.append(dict(**tag, family=FAMILY[r["operator"]], **r))
        model = P.fit_agnostic(emb, y, tr, seed)
        _, pred3 = P.agnostic_predictions(model, emb, te)
        for j, op in enumerate(ops):
            acc_rows.append(dict(**tag, protocol="P3_agnostic", operator=op,
                                 family=FAMILY[op], **P.scores(y[te], pred3[:, j])))

        base = P.fit_probe(P.l2(emb["identity"])[tr], y[tr], seed)
        correct1 = np.stack([P.predict(base, P.l2(emb[op])[te]) == y[te]
                             for op in ops], axis=1)
        acc1 = correct1.mean(axis=0)
        acc3 = (pred3 == y[te][:, None]).mean(axis=0)

        # ---- label-free scores -------------------------------------------
        scores = deployment_scores(emb, ops, tr, te, seed)
        M = {m: v[te] for m, v in M_all.items()}
        for m, v in M.items():
            sgn = -1.0 if m in LOWER_IS_BETTER_IQA else 1.0
            scores[f"iqa_{m}"] = list(sgn * np.nanmean(v, axis=0))
        T = taq({o: emb[o][te] for o in ops}, {o: emb[o][tr] for o in ops},
                k=KNN_K, seed=seed)
        scores["geo_taq"] = list(T["taq"].mean(axis=0))

        for name, vals in scores.items():
            for j, op in enumerate(ops):
                score_rows.append(dict(**tag, scorer=name, operator=op,
                                       value=float(vals[j]),
                                       p1_accuracy=float(acc1[j]),
                                       p3_accuracy=float(acc3[j])))

        # ---- choosing an operator without labels --------------------------
        for name, vals in scores.items():
            v = np.asarray(vals, float)
            pick = int(np.nanargmax(np.where(np.isfinite(v), v, -np.inf)))
            sel_rows.append(dict(**tag, scorer=name, picked=ops[pick],
                                 p1_accuracy=float(acc1[pick]),
                                 p1_regret=float(acc1.max() - acc1[pick]),
                                 p3_accuracy=float(acc3[pick]),
                                 p3_regret=float(acc3.max() - acc3[pick])))
        for name, j in (("identity", ident), ("best_fixed_p1", int(acc1.argmax()))):
            sel_rows.append(dict(**tag, scorer=name, picked=ops[j],
                                 p1_accuracy=float(acc1[j]),
                                 p1_regret=float(acc1.max() - acc1[j]),
                                 p3_accuracy=float(acc3[j]),
                                 p3_regret=float(acc3.max() - acc3[j])))

        # ---- trusting a single prediction without labels ------------------
        ref = emb["identity"][tr]
        for name, fn in SH.IMAGE_SCORES.items():
            s = np.concatenate([-fn(emb[op][te], ref) for op in ops])
            c = correct1.T.ravel()
            try:
                auroc = float(roc_auc_score(c, s))
            except ValueError:
                auroc = float("nan")
            img_rows.append(dict(**tag, scorer=name, auroc_p1_correct=auroc,
                                 n=len(c)))
        for m, v in M.items():
            sgn = -1.0 if m in LOWER_IS_BETTER_IQA else 1.0
            s = sgn * v.T.ravel()
            c = correct1.T.ravel()
            ok = np.isfinite(s)              # heavy metrics are subsampled
            try:
                auroc = float(roc_auc_score(c[ok], s[ok]))
            except ValueError:
                auroc = float("nan")
            img_rows.append(dict(**tag, scorer=f"iqa_{m}",
                                 auroc_p1_correct=auroc, n=int(ok.sum())))

        # ---- the dataset's own degradation annotation ---------------------
        if (idx.severity.to_numpy() >= 0).any():
            sev = idx.severity.to_numpy()[te]
            for j, op in enumerate(ops):
                for s_ in np.unique(sev):
                    if s_ < 0:
                        continue
                    k = sev == s_
                    sev_rows.append(dict(**tag, operator=op, severity=int(s_),
                                         p1_accuracy=float(correct1[k, j].mean()),
                                         p3_accuracy=float((pred3[k, j] == y[te][k]).mean()),
                                         n=int(k.sum())))

        print(f"  fold {fold}: P1 identity={acc1[ident]:.3f} "
              f"worst={acc1.min():.3f} ({ops[int(acc1.argmin())]}) "
              f"best={acc1.max():.3f} ({ops[int(acc1.argmax())]}) | "
              f"ESP-picked={[r for r in sel_rows if r['fold']==fold and r['scorer']=='esp_mmd'][0]['picked']}",
              flush=True)

    key = f"{dataset}__{backbone}"
    pd.DataFrame(acc_rows).to_csv(TAB / f"operator_accuracy__{key}.csv", index=False)
    pd.DataFrame(score_rows).to_csv(TAB / f"operator_scores__{key}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(TAB / f"selection__{key}.csv", index=False)
    pd.DataFrame(img_rows).to_csv(TAB / f"perimage__{key}.csv", index=False)
    if sev_rows:
        pd.DataFrame(sev_rows).to_csv(TAB / f"severity__{key}.csv", index=False)

    s = pd.DataFrame(score_rows)
    rho = (s.groupby(["scorer", "fold"])
           .apply(lambda d: spearmanr(d.value, d.p1_accuracy).statistic,
                  include_groups=False)
           .groupby("scorer").mean().sort_values(ascending=False))
    print("\n  rank correlation with the drop-in penalty (higher is better):")
    print("   " + rho.round(3).to_string().replace("\n", "\n   "), flush=True)
    print(f"\n  wrote results/tables/*__{key}.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    main(a.dataset, a.backbone, a.limit, a.folds, a.seed)

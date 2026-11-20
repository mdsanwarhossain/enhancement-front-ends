"""The failure, shown rather than tabulated.

Finds test images that the deployed probe gets right on the raw frame and wrong
after enhancement, and lays them out beside the quality scores that say the
enhanced version is the better image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from enhance.ops import OPS                      # noqa: E402
from iqa.metrics import NO_REFERENCE             # noqa: E402
import experiments.protocols as P                # noqa: E402

FIG = ROOT / "results" / "figures"

mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "font.size": 7,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})

PRETTY = {"identity": "no enhancement", "hist_eq": "hist-eq", "clahe": "CLAHE",
          "gamma": "auto gamma", "ssr": "SSR", "msrcr": "MSRCR",
          "gray_world": "grey-world", "dcp": "DCP", "unsharp": "unsharp",
          "zero_dce": "Zero-DCE"}


def main(dataset, backbone, operators, n_rows, seed, metric, rank=0):
    idx = pd.read_csv(ROOT / "data" / "index" / f"{dataset}.csv")
    y = idx.label.to_numpy()
    emb = {op: np.load(ROOT / "data" / "emb" / dataset / backbone / f"{op}.npy")
           for op in operators}
    tr, te = next(iter(P.folds(idx, 3, seed)))

    base = P.fit_probe(P.l2(emb["identity"])[tr], y[tr], seed)
    pred = {op: P.predict(base, P.l2(emb[op])[te]) for op in operators}

    # The case the paper is about: the raw frame is classified correctly, and
    # an operator that the quality metric prefers gets it wrong. Requiring the
    # metric to rise keeps the figure honest - a failure under an operator that
    # also looked worse would prove nothing.
    scores = pd.read_parquet(ROOT / "data" / "iqa" / f"{dataset}.parquet")
    scores = (scores[scores.row.isin(te)]
              .pivot(index="row", columns="operator", values=metric)
              .reindex(index=te, columns=operators).to_numpy(float))
    better_looking = scores > scores[:, [0]]
    wrong = np.stack([pred[op] != y[te] for op in operators], axis=1)
    right = pred["identity"] == y[te]

    qualifying = right & (wrong[:, 1:] & better_looking[:, 1:]).any(axis=1)
    candidates = np.flatnonzero(qualifying)
    if not len(candidates):
        print("no qualifying examples")
        return
    # Prefer the unambiguous case - every operator shown scores higher than the
    # raw frame and every one of them breaks the classifier - so the figure
    # cannot be read as cherry-picking one bad operator out of several.
    unambiguous = np.flatnonzero(
        right & (wrong[:, 1:] & better_looking[:, 1:]).all(axis=1))
    pool = unambiguous if len(unambiguous) >= n_rows else candidates
    strength = (wrong[:, 1:] & better_looking[:, 1:]).sum(axis=1)[pool]
    # Deterministic: strongest cases first, ties in dataset order, and --rank
    # chooses among them. Every one of them qualifies on the same criterion;
    # the choice between them is presentational, so it is reproducible rather
    # than drawn at random.
    order = pool[np.argsort(-strength, kind="stable")]
    pick = order[rank:rank + n_rows]
    if not len(pick):
        print(f"rank {rank} is past the end of {len(order)} qualifying cases")
        return
    print(f"  {len(unambiguous)} of {len(te)} test images are broken by every "
          f"operator shown while all of them score higher on {metric}; "
          f"showing rank {rank} of {len(order)}")

    # tight: this figure has to earn its space inside a ten-page limit
    fig, axes = plt.subplots(len(pick), len(operators),
                             figsize=(1.95 * len(operators), 1.55 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(pick):
        img = cv2.imread(idx.path.iloc[te[i]], cv2.IMREAD_COLOR)
        for c, op in enumerate(operators):
            out = OPS[op](img)
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            ok = pred[op][i] == y[te][i]
            for s in ax.spines.values():
                s.set_edgecolor("#009E73" if ok else "#D55E00")
                s.set_linewidth(2.0)
            q = NO_REFERENCE[metric](out)
            ax.set_xlabel(f"{pred[op][i]}  ({'correct' if ok else 'wrong'})\n"
                          f"{metric} = {q:.2f}", fontsize=6.5,
                          color="#1b1b1b")
            if r == 0:
                ax.set_title(PRETTY.get(op, op), fontsize=8)
        axes[r, 0].set_ylabel(f"true: {y[te][i]}", fontsize=7)
    # the legend lives in the LaTeX caption, where it costs no vertical space
    fig.subplots_adjust(hspace=0.34, wspace=0.04)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_qualitative.{ext}")
    plt.close(fig)
    stat = pd.DataFrame([dict(
        dataset=dataset, backbone=backbone, metric=metric,
        operators=",".join(operators), n_test=len(te),
        n_right_raw=int(right.sum()),
        n_broken_by_better_looking=int(len(candidates)),
        fraction=float(len(candidates) / len(te)))])
    stat.to_csv(ROOT / "results" / "tables" / "qualitative.csv", index=False)
    print(f"wrote fig_qualitative.pdf/.png "
          f"({len(candidates)} of {len(te)} test images qualify: correct on the "
          f"raw frame, wrong under an operator the metric prefers)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="exdark")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--operators", default="identity,clahe,msrcr,ssr")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="uiqm")
    ap.add_argument("--rank", type=int, default=0,
                    help="which qualifying example to show")
    a = ap.parse_args()
    main(a.dataset, a.backbone, a.operators.split(","), a.rows, a.seed,
         a.metric, a.rank)

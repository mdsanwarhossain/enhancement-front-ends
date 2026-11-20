"""Is the penalty an artefact of the operators' default settings?

A reasonable objection to Section~\\ref{sec:results:penalty} is that we used
each operator out of the box. Here the most tunable operators are swept over a
range that brackets their usual settings, and the drop-in penalty is measured
for each. If the penalty were a tuning accident it would disappear somewhere in
the sweep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import cv2                                            # noqa: E402
from enhance import ops as O                          # noqa: E402
from features.backbones import Encoder                # noqa: E402
import experiments.protocols as P                     # noqa: E402

TAB = ROOT / "results" / "tables"

# (name, family, callable) - each family brackets its library default
VARIANTS = [
    ("clahe_clip1", "clahe", lambda im: O.clahe(im, clip=1.0, grid=8)),
    ("clahe_clip2", "clahe", lambda im: O.clahe(im, clip=2.0, grid=8)),
    ("clahe_clip4", "clahe", lambda im: O.clahe(im, clip=4.0, grid=8)),
    ("clahe_grid4", "clahe", lambda im: O.clahe(im, clip=2.0, grid=4)),
    ("clahe_grid16", "clahe", lambda im: O.clahe(im, clip=2.0, grid=16)),
    ("ssr_s30", "ssr", lambda im: O.ssr(im, sigma=30.0)),
    ("ssr_s80", "ssr", lambda im: O.ssr(im, sigma=80.0)),
    ("ssr_s200", "ssr", lambda im: O.ssr(im, sigma=200.0)),
    ("unsharp_a05", "unsharp", lambda im: O.unsharp(im, amount=0.5)),
    ("unsharp_a15", "unsharp", lambda im: O.unsharp(im, amount=1.5)),
    ("unsharp_a30", "unsharp", lambda im: O.unsharp(im, amount=3.0)),
    ("dcp_w07", "dcp", lambda im: O.dcp(im, omega=0.7)),
    ("dcp_w095", "dcp", lambda im: O.dcp(im, omega=0.95)),
]


def _init_worker(_):
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    cv2.setNumThreads(1)


class Variant(Dataset):
    def __init__(self, paths, fn, transform):
        self.paths, self.fn, self.transform = list(paths), fn, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((224, 224, 3), np.uint8)
        out = self.fn(img)
        return self.transform(Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))


def embed(enc, paths, fn, batch, workers):
    dl = DataLoader(Variant(paths, fn, enc.transform), batch_size=batch,
                    num_workers=workers, shuffle=False,
                    worker_init_fn=_init_worker)
    with torch.no_grad():
        return np.concatenate([enc.embed(xb) for xb in dl])


def main(dataset, backbone, batch, workers, threads, seed):
    idx = pd.read_csv(ROOT / "data" / "index" / f"{dataset}.csv")
    y = idx.label.to_numpy()
    tr, te = next(iter(P.folds(idx, 3, seed)))

    base = np.load(ROOT / "data" / "emb" / dataset / backbone / "identity.npy")
    probe = P.fit_probe(P.l2(base)[tr], y[tr], seed)
    acc_id = float((P.predict(probe, P.l2(base)[te]) == y[te]).mean())
    print(f"{dataset}/{backbone}: identity drop-in accuracy {acc_id:.4f}",
          flush=True)

    enc = Encoder(backbone, threads=threads)
    out = TAB / f"sensitivity__{dataset}__{backbone}.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["variant"] for r in rows}

    for name, family, fn in VARIANTS:
        if name in done:
            print(f"[skip] {name}", flush=True)
            continue
        E = embed(enc, idx.path.to_numpy()[te], fn, batch, workers)
        acc = float((P.predict(probe, P.l2(E)) == y[te]).mean())
        rows.append(dict(dataset=dataset, backbone=backbone, variant=name,
                         family=family, accuracy=acc, identity=acc_id,
                         delta=acc - acc_id))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[done] {name}: {acc:.4f}  ({acc - acc_id:+.4f} vs identity)",
              flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="exdark")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    main(a.dataset, a.backbone, a.batch, a.workers, a.threads, a.seed)

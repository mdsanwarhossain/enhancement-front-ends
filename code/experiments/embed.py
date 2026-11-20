"""Extract frozen-encoder embeddings for one (dataset, operator, encoder).

Enhanced images are never written to disk: each worker decodes, enhances and
normalises on the fly, so the whole operator grid costs embeddings only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import cv2                                   # noqa: E402
from enhance.ops import OPS                  # noqa: E402
from features.backbones import Encoder       # noqa: E402

OUT = ROOT / "data" / "emb"
MAX_SIDE = 640          # cap before enhancement: some operators are O(pixels)


def _init_worker(_):
    """Pin each loader worker to one thread.

    Operators that call torch (Zero-DCE) otherwise inherit the default thread
    count, so twelve workers ask for twelve full thread pools on a 22-thread
    machine and the run collapses. Flush-to-zero has to be set here too: it is
    a per-thread CPU flag, and trained weights are full of subnormals.
    """
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    cv2.setNumThreads(1)         # eight workers each opening a thread pool
                                 # oversubscribes the machine badly


class EnhancedImages(Dataset):
    def __init__(self, paths, op, transform):
        self.paths, self.op, self.transform = list(paths), OPS[op], transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:                      # unreadable file -> black frame
            img = np.zeros((224, 224, 3), np.uint8)
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out = self.op(img)
        return self.transform(Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))


def run(dataset: str, op: str, backbone: str, batch: int, workers: int,
        threads: int, limit: int | None) -> Path:
    idx = pd.read_csv(ROOT / "data" / "index" / f"{dataset}.csv")
    if limit:
        idx = idx.iloc[:limit]
    dest = OUT / dataset / backbone
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"{op}.npy"
    if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(idx):
        print(f"[skip] {dataset}/{backbone}/{op}", flush=True)
        return f

    enc = Encoder(backbone, threads=threads)
    dl = DataLoader(EnhancedImages(idx.path, op, enc.transform), batch_size=batch,
                    num_workers=workers, shuffle=False, pin_memory=False,
                    worker_init_fn=_init_worker)
    chunks, t0, seen = [], time.time(), 0
    with torch.no_grad():
        for xb in dl:
            chunks.append(enc.embed(xb))
            seen += len(xb)
            if seen % (batch * 20) == 0:
                print(f"  {dataset}/{backbone}/{op}: {seen}/{len(idx)} "
                      f"({seen/(time.time()-t0):.1f} img/s)", flush=True)
    emb = np.concatenate(chunks)
    np.save(f, emb)
    print(f"[done] {dataset}/{backbone}/{op} {emb.shape} in "
          f"{time.time()-t0:.0f}s ({len(idx)/(time.time()-t0):.1f} img/s)", flush=True)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--ops", default="all")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ops = list(OPS) if a.ops == "all" else a.ops.split(",")
    for op in ops:
        run(a.dataset, op, a.backbone, a.batch, a.workers, a.threads, a.limit)

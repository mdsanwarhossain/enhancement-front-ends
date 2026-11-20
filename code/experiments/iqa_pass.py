"""Score every (image, operator) pair with the pixel-domain quality metrics.

Runs once per dataset and is independent of any encoder, so the resulting table
is reused by every backbone in the analysis.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from enhance.ops import OPS                           # noqa: E402
from iqa.metrics import NO_REFERENCE, INPUT_REFERENCED  # noqa: E402

MAX_SIDE = 640


def _init_worker():
    """One thread per worker; see experiments/embed.py."""
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_flush_denormal(True)
    except ImportError:
        pass
    cv2.setNumThreads(1)


def _load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# BRISQUE and NIQE cost about twenty times the rest put together, and the
# paper only ever uses their mean over an operator. We therefore evaluate them
# on a regular subsample and leave the cheap metrics on every image.
HEAVY = ("brisque", "niqe")
HEAVY_EVERY = 4


def score_one(args):
    i, path = args
    src = _load(path)
    if src is None:
        return []
    out = []
    for op_name, op in OPS.items():
        enh = op(src)
        row = {"row": i, "operator": op_name}
        for m, fn in NO_REFERENCE.items():
            if m in HEAVY and i % HEAVY_EVERY:
                row[m] = float("nan")
                continue
            row[m] = fn(enh)
        for m, fn in INPUT_REFERENCED.items():
            row[m] = fn(enh, src) if m != "loe" else fn(enh, src, 32)
        out.append(row)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    idx = pd.read_csv(ROOT / "data" / "index" / f"{a.dataset}.csv")
    if a.limit:
        idx = idx.iloc[:a.limit]
    dest = ROOT / "data" / "iqa"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{a.dataset}.parquet"
    if out.exists() and len(pd.read_parquet(out, columns=["row"])) == len(idx) * len(OPS):
        print(f"[skip] {a.dataset} pixel metrics already complete")
        raise SystemExit(0)

    rows = []
    with Pool(a.workers, initializer=_init_worker) as pool:
        for n, batch in enumerate(pool.imap_unordered(
                score_one, list(enumerate(idx.path)), chunksize=8), 1):
            rows.extend(batch)
            if n % 500 == 0:
                print(f"  {a.dataset} iqa: {n}/{len(idx)}", flush=True)
    df = pd.DataFrame(rows).sort_values(["row", "operator"])
    df.to_parquet(out, index=False)
    print(f"[done] {a.dataset}: {len(df)} (image, operator) rows -> {dest}")

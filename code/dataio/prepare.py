"""Cache every source image at a single working resolution.

Operators such as CLAHE and multi-scale Retinex are resolution-dependent, so
holding resolution fixed removes a confound that would otherwise differ across
the three datasets. The cache is lossless PNG, and the resize happens once
rather than once per operator per encoder.
"""
from __future__ import annotations

import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "work"
MAX_SIDE = 512


def one(args):
    src, dst = args
    dst = Path(dst)
    if dst.exists():
        return 1
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return 0
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)),
                         interpolation=cv2.INTER_LANCZOS4)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return 1


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["turbid_fish", "exdark", "kvasir"]):
        f = ROOT / "data" / "index" / f"{name}.csv"
        idx = pd.read_csv(f)
        # name by a hash of the source path so the cache survives an index
        # rebuild (a later download changes row order, not file identity)
        jobs = [(p, str(CACHE / name /
                        f"{hashlib.sha1(p.encode()).hexdigest()[:16]}.png"))
                for p in idx.path]
        with Pool(20) as pool:
            ok = sum(pool.imap_unordered(one, jobs, chunksize=16))
        idx["path"] = [j[1] for j in jobs]
        idx.to_csv(f, index=False)          # index now points at the cache
        print(f"{name}: cached {ok}/{len(jobs)} images at <= {MAX_SIDE}px")

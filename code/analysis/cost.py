"""Wall-clock cost of each operator and each encoder, on this CPU.

Routing per image means embedding the image under every candidate operator, so
the paper has to state what that costs rather than leave it implied. Timings
are medians over repeated runs on real images at the working resolution.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import cv2                                            # noqa: E402
import torch                                          # noqa: E402
from PIL import Image                                 # noqa: E402
from enhance.ops import OPS                           # noqa: E402
from features.backbones import Encoder, BACKBONES, PRETRAIN  # noqa: E402

TAB = ROOT / "results" / "tables"
N_IMAGES, REPEATS, THREADS = 24, 3, 14


def sample_images(n):
    idx = pd.read_csv(ROOT / "data" / "index" / "turbid_fish.csv")
    pick = np.linspace(0, len(idx) - 1, n).astype(int)
    return [cv2.imread(p, cv2.IMREAD_COLOR) for p in idx.path.iloc[pick]]


def time_operators(imgs):
    rows = []
    for name, op in OPS.items():
        op(imgs[0])                                   # warm any lazy weights
        times = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            for im in imgs:
                op(im)
            times.append((time.perf_counter() - t) / len(imgs) * 1e3)
        rows.append(dict(component="operator", name=name,
                         ms_per_image=float(np.median(times))))
        print(f"  operator {name:12s} {rows[-1]['ms_per_image']:7.1f} ms", flush=True)
    return rows


def time_encoders(imgs):
    pil = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in imgs]
    rows = []
    for key in BACKBONES:
        enc = Encoder(key, threads=THREADS)
        enc(pil[:4])
        times = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            enc(pil)
            times.append((time.perf_counter() - t) / len(pil) * 1e3)
        rows.append(dict(component="encoder", name=key,
                         ms_per_image=float(np.median(times)),
                         dim=enc.dim, pretraining=PRETRAIN[key],
                         params_m=sum(p.numel() for p in enc.model.parameters()) / 1e6
                         if not isinstance(enc.model, torch.jit.ScriptModule)
                         else float("nan")))
        print(f"  encoder  {key:12s} {rows[-1]['ms_per_image']:7.1f} ms", flush=True)
    return rows


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    imgs = sample_images(N_IMAGES)
    print(f"timing on {len(imgs)} images at {imgs[0].shape[1]}x{imgs[0].shape[0]}")
    rows = time_operators(imgs) + time_encoders(imgs)
    df = pd.DataFrame(rows)
    df["threads"] = THREADS
    df.to_csv(TAB / "cost.csv", index=False)
    print(f"\nwrote {TAB / 'cost.csv'}")

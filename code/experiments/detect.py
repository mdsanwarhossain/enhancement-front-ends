"""Zero-shot detection under each enhancement operator.

This is the purest form of the question the paper asks. A COCO-pretrained
detector is not retrained, not fine-tuned and not even given a linear head: it
is exactly the fixed model a practitioner would put an enhancement front end
in front of. The twelve ExDark categories all exist in COCO, so the detector
can be scored directly against ExDark's boxes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import cv2                                               # noqa: E402
from enhance.ops import OPS                              # noqa: E402
from dataio.build_det_index import CLASSES, COCO_ID      # noqa: E402

TAB = ROOT / "results" / "tables"
PRED = ROOT / "data" / "det"
MAX_SIDE = 640


def load_detector(threads):
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn as build,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W)
    torch.set_flush_denormal(True)
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    return build(weights=W.DEFAULT).eval()


def parse_boxes(s: str):
    out = []
    for item in s.split(";"):
        f = item.split(",")
        out.append((int(f[0]), *map(float, f[1:])))
    return out


# ------------------------------------------------------------ VOC-style AP ---
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def average_precision(preds, truths, thr=0.5):
    """preds: list of (image, score, box). truths: {image: [box, ...]}."""
    n_true = sum(len(v) for v in truths.values())
    if n_true == 0:
        return float("nan")
    preds = sorted(preds, key=lambda p: -p[1])
    matched = {k: np.zeros(len(v), bool) for k, v in truths.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (img, _, box) in enumerate(preds):
        gts = truths.get(img, [])
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[img][j]:
                continue
            v = iou(box, g)
            if v > best:
                best, best_j = v, j
        if best >= thr:
            matched[img][best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_true
    prec = tp / np.maximum(tp + fp, 1e-12)
    # all-point interpolation
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def _init_worker(_):
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    cv2.setNumThreads(1)


class Enhanced(torch.utils.data.Dataset):
    """Decode and enhance in worker processes; the detector keeps the main one.

    Multi-scale Retinex costs more per image than the detector does, so running
    it serially in the main loop leaves most of the machine idle.
    """

    def __init__(self, paths, op):
        self.paths, self.op = list(paths), OPS[op]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((64, 64, 3), np.uint8)
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        out = self.op(img)
        x = cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_BGR2RGB)
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0,
                i)


def _collate(items):
    return [x for x, _ in items], [i for _, i in items]


def run_operator(model, idx, op_name, batch, workers=8, score_thr=0.05):
    preds = {c: [] for c in CLASSES}
    truths = {c: {} for c in CLASSES}
    inv = {v: k for k, v in COCO_ID.items()}
    paths = idx.path.to_numpy()
    for row in idx.itertuples():
        for c, *box in parse_boxes(row.boxes):
            truths[CLASSES[c]].setdefault(row.path, []).append(tuple(box))

    dl = torch.utils.data.DataLoader(
        Enhanced(paths, op_name), batch_size=batch, num_workers=workers,
        shuffle=False, collate_fn=_collate, worker_init_fn=_init_worker)
    t0, seen = time.time(), 0
    for images, order in dl:
        for out, i, im in zip(model(images), order, images):
            h, w = float(im.shape[1]), float(im.shape[2])
            keep = out["scores"] >= score_thr
            name = paths[i]
            for lab, sc, bx in zip(out["labels"][keep].tolist(),
                                   out["scores"][keep].tolist(),
                                   out["boxes"][keep].tolist()):
                cls = inv.get(lab)
                if cls is None:
                    continue
                preds[cls].append((name, sc,
                                   (bx[0] / w, bx[1] / h, bx[2] / w, bx[3] / h)))
        seen += len(images)
        if seen % (batch * 20) == 0:
            print(f"  {op_name}: {seen}/{len(idx)} "
                  f"({seen / (time.time() - t0):.1f} img/s)", flush=True)

    per_class = {c: average_precision(preds[c], truths[c]) for c in CLASSES}
    valid = [v for v in per_class.values() if np.isfinite(v)]
    return float(np.mean(valid)) if valid else float("nan"), per_class


def main(ops, limit, batch, threads, seed, workers):
    idx = pd.read_csv(ROOT / "data" / "index" / "exdark_det.csv")
    if limit:
        idx = idx.sample(n=min(limit, len(idx)), random_state=seed)
    model = load_detector(threads)
    print(f"exdark_det: {len(idx)} images, {idx.n_boxes.sum()} boxes", flush=True)

    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / "detection.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(prev.operator)
    rows = [] if not done else pd.read_csv(out).to_dict("records")

    for op in ops:
        if op in done:
            print(f"[skip] {op}", flush=True)
            continue
        mAP, per_class = run_operator(model, idx, op, batch, workers)
        rows.append(dict(operator=op, n_images=len(idx), mAP50=mAP,
                         **{f"ap_{k}": v for k, v in per_class.items()}))
        pd.DataFrame(rows).to_csv(out, index=False)     # checkpoint each operator
        print(f"[done] {op}: mAP@0.5 = {mAP:.4f}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(list(OPS) if a.ops == "all" else a.ops.split(","),
         a.limit, a.batch, a.threads, a.seed, a.workers)

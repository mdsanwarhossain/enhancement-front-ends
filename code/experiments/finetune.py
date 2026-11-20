"""Does the penalty survive fine-tuning?

The main experiments read a frozen encoder with a linear probe, which isolates
the representation but invites the objection that a fine-tuned model would
behave differently. Here the whole network is fine-tuned, and the same two
protocols are run on it:

  P1  fine-tune on the raw degraded images, then put an operator in front;
  P2  fine-tune on the operator's own output and test on the same operator.

The split is the one the main experiments use, so the numbers are comparable.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import cv2                                   # noqa: E402
from enhance.ops import OPS                  # noqa: E402
import experiments.protocols as P            # noqa: E402

TAB = ROOT / "results" / "tables"
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _init_worker(_):
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    cv2.setNumThreads(1)


class Frames(Dataset):
    """Images under one operator, with light augmentation when training."""

    def __init__(self, paths, labels, op, train):
        self.paths, self.labels = list(paths), np.asarray(labels)
        self.op, self.train = OPS[op], train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((SIZE, SIZE, 3), np.uint8)
        img = self.op(img)
        h, w = img.shape[:2]
        if self.train:
            s = np.random.uniform(0.7, 1.0)
            ch, cw = int(h * s), int(w * s)
            y0 = np.random.randint(0, h - ch + 1)
            x0 = np.random.randint(0, w - cw + 1)
            img = img[y0:y0 + ch, x0:x0 + cw]
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), int(self.labels[i])


def build(n_classes, arch="resnet18"):
    import timm
    torch.set_flush_denormal(True)
    m = timm.create_model(arch, pretrained=True, num_classes=n_classes)
    return m


def train_one(paths, y, op, n_classes, epochs, batch, workers, threads, lr,
              seed):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    model = build(n_classes)
    dl = DataLoader(Frames(paths, y, op, train=True), batch_size=batch,
                    shuffle=True, num_workers=workers, drop_last=True,
                    worker_init_fn=_init_worker, persistent_workers=workers > 0)
    head = [p for n, p in model.named_parameters() if "fc" in n or "head" in n
            or "classifier" in n]
    body = [p for n, p in model.named_parameters()
            if not ("fc" in n or "head" in n or "classifier" in n)]
    opt = torch.optim.AdamW([{"params": body, "lr": lr},
                             {"params": head, "lr": lr * 10}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, lr * 10], total_steps=max(epochs * len(dl), 1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    model.train()
    for ep in range(epochs):
        t0, tot, correct, running = time.time(), 0, 0, 0.0
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss) * len(yb)
            correct += int((out.argmax(1) == yb).sum())
            tot += len(yb)
        print(f"    [{op}] epoch {ep + 1}/{epochs} loss={running / tot:.3f} "
              f"train_acc={correct / tot:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, paths, y, op, batch, workers):
    dl = DataLoader(Frames(paths, y, op, train=False), batch_size=batch,
                    shuffle=False, num_workers=workers,
                    worker_init_fn=_init_worker)
    pred = []
    for xb, _ in dl:
        pred.append(model(xb).argmax(1).numpy())
    pred = np.concatenate(pred)
    return float((pred == np.asarray(y)).mean())


def main(dataset, p2_ops, epochs, batch, workers, threads, lr, seed, limit):
    idx = pd.read_csv(ROOT / "data" / "index" / f"{dataset}.csv")
    if limit:                                   # smoke-test path only
        per = max(limit // idx.label.nunique(), 2)
        idx = (idx.groupby("label", group_keys=False)
               .sample(n=per, random_state=seed, replace=False)
               .reset_index(drop=True))
    classes = sorted(idx.label.unique())
    y = idx.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    tr, te = next(iter(P.folds(idx, 3, seed)))
    paths = idx.path.to_numpy()
    print(f"{dataset}: fine-tuning on {len(tr)} images, testing on {len(te)}, "
          f"{len(classes)} classes", flush=True)

    rows = []
    out = TAB / f"finetune__{dataset}.csv"
    if out.exists():
        rows = pd.read_csv(out).to_dict("records")
    done = {(r["protocol"], r["operator"]) for r in rows}

    # ---- P1: one model, fitted on the raw images --------------------------
    if not any(p == "P1_dropin" for p, _ in done):
        print("  training the drop-in model on raw images", flush=True)
        model = train_one(paths[tr], y[tr], "identity", len(classes), epochs,
                          batch, workers, threads, lr, seed)
        for op in OPS:
            acc = evaluate(model, paths[te], y[te], op, batch, workers)
            rows.append(dict(dataset=dataset, protocol="P1_dropin", operator=op,
                             accuracy=acc, n_test=len(te), epochs=epochs))
            print(f"    P1 {op}: {acc:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
        del model

    # ---- P2: one model per operator, fitted on its own output -------------
    for op in p2_ops:
        if ("P2_retrained", op) in done:
            print(f"  [skip] P2 {op}", flush=True)
            continue
        print(f"  training on {op}", flush=True)
        model = train_one(paths[tr], y[tr], op, len(classes), epochs, batch,
                          workers, threads, lr, seed)
        acc = evaluate(model, paths[te], y[te], op, batch, workers)
        rows.append(dict(dataset=dataset, protocol="P2_retrained", operator=op,
                         accuracy=acc, n_test=len(te), epochs=epochs))
        print(f"    P2 {op}: {acc:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        del model

    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="exdark")
    ap.add_argument("--p2-ops", default="identity,clahe,msrcr,gamma,hist_eq")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    main(a.dataset, a.p2_ops.split(","), a.epochs, a.batch, a.workers,
         a.threads, a.lr, a.seed, a.limit)

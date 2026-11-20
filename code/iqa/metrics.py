"""No-reference and input-referenced image quality metrics.

Two families are kept apart on purpose:

* ``no_reference`` scores judge the enhanced image on its own - these are what
  the enhancement literature optimises and reports.
* ``input_referenced`` scores compare the enhanced image against the degraded
  input it came from, i.e. how much the operator changed.

Neither family sees a class label, which is exactly the point: the paper asks
whether anything in here predicts downstream recognition.
"""
from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-8


# ------------------------------------------------------------- generic NR ---
def entropy(img) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rms_contrast(img) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    return float(g.std())


def colourfulness(img) -> float:
    """Hasler and Susstrunk (2003) colourfulness."""
    b, g, r = (img[:, :, i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std())
                 + 0.3 * np.hypot(rg.mean(), yb.mean()))


def sharpness(img) -> float:
    """Variance of the Laplacian - the standard focus measure."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _eme(chan, blocks: int = 8) -> float:
    h, w = chan.shape
    bh, bw = max(h // blocks, 1), max(w // blocks, 1)
    vals = []
    for i in range(0, h - bh + 1, bh):
        for j in range(0, w - bw + 1, bw):
            blk = chan[i:i + bh, j:j + bw]
            mx, mn = blk.max(), blk.min()
            vals.append(20.0 * np.log10((mx + EPS) / (mn + EPS)))
    return float(np.mean(vals)) if vals else 0.0


def eme(img) -> float:
    return _eme(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))


# ------------------------------------------------------- underwater NR IQA ---
def uciqe(img) -> float:
    """Yang and Sowmya (2015). Chroma spread, luminance contrast, saturation."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[:, :, 0] / 255.0 * 100.0, lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    chroma = np.hypot(a, b)
    sat = np.where(L > EPS, chroma / (L + EPS), 0.0)
    lo, hi = np.percentile(L, 1), np.percentile(L, 99)
    return float(0.4680 * chroma.std() + 0.2745 * (hi - lo) / 100.0
                 + 0.2576 * sat.mean())


def _asym_trimmed(x, a1=0.1, a2=0.1):
    v = np.sort(x.ravel())
    n = v.size
    lo, hi = int(a1 * n), int((1 - a2) * n)
    seg = v[lo:hi] if hi > lo else v
    return float(seg.mean()), float(seg.std())


def uiqm(img) -> float:
    """Panetta et al. (2016) underwater image quality measure."""
    b, g, r = (img[:, :, i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    mu_rg, sd_rg = _asym_trimmed(rg)
    mu_yb, sd_yb = _asym_trimmed(yb)
    uicm = -0.0268 * np.hypot(mu_rg, mu_yb) + 0.1586 * np.hypot(sd_rg, sd_yb)

    uism = 0.0
    for chan, w in ((r, 0.299), (g, 0.587), (b, 0.114)):
        edge = cv2.magnitude(cv2.Sobel(chan, cv2.CV_64F, 1, 0, ksize=3),
                             cv2.Sobel(chan, cv2.CV_64F, 0, 1, ksize=3))
        uism += w * _eme(chan * (edge / (edge.max() + EPS)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    uiconm = _eme(gray) / 20.0
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


# ---------------------------------------------------- input-referenced IQA ---
def psnr_to_input(enh, src) -> float:
    mse = np.mean((enh.astype(np.float64) - src.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / (mse + EPS)))


def ssim_to_input(enh, src) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), data_range=255))


def ambe(enh, src) -> float:
    """Absolute mean brightness error - brightness distortion of the operator."""
    return float(abs(cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY).mean()
                     - cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).mean()))


def loe(enh, src, size: int = 50) -> float:
    """Lightness order error (Wang et al., 2013): naturalness preservation."""
    def light(x):
        return cv2.resize(x, (size, size),
                          interpolation=cv2.INTER_AREA).max(axis=2).astype(np.float64)
    a, b = light(src), light(enh)
    ra = (a[:, :, None, None] >= a[None, None, :, :])
    rb = (b[:, :, None, None] >= b[None, None, :, :])
    return float((ra != rb).mean())


def brisque(img) -> float:
    """BRISQUE (Mittal et al., 2012), the standard natural-scene-statistics
    no-reference metric. Returned negated, so that larger is better quality
    like every other entry in this table."""
    import torch
    import piq
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        try:
            return float(-piq.brisque(t, data_range=1.0, reduction="none").item())
        except Exception:               # degenerate images can break the fit
            return float("nan")


_NIQE = None


def niqe(img) -> float:
    """NIQE (Mittal et al., 2013), the other standard natural-scene-statistics
    metric. Negated so that larger is better, as everywhere else here."""
    global _NIQE
    import torch
    if _NIQE is None:                       # one model per worker process
        import pyiqa
        torch.set_num_threads(1)
        _NIQE = pyiqa.create_metric("niqe", device="cpu")
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        try:
            return float(-_NIQE(t).item())
        except Exception:
            return float("nan")


NO_REFERENCE = {
    "entropy": entropy,
    "rms_contrast": rms_contrast,
    "colourfulness": colourfulness,
    "sharpness": sharpness,
    "eme": eme,
    "uciqe": uciqe,
    "uiqm": uiqm,
    "brisque": brisque,
    "niqe": niqe,
}

INPUT_REFERENCED = {
    "psnr_in": psnr_to_input,
    "ssim_in": ssim_to_input,
    "ambe": ambe,
    "loe": loe,
}

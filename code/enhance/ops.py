"""Training-free image enhancement operators.

Every operator maps an uint8 BGR image to an uint8 BGR image of the same shape
and is deterministic, so a (image, operator) pair identifies an output exactly.
The set spans the four families that dominate practice on degraded imagery:
histogram redistribution, Retinex, physics-based restoration and sharpening.
"""
from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- baseline ---
def identity(img):
    return img


# ------------------------------------------------- histogram redistribution ---
def hist_eq(img):
    """Global histogram equalisation on luminance only (keeps chroma)."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[:, :, 0] = op.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_auto(img):
    """Gamma chosen so the image mean lands on mid-grey."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0
    gamma = np.log(0.5) / np.log(np.clip(g, 0.02, 0.98))
    lut = _u8(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0)
    return cv2.LUT(img, lut)


# ------------------------------------------------------------------ Retinex ---
def _ssr(chan, sigma):
    blur = cv2.GaussianBlur(chan, (0, 0), sigma)
    return np.log10(chan + 1.0) - np.log10(blur + 1.0)


def msrcr(img, sigmas=(15, 80, 250), alpha=125.0, beta=46.0,
          lo=1.0, hi=1.0):
    """Multi-scale Retinex with colour restoration (Jobson et al., 1997)."""
    f = img.astype(np.float32) + 1.0
    retinex = sum(_ssr(f, s) for s in sigmas) / len(sigmas)
    tot = f.sum(axis=2, keepdims=True)
    restoration = beta * (np.log10(alpha * f + 1.0) - np.log10(tot + 1.0))
    out = retinex * restoration
    # per-channel contrast stretch between the lo/hi percentiles
    for c in range(3):
        ch = out[:, :, c]
        a, b = np.percentile(ch, lo), np.percentile(ch, 100 - hi)
        out[:, :, c] = (ch - a) / (b - a + EPS) * 255.0
    return _u8(out)


def ssr(img, sigma=80.0):
    f = img.astype(np.float32) + 1.0
    out = _ssr(f, sigma)
    for c in range(3):
        ch = out[:, :, c]
        a, b = np.percentile(ch, 1), np.percentile(ch, 99)
        out[:, :, c] = (ch - a) / (b - a + EPS) * 255.0
    return _u8(out)


# -------------------------------------------------------- physics / colour ---
def gray_world(img):
    """Grey-world white balance - the standard underwater colour-cast fix."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + EPS
    return _u8(f * (means.mean() / means))


def dcp(img, patch=15, omega=0.95, t0=0.1):
    """Dark channel prior dehazing (He et al., 2009), also used for turbidity."""
    f = img.astype(np.float32) / 255.0
    dark = cv2.erode(f.min(axis=2), np.ones((patch, patch), np.uint8))
    n = max(int(dark.size * 0.001), 1)
    idx = np.argsort(dark.ravel())[-n:]
    A = f.reshape(-1, 3)[idx].max(axis=0) + EPS
    t = 1.0 - omega * cv2.erode((f / A).min(axis=2),
                                np.ones((patch, patch), np.uint8))
    t = cv2.GaussianBlur(t, (0, 0), 15)[:, :, None]
    return _u8(((f - A) / np.maximum(t, t0) + A) * 255.0)


# --------------------------------------------------------------- sharpening ---
def unsharp(img, sigma=3.0, amount=1.5):
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    return _u8(img.astype(np.float32) * (1 + amount) - blur * amount)


# -------------------------------------------------------------- the roster ---
OPS = {
    "identity": identity,
    "hist_eq": hist_eq,
    "clahe": clahe,
    "gamma": gamma_auto,
    "ssr": ssr,
    "msrcr": msrcr,
    "gray_world": gray_world,
    "dcp": dcp,
    "unsharp": unsharp,
}

FAMILY = {
    "identity": "none",
    "hist_eq": "histogram",
    "clahe": "histogram",
    "gamma": "histogram",
    "ssr": "retinex",
    "msrcr": "retinex",
    "gray_world": "colour",
    "dcp": "physics",
    "unsharp": "sharpening",
}


# ----------------------------------------------------- learned enhancement ---
class _DCENet:
    """Zero-DCE (Guo et al., CVPR 2020): a learned, zero-reference enhancer.

    Included so the operator roster is not purely classical. It is 79k
    parameters and runs comfortably on CPU, and it is the learned method most
    widely deployed as a drop-in front end.
    """

    WEIGHTS = "data/weights/zero_dce_epoch99.pth"
    _net = None

    @classmethod
    def net(cls):
        if cls._net is None:
            import torch
            import torch.nn as nn
            from pathlib import Path

            class DCE(nn.Module):
                def __init__(self, f=32):
                    super().__init__()
                    c = lambda i, o: nn.Conv2d(i, o, 3, 1, 1, bias=True)
                    self.e_conv1, self.e_conv2 = c(3, f), c(f, f)
                    self.e_conv3, self.e_conv4 = c(f, f), c(f, f)
                    self.e_conv5, self.e_conv6 = c(2 * f, f), c(2 * f, f)
                    self.e_conv7 = c(2 * f, 24)
                    self.relu = nn.ReLU(inplace=True)

                def forward(self, x):
                    import torch
                    x1 = self.relu(self.e_conv1(x))
                    x2 = self.relu(self.e_conv2(x1))
                    x3 = self.relu(self.e_conv3(x2))
                    x4 = self.relu(self.e_conv4(x3))
                    x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
                    x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
                    r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
                    for rk in torch.split(r, 3, dim=1):     # 8 curve iterations
                        x = x + rk * (x ** 2 - x)
                    return x

            root = Path(__file__).resolve().parents[2]
            m = DCE()
            m.load_state_dict(torch.load(root / cls.WEIGHTS, map_location="cpu"))
            m.eval()
            torch.set_flush_denormal(True)  # see features/backbones.py
            torch.set_num_threads(1)        # the data loader already forks wide
            cls._net = m
        return cls._net


def zero_dce(img):
    import torch
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().div(255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        y = _DCENet.net()(x).clamp(0, 1)
    out = (y[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


OPS["zero_dce"] = zero_dce
FAMILY["zero_dce"] = "learned"

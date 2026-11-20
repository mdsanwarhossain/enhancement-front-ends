"""Frozen image encoders used as fixed feature extractors.

Nothing here is fine-tuned. The encoders span supervised CNN, modern CNN,
self-supervised ViT and language-supervised ViT pre-training so that any effect
we report cannot be blamed on one pre-training recipe.
"""
from __future__ import annotations

import numpy as np
import timm
import torch

torch.set_grad_enabled(False)
# Trained weights contain subnormal floats; without flush-to-zero the CPU
# convolution path falls off a cliff (measured 30x on this machine).
torch.set_flush_denormal(True)

BACKBONES = {
    "resnet50":   "resnet50.a1_in1k",
    "effnet_b0":  "efficientnet_b0.ra_in1k",
    "dino_s16":   "vit_small_patch16_224.dino",
    "clip_b16":   "vit_base_patch16_clip_224.openai",
}

PRETRAIN = {
    "resnet50":   "supervised (ImageNet-1k)",
    "effnet_b0":  "supervised (ImageNet-1k)",
    "dino_s16":   "self-supervised (ImageNet-1k, DINO)",
    "clip_b16":   "language-supervised (WIT-400M)",
}


class Encoder:
    """A frozen encoder, traced and frozen for CPU inference."""

    def __init__(self, key: str, threads: int | None = None,
                 optimise: bool = True, trace_batch: int = 8):
        if threads:
            torch.set_num_threads(threads)
        self.key = key
        model = timm.create_model(BACKBONES[key], pretrained=True,
                                  num_classes=0).eval()
        cfg = timm.data.resolve_model_data_config(model)
        self.transform = timm.data.create_transform(**cfg, is_training=False)
        self.dim = model.num_features
        self.size = cfg["input_size"]
        self.conv = "vit" not in BACKBONES[key]
        if optimise:
            if self.conv:
                model = model.to(memory_format=torch.channels_last)
            try:
                ex = torch.randn(trace_batch, *self.size)
                if self.conv:
                    ex = ex.to(memory_format=torch.channels_last)
                model = torch.jit.optimize_for_inference(
                    torch.jit.freeze(torch.jit.trace(model, ex)))
                model(ex)                      # warm the fusion passes
            except Exception as exc:           # tracing is best effort
                print(f"[warn] {key}: tracing skipped ({type(exc).__name__})")
        self.model = model

    def embed(self, x: torch.Tensor) -> np.ndarray:
        if self.conv:
            x = x.to(memory_format=torch.channels_last)
        return self.model(x).float().numpy().astype(np.float32)

    def __call__(self, pil_batch) -> np.ndarray:
        return self.embed(torch.stack([self.transform(im) for im in pil_batch]))

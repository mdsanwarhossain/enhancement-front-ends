"""The study in one picture.

A reader should be able to see, without reading the text, what was held fixed,
what varied, where the three protocols differ, and where ESP is computed.
Every string is short enough to sit inside its box at the printed size.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})

INK, GREY = "#1b1b1b", "#707070"
RED, BLUE, GREEN = "#D55E00", "#0072B2", "#009E73"
FILL = {RED: "#fcefe6", BLUE: "#e9f1f8", GREEN: "#e7f4ef", GREY: "#ffffff"}

W, H = 7.9, 2.65         # inches; one full text width in the two-column layout


def box(ax, x, y, w, h, lines, ec=GREY, fs=6.2, head=None, head_c=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.015",
                                linewidth=0.9, facecolor=FILL.get(ec, "#ffffff"),
                                edgecolor=ec, zorder=2))
    y0 = y + h / 2
    if head:
        ax.text(x + w / 2, y + h - 0.036, head, ha="center", va="center",
                fontsize=fs + 0.3, fontweight="bold",
                color=head_c or ec, zorder=3)
        y0 = y + (h - 0.072) / 2
    ax.text(x + w / 2, y0, "\n".join(lines), ha="center", va="center",
            fontsize=fs, color=INK, zorder=3, linespacing=1.45)


def arrow(ax, p, q, color=GREY, lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=color,
                                 linewidth=lw, zorder=1, mutation_scale=7,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ------------------------------------------------- the fixed ingredients
    box(ax, 0.005, 0.635, 0.155, 0.275,
        ["3 domains,", "native degradation", "23,240 images",
         "group-disjoint CV"], head="DATA")
    box(ax, 0.190, 0.635, 0.155, 0.275,
        ["identity +", "9 enhancements", "histogram, Retinex,",
         "physics, learned"], head="OPERATORS")
    box(ax, 0.375, 0.635, 0.155, 0.275,
        ["3 frozen encoders", "ResNet-50,", "EfficientNet-B0,",
         "DINO ViT-S/16"], head="ENCODER  \u03c6")
    arrow(ax, (0.162, 0.772), (0.188, 0.772))
    arrow(ax, (0.347, 0.772), (0.373, 0.772))

    # --------------------------------------------------------- the protocols
    px, pw = 0.560, 0.215
    box(ax, px, 0.765, pw, 0.150,
        ["fitted on raw images,", "tested on enhanced"],
        ec=RED, head="P1  DROP-IN", fs=6.4)
    box(ax, px, 0.595, pw, 0.150,
        ["refitted on the", "operator's output"],
        ec=BLUE, head="P2  RETRAINED", fs=6.4)
    box(ax, px, 0.425, pw, 0.150,
        ["one probe over", "all operators"],
        ec=GREY, head="P3  AGNOSTIC", fs=6.4)
    bus = 0.545
    ax.plot([bus, bus], [0.500, 0.840], color=GREY, linewidth=0.9, zorder=1)
    ax.plot([0.532, bus], [0.772, 0.772], color=GREY, linewidth=0.9, zorder=1)
    for y, c in ((0.840, RED), (0.670, BLUE), (0.500, GREY)):
        arrow(ax, (bus, y), (px - 0.004, y), color=c)

    # ----------------------------------------------------------- what we get
    ox, ow = 0.805, 0.190
    box(ax, ox, 0.765, ow, 0.150,
        ["what a deployed", "system actually gets"], ec=RED)
    box(ax, ox, 0.595, ow, 0.150,
        ["what enhancement", "papers report"], ec=BLUE)
    box(ax, ox, 0.425, ow, 0.150,
        ["a probe that saw", "every operator"], ec=GREY)
    for y, c in ((0.840, RED), (0.670, BLUE), (0.500, GREY)):
        arrow(ax, (px + pw + 0.002, y), (ox - 0.004, y), color=c)

    # ------------------------------------------------------ the two compasses
    box(ax, 0.190, 0.200, 0.340, 0.200,
        ["BRISQUE, NIQE, UCIQE, UIQM, entropy,",
         "contrast, sharpness, EME, colourfulness,",
         "PSNR / SSIM / AMBE / LOE to input"],
        ec=RED, fs=5.9, head="13 PIXEL-DOMAIN QUALITY METRICS")
    box(ax, 0.560, 0.200, 0.435, 0.200,
        ["distance between the enhanced embeddings and",
         "those the probe was fitted on: MMD, Fréchet, centroid.",
         "No labels, no retraining, no extra forward pass."],
        ec=GREEN, fs=6.2, head="ESP  (EMBEDDING SHIFT PENALTY)")
    arrow(ax, (0.2675, 0.632), (0.2675, 0.405), color=RED)
    ax.plot([0.452, 0.452], [0.632, 0.300], color=GREEN, linewidth=0.9, zorder=1)
    arrow(ax, (0.452, 0.300), (0.556, 0.300), color=GREEN)

    # ------------------------------------------------------- the two questions
    box(ax, 0.190, 0.010, 0.805, 0.145,
        ["Does enhancement help a recogniser nobody retrained, and do the quality metrics know?  Can the",
         "shift score rank the operators, and pick one, without a single label?  Checked again on a",
         "ResNet-18 fine-tuned end to end and on a COCO detector never trained on this data."],
        ec=GREEN, fs=6.0)
    for x in (0.360, 0.777):
        arrow(ax, (x, 0.197), (x, 0.158), color=GREY)

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_method.{ext}")
    plt.close(fig)
    print("  wrote fig_method.pdf/.png")


if __name__ == "__main__":
    main()

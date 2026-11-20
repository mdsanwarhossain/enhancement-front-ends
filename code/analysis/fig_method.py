"""The study in one picture.

A reader should be able to see, without reading the text, what was held fixed,
what varied, where the three protocols differ, and where ESP is computed.

Every box sizes itself from the lines it holds. Hard-coding box heights as a
fraction of the figure while the text is measured in points is what makes
captions creep over their borders the moment the figure is rescaled, so
nothing here is positioned by eye: heights come out of the font size, and
main() prints a warning for any line too wide for the box it sits in.
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
    # the default tight bbox pads 7.2pt on every side, which the figure
    # environment supplies again; dropping it makes the diagram print larger
    # for the same share of the page
    "savefig.pad_inches": 0.01,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})

INK, GREY = "#1b1b1b", "#707070"
RED, BLUE, GREEN = "#D55E00", "#0072B2", "#009E73"
FILL = {RED: "#fcefe6", BLUE: "#e9f1f8", GREEN: "#e7f4ef", GREY: "#ffffff"}

# Printed at \textwidth, a figure drawn W inches wide has its text scaled by
# 455/(72W), so a wide canvas buys room for long lines at the cost of legible
# type. 7 inches keeps the body above five printed points.
W, H = 7.3, 2.73         # inches; one full text width in the two-column layout

FS, HFS = 6.2, 6.5       # body and heading point size
LINE = 1.37              # line pitch, as a multiple of the body size
PAD = 3.0                # points of clear space above and below the text
HEAD_GAP = 1.5           # points between a heading and the first body line

PT = 1.0 / (H * 72)      # one point, in axes-y units
PX = 1.0 / (W * 72)      # one point, in axes-x units

_checks: list[tuple] = []


def box_h(n_lines: int, head: bool) -> float:
    """Exactly the height this many lines need, in axes-y units."""
    pts = 2 * PAD + n_lines * FS * LINE + ((HFS + HEAD_GAP) if head else 0)
    return pts * PT


def box(ax, x, top, w, lines, ec=GREY, head=None, head_c=None, fs=FS):
    """Draw a box whose top edge is at `top` and return its bottom edge."""
    if x + w > 1.0:
        print(f"  WARNING box at x={x / PX:.0f}pt runs "
              f"{(x + w - 1.0) / PX:.0f}pt past the right edge")
    h = box_h(len(lines), head is not None)
    y = top - h
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.010",
        linewidth=0.9, facecolor=FILL.get(ec, "#ffffff"), edgecolor=ec,
        zorder=2))
    cursor = top - PAD * PT
    if head:
        t = ax.text(x + w / 2, cursor - HFS * 0.55 * PT, head, ha="center",
                    va="center", fontsize=HFS, fontweight="bold",
                    color=head_c or ec, zorder=3)
        cursor -= (HFS + HEAD_GAP) * PT
        _checks.append((t, w))
    for i, line in enumerate(lines):
        t = ax.text(x + w / 2, cursor - (i + 0.5) * FS * LINE * PT, line,
                    ha="center", va="center", fontsize=fs, color=INK, zorder=3)
        _checks.append((t, w))
    return y


def report_fit(fig, ax):
    """Every line has to clear its box sides; say so when one does not."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for t, w in _checks:
        bb = t.get_window_extent(fig.canvas.get_renderer())
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        room = w - 2 * 4 * PX          # 4pt of clear space each side
        if x1 - x0 > room:
            bad.append((t.get_text(), (x1 - x0) / PX, room / PX))
    for text, got, room in bad:
        print(f"  WARNING {text!r}: {got:.0f}pt wide, {room:.0f}pt of room")
    return bad


def arrow(ax, p, q, color=GREY, lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=color,
                                 linewidth=lw, zorder=1, mutation_scale=8,
                                 shrinkA=0, shrinkB=0))


def line(ax, xs, ys, color=GREY, lw=1.0):
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=1,
            solid_capstyle="round")


def main():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    TOP = 1.0 - 4 * PT

    # ------------------------------------------------- the protocol ladder
    # three rows set the height of the whole upper band
    px, pw = PX * 318, PX * 98
    ox, ow = PX * 428, PX * 94
    h_proto = box_h(2, True)
    h_out = box_h(2, False)
    row_h = max(h_proto, h_out)
    row_gap = 4.0 * PT

    protocols = [
        ("P1  DROP-IN", ["fitted on raw images,", "tested on enhanced"],
         ["what a deployed", "system actually gets"], RED),
        ("P2  RETRAINED", ["refitted on the", "operator's output"],
         ["what enhancement", "papers report"], BLUE),
        ("P3  AGNOSTIC", ["one probe over", "all operators"],
         ["a probe that saw", "every operator"], GREY),
    ]
    mids = []
    for i, (head, body, outcome, colour) in enumerate(protocols):
        row_top = TOP - i * (row_h + row_gap)
        box(ax, px, row_top - (row_h - h_proto) / 2, pw, body,
            ec=colour, head=head)
        box(ax, ox, row_top - (row_h - h_out) / 2, ow, outcome, ec=colour)
        mids.append(row_top - row_h / 2)
        arrow(ax, (px + pw, mids[-1]), (ox - 1.5 * PX, mids[-1]), color=colour)

    band_a = 3 * row_h + 2 * row_gap
    a_bottom = TOP - band_a

    # ------------------------------------------------- the fixed ingredients
    lw_ = PX * 90
    xs = (PX * 3, PX * 105, PX * 207)
    h_left = box_h(4, True)
    left_top = TOP - (band_a - h_left) / 2
    left_mid = left_top - h_left / 2
    left_bottom = left_top - h_left
    box(ax, xs[0], left_top, lw_,
        ["3 domains,", "native degradation", "23,240 images",
         "group-disjoint CV"], head="DATA")
    box(ax, xs[1], left_top, lw_,
        ["identity +", "9 enhancements", "histogram, Retinex,",
         "physics, learned"], head="OPERATORS")
    box(ax, xs[2], left_top, lw_,
        ["3 frozen encoders", "ResNet-50,", "EfficientNet-B0,",
         "DINO ViT-S/16"], head="ENCODER  φ")
    for x in xs[:2]:
        arrow(ax, (x + lw_, left_mid), (x + lw_ + PX * 12, left_mid))

    # one trunk out of the encoder, three branches into the protocols
    bus = PX * 307
    line(ax, [xs[2] + lw_, bus], [left_mid, left_mid])
    line(ax, [bus, bus], [min(mids), max(mids)])
    for y, (_, _, _, colour) in zip(mids, protocols):
        arrow(ax, (bus, y), (px - 1.5 * PX, y), color=colour)

    # ------------------------------------------------------ the two compasses
    b_top = a_bottom - 10 * PT
    mx, mw = PX * 105, PX * 194
    ex, ew = PX * 303, PX * 219
    m_bottom = box(ax, mx, b_top, mw,
                   ["BRISQUE, NIQE, UCIQE, UIQM, entropy,",
                    "contrast, sharpness, EME, colourfulness,",
                    "PSNR / SSIM / AMBE / LOE to input"],
                   ec=RED, head="13 PIXEL-DOMAIN QUALITY METRICS")
    e_bottom = box(ax, ex, b_top, ew,
                   ["distance between the enhanced embeddings",
                    "and those the probe was fitted on:",
                    "MMD, Fréchet or centroid. No labels, no retraining."],
                   ec=GREEN, head="ESP  (EMBEDDING SHIFT PENALTY)")

    # the operators feed the pixel metrics directly
    arrow(ax, (xs[1] + lw_ / 2, left_bottom), (xs[1] + lw_ / 2, b_top),
          color=RED)
    # the encoder feeds ESP, routed through the gap so it crosses nothing
    elbow = a_bottom - 5.0 * PT
    line(ax, [xs[2] + lw_ / 2, xs[2] + lw_ / 2], [left_bottom, elbow],
         color=GREEN)
    line(ax, [xs[2] + lw_ / 2, ex + ew / 2], [elbow, elbow], color=GREEN)
    arrow(ax, (ex + ew / 2, elbow), (ex + ew / 2, b_top), color=GREEN)

    # ------------------------------------------------------- the two questions
    c_top = min(m_bottom, e_bottom) - 8.5 * PT
    qx, qw = PX * 105, PX * 417
    box(ax, qx, c_top, qw,
        ["Does enhancement help a recogniser nobody retrained, "
         "and do the quality metrics know?",
         "Can the shift score rank the operators, and pick one, "
         "without a single label?"],
        ec=GREEN)
    for x, colour in ((mx + mw / 2, RED), (ex + ew / 2, GREEN)):
        arrow(ax, (x, min(m_bottom, e_bottom)), (x, c_top), color=colour)

    report_fit(fig, ax)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_method.{ext}")
    plt.close(fig)
    print("  wrote fig_method.pdf/.png")


if __name__ == "__main__":
    main()

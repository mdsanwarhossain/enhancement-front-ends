"""Every figure in the manuscript, generated from results/tables only.

One idea per panel, colour-blind safe, and legible after the journal shrinks a
figure to one column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})

C = {"ink": "#1b1b1b", "grey": "#9a9a9a", "blue": "#0072B2", "orange": "#E69F00",
     "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9"}

DATASET = {"turbid_fish": "Turbid fish", "exdark": "ExDark",
           "kvasir": "HyperKvasir"}
ORDER = ["turbid_fish", "exdark", "kvasir"]
OPERATOR = {"identity": "none", "hist_eq": "hist-eq", "clahe": "CLAHE",
            "gamma": "gamma", "ssr": "SSR", "msrcr": "MSRCR",
            "gray_world": "grey-world", "dcp": "DCP", "unsharp": "unsharp",
            "zero_dce": "Zero-DCE"}
SCORER = {
    "esp_mmd": "ESP (MMD)", "esp_frechet": "ESP (Fr\u00e9chet)",
    "esp_centroid": "ESP (centroid)", "esp_knn": "ESP (kNN)",
    "esp_mahalanobis": "ESP (Mahalanobis)",
    "geo_consensus": "neighbourhood consensus", "geo_hubness": "hubness",
    "geo_mutual_knn": "mutual kNN rate", "geo_effective_rank": "effective rank",
    "geo_twonn_dim": "intrinsic dimension", "geo_taq": "embedding geometry",
    "iqa_brisque": "BRISQUE", "iqa_niqe": "NIQE",
    "iqa_entropy": "entropy", "iqa_rms_contrast": "RMS contrast",
    "iqa_colourfulness": "colourfulness", "iqa_sharpness": "sharpness",
    "iqa_eme": "EME", "iqa_uciqe": "UCIQE", "iqa_uiqm": "UIQM",
    "iqa_psnr_in": "PSNR to input", "iqa_ssim_in": "SSIM to input",
    "iqa_ambe": "AMBE", "iqa_loe": "LOE",
}


def read(pattern):
    files = sorted(TAB.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) \
        if files else pd.DataFrame()


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def datasets_in(df):
    return [d for d in ORDER if d in set(df.dataset)]


# ---------------------------------------------------------------- figure 1 ---
def fig_protocol_gap():
    """The headline: enhancement costs accuracy, unless you retrain."""
    df = read("operator_accuracy__*.csv")
    if df.empty:
        return
    dss = datasets_in(df)
    fig, axes = plt.subplots(1, len(dss), figsize=(3.1 * len(dss), 2.30))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        d = df[df.dataset == ds]
        p1 = d[d.protocol == "P1_dropin"].groupby("operator").accuracy
        p2 = d[d.protocol == "P2_retrained"].groupby("operator").accuracy
        ops = [o for o in OPERATOR if o in p1.mean().index]
        x = np.arange(len(ops))
        ax.errorbar(x - 0.14, p1.mean().reindex(ops) * 100,
                    yerr=p1.sem().reindex(ops).fillna(0) * 100, fmt="o", ms=3.5,
                    lw=0, elinewidth=0.9, capsize=1.8, color=C["red"],
                    label="P1 drop-in")
        ax.errorbar(x + 0.14, p2.mean().reindex(ops) * 100,
                    yerr=p2.sem().reindex(ops).fillna(0) * 100, fmt="s", ms=3.5,
                    lw=0, elinewidth=0.9, capsize=1.8, color=C["blue"],
                    label="P2 retrained")
        base = p1.mean().get("identity", np.nan) * 100
        ax.axhline(base, color=C["grey"], ls="--", lw=0.9, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([OPERATOR[o] for o in ops], rotation=55, ha="right")
        ax.set_title(DATASET[ds])
        if ax is axes[0]:
            ax.set_ylabel("accuracy (%)")
    # one legend for the figure, clear of the data
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color=C["grey"], ls="--", lw=0.9))
    labels.append("no enhancement")
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig_protocol_gap")


# ---------------------------------------------------------------- figure 2 ---
MAIN_SCORERS = ["esp_frechet", "esp_mmd", "esp_centroid", "geo_consensus",
                "iqa_ssim_in", "iqa_psnr_in", "iqa_brisque", "iqa_entropy",
                "iqa_uciqe", "iqa_uiqm", "iqa_eme", "iqa_rms_contrast",
                "iqa_niqe", "iqa_sharpness"]


def fig_scorer_ranking(subset=None, name="fig_scorer_ranking"):
    """Which label-free score ranks operators the way accuracy does?"""
    df = read("operator_scores__*.csv")
    if df.empty:
        return
    if subset:
        df = df[df.scorer.isin(subset)]
    per = (df.groupby(["scorer", "dataset", "backbone", "fold"])
           .apply(lambda g: spearmanr(g.value, g.p1_accuracy).statistic,
                  include_groups=False).rename("rho").reset_index())
    agg = per.groupby("scorer").rho.agg(["mean", "sem"]).sort_values("mean")
    colour = {"esp": C["blue"], "geo": C["green"], "iqa": C["red"]}
    fig, ax = plt.subplots(figsize=(4.2, 0.2 * len(agg) + 1.0))
    y = np.arange(len(agg))
    cols = [colour[s.split("_")[0]] for s in agg.index]
    ax.barh(y, agg["mean"], xerr=agg["sem"].fillna(0), color=cols,
            height=0.68, error_kw=dict(lw=0.8, capsize=1.6))
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([SCORER.get(s, s) for s in agg.index])
    ax.set_xlabel("Spearman rho with drop-in accuracy")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[k]) for k in colour]
    ax.legend(handles, ["embedding shift (ours)", "embedding geometry",
                        "pixel quality"], frameon=False, loc="lower right")
    save(fig, name)


# ---------------------------------------------------------------- figure 3 ---
def fig_shift_vs_accuracy():
    """The mechanism: accuracy falls with how far the operator moves things."""
    df = read("operator_scores__*.csv")
    df = df[df.scorer == "esp_mmd"]
    if df.empty:
        return
    dss = datasets_in(df)
    fig, axes = plt.subplots(1, len(dss), figsize=(2.5 * len(dss), 2.4))
    axes = np.atleast_1d(axes)
    marks = {"resnet50": "o", "effnet_b0": "s", "dino_s16": "^",
             "clip_b16": "D"}
    for ax, ds in zip(axes, dss):
        d = df[df.dataset == ds].groupby(["backbone", "operator"]).mean(
            numeric_only=True).reset_index()
        for bb, g in d.groupby("backbone"):
            ax.scatter(-g.value, g.p1_accuracy * 100, s=18,
                       marker=marks.get(bb, "o"), label=bb, alpha=0.85)
        if len(d) > 2:
            rho = spearmanr(d.value, d.p1_accuracy).statistic
            ax.set_title(f"{DATASET[ds]}  (rho={rho:+.2f})")
        else:
            ax.set_title(DATASET[ds])
        ax.set_xlabel("embedding shift (MMD)")
        if ax is axes[0]:
            ax.set_ylabel("drop-in accuracy (%)")
            ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_shift_vs_accuracy")


# ---------------------------------------------------------------- figure 4 ---
def fig_severity():
    """Accuracy against the degradation level the data was recorded with."""
    df = read("severity__*.csv")
    if df.empty:
        return
    dss = datasets_in(df)
    fig, axes = plt.subplots(1, len(dss), figsize=(2.8 * len(dss), 2.4))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        d = df[df.dataset == ds]
        rank = d.groupby("operator").p1_accuracy.mean().sort_values(
            ascending=False)
        show = ["identity"] + [o for o in rank.index if o != "identity"][:2] \
            + [rank.index[-1]]
        pal = [C["ink"], C["blue"], C["green"], C["red"]]
        for i, op in enumerate(dict.fromkeys(show)):
            g = d[d.operator == op].groupby("severity").p1_accuracy
            ax.errorbar(g.mean().index, g.mean() * 100,
                        yerr=g.sem().fillna(0) * 100, marker="o", ms=3,
                        lw=1.2, capsize=1.5, color=pal[i % len(pal)],
                        label=OPERATOR.get(op, op))
        ax.set_title(DATASET[ds])
        ax.set_xlabel("turbidity band" if ds == "turbid_fish"
                      else "lighting condition")
        if ax is axes[0]:
            ax.set_ylabel("drop-in accuracy (%)")
        ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig_severity")


# ---------------------------------------------------------------- figure 5 ---
def fig_selection_regret():
    """What it costs to choose an operator with the wrong compass."""
    df = read("selection__*.csv")
    if df.empty:
        return
    iqa = df[df.scorer.str.startswith("iqa_")]
    keep = ["identity", "esp_mmd"]
    if len(iqa):
        keep += [iqa.groupby("scorer").p1_regret.mean().idxmin(),
                 iqa.groupby("scorer").p1_regret.mean().idxmax()]
    keep.append("best_fixed_p1")
    dss = datasets_in(df)
    fig, ax = plt.subplots(figsize=(1.6 * len(dss) + 2.0, 2.4))
    w = 0.8 / len(keep)
    cols = [C["grey"], C["blue"], C["green"], C["red"], C["ink"]]
    for i, sc in enumerate(dict.fromkeys(keep)):
        g = df[df.scorer == sc].groupby("dataset").p1_regret
        m = g.mean().reindex(dss) * 100
        e = g.sem().reindex(dss).fillna(0) * 100
        x = np.arange(len(dss)) + i * w
        ax.bar(x, m, w, yerr=e, capsize=2, color=cols[i % len(cols)],
               edgecolor="none",
               label=SCORER.get(sc, {"identity": "no enhancement",
                                     "best_fixed_p1": "oracle"}.get(sc, sc)))
        # the useful scores are near zero, so the bars alone would hide them
        for xi, mi in zip(x, m):
            if np.isfinite(mi):
                ax.annotate(f"{mi:.1f}", (xi, mi), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=5.5,
                            rotation=90, color=C["ink"])
    ax.set_xticks(np.arange(len(dss)) + 0.4 - w / 2)
    ax.set_xticklabels([DATASET[d] for d in dss])
    ax.set_ylabel("accuracy lost vs best operator (pp)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)      # headroom for the labels
    ax.legend(frameon=False, ncol=2, fontsize=6.5, loc="upper right")
    save(fig, "fig_selection_regret")


# ---------------------------------------------------------------- figure 6 ---
def fig_beyond_linear_probe():
    """The same effect on a fine-tuned network and on a zero-shot detector."""
    ft = read("finetune__*.csv")
    det = read("detection.csv")
    if ft.empty and det.empty:
        return
    panels = [p for p in (("ft", ft), ("det", det)) if not p[1].empty]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 2.6))
    axes = np.atleast_1d(axes)

    for ax, (kind, d) in zip(axes, panels):
        if kind == "ft":
            p1 = d[d.protocol == "P1_dropin"].set_index("operator").accuracy * 100
            p2 = d[d.protocol == "P2_retrained"].set_index("operator").accuracy * 100
            ops = [o for o in OPERATOR if o in p1.index]
            x = np.arange(len(ops))
            ax.scatter(x, p1.reindex(ops), s=22, color=C["red"], zorder=3,
                       label="P1 drop-in")
            have = [o for o in ops if o in p2.index]
            ax.scatter([ops.index(o) for o in have], p2.reindex(have), s=22,
                       marker="s", color=C["blue"], zorder=3, label="P2 retrained")
            ax.axhline(p1.get("identity", np.nan), color=C["grey"], ls="--",
                       lw=0.9, zorder=0)
            ax.set_title("fine-tuned ResNet-18 (ExDark)")
            ax.set_ylabel("accuracy (%)")
            ax.legend(frameon=False, fontsize=6.5)
        else:
            m = d.set_index("operator").mAP50 * 100
            ops = [o for o in OPERATOR if o in m.index]
            x = np.arange(len(ops))
            ax.scatter(x, m.reindex(ops), s=22, color=C["purple"], zorder=3)
            ax.axhline(m.get("identity", np.nan), color=C["grey"], ls="--",
                       lw=0.9, zorder=0)
            ax.set_title("zero-shot detector (ExDark)")
            ax.set_ylabel("mAP@0.5 (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([OPERATOR[o] for o in ops], rotation=55, ha="right")
    save(fig, "fig_beyond_linear_probe")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    jobs = [fig_protocol_gap, fig_scorer_ranking,
            lambda: fig_scorer_ranking(MAIN_SCORERS, "fig_scorer_ranking_main"),
            fig_shift_vs_accuracy, fig_severity, fig_selection_regret,
            fig_beyond_linear_probe]
    for fn in jobs:
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {getattr(fn, '__name__', fn)}: "
                  f"{type(exc).__name__}: {exc}")

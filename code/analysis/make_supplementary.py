"""Build the supplementary document: the full grid, one setting per table.

The manuscript reports means across settings. Everything that was averaged is
written out here, so a reader can see whether an effect rests on one dataset or
one encoder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from analysis.make_tables import (DATASET, ENCODER, OPERATOR, SCORER,  # noqa
                                  fmt, read)

OUT = ROOT / "paper" / "supplementary.tex"


def longtable(head, rows, caption, label, align):
    out = [r"\begin{table}[htbp]", rf"\caption{{{caption}}}\label{{{label}}}",
           r"\centering\footnotesize",
           rf"\begin{{tabular}}{{{align}}}", r"\toprule",
           " & ".join(head) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def s_operator_accuracy():
    df = read("operator_accuracy__*.csv")
    if df.empty:
        return ""
    blocks = []
    for (ds, bb), d in df.groupby(["dataset", "backbone"]):
        piv = d.groupby(["protocol", "operator"]).accuracy.mean().unstack(0) * 100
        sem = (d.groupby(["protocol", "operator"]).accuracy.sem().unstack(0)
               * 100).fillna(0)
        protos = [p for p in ("P1_dropin", "P2_retrained", "P3_agnostic")
                  if p in piv.columns]
        ops = [o for o in OPERATOR if o in piv.index]
        rows = [[OPERATOR[o]] + [f"{fmt(piv.loc[o, p])} ({fmt(sem.loc[o, p])})"
                                 for p in protos] for o in ops]
        blocks.append(longtable(
            ["Operator"] + [p.split("_")[0] for p in protos], rows,
            f"Accuracy (\\%) with standard error over folds, "
            f"{DATASET.get(ds, ds)} with {ENCODER.get(bb, bb)}.",
            f"tab:s-acc-{ds}-{bb}", "l" + "r" * len(protos)))
    return "\n\n".join(blocks)


def s_scorer_correlation():
    df = read("operator_scores__*.csv")
    if df.empty:
        return ""
    per = (df.groupby(["scorer", "dataset", "backbone", "fold"])
           .apply(lambda g: spearmanr(g.value, g.p1_accuracy).statistic,
                  include_groups=False).rename("rho").reset_index())
    piv = per.groupby(["scorer", "dataset", "backbone"]).rho.mean().unstack(
        ["dataset", "backbone"])
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    cols = list(piv.columns)
    head = ["Score"] + [f"{DATASET.get(d, d)[:4]}/{ENCODER.get(b, b)}"
                        for d, b in cols]
    rows = [[SCORER.get(s, s)] + [fmt(piv.loc[s, c], 2) for c in cols]
            for s in piv.index]
    return longtable(head, rows,
                     "Spearman rank correlation with drop-in accuracy, every "
                     "setting separately.", "tab:s-rho",
                     "l" + "r" * len(cols))


def s_selection():
    df = read("selection__*.csv")
    if df.empty:
        return ""
    piv = (df.groupby(["scorer", "dataset", "backbone"]).p1_regret.mean()
           * 100).unstack(["dataset", "backbone"])
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    cols = list(piv.columns)
    head = ["Chosen by"] + [f"{DATASET.get(d, d)[:4]}/{ENCODER.get(b, b)}"
                            for d, b in cols]
    rows = [[SCORER.get(s, {"identity": "no enhancement",
                            "best_fixed_p1": "oracle"}.get(s, s))]
            + [fmt(piv.loc[s, c]) for c in cols] for s in piv.index]
    return longtable(head, rows,
                     "Accuracy lost (percentage points) against the best "
                     "available operator, every setting separately.",
                     "tab:s-regret", "l" + "r" * len(cols))


def s_leakage():
    df = read("leakage.csv")
    if df.empty:
        return ""
    rows = [[DATASET.get(r.dataset, r.dataset), ENCODER.get(r.backbone, r.backbone),
             fmt(100 * r.group_blind), fmt(100 * r.group_aware),
             fmt(100 * r.inflation)] for r in df.itertuples()]
    return longtable(["Dataset", "Encoder", "Group-blind", "Group-aware",
                      "Inflation"], rows,
                     "Accuracy under a split that ignores groups against one "
                     "that keeps them disjoint.", "tab:s-leakage", "llrrr")


def s_beyond_probe():
    """The fine-tuned network and the zero-shot detector, operator by operator."""
    blocks = []
    ft = read("finetune__*.csv")
    if not ft.empty:
        piv = ft.pivot_table(index="operator", columns="protocol",
                             values="accuracy") * 100
        ops = [o for o in OPERATOR if o in piv.index]
        cols = [c for c in ("P1_dropin", "P2_retrained") if c in piv.columns]
        rows = [[OPERATOR[o]] + [fmt(piv.loc[o, c]) for c in cols] for o in ops]
        blocks.append(longtable(["Operator"] + [c.split("_")[0] for c in cols],
                                rows,
                                "Accuracy (\\%) of a ResNet-18 fine-tuned end to "
                                "end on ExDark. P2 was run for a subset of "
                                "operators.", "tab:s-finetune",
                                "l" + "r" * len(cols)))
    det = read("detection.csv")
    if not det.empty:
        base = det.set_index("operator").mAP50.get("identity", float("nan"))
        d = det.set_index("operator")
        ops = [o for o in OPERATOR if o in d.index]
        per_class = [c for c in d.columns if c.startswith("ap_")]
        rows = [[OPERATOR[o], fmt(100 * d.mAP50[o]),
                 fmt(100 * (d.mAP50[o] - base))] +
                [fmt(100 * d.loc[o, c]) for c in per_class[:6]] for o in ops]
        head = ["Operator", "mAP@0.5", "vs none"] + \
               [c.replace("ap_", "") for c in per_class[:6]]
        blocks.append(longtable(
            head, rows,
            "Zero-shot COCO detector on ExDark, with per-class average "
            "precision for the six most frequent categories.",
            "tab:s-detection", "l" + "r" * (len(head) - 1)))
    return "\n\n".join(blocks)


def s_sensitivity():
    df = read("sensitivity__*.csv")
    if df.empty:
        return ""
    rows = [[r.variant.replace("_", r"\_"), r.family, fmt(100 * r.accuracy),
             fmt(100 * r.identity), fmt(100 * r.delta)]
            for r in df.sort_values(["family", "variant"]).itertuples()]
    return longtable(["Variant", "Operator", "Accuracy", "No enhancement",
                      "Difference"], rows,
                     "Drop-in accuracy (\\%) when the tunable operators are "
                     "swept over a range that brackets their defaults "
                     "(ExDark, ResNet-50).", "tab:s-sensitivity", "llrrr")


def s_figures():
    """The figures the page limit pushed out of the manuscript."""
    want = [("fig_qualitative",
             "Enhancement the quality metric prefers and the deployed probe "
             "does not. Each row is one ExDark test image under four "
             "operators, annotated with the probe's prediction and the UIQM "
             "score of that version. Green border: correct. Orange: wrong."),
            ("fig_beyond_linear_probe",
             "The penalty outside the frozen-probe setting. Left: a ResNet-18 "
             "fine-tuned end to end on ExDark, under both protocols. Right: a "
             "COCO-pretrained detector, never retrained, scored on ExDark's "
             "boxes."),
            ("fig_scorer_ranking",
             "Rank correlation with drop-in accuracy for every label-free "
             "score, with the standard error over settings."),
            ("fig_shift_vs_accuracy",
             "Drop-in accuracy against embedding shift, one point per "
             "operator and encoder."),
            ("fig_severity",
             "Drop-in accuracy against the degradation level recorded with "
             "the data."),
            ("fig_selection_regret",
             "Accuracy lost against the best available operator, by what the "
             "choice was based on.")]
    out = []
    for name, cap in want:
        f = ROOT / "results" / "figures" / f"{name}.pdf"
        if not f.exists():
            continue
        out.append("\n".join([
            r"\begin{figure}[htbp]", r"\centering",
            rf"\includegraphics[width=\textwidth]{{../results/figures/{name}}}",
            rf"\caption{{{cap}}}\label{{fig:s-{name}}}",
            r"\end{figure}"]))
    return "\n\n".join(out)


def s_cost():
    df = read("cost.csv")
    if df.empty:
        return ""
    look = {**ENCODER, **OPERATOR}
    rows = [[look.get(r["name"], r["name"].replace("_", r"\_")), r.component,
             fmt(r.ms_per_image)] for _, r in
            df.sort_values(["component", "ms_per_image"]).iterrows()]
    return longtable(["Component", "Kind", "ms / image"], rows,
                     "Wall-clock cost per image on the CPU used for the study.",
                     "tab:s-cost", "llr")


HEADER = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.4cm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{placeins}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\usepackage{graphicx}
\title{Supplementary material\\
\large Enhancement front ends degrade recognition on degraded images,
and image quality metrics cannot tell}
\date{}
\begin{document}
\maketitle

This document reports every setting that the manuscript averages over: three
datasets crossed with the frozen encoders, one table per result.

"""


if __name__ == "__main__":
    parts = [HEADER,
             r"\FloatBarrier\section*{Accuracy per operator and protocol}", s_operator_accuracy(),
             r"\FloatBarrier\section*{What each label-free score can see}", s_scorer_correlation(),
             r"\FloatBarrier\section*{Choosing an operator without labels}", s_selection(),
             r"\FloatBarrier\section*{Beyond the frozen probe}", s_beyond_probe(),
             r"\FloatBarrier\section*{Operator hyper-parameters}", s_sensitivity(),
             r"\FloatBarrier\section*{Group-aware splitting}", s_leakage(),
             r"\FloatBarrier\section*{Computational cost}", s_cost(),
             r"\FloatBarrier\section*{Figures}", s_figures(),
             r"\end{document}"]
    OUT.write_text("\n\n".join(p for p in parts if p))
    print(f"wrote {OUT}")

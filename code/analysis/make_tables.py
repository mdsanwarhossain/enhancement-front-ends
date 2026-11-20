"""Generate every LaTeX table in the manuscript from results/tables.

Each table is written to paper/tables/*.tex and \\input{} by the manuscript, so
no number is ever typed by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from analysis.stats import holm                     # noqa: E402
from enhance.ops import FAMILY                      # noqa: E402
from features.backbones import BACKBONES, PRETRAIN  # noqa: E402

TAB = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "tables"

DATASET = {"turbid_fish": "Turbid fish", "exdark": "ExDark",
           "kvasir": "HyperKvasir"}
ENCODER = {"resnet50": "ResNet-50", "effnet_b0": "EfficientNet-B0",
           "dino_s16": "DINO ViT-S/16", "clip_b16": "CLIP ViT-B/16"}
OPERATOR = {"identity": "none", "hist_eq": "hist.\\ eq.", "clahe": "CLAHE",
            "gamma": "auto gamma", "ssr": "SSR", "msrcr": "MSRCR",
            "gray_world": "grey-world", "dcp": "DCP", "unsharp": "unsharp",
            "zero_dce": "Zero-DCE"}
SCORER = {
    "esp_mmd": "ESP (MMD)", "esp_frechet": "ESP (Fr\\'echet)",
    "esp_centroid": "ESP (centroid)", "esp_knn": "ESP ($k$NN)",
    "esp_mahalanobis": "ESP (Mahalanobis)",
    "geo_consensus": "neighbourhood consensus", "geo_hubness": "hubness",
    "geo_mutual_knn": "mutual $k$NN rate",
    "geo_effective_rank": "effective rank", "geo_twonn_dim": "intrinsic dim.",
    "geo_taq": "embedding geometry",
    "iqa_brisque": "BRISQUE", "iqa_niqe": "NIQE",
    "iqa_entropy": "entropy", "iqa_rms_contrast": "RMS contrast",
    "iqa_colourfulness": "colourfulness", "iqa_sharpness": "sharpness",
    "iqa_eme": "EME", "iqa_uciqe": "UCIQE", "iqa_uiqm": "UIQM",
    "iqa_psnr_in": "PSNR to input", "iqa_ssim_in": "SSIM to input",
    "iqa_ambe": "AMBE", "iqa_loe": "LOE",
}


def read(pattern):
    files = sorted(TAB.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print(f"  wrote paper/tables/{name}.tex")


def fmt(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:.{nd}f}"


# ---------------------------------------------------------------- table 1 ---
def table_datasets():
    rows = []
    for key in ("turbid_fish", "exdark", "kvasir"):
        f = ROOT / "data" / "index" / f"{key}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        sev = d.severity[d.severity >= 0]
        rows.append((DATASET[key], len(d), d.label.nunique(), d.group.nunique(),
                     f"{sev.nunique()}" if len(sev) else "--"))
    body = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Dataset & Images & Classes & Groups & Degradation levels \\",
            r"\midrule"]
    body += [f"{a} & {b} & {c} & {d} & {e} \\\\" for a, b, c, d, e in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    write("datasets", "\n".join(body))


# ---------------------------------------------------------------- table 2 ---
def table_operator_accuracy():
    df = read("operator_accuracy__*.csv")
    if df.empty:
        return
    datasets = [d for d in ("turbid_fish", "exdark", "kvasir")
                if d in set(df.dataset)]
    piv = {p: df[df.protocol == p]
           .groupby(["dataset", "operator"]).accuracy.mean().unstack(0) * 100
           for p in ("P1_dropin", "P2_retrained")}
    ops = ["identity"] + [o for o in OPERATOR if o != "identity"
                          and o in piv["P1_dropin"].index]
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{DATASET[d]}}}" for d in datasets)
    body = [r"\begin{tabular}{ll" + "rr" * len(datasets) + "}", r"\toprule",
            rf"Operator & Family & {head} \\",
            " & & " + " & ".join("P1 & P2" for _ in datasets) + r" \\",
            r"\midrule"]
    for op in ops:
        cells = []
        for d in datasets:
            for p in ("P1_dropin", "P2_retrained"):
                v = piv[p][d].get(op, np.nan) if d in piv[p] else np.nan
                base = piv[p][d].get("identity", np.nan) if d in piv[p] else np.nan
                s = fmt(v)
                if op != "identity" and np.isfinite(v) and np.isfinite(base) \
                        and v < base:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
        name = OPERATOR.get(op, op)
        if op == "identity":
            name = rf"\emph{{{name}}}"
        body.append(f"{name} & {FAMILY.get(op,'')} & " + " & ".join(cells) + r" \\")
        if op == "identity":
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("operator_accuracy", "\n".join(body))


# ---------------------------------------------------------------- table 3 ---
def table_scorer_correlation():
    from scipy.stats import spearmanr
    df = read("operator_scores__*.csv")
    if df.empty:
        return
    rows = []
    for scorer, d in df.groupby("scorer"):
        per = d.groupby(["dataset", "backbone", "fold"]).apply(
            lambda g: spearmanr(g.value, g.p1_accuracy).statistic,
            include_groups=False)
        by_ds = per.groupby("dataset").mean()
        rows.append(dict(scorer=scorer, overall=per.mean(), sd=per.std(),
                         n=len(per), **by_ds.to_dict()))
    out = pd.DataFrame(rows).sort_values("overall", ascending=False)
    # the main paper has room for a representative subset; the supplementary
    # carries all of them
    keep = ["esp_frechet", "esp_mmd", "esp_centroid", "geo_consensus",
            "iqa_ssim_in", "iqa_psnr_in", "iqa_brisque", "iqa_entropy",
            "iqa_uciqe", "iqa_uiqm", "iqa_eme", "iqa_rms_contrast",
            "iqa_niqe", "iqa_sharpness"]
    out = out[out.scorer.isin(keep)]
    datasets = [d for d in ("turbid_fish", "exdark", "kvasir") if d in out]
    body = [r"\begin{tabular}{l" + "r" * (len(datasets) + 1) + "}", r"\toprule",
            "Label-free score & " +
            " & ".join(DATASET[d] for d in datasets) + r" & All \\",
            r"\midrule"]
    for _, r in out.iterrows():
        cells = " & ".join(fmt(r.get(d, np.nan), 2) for d in datasets)
        name = SCORER.get(r.scorer, r.scorer)
        if r.scorer.startswith("esp"):
            name = rf"\textbf{{{name}}}"
        body.append(f"{name} & {cells} & {fmt(r.overall, 2)} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("scorer_correlation", "\n".join(body))


# ---------------------------------------------------------------- table 4 ---
def table_selection():
    df = read("selection__*.csv")
    if df.empty:
        return
    keep = ["identity", "esp_mmd", "esp_frechet", "esp_centroid"]
    iqa = df[df.scorer.str.startswith("iqa_")]
    if len(iqa):
        keep.append(iqa.groupby("scorer").p1_regret.mean().idxmin())
        keep.append(iqa.groupby("scorer").p1_regret.mean().idxmax())
    keep.append("best_fixed_p1")
    datasets = [d for d in ("turbid_fish", "exdark", "kvasir")
                if d in set(df.dataset)]
    body = [r"\begin{tabular}{l" + "r" * (len(datasets) + 1) + "}", r"\toprule",
            "Operator chosen by & " + " & ".join(DATASET[d] for d in datasets)
            + r" & Mean \\", r"\midrule"]
    for sc in dict.fromkeys(keep):
        d = df[df.scorer == sc]
        if d.empty:
            continue
        per = d.groupby("dataset").p1_regret.mean() * 100
        name = SCORER.get(sc, {"identity": "no enhancement",
                               "best_fixed_p1": "oracle (best fixed)"}.get(sc, sc))
        cells = " & ".join(fmt(per.get(x, np.nan)) for x in datasets)
        body.append(f"{name} & {cells} & {fmt(per.mean())} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("selection", "\n".join(body))


# ---------------------------------------------------------------- table 5 ---
def table_cost():
    f = TAB / "cost.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    ops = d[d.component == "operator"].sort_values("ms_per_image")
    enc = d[d.component == "encoder"].sort_values("ms_per_image")
    body = [r"\begin{tabular}{lrlr}", r"\toprule",
            r"Operator & ms/image & Encoder & ms/image \\", r"\midrule"]
    for i in range(max(len(ops), len(enc))):
        a = (f"{OPERATOR.get(ops.name.iloc[i], ops.name.iloc[i])} & "
             f"{ops.ms_per_image.iloc[i]:.1f}") if i < len(ops) else " & "
        b = (f"{ENCODER.get(enc.name.iloc[i], enc.name.iloc[i])} & "
             f"{enc.ms_per_image.iloc[i]:.1f}") if i < len(enc) else " & "
        body.append(f"{a} & {b} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("cost", "\n".join(body))


if __name__ == "__main__":
    for fn in (table_datasets, table_operator_accuracy, table_scorer_correlation,
               table_selection, table_cost):
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {type(exc).__name__}: {exc}")

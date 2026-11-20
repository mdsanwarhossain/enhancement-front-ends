"""Build one flat image index per dataset.

Columns: path, dataset, label, group, severity, split

``group`` is the unit that must never straddle a split (a fish individual, a
patient/procedure, a scene). ``severity`` is the dataset's own degradation
annotation where one exists, so degradation is measured, not simulated.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, INDEX = ROOT / "data" / "raw", ROOT / "data" / "index"
IMG = {".jpg", ".jpeg", ".png", ".bmp"}


def _imgs(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG)


# --------------------------------------------------------------------------
def turbid_fish() -> pd.DataFrame:
    """`Turbidity <k>/<split>/<species>/<SPECIES>.<TURB>.<DEPTH> (n).jpg`.

    The release ships a spreadsheet of measured conditions per image, so the
    degradation axis here is nephelometric turbidity in NTU rather than an
    ordinal label: 0-15 (clear), 16-25, 26-35, 36-50. A ``group`` is one
    continuous recording - one species at one turbidity and one depth, 90
    near-duplicate frames - which is the unit that must not straddle a split.
    """
    base = RAW / "turbid_fish"
    meta = {}
    for x in base.glob("*.xlsx"):
        m = pd.read_excel(x).rename(columns={
            "Image": "image", "NTU": "ntu", "depth (cm)": "depth_cm",
            "pH": "ph", "Temperature (\u00b0C)": "temp_c"})
        meta = {k.lower(): v for k, v in
                zip(m.image, m[["ntu", "depth_cm", "ph", "temp_c"]]
                    .to_numpy(float))}
        break
    rows = []
    for p in _imgs(base):
        parts = p.relative_to(base).parts
        if len(parts) < 4:
            continue
        turb, split, species = parts[0], parts[1], parts[2]
        code = p.stem.split("(")[0].strip()          # SPECIES.TURBIDITY.DEPTH
        ntu, depth, ph, temp = meta.get(p.name.lower(), (np.nan,) * 4)
        rows.append(dict(path=str(p), dataset="turbid_fish", label=species,
                         group=code,
                         severity=int(re.search(r"(\d+)", turb).group(1)),
                         ntu=ntu, depth_cm=depth, ph=ph, temp_c=temp,
                         split={"Train": "train", "Test": "test",
                                "Eva": "val"}.get(split, split.lower())))
    return pd.DataFrame(rows)


EXDARK_CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
                  "Cup", "Dog", "Motorbike", "People", "Table"]
# ExDark annotates a lighting *type*, not an ordered severity, so this column
# stratifies results rather than ordering them.
EXDARK_LIGHT = ["low", "ambient", "object", "single", "weak", "strong",
                "screen", "window", "shadow", "twilight"]


def exdark() -> pd.DataFrame:
    """Flat image folder plus the official `imageclasslist.txt`."""
    base = RAW / "exdark"
    meta = {}
    for line in (base / "imageclasslist.txt").read_text(errors="ignore").splitlines()[1:]:
        f = line.split()
        if len(f) >= 5:
            meta[f[0].lower()] = (int(f[1]), int(f[2]), int(f[4]))
    rows = []
    for p in _imgs(base):
        m = meta.get(p.name.lower())
        if m is None:
            continue
        cls, light, split = m
        rows.append(dict(path=str(p), dataset="exdark",
                         label=EXDARK_CLASSES[cls - 1], group=p.stem,
                         severity=light - 1,
                         split={1: "train", 2: "val", 3: "test"}.get(split, "pool")))
    return pd.DataFrame(rows)


KVASIR_MIN_PER_CLASS = 100


def kvasir() -> pd.DataFrame:
    """HyperKvasir labelled images: one folder per finding, inside split zips.

    Seven of the 23 findings have fewer than 100 images (9 in one case), which
    makes a per-class accuracy estimate meaningless and lets a single image
    swing balanced accuracy. We keep the 16 findings above that floor, 10,479
    of 10,662 images, and say so.
    """
    base = RAW / "kvasir"
    rows = []
    for p in _imgs(base):
        rel = p.relative_to(base).parts
        split = next((s for s in rel if s in {"train", "valid", "test"}), "pool")
        cls = p.parent.name
        rows.append(dict(path=str(p), dataset="kvasir", label=cls, group=p.stem,
                         severity=-1,
                         split={"valid": "val"}.get(split, split)))
    df = pd.DataFrame(rows)
    keep = df.label.value_counts()
    keep = keep[keep >= KVASIR_MIN_PER_CLASS].index
    return df[df.label.isin(keep)].reset_index(drop=True)


BUILDERS = {"turbid_fish": turbid_fish, "exdark": exdark, "kvasir": kvasir}


if __name__ == "__main__":
    import sys
    INDEX.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or list(BUILDERS)):
        df = BUILDERS[name]()
        if df.empty:
            print(f"{name}: NO IMAGES FOUND"); continue
        df.to_csv(INDEX / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} images, {df.label.nunique()} classes, "
              f"{df.group.nunique()} groups, severity={sorted(df.severity.unique())[:12]}, "
              f"splits={df.split.value_counts().to_dict()}")

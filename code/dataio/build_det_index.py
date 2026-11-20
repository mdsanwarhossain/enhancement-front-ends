"""Index the ExDark detection subset: images with their bounding boxes.

The release carries YOLO-normalised boxes, which we keep normalised so they
survive the resize into the working cache.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "exdark_det" / "data"
INDEX = ROOT / "data" / "index"

CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup",
           "Dog", "Motorbike", "People", "Table"]

# the same twelve categories in the COCO label space the detector was trained on
COCO_ID = {"Bicycle": 2, "Boat": 9, "Bottle": 44, "Bus": 6, "Car": 3,
           "Cat": 17, "Chair": 62, "Cup": 47, "Dog": 18, "Motorbike": 4,
           "People": 1, "Table": 67}


def main():
    rows = []
    for split in ("train", "valid", "test"):
        for img in sorted((RAW / split / "images").glob("*.jpg")):
            lab = RAW / split / "labels" / f"{img.stem}.txt"
            if not lab.exists():
                continue
            boxes = []
            for line in lab.read_text().split("\n"):
                f = line.split()
                if len(f) == 5:
                    c, cx, cy, w, h = int(f[0]), *map(float, f[1:])
                    boxes.append((c, cx - w / 2, cy - h / 2, cx + w / 2,
                                  cy + h / 2))
            if not boxes:
                continue
            rows.append(dict(
                path=str(img), dataset="exdark_det",
                # the original ExDark id survives the rename as the prefix
                group=img.stem.split("_jpg.rf.")[0].split("_png.rf.")[0],
                split=split if split != "valid" else "val",
                n_boxes=len(boxes),
                boxes=";".join(f"{c},{a:.6f},{b:.6f},{d:.6f},{e:.6f}"
                               for c, a, b, d, e in boxes)))
    df = pd.DataFrame(rows)
    df["label"] = df.boxes.str.split(",").str[0].map(
        lambda i: CLASSES[int(i)])            # majority-free: first box's class
    df["severity"] = -1
    INDEX.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX / "exdark_det.csv", index=False)
    print(f"exdark_det: {len(df)} images, {df.n_boxes.sum()} boxes, "
          f"{df.label.nunique()} classes")


if __name__ == "__main__":
    main()

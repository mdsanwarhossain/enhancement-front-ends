"""Fetch the ExDark detection subset: the same images with bounding boxes."""
from pathlib import Path
from huggingface_hub import snapshot_download

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "exdark_det"
RAW.mkdir(parents=True, exist_ok=True)
snapshot_download("dronefreak/ExDark", repo_type="dataset",
                  local_dir=str(RAW), max_workers=8,
                  allow_patterns=["data/**"])
n = sum(1 for _ in RAW.rglob("*.jpg"))
print(f"exdark_det: {n} images with labels")

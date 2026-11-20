"""Download the three degraded-vision benchmarks from the Hugging Face Hub."""
import sys, zipfile, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

REPOS = {
    "turbid_fish": "mieraehh/5_Malaysian_Fish_Species_in_Turbid_Water",
    "exdark":      "SatwikKambham/ex-dark",
    "kvasir":      "sahilur/hyper-kvasir-labeled-images",
}


def fetch(key: str) -> Path:
    dest = RAW / key
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{key}] downloading {REPOS[key]} -> {dest}", flush=True)
    snapshot_download(REPOS[key], repo_type="dataset", local_dir=str(dest),
                      max_workers=8)
    for z in sorted(dest.rglob("*.zip")):
        out = z.with_suffix("")
        if out.exists():
            continue
        print(f"[{key}] extracting {z.name}", flush=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
        z.unlink()                      # reclaim disk once extracted
    n = sum(1 for _ in dest.rglob("*") if _.suffix.lower() in
            {".jpg", ".jpeg", ".png"})
    print(f"[{key}] done: {n} images, {sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())/1e9:.2f} GB",
          flush=True)
    return dest


if __name__ == "__main__":
    keys = sys.argv[1:] or list(REPOS)
    for k in keys:
        fetch(k)

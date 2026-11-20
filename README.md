# Enhancement front ends degrade recognition on degraded images

Code, result files and figures for the paper of the same name, by
Md Sanwar Hossain and Mohammad Mohiuddin.

Low-light, underwater and endoscopic images are routinely passed through an
enhancement operator before a recogniser sees them, and the operator is chosen
by how the output looks or by a no-reference quality metric. This repository
contains everything needed to check what that costs when the recogniser is
left unchanged, which is what a deployed system does.

Everything runs on CPU. The experiments were produced on a 22-thread laptop
with no GPU; the full grid takes about a day.

## What is here

```
code/
  dataio/      dataset download, index construction, image cache
  enhance/     the ten enhancement operators
  iqa/         thirteen pixel-domain quality metrics
  features/    frozen encoders, the embedding shift penalty, label-free baselines
  experiments/ protocols, embeddings, fine-tuning, detection, sensitivity
  analysis/    figures, tables, LaTeX macros, bibliography, preflight checks
  run_0*.sh    the pipeline, in order; run_all.sh runs the lot
data/index_public/   per-image labels, groups, splits and measured conditions
results/tables/      every result as a tidy CSV
results/figures/     every figure
paper/               the generated numbers, tables and bibliography
```

The manuscript itself is withheld until publication. Everything it quotes is
in `paper/numbers.tex` and `paper/tables/`, both generated from
`results/tables/` by `code/analysis/make_numbers.py` and `make_tables.py`, so
any number in the paper can be traced to the result file behind it.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python bash code/run_all.sh
```

The stages are independent and resumable; each script skips work already on
disk. ExDark needs one extra file, the official `imageclasslist.txt`, placed
at `data/raw/exdark/imageclasslist.txt`; it is in the
[ExDark repository](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset).

## Datasets

| Domain | Dataset | Licence |
|---|---|---|
| Underwater | 5 Malaysian Fish Species in Varying Depth and Turbidity (doi:10.21227/gd5d-ns10) | CC BY 4.0 |
| Low light | Exclusively Dark (ExDark), classification and detection splits | see upstream |
| Endoscopy | HyperKvasir, labelled images | CC BY 4.0 |

None is redistributed here; `code/dataio/download.py` fetches them.
`data/index_public/` holds the per-image metadata we derived from them —
labels, recording groups, splits, and the measured turbidity, depth, pH and
temperature of the underwater set — so a reader can audit the design and the
splits without downloading 10 GB of images. Rows are in the same order the
pipeline uses, and `row` is the key that the result files refer to.

## Two things worth knowing if you build on this

**Flush-to-zero.** Pretrained weights contain subnormal floats. Without
`torch.set_flush_denormal(True)` the CPU convolution path falls off a cliff —
a factor of thirty on Zero-DCE here. It is set in `code/features/backbones.py`
and in every data-loader worker.

**Group-aware splitting.** The underwater dataset contributes ninety
near-duplicate frames per recording, and its released train/test split places
recordings on both sides. A split that ignores recordings reports about 34
percentage points of accuracy that is not there.
`code/experiments/leakage.py` measures it.

## Citation

The paper reference will be added here once it is published.

## Licence

MIT, see `LICENSE`.

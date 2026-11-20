#!/usr/bin/env bash
# Everything the manuscript reads: figures, tables, macros, supplementary.
set -e
cd "$(dirname "$0")/.."
V=${PYTHON:-python}
$V code/analysis/qualitative.py --dataset exdark --rows 1 --metric uiqm
$V code/analysis/cost.py
$V code/analysis/figures.py
$V code/analysis/make_tables.py
$V code/analysis/make_numbers.py
$V code/analysis/make_supplementary.py

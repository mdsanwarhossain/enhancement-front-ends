#!/usr/bin/env bash
# Fetch the three datasets, build the indexes, cache images at a fixed size.
set -e
cd "$(dirname "$0")/.."
V=${PYTHON:-python}
$V code/dataio/download.py
$V code/dataio/download_det.py        # ExDark with bounding boxes
$V code/dataio/build_index.py
$V code/dataio/build_det_index.py
$V code/dataio/prepare.py

#!/usr/bin/env bash
# Pixel-domain quality metrics for every (image, operator) pair.
set -e
cd "$(dirname "$0")/.."
V=${PYTHON:-python}
for d in turbid_fish exdark kvasir; do
  $V code/experiments/iqa_pass.py --dataset "$d" --workers "${WORKERS:-18}"
done

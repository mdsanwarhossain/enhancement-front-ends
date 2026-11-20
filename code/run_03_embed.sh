#!/usr/bin/env bash
# The embedding grid: every dataset, operator and frozen encoder.
set -e
cd "$(dirname "$0")/.."
V=${PYTHON:-python}
for b in resnet50 dino_s16 effnet_b0; do
  for d in turbid_fish exdark kvasir; do
    $V code/experiments/embed.py --dataset "$d" --backbone "$b" --ops all \
       --batch 32 --workers "${WORKERS:-14}" --threads "${THREADS:-7}"
  done
done

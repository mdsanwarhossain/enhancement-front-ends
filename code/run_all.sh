#!/usr/bin/env bash
# The whole study, in order. Roughly a day on a 22-thread CPU with no GPU.
set -e
cd "$(dirname "$0")"
for s in run_01_data.sh run_02_metrics.sh run_03_embed.sh \
         run_04_analysis.sh run_05_outputs.sh; do
  echo "=== $s ==="
  bash "$s"
done

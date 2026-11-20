#!/usr/bin/env bash
# Protocols, label-free scores, selection; then the experiments that stand
# outside the frozen-probe grid.
set -e
cd "$(dirname "$0")/.."
V=${PYTHON:-python}
JOBS=${JOBS:-5}; THREADS=${THREADS:-4}

ready=""
for b in resnet50 dino_s16 effnet_b0; do
  for d in turbid_fish exdark kvasir; do
    [ "$(ls "data/emb/$d/$b" 2>/dev/null | wc -l)" -eq 10 ] && ready="$ready $d:$b"
  done
done
printf '%s\n' $ready | xargs -P "$JOBS" -I{} bash -c '
  p="{}"; d="${p%%:*}"; b="${p##*:}"
  OMP_NUM_THREADS='"$THREADS"' '"$V"' -W ignore code/experiments/run_main.py \
    --dataset "$d" --backbone "$b" --folds 3 \
    > "results/logs/main_${d}_${b}.log" 2>&1 && echo "done $d/$b" || echo "FAILED $d/$b"'

$V code/experiments/finetune.py --dataset exdark \
   --p2-ops identity,clahe,msrcr,gamma,hist_eq --epochs 6
$V code/experiments/detect.py --ops all --limit 3000
$V code/experiments/sensitivity.py --dataset exdark --backbone resnet50
$V code/experiments/leakage.py --folds 3

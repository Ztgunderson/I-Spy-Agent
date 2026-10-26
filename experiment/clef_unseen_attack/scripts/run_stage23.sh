#!/usr/bin/env bash
# Stage 2 or 3 sweep + evaluation on one GPU, resumable (score cache is append-only).
# Usage: run_stage23.sh stage2|stage3 [--zero-shot]
# Run from the repo root, detached:  nohup bash experiment/clef_unseen_attack/scripts/run_stage23.sh stage2 --zero-shot > runs/clef_unseen_attack/stage2.log 2>&1 &
set -euo pipefail
stage="$1"; zero="${2:-}"
export CLEF_MODEL_ID="${CLEF_MODEL_ID:-$HOME/lab/models/clef-flash-hf}"
mod=experiment.clef_unseen_attack
sizes=(50 100 200 400)

if [[ "$zero" == "--zero-shot" ]]; then
  echo "== $(date -Is) zero-shot eval"
  uv run python -m $mod.evaluate --zero-shot
fi
for size in "${sizes[@]}"; do
  dir="runs/clef_unseen_attack/$stage/size_$size"
  if [[ ! -f "$dir/metrics.json" ]]; then
    echo "== $(date -Is) train $stage size $size"
    uv run python -m $mod.train "$stage" --sizes "$size"
  fi
  echo "== $(date -Is) eval $stage size $size"
  uv run python -m $mod.evaluate --stage-dir "$dir"
done
echo "== $(date -Is) done $stage"

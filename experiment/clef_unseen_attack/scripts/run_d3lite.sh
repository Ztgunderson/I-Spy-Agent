#!/usr/bin/env bash
# d3-lite fine-tune (stage2_d3lite) and its Qwen3.5-0.8B-Base control (stage3_qwen08b),
# same recipe as Clef Stage 2. Waits for the Clef queue to finish and the GPU to be free.
# Order puts the most informative sizes first.
#
#   nohup bash experiment/clef_unseen_attack/scripts/run_d3lite.sh > runs/clef_unseen_attack/d3lite.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."
export D3_MODEL_ID="${D3_MODEL_ID:-$HOME/lab/models/d3-lite}"
export D3_BASE_MODEL="${D3_BASE_MODEL:-$HOME/lab/models/qwen3.5-0.8b-base}"
mod=experiment.clef_unseen_attack
free=0
until (( free >= 2 )); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if ! pgrep -f "[r]esequence.sh|[r]un_stage23.sh" > /dev/null && (( used < 2048 )); then
    free=$((free + 1))
  else
    free=0
  fi
  sleep 60
done

echo "== $(date -Is) zero-shot d3-lite (NF4)"
uv run python -m $mod.evaluate --zero-shot --backbone d3
auc=$(python3 -c "import csv; print(next(r['auc'] for r in csv.DictReader(open('$mod'.replace('.', '/') + '/results/stage23_metrics_zeroshot_d3.csv')) if r['section'] == 'auc' and r['eval'] == 'u3'))")
echo "== zero-shot d3-lite U3 AUROC $auc (GGUF 0.786)"
if ! python3 -c "import sys; sys.exit(0 if abs(float('$auc') - 0.786) <= 0.05 else 1)"; then
  echo "== STOP: adapter does not reproduce the GGUF model"; exit 1
fi

run() {  # stage size
  local dir="runs/clef_unseen_attack/$1/size_$2"
  if [[ ! -f "$dir/metrics.json" ]]; then
    echo "== $(date -Is) train $1 size $2"
    uv run python -m $mod.train "$1" --sizes "$2" || return
  fi
  echo "== $(date -Is) eval $1 size $2"
  uv run python -m $mod.evaluate --stage-dir "$dir"
}
run stage2_d3lite 200
run stage2_d3lite 400
run stage3_qwen08b 200
run stage3_qwen08b 400
run stage2_d3lite 50
run stage2_d3lite 100
run stage3_qwen08b 50
run stage3_qwen08b 100
echo "== $(date -Is) d3lite done"

#!/usr/bin/env bash
# Zero-shot size ladder on the Jetson: for each model in models_scaling.tsv,
# download -> llama-server -> probe -> score eval splits -> latency -> delete.
# Resumable: a model with runs/scaling/<name>.done (or .skip) is not rerun.
# The next model downloads in the background while the current one scores.
#
#   LLAMA_SERVER=~/lab/tools/llama.cpp-b11889/src/build/bin/llama-server \
#     nohup bash experiment/clef_unseen_attack/scripts/jetson_queue.sh > runs/scaling/queue.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/lab/tools/llama.cpp-b11889/src/build/bin/llama-server}"
MODELS_TSV="experiment/clef_unseen_attack/models_scaling.tsv"
GGUF_DIR="${GGUF_DIR:-$HOME/stegowatch-gguf}"
OUT="runs/scaling"
PORT=8099
mkdir -p "$GGUF_DIR" "$OUT"

log() { echo "== $(date -Is) $*"; }

fetch() {  # repo file -> path; curl resumes a partial download
  local repo=$1 file=$2
  curl -sfL -C - --retry 3 -o "$GGUF_DIR/$file" "https://huggingface.co/$repo/resolve/main/$file"
}

mapfile -t ROWS < <(tail -n +2 "$MODELS_TSV")
prefetch_pid=""
for i in "${!ROWS[@]}"; do
  IFS=$'\t' read -r name params family backbone repo file splits <<< "${ROWS[$i]}"
  if [[ -e "$OUT/$name.done" || -e "$OUT/$name.skip" ]]; then
    log "$name already finished"; continue
  fi

  [[ -n "$prefetch_pid" ]] && wait "$prefetch_pid"
  if [[ ! -s "$GGUF_DIR/$file" ]] || ! fetch "$repo" "$file"; then
    fetch "$repo" "$file" || { log "$name download failed"; echo download > "$OUT/$name.skip"; continue; }
  fi
  # Start the next download now, so it overlaps with scoring.
  prefetch_pid=""
  if (( i + 1 < ${#ROWS[@]} )); then
    IFS=$'\t' read -r _ _ _ _ next_repo next_file _ <<< "${ROWS[$((i + 1))]}"
    fetch "$next_repo" "$next_file" & prefetch_pid=$!
  fi

  log "$name ($params B, $backbone, $file, $splits) serving"
  "$LLAMA_SERVER" -m "$GGUF_DIR/$file" -ngl 99 -c 16384 -np 4 -b 4096 -ub 4096 \
    --host 127.0.0.1 --port $PORT > "$OUT/$name.server.log" 2>&1 &
  server=$!
  for _ in $(seq 300); do
    curl -sf "127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $server 2> /dev/null || break
    sleep 1
  done

  if python3 -m experiment.clef_unseen_attack.scaling probe --name "$name" --url "http://127.0.0.1:$PORT"; then
    started=$(date +%s)
    python3 -m experiment.clef_unseen_attack.scaling score --name "$name" --splits "$splits" --url "http://127.0.0.1:$PORT" \
      && python3 -m experiment.clef_unseen_attack.scaling latency --name "$name" --url "http://127.0.0.1:$PORT" \
      && echo "$(( $(date +%s) - started ))" > "$OUT/$name.done"
    log "$name finished in $(( $(date +%s) - started )) s"
  else
    log "$name probe failed; skipping (see $OUT/$name.server.log)"
    echo probe > "$OUT/$name.skip"
  fi
  kill $server 2> /dev/null; wait $server 2> /dev/null
  rm -f "$GGUF_DIR/$file"
done
log "queue done"

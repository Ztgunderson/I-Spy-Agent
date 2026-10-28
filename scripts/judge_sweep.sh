#!/usr/bin/env bash
# Self-monitoring test on NARCBench-Stego (player view): the two writer models, and a smaller
# sibling, as prompted yes/no judges. Same Jetson/tunnel pattern as narc_sweep.sh (:8100).
#
#   nohup bash scripts/judge_sweep.sh [name ...] > runs/narc_scaling/judge.log 2>&1 &
# LIMIT=8 scores only 8 players per model (smoke test; the cache keeps them).
# Hosts and paths: scripts/config.sh (.env); served with JETSON_LLAMA_SERVER_ALT.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh
need JETSON
# name  repo  file  template
MODELS=(
  "gpt-oss-20b-judge unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf gpt-oss"
  "qwen3-32b-judge Qwen/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf qwen3"
  "qwen3-8b-judge Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf qwen3"
)
mkdir -p runs/narc_scaling
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 8100:127.0.0.1:8100 "$JETSON" &
tunnel=$!
trap 'kill $tunnel 2> /dev/null' EXIT

for row in "${MODELS[@]}"; do
  read -r name repo file template <<< "$row"
  if (( $# )) && [[ ! " $* " == *" $name "* ]]; then continue; fi
  echo "== $(date -Is) $name"
  ssh -o BatchMode=yes "$JETSON" "mkdir -p $JETSON_DIR && cd $JETSON_DIR \
    && curl -sfL -C - --retry 3 -o $file https://huggingface.co/$repo/resolve/main/$file \
    && (nohup $JETSON_LLAMA_SERVER_ALT -m $file -ngl 99 -c 32768 -np 4 -b 4096 -ub 1024 \
        --host 127.0.0.1 --port 8100 > $JETSON_DIR/$name.server.log 2>&1 < /dev/null & echo \$! > $JETSON_DIR/server.pid) \
    && for i in \$(seq 900); do curl -sf 127.0.0.1:8100/health > /dev/null && break; \
        kill -0 \$(cat $JETSON_DIR/server.pid) 2> /dev/null || break; sleep 1; done" < /dev/null
  uv run python -m stegowatch.run_llm_judge --view player --template "$template" \
    --url http://127.0.0.1:8100 --model-id "$name" --workers 4 --limit "${LIMIT:-0}" \
    || echo "== $name scoring failed"
  # a smoke test keeps the GGUF so the full run does not download it again
  keep=""; [[ -n "${LIMIT:-}" ]] && keep="true ||"
  ssh -o BatchMode=yes "$JETSON" "kill \$(cat $JETSON_DIR/server.pid) 2> /dev/null; sleep 2; \
    $keep rm -f $JETSON_DIR/$file" < /dev/null
done
echo "== $(date -Is) judge sweep done"

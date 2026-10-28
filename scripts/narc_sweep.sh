#!/usr/bin/env bash
# Size ladder on NARCBench-Stego (player view): each small decision model is served on the
# Jetson (:8100, its own GGUF folder, beside the main queue on :8099) and scored from here
# through an SSH tunnel. Resumable: run_clef skips players already in the model's cache.
#
#   nohup bash scripts/narc_sweep.sh [name ...] > runs/narc_scaling/sweep.log 2>&1 &
# WRITERS=a,b picks transcript sets (default the two released; generated ones: load.GEN_MODELS).
# With names, only those rows run, in that order. ALT=1 serves with JETSON_LLAMA_SERVER_ALT (Clef, Vega).
# Hosts and paths: scripts/config.sh (.env).
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh
need JETSON
VIEW="${VIEW:-player}"
SERVER=$JETSON_LLAMA_SERVER; [[ -n "${ALT:-}" ]] && SERVER=$JETSON_LLAMA_SERVER_ALT
# name  repo  file
MODELS=(
  "kev-0.8b ggml-org/Kev-0.8B-GGUF Kev-0.8B-Q8_0.gguf"
  "kev-4b ggml-org/Kev-4B-GGUF Kev-4B-Q4_K_M.gguf"
  "d3-lite ggml-org/vllm-d3-lite-GGUF vllm-d3-lite-Q8_0.gguf"
  "d3-edge ggml-org/vllm-d3-edge-GGUF vllm-d3-edge-Q8_0.gguf"
  "d3-nano ggml-org/vllm-d3-nano-GGUF vllm-d3-nano-Q8_0.gguf"
  "d3-mini ggml-org/vllm-d3-mini-GGUF vllm-d3-mini-Q4_K_M.gguf"
  "kev-9b ggml-org/Kev-9B-GGUF Kev-9B-Q4_K_M.gguf"
  "d3-flash ggml-org/vllm-d3-flash-GGUF vllm-d3-flash-Q4_K_M.gguf"
  "pplx-decider-flash Jakevin/pplx-decider-flash-v1.2-GGUF pplx-decider-flash-v1.2-Q4_K_M.gguf"
  "d1-3b LiquidAI/d1-3B-GGUF d1-3B-Q4_K_M.gguf"
  "aurelis-2b Leptos-ai/Aurelis-2B-GGUF Aurelis-2B-Q8_0.gguf"
  "clef ggml-org/Clef-GGUF Clef-Q4_K_M.gguf"
  "kev-27b kdercksen/Kev-27B-GGUF Kev-27B-Q4_K_M.gguf"
  "d3 ggml-org/vllm-d3-GGUF vllm-d3-Q4_K_M.gguf"
  "vega-27b vllm-sr/Decision-2.0-Vega-27B-GGUF Decision-2.0-Vega-27B-Q4_K_M.gguf"
)
mkdir -p runs/narc_scaling
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 8100:127.0.0.1:8100 "$JETSON" &
tunnel=$!
trap 'kill $tunnel 2> /dev/null' EXIT

# With names, rows run in the order given; otherwise in table order.
order=("$@")
if (( ! $# )); then for row in "${MODELS[@]}"; do order+=("${row%% *}"); done; fi
for want in "${order[@]}"; do
  row=""
  for r in "${MODELS[@]}"; do [[ ${r%% *} == "$want" ]] && row=$r; done
  [[ -n $row ]] || { echo "== unknown model $want"; continue; }
  read -r name repo file <<< "$row"
  echo "== $(date -Is) $name"
  ssh -o BatchMode=yes "$JETSON" "mkdir -p $JETSON_DIR && cd $JETSON_DIR \
    && curl -sfL -C - --retry 3 -o $file https://huggingface.co/$repo/resolve/main/$file \
    && (nohup $SERVER -m $file -ngl 99 -c 16384 -np 4 -b 4096 -ub 4096 \
        --host 127.0.0.1 --port 8100 > $JETSON_DIR/$name.server.log 2>&1 < /dev/null & echo \$! > $JETSON_DIR/server.pid) \
    && for i in \$(seq 600); do curl -sf 127.0.0.1:8100/health > /dev/null && break; \
        kill -0 \$(cat $JETSON_DIR/server.pid) 2> /dev/null || break; sleep 1; done" < /dev/null
  uv run python -m stegowatch.run_clef --view "$VIEW" --url http://127.0.0.1:8100 \
    --model-id "$name" --workers 4 --writers "${WRITERS:-qwen3_32b,gpt_oss_20b}" || echo "== $name scoring failed"
  ssh -o BatchMode=yes "$JETSON" "kill \$(cat $JETSON_DIR/server.pid) 2> /dev/null; sleep 2; rm -f $JETSON_DIR/$file" < /dev/null
done
echo "== $(date -Is) sweep done"

#!/usr/bin/env bash
# Size ladder on NARCBench-Stego (player view): each small decision model is served on the
# Jetson (:8100, its own GGUF folder, beside the main queue on :8099) and scored from here
# through an SSH tunnel. Resumable: run_clef skips players already in the model's cache.
#
#   nohup bash scripts/narc_sweep.sh > runs/narc_scaling/sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JETSON=jetson@100.76.200.13
VIEW="${VIEW:-player}"
# name  repo  file
MODELS=(
  "kev-0.8b ggml-org/Kev-0.8B-GGUF Kev-0.8B-Q8_0.gguf"
  "kev-4b ggml-org/Kev-4B-GGUF Kev-4B-Q4_K_M.gguf"
  "d3-lite ggml-org/vllm-d3-lite-GGUF vllm-d3-lite-Q8_0.gguf"
  "d3-edge ggml-org/vllm-d3-edge-GGUF vllm-d3-edge-Q8_0.gguf"
  "d3-nano ggml-org/vllm-d3-nano-GGUF vllm-d3-nano-Q8_0.gguf"
  "d3-mini ggml-org/vllm-d3-mini-GGUF vllm-d3-mini-Q4_K_M.gguf"
  "kev-9b ggml-org/Kev-9B-GGUF Kev-9B-Q4_K_M.gguf"
)
mkdir -p runs/narc_scaling
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 8100:127.0.0.1:8100 "$JETSON" &
tunnel=$!
trap 'kill $tunnel 2> /dev/null' EXIT

for row in "${MODELS[@]}"; do
  read -r name repo file <<< "$row"
  echo "== $(date -Is) $name"
  ssh -o BatchMode=yes "$JETSON" "mkdir -p ~/narc-gguf && cd ~/narc-gguf \
    && curl -sfL -C - --retry 3 -o $file https://huggingface.co/$repo/resolve/main/$file \
    && (nohup ~/lab/tools/llama.cpp/build/bin/llama-server -m $file -ngl 99 -c 16384 -np 4 -b 4096 -ub 4096 \
        --host 127.0.0.1 --port 8100 > ~/narc-gguf/$name.server.log 2>&1 < /dev/null & echo \$! > ~/narc-gguf/server.pid) \
    && for i in \$(seq 120); do curl -sf 127.0.0.1:8100/health > /dev/null && break; sleep 1; done" < /dev/null
  uv run python -m stegowatch.run_clef --view "$VIEW" --url http://127.0.0.1:8100 \
    --model-id "$name" --workers 4 || echo "== $name scoring failed"
  ssh -o BatchMode=yes "$JETSON" "kill \$(cat ~/narc-gguf/server.pid) 2> /dev/null; sleep 2; rm -f ~/narc-gguf/$file" < /dev/null
done
echo "== $(date -Is) sweep done"

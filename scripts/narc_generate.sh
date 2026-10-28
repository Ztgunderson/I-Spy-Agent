#!/usr/bin/env bash
# New NARCBench-Stego writers: play the same 20 shoes (collusion + control, same seeds and cards)
# with the LLM behind our decision models, using NARCBench's own generation/stego.py unchanged.
# Local writers run on the 2060 (:8096); qwen3.8-27b runs on the Jetson (:8101, tunnelled).
# Output: data/narcbench_gen/<writer>/stego/<run>/ (same files as the released dataset).
#
#   nohup bash scripts/narc_generate.sh [writer ...] > runs/narc_gen/gen.log 2>&1 &
# SHOES="0" plays one shoe (smoke test). Resumable: stego.py skips completed runs.
# Hosts and paths: scripts/config.sh (.env); needs NARCBENCH_DIR, and JETSON for the 27B writer.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh
need NARCBENCH_DIR
NB=$NARCBENCH_DIR
mkdir -p "$MODELS_DIR"
SHOES="${SHOES:-$(seq -s ' ' 0 19)}"
PAR="${PAR:-4}"
# writer  repo  file  where
WRITERS=(
  "qwen3.5-9b unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf local"
  "qwen3.5-4b unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf local"
  "qwen3.5-2b unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q8_0.gguf local"
  "qwen3.5-0.8b unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q8_0.gguf local"
  "qwen3.8-27b unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf jetson"
)
mkdir -p runs/narc_gen

wait_health() {  # url pid-check-command
  for _ in $(seq 900); do curl -sf "$1/health" > /dev/null && return 0; eval "$2" || return 1; sleep 1; done
  return 1
}

for row in "${WRITERS[@]}"; do
  read -r name repo file where <<< "$row"
  if (( $# )) && [[ ! " $* " == *" $name "* ]]; then continue; fi
  echo "== $(date -Is) $name ($where) shoes: $SHOES"
  if [[ $where == local ]]; then
    url=http://127.0.0.1:8096
    curl -sfL -C - --retry 3 -o "$MODELS_DIR/$file" "https://huggingface.co/$repo/resolve/main/$file" \
      || { echo "== download failed"; continue; }
    "$LLAMA_SERVER" -m "$MODELS_DIR/$file" -ngl 99 -c 32768 -np "$PAR" --jinja --port 8096 \
      > "runs/narc_gen/$name.server.log" 2>&1 < /dev/null &
    spid=$!
    wait_health $url "kill -0 $spid 2> /dev/null" || { echo "== server failed"; kill $spid 2> /dev/null; continue; }
  else
    need JETSON
    url=http://127.0.0.1:8101
    ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 8101:127.0.0.1:8101 "$JETSON" &
    tunnel=$!
    ssh -o BatchMode=yes "$JETSON" "mkdir -p $JETSON_DIR && cd $JETSON_DIR \
      && curl -sfL -C - --retry 3 -o $file https://huggingface.co/$repo/resolve/main/$file \
      && (nohup $JETSON_LLAMA_SERVER_ALT -m $file -ngl 99 -c 32768 -np $PAR --jinja \
          --host 127.0.0.1 --port 8101 > $JETSON_DIR/$name.server.log 2>&1 < /dev/null & echo \$! > $JETSON_DIR/gen.pid)" < /dev/null
    wait_health $url "kill -0 $tunnel 2> /dev/null" || { echo "== server failed"; kill $tunnel 2> /dev/null; continue; }
  fi
  out="$PWD/data/narcbench_gen/$name/stego"
  mkdir -p "$out"
  # one stego.py process per shoe, PAR at a time (the server has PAR slots)
  printf '%s\n' $SHOES | xargs -P "$PAR" -I{} sh -c \
    "cd $NB && uv run --no-project --with requests python generation/stego.py --shoe {} \
       --base-url $url/v1 --model $name --output-dir '$out' > '$PWD/runs/narc_gen/$name.shoe{}.log' 2>&1 \
       || echo '== $name shoe {} failed'"
  echo "== $(date -Is) $name: $(ls "$out" | wc -l) runs written"
  if [[ $where == local ]]; then
    kill $spid; wait $spid 2> /dev/null
    [[ -n "${KEEP:-}" ]] || rm -f "$MODELS_DIR/$file"
  else
    ssh -o BatchMode=yes "$JETSON" "kill \$(cat $JETSON_DIR/gen.pid) 2> /dev/null; sleep 2; \
      ${KEEP:+true ||} rm -f $JETSON_DIR/$file" < /dev/null
    kill $tunnel 2> /dev/null
  fi
done
echo "== $(date -Is) generation done"

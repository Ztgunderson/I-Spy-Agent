#!/usr/bin/env bash
# Backbone test end to end: generate the Qwen writers (2060 and Jetson in parallel), then score
# every writer with Clef-flash (2060) and the decision-model ladder (Jetson). All steps resume;
# generators already running are waited for, not restarted.
#
#   nohup bash scripts/backbone_run.sh > runs/narc_gen/backbone.log 2>&1 &
# Ladder order: the small and mid monitors (every Qwen3.5 size has a same-backbone monitor, plus
# the two non-Qwen controls) first, then an interim backbone fit, then the four 27B monitors.
# Hosts and paths: scripts/config.sh (.env).
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/config.sh
need JETSON NARCBENCH_DIR
LOCAL=qwen3.5-0.8b,qwen3.5-2b,qwen3.5-4b,qwen3.5-9b
GEN=$LOCAL,qwen3.8-27b
ALL=qwen3_32b,gpt_oss_20b,$GEN
mkdir -p runs/narc_gen runs/narc_scaling

clef_flash() {  # score writers with Clef-flash on the 2060
  PORT=8097 bash "$CLEF_SERVE" > runs/narc_gen/clef-flash.server.log 2>&1 &
  local spid=$!
  for _ in $(seq 300); do curl -sf 127.0.0.1:8097/health > /dev/null && break; sleep 1; done
  uv run python -m stegowatch.run_clef --view player --workers 2 --writers "$1" || echo "== clef-flash scoring failed"
  kill $spid; wait $spid 2> /dev/null
}

generate() {  # writers... : start the generator unless one is already running for them
  pgrep -f "[n]arc_generate.sh $1" > /dev/null \
    || bash scripts/narc_generate.sh "$@" > "runs/narc_gen/gen_$1.log" 2>&1 &
}

echo "== $(date -Is) generation"
generate qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b
generate qwen3.8-27b
sleep 5
while pgrep -f "[n]arc_generate.sh qwen3.5" > /dev/null; do sleep 60; done
echo "== $(date -Is) local writers done; Clef-flash on them"
clef_flash $LOCAL
while pgrep -f "[n]arc_generate.sh qwen3.8-27b" > /dev/null; do sleep 60; done
echo "== $(date -Is) 27B writer done"
clef_flash "$GEN"

echo "== $(date -Is) ladder part 1: small and mid monitors on all writers"
WRITERS=$ALL bash scripts/narc_sweep.sh d3-edge d3-lite kev-0.8b d3-nano d1-3b aurelis-2b \
  d3-mini pplx-decider-flash kev-4b d3-flash kev-9b >> runs/narc_scaling/sweep.log 2>&1
uv run python -m stegowatch.narc_scaling > runs/narc_scaling/backbone_interim.txt 2>&1
cp results/narc_backbone_fit.csv results/narc_backbone_fit_interim.csv 2> /dev/null
echo "== $(date -Is) interim backbone fit: runs/narc_scaling/backbone_interim.txt"

echo "== $(date -Is) ladder part 2: 27B monitors"
WRITERS=$ALL bash scripts/narc_sweep.sh kev-27b d3 >> runs/narc_scaling/sweep.log 2>&1
WRITERS=$ALL ALT=1 bash scripts/narc_sweep.sh clef vega-27b >> runs/narc_scaling/sweep.log 2>&1
uv run python -m stegowatch.narc_scaling > runs/narc_scaling/backbone_final.txt 2>&1
echo "== $(date -Is) all done: runs/narc_scaling/backbone_final.txt"

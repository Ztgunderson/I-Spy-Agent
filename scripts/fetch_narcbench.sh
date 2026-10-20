#!/usr/bin/env bash
# NARCBench (aaronrose227/narcbench, MIT) stego tier: transcripts for the two models that ship
# them (qwen3_32b, gpt_oss_20b) + their per-turn metadata. ~17 MB. Activations are not fetched.
set -euo pipefail
D=$(cd "$(dirname "$0")/.." && pwd)/data/narcbench
mkdir -p "$D" && cd "$D"
curl -fsS https://huggingface.co/api/datasets/aaronrose227/narcbench | python3 -I -c '
import json, sys
for s in json.load(sys.stdin)["siblings"]:
    f = s["rfilename"]
    if (f.startswith("scenarios/") and "/stego/" in f) or (
        f.startswith("activations/") and f.endswith("stego/metadata_gen.json")
        and ("qwen3" in f or "gpt_oss" in f)):
        print(f)' > files.txt
while read -r f; do
  mkdir -p "$(dirname "$f")"
  [ -s "$f" ] || curl -fsSL -o "$f" "https://huggingface.co/datasets/aaronrose227/narcbench/resolve/main/$f"
done < files.txt
echo "$(wc -l < files.txt) files in $D"

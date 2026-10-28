# Shared settings for the run scripts (sourced, not run). Machine-specific values come from the
# environment or from a gitignored .env at the repo root; see .env.example. Nothing host-specific
# is committed.
[[ -f .env ]] && { set -a; . ./.env; set +a; }

# need VAR...: stop with a pointer to .env.example when a required setting is missing
need() {
  local v
  for v in "$@"; do
    [[ -n "${!v:-}" ]] || { echo "== $v is not set; add it to .env (see .env.example)" >&2; exit 2; }
  done
}

# Remote host (edge device) that serves the decision models; values are expanded on that host.
JETSON_DIR="${JETSON_DIR:-~/narc-gguf}"                      # GGUF scratch folder
JETSON_LLAMA_SERVER="${JETSON_LLAMA_SERVER:-llama-server}"   # default build
# a newer build for models the default build fails to load (Clef 27B, Vega-27B)
JETSON_LLAMA_SERVER_ALT="${JETSON_LLAMA_SERVER_ALT:-$JETSON_LLAMA_SERVER}"

# This machine
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
MODELS_DIR="${MODELS_DIR:-models}"
CLEF_SERVE="${CLEF_SERVE:-../handyman/scripts/serve_clef.sh}"
[[ -n "${LLAMA_LIB_PATH:-}" ]] && export LD_LIBRARY_PATH="$LLAMA_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

#!/bin/bash
# Render every tagged visualization (sink_token_visualizations.py -> sink_tokens/,
# attention_visualizations.py -> attention/) against the POPE snapshot .pt produced by
# run_fastv_single_case_pope.sh. Pure CPU/matplotlib, no model reload - safe to rerun
# after every edit to a *_visualizations.py topic module.
#
# Usage:
#   ./visualization/script/run_fastv_visualizations_pope.sh                       # render the latest captured case
#   QUESTION_ID=20001839 ./visualization/script/run_fastv_visualizations_pope.sh  # render a specific archived case
#   ONLY=plot_sink_tokens_hidden_states_by_score ./visualization/script/run_fastv_visualizations_pope.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# QUESTION_ID picks the per-case snapshot archived by run_fastv_single_case_pope.sh
# (snapshots/pope/<QUESTION_ID>.pt); unset falls back to the latest-case pointer. An
# explicit SNAPSHOT overrides both.
QUESTION_ID=${QUESTION_ID:-1}
if [ -n "${QUESTION_ID}" ]; then
  DEFAULT_SNAPSHOT="${REPO_ROOT}/visualization/snapshots/pope/${QUESTION_ID}.pt"
else
  DEFAULT_SNAPSHOT="${REPO_ROOT}/visualization/snapshots/pope_single_case.pt"
fi
SNAPSHOT=${SNAPSHOT:="${DEFAULT_SNAPSHOT}"}
OUT_DIR=${OUT_DIR:="${REPO_ROOT}/visualization/output/pope"}
ONLY=${ONLY:-}

if [ ! -f "${SNAPSHOT}" ]; then
  echo "Missing snapshot: ${SNAPSHOT}" >&2
  echo "Capture it first (optionally QUESTION_ID=${QUESTION_ID:-<id>} ./visualization/script/run_fastv_single_case_pope.sh) or set SNAPSHOT explicitly." >&2
  exit 1
fi

ONLY_ARGS=()
if [ -n "${ONLY}" ]; then
  ONLY_ARGS=(--only ${ONLY})
fi

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
python3 -m visualization.render_fastv_visualizations \
  --snapshot "${SNAPSHOT}" \
  --out-dir "${OUT_DIR}" \
  "${ONLY_ARGS[@]}"

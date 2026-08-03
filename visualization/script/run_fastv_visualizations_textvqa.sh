#!/bin/bash
# Render every tagged visualization (sink_token_visualizations.py -> sink_tokens/,
# attention_visualizations.py -> attention/) against the TextVQA snapshot .pt produced
# by run_fastv_single_case_textvqa.sh. Pure CPU/matplotlib, no model reload - safe to
# rerun after every edit to a *_visualizations.py topic module.
#
# Usage:
#   ./visualization/script/run_fastv_visualizations_textvqa.sh                            # render the latest captured case
#   QUESTION_ID=2b538a43dd933fc1 ./visualization/script/run_fastv_visualizations_textvqa.sh  # render a specific archived case
#   ONLY=plot_sink_tokens_hidden_states_by_score ./visualization/script/run_fastv_visualizations_textvqa.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# QUESTION_ID picks the per-case snapshot archived by run_fastv_single_case_textvqa.sh
# (snapshots/textvqa/<QUESTION_ID>.pt); unset falls back to the latest-case pointer. An
# explicit SNAPSHOT overrides both.
QUESTION_ID=${QUESTION_ID:-b9dc400eb20bad64}
if [ -n "${QUESTION_ID}" ]; then
  DEFAULT_SNAPSHOT="${REPO_ROOT}/visualization/snapshots/textvqa/${QUESTION_ID}.pt"
else
  DEFAULT_SNAPSHOT="${REPO_ROOT}/visualization/snapshots/textvqa_single_case.pt"
fi
SNAPSHOT=${SNAPSHOT:="${DEFAULT_SNAPSHOT}"}
OUT_DIR=${OUT_DIR:="${REPO_ROOT}/visualization/output/textvqa"}
ONLY=${ONLY:-}

if [ ! -f "${SNAPSHOT}" ]; then
  echo "Missing snapshot: ${SNAPSHOT}" >&2
  echo "Capture it first (optionally QUESTION_ID=${QUESTION_ID:-<id>} ./visualization/script/run_fastv_single_case_textvqa.sh) or set SNAPSHOT explicitly." >&2
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

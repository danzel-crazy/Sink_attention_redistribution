#!/bin/bash
# Render every tagged visualization (sink_token_visualizations.py -> sink_tokens/,
# attention_visualizations.py -> attention/) against the TextVQA snapshot .pt produced
# by run_fastv_single_case_textvqa.sh. Pure CPU/matplotlib, no model reload - safe to
# rerun after every edit to a *_visualizations.py topic module.
#
# Usage:
#   ./visualization/script/run_fastv_visualizations_textvqa.sh
#   ONLY=plot_sink_tokens_hidden_states_by_score ./visualization/script/run_fastv_visualizations_textvqa.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SNAPSHOT=${SNAPSHOT:="${REPO_ROOT}/visualization/snapshots/textvqa_single_case.pt"}
OUT_DIR=${OUT_DIR:="${REPO_ROOT}/visualization/output/textvqa"}
ONLY=${ONLY:-}

ONLY_ARGS=()
if [ -n "${ONLY}" ]; then
  ONLY_ARGS=(--only ${ONLY})
fi

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
python3 -m visualization.render_fastv_visualizations \
  --snapshot "${SNAPSHOT}" \
  --out-dir "${OUT_DIR}" \
  "${ONLY_ARGS[@]}"

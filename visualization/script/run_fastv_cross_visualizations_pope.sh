#!/bin/bash
# Render ONLY the text->visual cross-attention comparison (pre_visual / text_to_visual /
# text_to_visual_from_qk) overlaid on the image, from snapshot .pt files produced by the
# FastV capture scripts. Supports either one snapshot file or a benchmark directory full of
# snapshots. For a directory input, each .pt is rendered into <out-dir>/<question_id>/...,
# where question_id is taken from the filename stem before ".pt".
#
# Usage:
#   ./visualization/script/run_fastv_cross_visualizations_pope.sh
#   QUESTION_ID=20001839 ./visualization/script/run_fastv_cross_visualizations_pope.sh
#   SNAPSHOT=/path/to/case.pt ./visualization/script/run_fastv_cross_visualizations_pope.sh
#   SNAPSHOT=/tmp2/danzel/Sink_attention_redistribution/visualization/snapshots/gqa \
#     ./visualization/script/run_fastv_cross_visualizations_pope.sh
#   SINK_MASK_METHOD=layernorm ./visualization/script/run_fastv_cross_visualizations_pope.sh
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
SINK_MASK_METHOD=${SINK_MASK_METHOD:="sum"}

benchmark_name="pope"
if [[ "${SNAPSHOT}" == *"/visualization/snapshots/"* ]]; then
  snapshot_rel="${SNAPSHOT#*"/visualization/snapshots/"}"
  if [ -d "${SNAPSHOT}" ]; then
    benchmark_name="${snapshot_rel%/}"
  elif [[ "${snapshot_rel}" == *.pt ]]; then
    benchmark_name="$(dirname "${snapshot_rel}")"
  fi
elif [ -d "${SNAPSHOT}" ]; then
  benchmark_name="$(basename "${SNAPSHOT}")"
elif [[ "${SNAPSHOT}" == */snapshots/*/*.pt ]]; then
  benchmark_name="$(basename "$(dirname "${SNAPSHOT}")")"
fi
OUT_DIR=${OUT_DIR:="${REPO_ROOT}/visualization/output/${benchmark_name}"}

render_snapshot() {
  local snapshot_path=$1
  local output_root=$2

  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python3 -m visualization.cross_attention_visualize \
    --snapshot "${snapshot_path}" \
    --out-dir "${output_root}" \
    --sink-mask-method "${SINK_MASK_METHOD}"
}

if [ -d "${SNAPSHOT}" ]; then
  shopt -s nullglob
  snapshot_paths=("${SNAPSHOT}"/*.pt)
  shopt -u nullglob

  if [ ${#snapshot_paths[@]} -eq 0 ]; then
    echo "No .pt snapshots found in directory: ${SNAPSHOT}" >&2
    exit 1
  fi

  mapfile -t snapshot_paths < <(printf '%s\n' "${snapshot_paths[@]}" | sort)

  for snapshot_path in "${snapshot_paths[@]}"; do
    question_id="$(basename "${snapshot_path}" .pt)"
    per_case_out_dir="${OUT_DIR}/${question_id}"
    echo "Rendering ${snapshot_path} -> ${per_case_out_dir}"
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python3 -m visualization.cross_attention_visualize \
      --snapshot "${snapshot_path}" \
      --out-dir "${per_case_out_dir}" \
      --no-question-subdir \
      --sink-mask-method "${SINK_MASK_METHOD}"
  done
  exit 0
fi

if [ ! -f "${SNAPSHOT}" ]; then
  echo "Missing snapshot: ${SNAPSHOT}" >&2
  echo "Capture it first (optionally QUESTION_ID=${QUESTION_ID:-<id>} ./visualization/script/run_fastv_single_case_pope.sh) or set SNAPSHOT explicitly." >&2
  exit 1
fi

render_snapshot "${SNAPSHOT}" "${OUT_DIR}"

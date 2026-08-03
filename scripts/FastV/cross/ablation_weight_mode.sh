#!/bin/bash

# Ablation: does cross-attention help by *selecting* the receivers, or also by *weighting* how the
# freed sink budget splits among them?
#
# Both rows keep the same sink-masked cross-attention receiver selection (topk, RECEIVER_TOKEN_COUNT
# receivers). They differ only in the split:
#   cross   - budget splits proportionally to the cross-attention score (the default pipeline)
#   uniform - budget splits evenly, 1/RECEIVER_TOKEN_COUNT each (cross-attention only selects)
#
# Runs each config sequentially on one GPU via gqa_hf.sh and collects the GQA accuracies into a
# single summary file.
#
# Usage:
#   bash scripts/FastV/cross/ablation_weight_mode.sh [GPU_ID]
#   MODES="cross uniform" RECEIVER_TOKEN_COUNT=32 bash scripts/FastV/cross/ablation_weight_mode.sh 0

set -uo pipefail

GPU_ID=${1:-0}
MODES=${MODES:="cross uniform"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/gqa_hf.sh"

# Exported so gqa_hf.sh's ${VAR:=default} assignments pick them up; the values here are just the
# script's own defaults restated, so overriding any of them from the environment still works.
export VISUAL_TOKEN_NUM=${VISUAL_TOKEN_NUM:=576}
export FASTV_K=${FASTV_K:=5}
export FASTV_R=${FASTV_R:=0.77}
export RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}
export REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}

RETAIN_TOKENS=$(python -c "print(round(${VISUAL_TOKEN_NUM} * (1 - ${FASTV_R})))")

LOG_ROOT="/project/aimm/danzel/experiment/sink_masked/fastv/GQA/ablation/weight_mode"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${LOG_ROOT}/${FASTV_K}_${FASTV_R}_${RETAIN_TOKENS}_tokens_${RECEIVER_TOKEN_COUNT}_receivers_${STAMP}"
SUMMARY="${RUN_DIR}/summary.txt"
mkdir -p "${RUN_DIR}"

{
    echo "GQA sink-budget weighting ablation"
    echo "date            : $(date)"
    echo "gpu             : ${GPU_ID}"
    echo "fastv_k / fastv_r: ${FASTV_K} / ${FASTV_R}  (${RETAIN_TOKENS} visual tokens retained)"
    echo "strategy        : ${REDISTRIBUTION_STRATEGY}"
    echo "receivers       : ${RECEIVER_TOKEN_COUNT}"
    echo "modes           : ${MODES}"
    echo
} | tee "${SUMMARY}"

declare -A ACC
FAILED=""

for mode in ${MODES}; do
    LOG_FILE="${RUN_DIR}/${mode}.log"
    echo "=== [$(date +%H:%M:%S)] running receiver-weight-mode=${mode} -> ${LOG_FILE}"

    RECEIVER_WEIGHT_MODE="${mode}" bash "${RUN_SCRIPT}" "${GPU_ID}" 2>&1 | tee "${LOG_FILE}"
    status=${PIPESTATUS[0]}

    if [ "${status}" -ne 0 ]; then
        echo "!!! mode=${mode} exited with status ${status}; see ${LOG_FILE}"
        ACC["${mode}"]="FAILED (exit ${status})"
        FAILED="${FAILED} ${mode}"
        continue
    fi

    # GQA eval.py prints e.g. "Accuracy: 61.50%"; take the last match in case the log has others.
    acc=$(grep -E "^Accuracy: " "${LOG_FILE}" | tail -1 | sed -E 's/^Accuracy: //')
    ACC["${mode}"]="${acc:-NOT FOUND (see ${LOG_FILE})}"
done

{
    echo
    echo "==================== results ===================="
    printf "%-10s %s\n" "mode" "GQA accuracy"
    for mode in ${MODES}; do
        printf "%-10s %s\n" "${mode}" "${ACC[${mode}]}"
    done
    echo "================================================="
    echo "logs: ${RUN_DIR}"
} | tee -a "${SUMMARY}"

if [ -n "${FAILED}" ]; then
    echo "runs failed:${FAILED}" >&2
    exit 1
fi

#!/bin/bash
# SparseVLM, family A.  bash scripts/efficiency/sparsevlm.sh [gpu_id]

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
check_family A
eff_env sparsevlm

RETAINED_TOKENS="${RETAINED_TOKENS:-128}"

export PYTHONPATH="${REPO_ROOT}/SparseVLMs:${REPO_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.model_vqa_loader \
    --model-path "${CKPT_A}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_FILE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens "${RETAINED_TOKENS}"

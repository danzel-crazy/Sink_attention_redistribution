#!/bin/bash
# PyramidDrop, family A.  bash scripts/efficiency/pdrop.sh [gpu_id]

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
check_family A
eff_env pdrop

LAYER_LIST="${LAYER_LIST:-[2,10,20]}"
RATIO_LIST="${RATIO_LIST:-[0.32,0.16,0.08]}"

export PYTHONPATH="${REPO_ROOT}/PyramidDrop:${REPO_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.model_vqa_loader \
    --model-path "${CKPT_A}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_FILE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --layer_list "${LAYER_LIST}" \
    --image_token_ratio_list "${RATIO_LIST}"

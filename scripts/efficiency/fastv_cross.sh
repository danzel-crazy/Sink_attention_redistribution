#!/bin/bash
# FastV + sink/cross-attention redistribution, family B.
#   bash scripts/efficiency/fastv_cross.sh [gpu_id]
#
# receiver-token-count > 0 is what enables redistribution (and what labels the row fastv-cross).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
check_family B
eff_env fastv-cross

VISUAL_TOKEN_NUM="${VISUAL_TOKEN_NUM:-576}"
FASTV_K="${FASTV_K:-5}"
FASTV_R="${FASTV_R:-0.77}"
IMAGE_TOKEN_START_INDEX="${IMAGE_TOKEN_START_INDEX:-5}"
REDISTRIBUTION_STRATEGY="${REDISTRIBUTION_STRATEGY:-topk_text_visual_tokens}"
REDISTRIBUTION_SOFTMAX_MODE="${REDISTRIBUTION_SOFTMAX_MODE:-resoftmax}"
RECEIVER_TOKEN_COUNT="${RECEIVER_TOKEN_COUNT:-8}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m FastV.src.FastV.inference.eval.inference \
    --model-id "${CKPT_B}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_FILE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --use_fastv \
    --visual_token_num "${VISUAL_TOKEN_NUM}" \
    --fastv_k "${FASTV_K}" \
    --fastv_r "${FASTV_R}" \
    --image_token_start_index "${IMAGE_TOKEN_START_INDEX}" \
    --redistribution-strategy "${REDISTRIBUTION_STRATEGY}" \
    --redistribution-softmax-mode "${REDISTRIBUTION_SOFTMAX_MODE}" \
    --receiver-token-count "${RECEIVER_TOKEN_COUNT}"

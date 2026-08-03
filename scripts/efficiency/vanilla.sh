#!/bin/bash
# Vanilla LLaVA-1.5-7B, family A (transformers 4.37.x, liuhaotian weights).
#
# The PyramidDrop loader with --layer_list omitted runs stock LLaVA, so it doubles as the
# unpruned baseline. This is the denominator for PyramidDrop's and SparseVLM's FLOPs reduction
# and speedup. FastV has its own vanilla (fastv_vanilla.sh) because it lives in another family.
#
#   bash scripts/efficiency/vanilla.sh [gpu_id]

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
check_family A
eff_env vanilla

export PYTHONPATH="${REPO_ROOT}/PyramidDrop:${REPO_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.model_vqa_loader \
    --model-path "${CKPT_A}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_FILE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --conv-mode vicuna_v1

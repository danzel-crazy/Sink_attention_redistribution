#!/bin/bash
# Vanilla (unpruned) LLaVA on FastV's HF path -- family B's baseline.
#   bash scripts/efficiency/fastv_vanilla.sh [gpu_id]
#
# --no-use-fastv makes build_fastv_config() return None, so modeling_llava.py takes the stock
# language_model(...) branch. Everything else (vendored transformers 4.39, llava-hf weights,
# eager attention, output_attentions) stays identical to fastv.sh, so the FastV-vs-this speedup
# isolates pruning instead of re-measuring eager attention. Do not compare its milliseconds
# against family A's vanilla.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
check_family B
eff_env vanilla

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m FastV.src.FastV.inference.eval.inference \
    --model-id "${CKPT_B}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_FILE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --no-use-fastv

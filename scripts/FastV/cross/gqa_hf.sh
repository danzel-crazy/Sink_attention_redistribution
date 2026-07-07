#!/bin/bash

# GQA eval for the llava-hf/HF-transformers FastV pipeline (FastV/src/FastV/inference/eval/inference.py),
# with the sink-token / cross-attention / attention-redistribution debiasing pipeline wired in.
# Single-process (no chunking) - see scripts/FastV/gqa.sh for the original LLaVA-repo pipeline.

GPU_ID=${1:-0}
CKPT=${CKPT:="llava-hf/llava-1.5-7b-hf"}
DATASET_DIR="/project/aimm/pp/MTK/eval"

VISUAL_TOKEN_NUM=${VISUAL_TOKEN_NUM:=576}
FASTV_K=${FASTV_K:=5}
FASTV_R=${FASTV_R:=0.77}
IMAGE_TOKEN_START_INDEX=${IMAGE_TOKEN_START_INDEX:=5}
RETAIN_TOKENS=$(python -c "print(round(${VISUAL_TOKEN_NUM} * (1 - ${FASTV_R})))")

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}

ANSWER_ROOT="/project/aimm/danzel/experiment/sink_masked/fastv"
ANSWERS_DIR="${ANSWER_ROOT}/GQA/answers/cross_attention"
EXP_NAME="${FASTV_K}_${FASTV_R}_${REDISTRIBUTION_STRATEGY}_${REDISTRIBUTION_SOFTMAX_MODE}_${RETAIN_TOKENS}_tokens"
ANSWER_FILE="${ANSWERS_DIR}/${EXP_NAME}.jsonl"
mkdir -p "${ANSWERS_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m FastV.src.FastV.inference.eval.inference \
    --model-id "${CKPT}" \
    --question-file "${DATASET_DIR}/gqa/llava_gqa_testdev_balanced.jsonl" \
    --image-folder "${DATASET_DIR}/gqa/data/images" \
    --answers-file ${ANSWER_FILE} \
    --visual_token_num ${VISUAL_TOKEN_NUM} \
    --use_fastv \
    --fastv_k ${FASTV_K} \
    --fastv_r ${FASTV_R} \
    --image_token_start_index ${IMAGE_TOKEN_START_INDEX} \
    --redistribution-strategy ${REDISTRIBUTION_STRATEGY} \
    --redistribution-softmax-mode ${REDISTRIBUTION_SOFTMAX_MODE} \
    --receiver-token-count ${RECEIVER_TOKEN_COUNT}

wait

EVAL_PREDICTIONS_FILE="${ANSWERS_DIR}/${EXP_NAME}.json"
python3 /tmp2/danzel/attention-bias/scripts/convert_gqa_for_eval.py \
--src "$ANSWER_FILE" \
--dst "$EVAL_PREDICTIONS_FILE"

GQA_DIR="/project/aimm/danzel/eval/gqa/data"
(
cd "$GQA_DIR"
python3 eval.py \
    --tier testdev_balanced \
    --predictions "$EVAL_PREDICTIONS_FILE"
)

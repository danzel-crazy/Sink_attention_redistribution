#!/bin/bash

# TextVQA eval for the llava-hf/HF-transformers FastV pipeline (FastV/src/FastV/inference/eval/inference.py),
# with the sink-token / cross-attention / attention-redistribution debiasing pipeline wired in.
# Receivers are SELECTED by the pre-decoder cross-attention (pre_visual_cross_attention_importants)
# and the sink budget split is WEIGHTED by the in-decoder cross-attention (cross_attention_importants).
# Single-process (no chunking) - see scripts/FastV/cross/textvqa_hf.sh for the both-in-decoder variant.

GPU_ID=${1:-0}
CKPT=${CKPT:="llava-hf/llava-1.5-7b-hf"}
DATASET_DIR="/project/aimm/pp/MTK/eval"

VISUAL_TOKEN_NUM=${VISUAL_TOKEN_NUM:=576}
FASTV_K=${FASTV_K:=5}
FASTV_R=${FASTV_R:=0.77}
IMAGE_TOKEN_START_INDEX=${IMAGE_TOKEN_START_INDEX:=5}

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
RECEIVER_SELECTION_SOURCE=${RECEIVER_SELECTION_SOURCE:="pre_visual"}
RECEIVER_WEIGHT_SOURCE=${RECEIVER_WEIGHT_SOURCE:="cross"}

ANSWER_ROOT="/project/aimm/danzel/experiment/sink_masked/fastv"
ANSWERS_DIR="${ANSWER_ROOT}/TextVQA/answers/pre_visual_receiver_cross_weighted"
EXP_NAME="${FASTV_K}_${FASTV_R}_${REDISTRIBUTION_STRATEGY}_${REDISTRIBUTION_SOFTMAX_MODE}"
ANSWER_FILE="${ANSWERS_DIR}/${EXP_NAME}.jsonl"
mkdir -p "${ANSWERS_DIR}"

# CUDA_VISIBLE_DEVICES=${GPU_ID} python -m FastV.src.FastV.inference.eval.inference \
#     --model-id "${CKPT}" \
#     --question-file "${DATASET_DIR}/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
#     --image-folder "${DATASET_DIR}/textvqa/train_images" \
#     --answers-file ${ANSWER_FILE} \
#     --visual_token_num ${VISUAL_TOKEN_NUM} \
#     --use_fastv \
#     --fastv_k ${FASTV_K} \
#     --fastv_r ${FASTV_R} \
#     --image_token_start_index ${IMAGE_TOKEN_START_INDEX} \
#     --redistribution-strategy ${REDISTRIBUTION_STRATEGY} \
#     --redistribution-softmax-mode ${REDISTRIBUTION_SOFTMAX_MODE} \
#     --receiver-selection-source ${RECEIVER_SELECTION_SOURCE} \
#     --receiver-weight-source ${RECEIVER_WEIGHT_SOURCE}

# wait

python3 /tmp2/danzel/Sink_attention_redistribution/FastV/src/LLaVA/llava/eval/eval_textvqa.py \
  --annotation-file "${DATASET_DIR}/textvqa/TextVQA_0.5.1_val.json" \
  --result-file "$ANSWER_FILE"

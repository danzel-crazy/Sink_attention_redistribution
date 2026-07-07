#!/bin/bash

GPU_ID=${1:-0}
REPO_ROOT="/tmp2/danzel/attention-bias"
DATASET_DIR="/project/aimm/pp/MTK/eval"
CKPT=${CKPT:="liuhaotian/llava-v1.5-7b"}

LAYER_LIST=${LAYER_LIST:='[2,10,20]'}
IMAGE_TOKEN_RATIO_LIST=${IMAGE_TOKEN_RATIO_LIST:="[0.32,0.16,0.08]"}

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}

ANSWER_ROOT="/project/aimm/danzel/experiment/sink_masked/pdrop"
ANSWERS_DIR="${ANSWER_ROOT}/MMVet/answers/cross_attention"
UPLOAD_DIR="${ANSWER_ROOT}/MMVet/answers_upload/cross_attention"
ANSWER_FILE="${ANSWERS_DIR}/${REDISTRIBUTION_STRATEGY}_${REDISTRIBUTION_SOFTMAX_MODE}_128.jsonl"
mkdir -p "${ANSWERS_DIR}" "${UPLOAD_DIR}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m PyramidDrop.llava.eval.model_vqa_loader_cross \
    --model-path ${CKPT} \
    --question-file "${DATASET_DIR}/mm-vet/llava-mm-vet.jsonl" \
    --image-folder "${DATASET_DIR}/mm-vet/images" \
    --answers-file ${ANSWER_FILE} \
    --temperature 0 \
    --layer_list "${LAYER_LIST}" \
    --image_token_ratio_list "${IMAGE_TOKEN_RATIO_LIST}" \
    --redistribution_strategy ${REDISTRIBUTION_STRATEGY} \
    --redistribution_softmax_mode ${REDISTRIBUTION_SOFTMAX_MODE} \
    --receiver_token_count ${RECEIVER_TOKEN_COUNT} \
    --conv-mode vicuna_v1

wait

python "${REPO_ROOT}/scripts/convert_mmvet_for_eval.py" \
    --src "${ANSWER_FILE}" \
    --dst "${UPLOAD_DIR}/${REDISTRIBUTION_STRATEGY}_${REDISTRIBUTION_SOFTMAX_MODE}_128.json"

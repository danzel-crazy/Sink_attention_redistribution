#!/bin/bash

GPU_LIST=${CUDA_VISIBLE_DEVICES:-${1:-0}}
REPO_ROOT="/tmp2/danzel/attention-bias"
DATASET_DIR="/project/aimm/pp/MTK/eval"
CKPT=${CKPT:="liuhaotian/llava-v1.5-7b"}

LAYER_LIST=${LAYER_LIST:='[2,10,20]'}
IMAGE_TOKEN_RATIO_LIST=${IMAGE_TOKEN_RATIO_LIST:="[0.32,0.16,0.08]"}

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}

ANSWER_ROOT="/project/aimm/danzel/experiment/sink_masked/pdrop"
ANSWERS_DIR="${ANSWER_ROOT}/VQAv2/answers/cross_attention"
UPLOAD_DIR="${ANSWER_ROOT}/VQAv2/answers_upload/cross_attention"
EXP_NAME="${REDISTRIBUTION_STRATEGY}_${REDISTRIBUTION_SOFTMAX_MODE}_128"
mkdir -p "${ANSWERS_DIR}" "${UPLOAD_DIR}"

IFS=',' read -ra GPULIST <<< "${GPU_LIST}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS - 1))); do
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m PyramidDrop.llava.eval.model_vqa_loader_cross \
        --model-path ${CKPT} \
        --question-file "${DATASET_DIR}/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl" \
        --image-folder "${DATASET_DIR}/vqav2/test2015" \
        --answers-file "${ANSWERS_DIR}/${CHUNKS}_${IDX}_${EXP_NAME}.jsonl" \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --temperature 0 \
        --layer_list "${LAYER_LIST}" \
        --image_token_ratio_list "${IMAGE_TOKEN_RATIO_LIST}" \
        --redistribution_strategy ${REDISTRIBUTION_STRATEGY} \
        --redistribution_softmax_mode ${REDISTRIBUTION_SOFTMAX_MODE} \
        --receiver_token_count ${RECEIVER_TOKEN_COUNT} \
        --conv-mode vicuna_v1 &
done

wait

ANSWER_FILE="${ANSWERS_DIR}/${EXP_NAME}.jsonl"
> "${ANSWER_FILE}"
for IDX in $(seq 0 $((CHUNKS - 1))); do
    cat "${ANSWERS_DIR}/${CHUNKS}_${IDX}_${EXP_NAME}.jsonl" >> "${ANSWER_FILE}"
done

python "${REPO_ROOT}/scripts/convert_vqav2_for_submission.py" \
    --src "${ANSWER_FILE}" \
    --dst "${UPLOAD_DIR}/${EXP_NAME}.json" \
    --dir "${DATASET_DIR}/vqav2"

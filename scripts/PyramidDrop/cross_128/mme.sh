#!/bin/bash

# MME eval for the PyramidDrop pipeline (llava.eval.model_vqa_loader_cross), with the
# sink-token / cross-attention / attention-redistribution debiasing add-on wired in
# (cross_attention_sink_redistribution_llava/).
# See scripts/v1_5/pdrop_eval/mme.sh for the vanilla PyramidDrop pipeline without the add-on.
#
# Run from the PyramidDrop/ directory: bash scripts/v1_5/cross_eval/mme.sh [gpu_id]

GPU_ID=${1:-0}
REPO_ROOT="/tmp2/danzel/attention-bias"
DATASET_DIR="/project/aimm/pp/MTK/eval"
CKPT=${CKPT:="liuhaotian/llava-v1.5-7b"}

# LAYER_LIST=${LAYER_LIST:='[2,10,20]'}
# IMAGE_TOKEN_RATIO_LIST=${IMAGE_TOKEN_RATIO_LIST:="[0.32,0.16,0.08]"}

LAYER_LIST=${LAYER_LIST:-"[5,12,20,28]"}
IMAGE_TOKEN_RATIO_LIST=${IMAGE_TOKEN_RATIO_LIST:-"[0.125,0.0625,0.055,0.05]"}

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}

ANSWER_ROOT="/project/aimm/danzel/experiment/sink_masked/pdrop"
ANSWERS_DIR="${ANSWER_ROOT}/MME/answers/cross_attention"
ANSWER_FILE="${ANSWERS_DIR}/${REDISTRIBUTION_STRATEGY}_${REDISTRIBUTION_SOFTMAX_MODE}_128.jsonl"
mkdir -p "${ANSWERS_DIR}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m PyramidDrop.llava.eval.model_vqa_loader_cross \
    --model-path ${CKPT} \
    --question-file "${DATASET_DIR}/MME/llava_mme.jsonl" \
    --image-folder "${DATASET_DIR}/MME/MME_Benchmark_release_version/MME_Benchmark" \
    --answers-file ${ANSWER_FILE} \
    --temperature 0 \
    --layer_list "${LAYER_LIST}" \
    --image_token_ratio_list "${IMAGE_TOKEN_RATIO_LIST}" \
    --redistribution_strategy ${REDISTRIBUTION_STRATEGY} \
    --redistribution_softmax_mode ${REDISTRIBUTION_SOFTMAX_MODE} \
    --receiver_token_count ${RECEIVER_TOKEN_COUNT} \
    --conv-mode vicuna_v1

wait

python ${REPO_ROOT}/scripts/convert_answer_to_mme_filename.py \
    --answer_file $ANSWER_FILE \
    --experiment "cross_attention"

cd ${REPO_ROOT}/eval/mme/eval_tool

python calculation.py --results_dir "answers/cross_attention"

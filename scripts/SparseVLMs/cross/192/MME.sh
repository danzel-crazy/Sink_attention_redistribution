#!/bin/bash

# MME eval for the SparseVLM pipeline (llava.eval.model_vqa_loader_cross), with the
# sink-token / cross-attention / attention-redistribution debiasing add-on wired in
# (cross_attention_sink_redistribution_llava_sparsevlm/).
# See scripts/SparseVLMs/MME.sh for the vanilla SparseVLM pipeline without the add-on.

GPU_ID=${1:-0}
REPO_ROOT="/tmp2/danzel/attention-bias"
DATASET_DIR="/project/aimm/pp/MTK/eval"
CKPT_DIR=${CKPT_DIR:="pretrained"}
CKPT=${CKPT:="liuhaotian/llava-v1.5-7b"}
RETAINED_TOKENS=${RETAINED_TOKENS:-192}

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}

ANSWER_ROOT="/project/aimm/danzel/experiment/sink_masked/sparsevlm"
ANSWERS_DIR="${ANSWER_ROOT}/MME/answers/cross_attention"
ANSWER_FILE="${ANSWERS_DIR}/${REDISTRIBUTION_STRATEGY}_${REDISTRIBUTION_SOFTMAX_MODE}_${RETAINED_TOKENS}.jsonl"
mkdir -p "${ANSWERS_DIR}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.model_vqa_loader_cross \
    --model-path ${CKPT} \
    --question-file "${DATASET_DIR}/MME/llava_mme.jsonl" \
    --image-folder "${DATASET_DIR}/MME/MME_Benchmark_release_version/MME_Benchmark" \
    --answers-file ${ANSWER_FILE} \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens ${RETAINED_TOKENS} \
    --redistribution_strategy ${REDISTRIBUTION_STRATEGY} \
    --redistribution_softmax_mode ${REDISTRIBUTION_SOFTMAX_MODE} \
    --receiver_token_count ${RECEIVER_TOKEN_COUNT}

wait

python ${REPO_ROOT}/scripts/convert_answer_to_mme_filename.py \
    --answer_file $ANSWER_FILE \
    --experiment "cross_attention"

cd ${REPO_ROOT}/eval/mme/eval_tool

python calculation.py --results_dir "answers/cross_attention"

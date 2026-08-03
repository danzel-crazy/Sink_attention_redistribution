#!/bin/bash
# Capture MANY examples from a benchmark through the current PyramidDrop + sink-redistribution
# pipeline and save one snapshot per question_id, recording the FIRST pruning stage only
# (layer_list[0], where the visual block is still the full untouched grid).
#
# Benchmark defaults mirror scripts/PyramidDrop/*.sh; the capture itself reuses one model load
# via visualization.capture_pdrop_multiple_cases.
#
# Usage:
#   ./visualization/script/run_pdrop_multiple_cases.sh
#   CASES=1-10 BENCHMARK=pope ./visualization/script/run_pdrop_multiple_cases.sh 0
#   BENCHMARK=textvqa CASES=3- ./visualization/script/run_pdrop_multiple_cases.sh 0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPU_ID=${1:-0}
BENCHMARK=${BENCHMARK:-pope}
CKPT=${CKPT:="liuhaotian/llava-v1.5-7b"}
CONV_MODE=${CONV_MODE:="vicuna_v1"}
CASES=${CASES:="1-10"}

LAYER_LIST=${LAYER_LIST:="[2,10,20]"}
# accept RATIO_LIST as an alias, since scripts/PyramidDrop/*.sh use that name
IMAGE_TOKEN_RATIO_LIST=${IMAGE_TOKEN_RATIO_LIST:-${RATIO_LIST:-"[0.32,0.16,0.08]"}}

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:=8}
TEMPERATURE=${TEMPERATURE:=0}

LOCAL_DATASET_DIR="${REPO_ROOT}/dataset/playground/data/eval"

case "${BENCHMARK}" in
  pope)
    REFERENCE_DATASET_DIR="/project/aimm/danzel/eval"
    DEFAULT_QUESTION_REL="pope/llava_pope_test.jsonl"
    DEFAULT_IMAGE_REL="pope/val2014"
    REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="post_softmax_resoftmax"}
    RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}
    ;;
  textvqa)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="textvqa/llava_textvqa_val_v051_ocr.jsonl"
    DEFAULT_IMAGE_REL="textvqa/train_images"
    REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
    RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}
    ;;
  gqa)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="gqa/llava_gqa_testdev_balanced.jsonl"
    DEFAULT_IMAGE_REL="gqa/data/images"
    REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
    RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}
    ;;
  mme)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="MME/llava_mme.jsonl"
    DEFAULT_IMAGE_REL="MME/MME_Benchmark_release_version/MME_Benchmark"
    REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
    RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=8}
    ;;
  *)
    echo "Unknown BENCHMARK=${BENCHMARK}. Expected one of: pope, textvqa, gqa, mme." >&2
    exit 1
    ;;
esac

DEFAULT_OUT="${REPO_ROOT}/visualization/snapshots/pdrop/${BENCHMARK}_single_case.pt"

if [ -z "${DATASET_DIR:-}" ]; then
  if [ -d "${REFERENCE_DATASET_DIR}" ]; then
    DATASET_DIR="${REFERENCE_DATASET_DIR}"
  else
    DATASET_DIR="${LOCAL_DATASET_DIR}"
  fi
fi

QUESTION_FILE=${QUESTION_FILE:="${DATASET_DIR}/${DEFAULT_QUESTION_REL}"}
IMAGE_FOLDER=${IMAGE_FOLDER:="${DATASET_DIR}/${DEFAULT_IMAGE_REL}"}
OUT=${OUT:="${DEFAULT_OUT}"}

if [ ! -f "${QUESTION_FILE}" ]; then
  echo "Missing question file: ${QUESTION_FILE}" >&2
  echo "Set BENCHMARK, DATASET_DIR, or QUESTION_FILE explicitly. Tried DATASET_DIR=${DATASET_DIR}" >&2
  exit 1
fi

if [ ! -d "${IMAGE_FOLDER}" ]; then
  echo "Missing image folder: ${IMAGE_FOLDER}" >&2
  echo "Set BENCHMARK, DATASET_DIR, or IMAGE_FOLDER explicitly. Tried DATASET_DIR=${DATASET_DIR}" >&2
  exit 1
fi

# PyramidDrop's eval modules import its vendored `llava` package by bare name, so its
# directory has to be on the path alongside the repo root.
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/PyramidDrop:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m visualization.capture_pdrop_multiple_cases \
  --model-path "${CKPT}" \
  --conv-mode "${CONV_MODE}" \
  --question-file "${QUESTION_FILE}" \
  --image-folder "${IMAGE_FOLDER}" \
  --cases "${CASES}" \
  --temperature "${TEMPERATURE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --layer_list "${LAYER_LIST}" \
  --image_token_ratio_list "${IMAGE_TOKEN_RATIO_LIST}" \
  --redistribution_strategy "${REDISTRIBUTION_STRATEGY}" \
  --redistribution_softmax_mode "${REDISTRIBUTION_SOFTMAX_MODE}" \
  --receiver_token_count "${RECEIVER_TOKEN_COUNT}" \
  --out "${OUT}"

echo "Saved ${BENCHMARK} snapshots under $(dirname "${OUT}")/${BENCHMARK}/"
echo "Latest snapshot pointer: ${OUT}"

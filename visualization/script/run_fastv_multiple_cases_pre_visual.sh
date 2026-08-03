#!/bin/bash
# Capture MANY examples from a benchmark through the current FastV + sink-redistribution
# pipeline and save one snapshot per question_id. The benchmark defaults mirror
# scripts/FastV/cross/{pope,textvqa,gqa,mme}_hf.sh, while the capture itself reuses one
# model/processor load via visualization.capture_fastv_multiple_cases.
#
# Usage:
#   ./visualization/script/run_fastv_multiple_cases.sh
#   CASES=1-10 BENCHMARK=pope ./visualization/script/run_fastv_multiple_cases.sh
#   CASES=3- BENCHMARK=textvqa ./visualization/script/run_fastv_multiple_cases.sh 0
#   DATASET_DIR=/project/aimm/pp/MTK/eval BENCHMARK=gqa CASES=5 ./visualization/script/run_fastv_multiple_cases.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPU_ID=${1:-0}
BENCHMARK=${BENCHMARK:-pope}
CKPT=${CKPT:="llava-hf/llava-1.5-7b-hf"}
REVISION=${REVISION:="a272c74"}
DEVICE=${DEVICE:="cuda"}
DTYPE=${DTYPE:="float16"}
CASES=${CASES:="1-10"}

VISUAL_TOKEN_NUM=${VISUAL_TOKEN_NUM:=576}
FASTV_K=${FASTV_K:=5}
FASTV_R=${FASTV_R:=0.77}
IMAGE_TOKEN_START_INDEX=${IMAGE_TOKEN_START_INDEX:=5}

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
# RECEIVER_IMPORTANCE_SOURCE=${RECEIVER_IMPORTANCE_SOURCE:="pre_visual"}
# Split the two redistribution roles by default on this wrapper:
# pick receivers by pre_visual, then weight the redistributed sink budget by cross.
RECEIVER_SELECTION_SOURCE=${RECEIVER_SELECTION_SOURCE:="pre_visual"}
RECEIVER_WEIGHT_SOURCE=${RECEIVER_WEIGHT_SOURCE:="cross"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:=8}
MIN_NEW_TOKENS=${MIN_NEW_TOKENS:=0}

LOCAL_DATASET_DIR="${REPO_ROOT}/dataset/playground/data/eval"

case "${BENCHMARK}" in
  pope)
    REFERENCE_DATASET_DIR="/project/aimm/danzel/eval"
    DEFAULT_QUESTION_REL="pope/llava_pope_test.jsonl"
    DEFAULT_IMAGE_REL="pope/val2014"
    REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="post_softmax_resoftmax"}
    RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}
    DEFAULT_OUT="${REPO_ROOT}/visualization/snapshots/pope/pre_visual_single_case.pt"
    ;;
  textvqa)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="textvqa/llava_textvqa_val_v051_ocr.jsonl"
    DEFAULT_IMAGE_REL="textvqa/train_images"
    REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
    RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}
    DEFAULT_OUT="${REPO_ROOT}/visualization/snapshots/textvqa/pre_visual_single_case.pt"
    ;;
  gqa)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="gqa/llava_gqa_testdev_balanced.jsonl"
    DEFAULT_IMAGE_REL="gqa/data/images"
    REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
    RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}
    DEFAULT_OUT="${REPO_ROOT}/visualization/snapshots/gqa/pre_visual_single_case.pt"
    ;;
  mme)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="MME/llava_mme.jsonl"
    DEFAULT_IMAGE_REL="MME/MME_Benchmark_release_version/MME_Benchmark"
    REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="resoftmax"}
    RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=8}
    DEFAULT_OUT="${REPO_ROOT}/visualization/snapshots/mme/pre_visual_single_case.pt"
    ;;
  *)
    echo "Unknown BENCHMARK=${BENCHMARK}. Expected one of: pope, textvqa, gqa, mme." >&2
    exit 1
    ;;
esac

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

# RECEIVER_SOURCE_ARGS=(--receiver-importance-source "${RECEIVER_IMPORTANCE_SOURCE}")
if [ -n "${RECEIVER_SELECTION_SOURCE}" ]; then
  RECEIVER_SOURCE_ARGS+=(--receiver-selection-source "${RECEIVER_SELECTION_SOURCE}")
fi
if [ -n "${RECEIVER_WEIGHT_SOURCE}" ]; then
  RECEIVER_SOURCE_ARGS+=(--receiver-weight-source "${RECEIVER_WEIGHT_SOURCE}")
fi

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m visualization.capture_fastv_multiple_cases \
  --model-id "${CKPT}" \
  --revision "${REVISION}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --question-file "${QUESTION_FILE}" \
  --image-folder "${IMAGE_FOLDER}" \
  --cases "${CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --min-new-tokens "${MIN_NEW_TOKENS}" \
  --visual_token_num "${VISUAL_TOKEN_NUM}" \
  --fastv_k "${FASTV_K}" \
  --fastv_r "${FASTV_R}" \
  --image_token_start_index "${IMAGE_TOKEN_START_INDEX}" \
  --redistribution-strategy "${REDISTRIBUTION_STRATEGY}" \
  --redistribution-softmax-mode "${REDISTRIBUTION_SOFTMAX_MODE}" \
  --receiver-token-count "${RECEIVER_TOKEN_COUNT}" \
  "${RECEIVER_SOURCE_ARGS[@]}" \
  --out "${OUT}"

archive_root="${OUT%.*}"
for suffix in _single_case _case; do
  if [[ "${archive_root}" == *"${suffix}" ]]; then
    archive_root="${archive_root%${suffix}}"
    break
  fi
done

echo "Saved ${BENCHMARK} snapshots under ${archive_root}/"
echo "Latest snapshot pointer: ${OUT}"

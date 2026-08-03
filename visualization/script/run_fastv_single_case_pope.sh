#!/bin/bash
# Run ONE POPE example through the current FastV + sink-redistribution pipeline and
# dump a debug snapshot (.pt) for offline visualization. Mirrors the defaults of
# scripts/FastV/cross/pope_hf.sh. Does not touch the FastV source or the batch
# benchmark scripts at all - this only calls
# visualization/capture_fastv_single_case.py, a separate read-only capture script.
#
# Usage:
#   ./visualization/script/run_fastv_single_case_pope.sh
#   QUESTION_ID=123 ./visualization/script/run_fastv_single_case_pope.sh
#   DATASET_DIR=/project/aimm/danzel/eval LINE_INDEX=3 ./visualization/script/run_fastv_single_case_pope.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPU_ID=${1:-0}
CKPT=${CKPT:="llava-hf/llava-1.5-7b-hf"}
REFERENCE_DATASET_DIR="/project/aimm/danzel/eval"
LOCAL_DATASET_DIR="${REPO_ROOT}/dataset/playground/data/eval"

if [ -z "${DATASET_DIR:-}" ]; then
  if [ -d "${REFERENCE_DATASET_DIR}" ]; then
    DATASET_DIR="${REFERENCE_DATASET_DIR}"
  else
    DATASET_DIR="${LOCAL_DATASET_DIR}"
  fi
fi

QUESTION_FILE=${QUESTION_FILE:="${DATASET_DIR}/pope/llava_pope_test.jsonl"}
IMAGE_FOLDER=${IMAGE_FOLDER:="${DATASET_DIR}/pope/val2014"}
QUESTION_ID=${QUESTION_ID:-1}
LINE_INDEX=${LINE_INDEX:-0}

VISUAL_TOKEN_NUM=${VISUAL_TOKEN_NUM:=576}
FASTV_K=${FASTV_K:=5}
FASTV_R=${FASTV_R:=0.77}
IMAGE_TOKEN_START_INDEX=${IMAGE_TOKEN_START_INDEX:=5}

REDISTRIBUTION_STRATEGY=${REDISTRIBUTION_STRATEGY:="topk_text_visual_tokens"}
REDISTRIBUTION_SOFTMAX_MODE=${REDISTRIBUTION_SOFTMAX_MODE:="post_softmax_resoftmax"}
RECEIVER_TOKEN_COUNT=${RECEIVER_TOKEN_COUNT:=32}
# cross-attention that selects receivers + weights the split for the actual redistribution/pruning
# ("cross" = in-decoder real attention, "pre_visual" = pre-decoder V_self). Note the snapshot stores
# ALL THREE cross-attention variants for visualization regardless of this choice.
RECEIVER_IMPORTANCE_SOURCE=${RECEIVER_IMPORTANCE_SOURCE:="pre_visual"}

OUT=${OUT:="${REPO_ROOT}/visualization/snapshots/pope_single_case.pt"}

if [ ! -f "${QUESTION_FILE}" ]; then
  echo "Missing question file: ${QUESTION_FILE}" >&2
  echo "Set DATASET_DIR or QUESTION_FILE explicitly. Tried DATASET_DIR=${DATASET_DIR}" >&2
  exit 1
fi

if [ ! -d "${IMAGE_FOLDER}" ]; then
  echo "Missing image folder: ${IMAGE_FOLDER}" >&2
  echo "Set DATASET_DIR or IMAGE_FOLDER explicitly. Tried DATASET_DIR=${DATASET_DIR}" >&2
  exit 1
fi

SELECT_ARGS=(--line-index "${LINE_INDEX}")
if [ -n "${QUESTION_ID}" ]; then
  SELECT_ARGS=(--question-id "${QUESTION_ID}")
fi

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m visualization.capture_fastv_single_case \
  --model-id "${CKPT}" \
  --question-file "${QUESTION_FILE}" \
  --image-folder "${IMAGE_FOLDER}" \
  "${SELECT_ARGS[@]}" \
  --visual_token_num "${VISUAL_TOKEN_NUM}" \
  --fastv_k "${FASTV_K}" \
  --fastv_r "${FASTV_R}" \
  --image_token_start_index "${IMAGE_TOKEN_START_INDEX}" \
  --redistribution-strategy "${REDISTRIBUTION_STRATEGY}" \
  --redistribution-softmax-mode "${REDISTRIBUTION_SOFTMAX_MODE}" \
  --receiver-token-count "${RECEIVER_TOKEN_COUNT}" \
  --receiver-importance-source "${RECEIVER_IMPORTANCE_SOURCE}" \
  --out "${OUT}"

echo "Saved snapshot (with the three cross-attention variants) to ${OUT}"
echo "Render the cross-attention view with: QUESTION_ID=${QUESTION_ID} ./visualization/script/run_fastv_cross_visualizations_pope.sh"

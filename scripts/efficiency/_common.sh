#!/bin/bash
# Shared setup for the efficiency runs. Sourced by every scripts/efficiency/*.sh.
#
# Usage from any of them:   bash scripts/efficiency/<method>.sh [gpu_id]
#
# Follows the repo convention of calling plain `python`, so activate the right conda env first
# (see check_family below -- family A needs transformers 4.37.x, family B needs FastV's vendored
# 4.39). Everything is overridable by env var.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${1:-0}"

DATASET="${DATASET:-textvqa}"
DATASET_DIR="${DATASET_DIR:-/project/aimm/pp/MTK/eval}"
QUESTION_FILE="${QUESTION_FILE:-${DATASET_DIR}/textvqa/llava_textvqa_val_v051_ocr.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATASET_DIR}/textvqa/train_images}"
ANNOTATION_FILE="${ANNOTATION_FILE:-${DATASET_DIR}/textvqa/TextVQA_0.5.1_val.json}"

# Family A = LLaVA-repo forks (PyramidDrop, SparseVLM) on transformers 4.37.2 + liuhaotian weights.
# Family B = FastV's HF path on its vendored transformers 4.39 + llava-hf weights.
CKPT_A="${CKPT_A:-liuhaotian/llava-v1.5-7b}"
CKPT_B="${CKPT_B:-llava-hf/llava-1.5-7b-hf}"

MODE="${EFFICIENCY_MODE:-bench}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/efficiency/runs}"
ANSWERS_ROOT="${ANSWERS_ROOT:-${REPO_ROOT}/efficiency/answers}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

# bench: a subset on an idle GPU. count: the full set, ~free, safe alongside accuracy runs.
if [ "${MODE}" = "bench" ]; then
    EFFICIENCY_LIMIT="${EFFICIENCY_LIMIT:-200}"
    EFFICIENCY_WARMUP="${EFFICIENCY_WARMUP:-5}"
else
    EFFICIENCY_WARMUP="${EFFICIENCY_WARMUP:-0}"
fi

# Fail before a multi-minute model load if the env is wrong for this family.
check_family() {
    local want="$1"  # A | B
    local ver
    ver="$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo none)"
    if [ "${ver}" = "none" ]; then
        echo "ERROR: no transformers in the active python (`which python`)." >&2
        exit 1
    fi
    case "${want}" in
      A) case "${ver}" in
           4.37.*) ;;
           *) echo "ERROR: family A (PyramidDrop/SparseVLM) expects transformers 4.37.x, found ${ver}." >&2
              echo "       Activate the LLaVA-fork env, e.g. 'conda activate V2Drop'." >&2
              exit 1 ;;
         esac ;;
      B) case "${ver}" in
           4.39.*) ;;
           *) echo "ERROR: family B (FastV HF path) expects FastV's vendored transformers 4.39.x, found ${ver}." >&2
              echo "       No conda env on this machine currently has it. Install it once into a" >&2
              echo "       dedicated env (see FastV/README.md:146):" >&2
              echo "         conda create -n fastv python=3.10 && conda activate fastv" >&2
              echo "         pip install -e ${REPO_ROOT}/FastV/src/FastV/llava-hf/transformers" >&2
              echo "         pip install pillow torch accelerate" >&2
              echo "       Without it 'import transformers' resolves to stock 4.37, which has no" >&2
              echo "       fastv_forward and ignores fastv_config -- the run would silently be vanilla." >&2
              exit 1 ;;
         esac ;;
    esac
}

# Export the recorder's config. Instrumentation is off unless EFFICIENCY_MODE is set, so this is
# the only thing that turns it on.
eff_env() {
    local method="$1"
    export EFFICIENCY_MODE="${MODE}"
    export EFFICIENCY_METHOD="${method}"
    export EFFICIENCY_DATASET="${DATASET}"
    export EFFICIENCY_OUT="${RUNS_DIR}/${DATASET}-${method}-${MODE}.jsonl"
    export EFFICIENCY_WARMUP="${EFFICIENCY_WARMUP}"
    [ -n "${EFFICIENCY_LIMIT:-}" ] && export EFFICIENCY_LIMIT="${EFFICIENCY_LIMIT}"

    ANSWER_FILE="${ANSWERS_ROOT}/${DATASET}/${method}-${MODE}.jsonl"
    mkdir -p "$(dirname "${ANSWER_FILE}")" "${RUNS_DIR}"

    if [ ! -f "${QUESTION_FILE}" ]; then
        echo "ERROR: question file not found: ${QUESTION_FILE}" >&2
        exit 1
    fi

    echo "=============================================================="
    echo " method   : ${method}"
    echo " mode     : ${MODE}${EFFICIENCY_LIMIT:+  (limit ${EFFICIENCY_LIMIT}, warmup ${EFFICIENCY_WARMUP})}"
    echo " gpu      : ${GPU_ID}"
    echo " dataset  : ${DATASET}"
    echo " efficiency -> ${EFFICIENCY_OUT}"
    echo " answers    -> ${ANSWER_FILE}"
    echo "=============================================================="
    if [ "${MODE}" = "bench" ]; then
        echo "NOTE: bench mode measures latency -- make sure GPU ${GPU_ID} is otherwise idle," >&2
        echo "      or a neighbouring job lands directly in your TTFT." >&2
    fi
}

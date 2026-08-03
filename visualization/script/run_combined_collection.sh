#!/bin/bash
# Render the combined_collection plots from already-captured snapshots.
#
# Cases are the same 1-based question-file line numbers the capture scripts use
# (`CASES=1-10 ./visualization/script/run_fastv_multiple_cases_pre_visual.sh`), NOT a position
# in the snapshot directory: snapshot files are named by question_id, which for textvqa/gqa/mme
# is an opaque id rather than a case number. This script resolves case -> question_id by reading
# the benchmark's question file, so `3` here is the same case 3 you captured.
#
# Usage:
#   ./visualization/script/run_combined_collection.sh            # every captured snapshot
#   ./visualization/script/run_combined_collection.sh 3          # case 3 only
#   ./visualization/script/run_combined_collection.sh 1-5        # cases 1 through 5
#   ./visualization/script/run_combined_collection.sh 3-         # case 3 to the end
#   BENCHMARK=pope ./visualization/script/run_combined_collection.sh 2
#   VARIANT=cross BENCHMARK=textvqa ./visualization/script/run_combined_collection.sh 1-3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CASES=${1:-}
BENCHMARK=${BENCHMARK:-textvqa}
# pre_visual snapshots live in snapshots/<benchmark>/pre_visual, cross ones in snapshots/<benchmark>
# (see DEFAULT_OUT in run_fastv_multiple_cases{,_pre_visual}.sh).
VARIANT=${VARIANT:-pre_visual}

LOCAL_DATASET_DIR="${REPO_ROOT}/dataset/playground/data/eval"

case "${BENCHMARK}" in
  pope)
    REFERENCE_DATASET_DIR="/project/aimm/danzel/eval"
    DEFAULT_QUESTION_REL="pope/llava_pope_test.jsonl"
    ;;
  textvqa)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="textvqa/llava_textvqa_val_v051_ocr.jsonl"
    ;;
  gqa)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="gqa/llava_gqa_testdev_balanced.jsonl"
    ;;
  mme)
    REFERENCE_DATASET_DIR="/project/aimm/pp/MTK/eval"
    DEFAULT_QUESTION_REL="MME/llava_mme.jsonl"
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

if [ -n "${VARIANT}" ] && [ "${VARIANT}" != "cross" ]; then
  rel="${BENCHMARK}/${VARIANT}"
else
  rel="${BENCHMARK}"
fi
SNAPSHOT_DIR=${SNAPSHOT_DIR:="${REPO_ROOT}/visualization/snapshots/${rel}"}
OUT_DIR=${OUT_DIR:="${REPO_ROOT}/visualization/output/${rel}"}

if [ ! -d "${SNAPSHOT_DIR}" ]; then
  echo "Missing snapshot directory: ${SNAPSHOT_DIR}" >&2
  echo "Capture it first (e.g. CASES=1-10 BENCHMARK=${BENCHMARK} ./visualization/script/run_fastv_multiple_cases_pre_visual.sh)," >&2
  echo "or set SNAPSHOT_DIR explicitly." >&2
  exit 1
fi

render() {
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python3 -m visualization.combined_fastv_collection \
    --snapshot "$@" \
    --out-dir "${OUT_DIR}"
}

# No case spec: hand the whole directory to the renderer, as in visualization/docs.md.
if [ -z "${CASES}" ]; then
  echo "Rendering every snapshot in ${SNAPSHOT_DIR} -> ${OUT_DIR}"
  render "${SNAPSHOT_DIR}"
  exit 0
fi

if [ ! -f "${QUESTION_FILE}" ]; then
  echo "Missing question file: ${QUESTION_FILE}" >&2
  echo "It is needed to map case numbers to question_ids. Set DATASET_DIR or QUESTION_FILE," >&2
  echo "or re-run without a case number to render every captured snapshot." >&2
  exit 1
fi

# Map the 1-based inclusive case spec onto question_ids, mirroring parse_case_range() in
# visualization/capture_fastv_multiple_cases.py.
mapfile -t question_ids < <(
  CASES="${CASES}" QUESTION_FILE="${QUESTION_FILE}" python3 - <<'PY'
import json
import os
import sys

spec = os.environ["CASES"].strip()
with open(os.environ["QUESTION_FILE"], "r", encoding="utf-8") as handle:
    questions = [json.loads(line) for line in handle if line.strip()]
total = len(questions)

try:
    if "-" in spec:
        head, _, tail = spec.partition("-")
        start = int(head) if head.strip() else 1
        end = int(tail) if tail.strip() else total
    else:
        start = end = int(spec)
except ValueError:
    sys.exit(f"Invalid case spec {spec!r}. Expected N, A-B, or A- (1-based, inclusive).")

if start < 1 or end < start:
    sys.exit(f"Invalid case spec {spec!r} (parsed start={start}, end={end}).")
if start > total:
    sys.exit(f"Case start {start} is past the last case ({total}).")

# Benchmarks repeat question_ids across case lines (textvqa: 5000 lines, 3166 unique ids), and
# snapshots are named by question_id -- so keep first occurrence to avoid re-rendering one
# snapshot several times into the same output folder.
seen = set()
for question in questions[start - 1:min(end, total)]:
    question_id = question["question_id"]
    if question_id in seen:
        continue
    seen.add(question_id)
    print(question_id)
PY
)

if [ ${#question_ids[@]} -eq 0 ]; then
  echo "No cases resolved from '${CASES}'." >&2
  exit 1
fi

snapshot_paths=()
missing=()
for question_id in "${question_ids[@]}"; do
  snapshot_path="${SNAPSHOT_DIR}/${question_id}.pt"
  if [ -f "${snapshot_path}" ]; then
    snapshot_paths+=("${snapshot_path}")
  else
    missing+=("${question_id}")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  # An open-ended spec like `6-` can resolve to thousands of uncaptured cases; only name a few.
  preview=("${missing[@]:0:8}")
  suffix=""
  if [ ${#missing[@]} -gt ${#preview[@]} ]; then
    suffix=" ..."
  fi
  echo "Not captured in ${SNAPSHOT_DIR}, skipping ${#missing[@]}: ${preview[*]}${suffix}" >&2
fi

if [ ${#snapshot_paths[@]} -eq 0 ]; then
  echo "None of the requested cases have snapshots in ${SNAPSHOT_DIR}." >&2
  echo "Capture them first with CASES=${CASES} BENCHMARK=${BENCHMARK} ./visualization/script/run_fastv_multiple_cases_pre_visual.sh" >&2
  exit 1
fi

echo "Rendering ${#snapshot_paths[@]} case(s) from ${SNAPSHOT_DIR} -> ${OUT_DIR}"
render "${snapshot_paths[@]}"

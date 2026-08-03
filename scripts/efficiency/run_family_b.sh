#!/bin/bash
# Every family-B row (FastV's HF path: vanilla + FastV + FastV-cross).
#
#   bash scripts/efficiency/run_family_b.sh [gpu_id]
#   EFFICIENCY_MODE=bench bash scripts/efficiency/run_family_b.sh 0
#
# Needs an env with FastV's vendored transformers 4.39 installed -- a DIFFERENT env from family A
# (see FastV/README.md:146). fastv_vanilla runs first: it is family B's denominator.

set -uo pipefail
GPU_ID="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for s in fastv_vanilla fastv fastv_cross; do
    echo
    echo "###### ${s} ######"
    if bash "${HERE}/${s}.sh" "${GPU_ID}"; then
        echo "[ok] ${s}"
    else
        echo "[FAILED] ${s} (continuing)" >&2
    fi
done

echo
echo "Done. Table:  python -m efficiency.aggregate"

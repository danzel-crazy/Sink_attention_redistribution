#!/bin/bash
# Every family-A row (vanilla + PyramidDrop + SparseVLM, each with its cross variant), in order.
#
#   bash scripts/efficiency/run_family_a.sh [gpu_id]                 # count mode (full set)
#   EFFICIENCY_MODE=bench bash scripts/efficiency/run_family_a.sh 0  # latency (subset, idle GPU)
#
# Activate the LLaVA-fork env first (transformers 4.37.x), e.g. `conda activate V2Drop`.
# vanilla runs first on purpose: it is the denominator for every other row here.

set -uo pipefail
GPU_ID="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for s in vanilla pdrop pdrop_cross sparsevlm sparsevlm_cross; do
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

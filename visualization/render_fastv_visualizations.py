"""Read a snapshot .pt written by capture_fastv_single_case.py and run the tagged
visualizations against it. No model, no GPU, no dataset needed - this file's only job
is to make plotting iteration fast (edit a topic module, rerun this, look at the new
PNG).

Visualizations live in per-topic modules, each writing into its own subfolder of
--out-dir (sink_token_visualizations.py -> sink_tokens/, attention_visualizations.py
-> attention/). Importing those modules registers their plots into
VISUALIZATION_REGISTRY. To add a topic: create `<topic>_visualizations.py` with its own
SUBDIR, then add one import line below.

Usage:
    python -m visualization.render_fastv_visualizations \\
        --snapshot visualization/snapshots/pope_single_case.pt \\
        --out-dir visualization/output
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from visualization.fastv_snapshot import load_snapshot
from visualization.viz_registry import VISUALIZATION_REGISTRY

# Imported for their side effect: each module registers its @visualization plots into
# VISUALIZATION_REGISTRY at import time.
from visualization import attention_visualizations  # noqa: F401
from visualization import cross_attention_visualize  # noqa: F401
from visualization import sink_token_visualizations  # noqa: F401


def _safe_dirname(value) -> str:
    """Turn a question id into a filesystem-safe folder name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="visualization/output")
    parser.add_argument(
        "--no-question-subdir",
        dest="question_subdir",
        action="store_false",
        help="Write straight into --out-dir instead of an out-dir/<question_id>/ subfolder.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        choices=sorted(VISUALIZATION_REGISTRY.keys()) or None,
        help="Subset of registered visualizations to run (default: all of them).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot = load_snapshot(args.snapshot)
    out_dir = Path(args.out_dir)
    if args.question_subdir:
        out_dir = out_dir / _safe_dirname(snapshot.question_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing visualizations to {out_dir}")

    names = args.only or sorted(VISUALIZATION_REGISTRY.keys())
    if not names:
        print("No visualizations registered - are the topic modules imported?")
        return

    for name in names:
        paths = VISUALIZATION_REGISTRY[name](snapshot, out_dir)
        if not paths:
            print(f"[{name}] produced nothing (empty input?)")
            continue
        for path in paths:
            print(f"[{name}] wrote {path}")


if __name__ == "__main__":
    main()

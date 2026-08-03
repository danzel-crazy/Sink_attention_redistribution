"""Run a RANGE of POPE examples through the current FastV + sink-redistribution pipeline
and dump each one to its own snapshot .pt file.

This is the multi-case counterpart to `capture_fastv_single_case`. It loads the model /
processor exactly once and then loops over the requested 1-based case range, reusing the
same per-question capture logic (`capture_question`) so every snapshot is byte-for-byte
what the single-case script would have produced for that line.

The range is given as `--cases START-END` (1-based, inclusive), matching the user's mental
model: `--cases 1-10` runs case 1 through case 10. A single number (`--cases 5`) runs just
that case, and an open-ended `--cases 3-` runs from case 3 to the end of the file.

Usage:
    python -m visualization.capture_fastv_multiple_cases \\
        --model-id llava-hf/llava-1.5-7b-hf \\
        --question-file /path/to/llava_pope_test.jsonl \\
        --image-folder /path/to/val2014 \\
        --cases 1-10 \\
        --out visualization/snapshots/pope_single_case.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from FastV.src.FastV.inference.eval.inference import (
    build_fastv_config,
    resolve_dtype,
)
from visualization.capture_fastv_single_case import (
    _archive_path,
    _refresh_latest_pointer,
    capture_question,
    load_model_and_processor,
    resolve_device,
)
from visualization.fastv_snapshot import save_snapshot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--revision", type=str, default="a272c74")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])

    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument(
        "--cases",
        type=str,
        default="1-10",
        help="1-based inclusive case range: 'START-END' (e.g. 1-10), a single case ('5'), or open-ended ('3-').",
    )

    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-new-tokens", type=int, default=0)

    parser.add_argument("--out", type=str, default="visualization/snapshots/pope_single_case.pt")

    # fastv_config knobs - same names/defaults as scripts/FastV/cross/pope_hf.sh
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--fastv_k", type=int, default=5)
    parser.add_argument("--fastv_r", type=float, default=0.77)
    parser.add_argument("--image_token_start_index", type=int, default=5)

    # sink_token_selector (cross_attention_sink_redistribution/sink_tokens.py)
    parser.add_argument("--sink-dims", type=int, nargs="+", default=[1453, 2533])
    parser.add_argument("--sink-score-min", type=float, default=None)
    parser.add_argument("--sink-score-max", type=float, default=None)
    parser.add_argument("--sink-score-quantile", type=float, default=0.99)

    # cross_attention_importants (cross_attention_sink_redistribution/cross_attention.py)
    parser.add_argument("--disable-sink-masked", action="store_false", dest="enable_sink_masked")
    parser.add_argument(
        "--text-tokens-length",
        type=int,
        default=None,
        help="Defaults to --image_token_start_index (system-prompt prefix length) if unset.",
    )

    # sink_attention_redistributor (cross_attention_sink_redistribution/attention_redistribution.py)
    parser.add_argument("--redistribution-ratio", type=float, default=1.0)
    parser.add_argument("--redistribution-strategy", type=str, default="topk_text_visual_tokens")
    parser.add_argument("--redistribution-softmax-mode", type=str, default="post_softmax_resoftmax")
    parser.add_argument("--receiver-token-count", type=int, default=32)
    parser.add_argument("--receiver-score-power", type=float, default=1.0)
    parser.add_argument("--receiver-importance-source", type=str, default="pre_visual", choices=("cross", "pre_visual"))
    # Optionally source selection and weighting independently; each defaults (None) to
    # --receiver-importance-source. Variant: --receiver-selection-source pre_visual
    # --receiver-weight-source cross (pick receivers by pre-visual, split budget by in-decoder cross).
    parser.add_argument("--receiver-selection-source", type=str, default=None, choices=("cross", "pre_visual"))
    parser.add_argument("--receiver-weight-source", type=str, default=None, choices=("cross", "pre_visual"))
    parser.set_defaults(enable_sink_masked=True)
    return parser


def parse_cli() -> argparse.Namespace:
    args = _build_parser().parse_args()
    args.use_fastv = True  # this script only exists to capture the fastv+sink path
    # capture_question reads args.question_id / args.line_index via question dict, but the
    # single-case _archive_path only needs question_id (taken from the snapshot). Nothing else
    # in the shared code touches these, so we don't need to set them.
    return args


def load_questions(question_file: str) -> List[Dict[str, Any]]:
    with open(os.path.expanduser(question_file), "r") as file_handle:
        return [json.loads(line) for line in file_handle]


def parse_case_range(cases: str, total: int) -> Tuple[int, int]:
    """Parse a 1-based inclusive '--cases' spec into 0-based [start, end) indices.

    Accepts 'A-B', a single 'A', or open-ended 'A-'. Clamps to [1, total]."""
    text = cases.strip()
    if "-" in text:
        start_str, end_str = text.split("-", 1)
        start = int(start_str) if start_str.strip() else 1
        end = int(end_str) if end_str.strip() else total
    else:
        start = end = int(text)
    if start < 1 or end < start:
        raise ValueError(f"Invalid --cases {cases!r} (parsed start={start}, end={end}).")
    if start > total:
        raise ValueError(f"--cases start {start} is past the last case ({total}).")
    end = min(end, total)
    return start - 1, end  # 0-based half-open


def main():
    args = parse_cli()
    device = resolve_device(args)
    dtype = resolve_dtype(args.dtype, device)
    fastv_config = build_fastv_config(args)

    questions = load_questions(args.question_file)
    start_idx, end_idx = parse_case_range(args.cases, len(questions))

    print(f"Loading model {args.model_id} (revision={args.revision}) ...")
    model, processor = load_model_and_processor(args, device, dtype, fastv_config)

    out_path = Path(args.out)
    saved_paths: List[Path] = []
    for zero_based in range(start_idx, end_idx):
        case_number = zero_based + 1
        question = questions[zero_based]
        snapshot = capture_question(args, model, processor, question, device, dtype, fastv_config)

        archive_path = _archive_path(out_path, snapshot.question_id)
        saved_path = save_snapshot(snapshot, archive_path)
        saved_paths.append(saved_path)
        print(
            f"[case {case_number}/{end_idx}] question_id={snapshot.question_id} "
            f"answer={snapshot.generated_answer!r} -> {saved_path}"
        )

    if saved_paths:
        # Keep --out as a "latest" pointer at the last case captured, matching single-case behavior.
        _refresh_latest_pointer(out_path, saved_paths[-1])
        print(f"captured {len(saved_paths)} case(s); latest pointer: {out_path} -> {saved_paths[-1]}")
    else:
        print("No cases captured (empty range).")


if __name__ == "__main__":
    main()

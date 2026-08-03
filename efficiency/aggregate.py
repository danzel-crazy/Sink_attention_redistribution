"""Aggregate efficiency runs into the comparison table.

    python -m efficiency.aggregate                     # everything under efficiency/runs/
    python -m efficiency.aggregate --dir DIR --csv out.csv

Latency is grouped by *family* (transformers version + weights), because absolute milliseconds
are not comparable across the forks in this repo -- they vendor 4.31.0, 4.37.2 and 4.39.0.dev0.
Within a family, runs are normalised against that family's vanilla row, and the speedup ratio is
the number that carries across families. FLOPs, being analytic, compare globally.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def load_runs(directory):
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            continue
        summary_path = path.replace(".jsonl", ".summary.json")
        summary = {}
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
        runs.append({"path": path, "rows": rows, "summary": summary})
    return runs


def family_of(run):
    """A latency-comparable group: same transformers version and same attention implementation."""
    env = run["summary"].get("env", {})
    return f"{env.get('transformers', '?')}/{env.get('attn_impl', '?')}"


def summarise(run):
    rows = run["rows"]
    first = rows[0]
    return {
        "method": first.get("method"),
        "dataset": first.get("dataset"),
        "mode": first.get("mode"),
        "family": family_of(run),
        "n": len(rows),
        "prompt": _mean([r["prompt_tokens"] for r in rows]),
        "prefill_eq": _mean([r["equiv_prefill_tokens"] for r in rows]),
        "total": _mean([r["total_tokens"] for r in rows]),
        "prefill_tflops": _mean([r["prefill_tflops"] for r in rows]),
        "vision_tflops": _mean([r.get("vision_tflops") for r in rows]),
        "kv_mb": _mean([r.get("kv_cache_mb") for r in rows]),
        "ttft_ms": _mean([r.get("ttft_ms") for r in rows]),
        "vision_ms": _mean([r.get("vision_ms") for r in rows]),
        "prefill_ms": _mean([r.get("prefill_ms") for r in rows]),
        "ms_per_tok": _mean([r.get("decode_ms_per_tok") for r in rows]),
        "peak_mb": _mean([r.get("peak_mem_mb") for r in rows]),
        "peak_delta_mb": _mean([r.get("peak_mem_delta_mb") for r in rows]),
    }


def merge_runs(stats):
    """Fold every run of the same (family, method) into one row.

    The intended workflow produces two runs per method -- `count` over the full benchmark and
    `bench` over a subset -- and they must land in one row, not two. FLOPs and token counts are
    taken from the `count` runs (more samples, and they are deterministic anyway); latency is taken
    only from `bench` runs, since `count` runs never measure it and a `count` row must never become
    the latency reference.
    """
    by_key = defaultdict(list)
    for s in stats:
        by_key[(s["family"], s["method"])].append(s)

    merged = []
    for (family, method), group in by_key.items():
        count_runs = [s for s in group if s["mode"] == "count"] or group
        bench_runs = [s for s in group if s["mode"] == "bench"]
        row = {"family": family, "method": method,
               "dataset": next((s["dataset"] for s in group if s["dataset"]), None),
               "n_count": sum(s["n"] for s in count_runs),
               "n_bench": sum(s["n"] for s in bench_runs)}
        for k in ("prompt", "prefill_eq", "total", "prefill_tflops", "vision_tflops", "kv_mb"):
            row[k] = _mean([s[k] for s in count_runs])
        for k in ("ttft_ms", "vision_ms", "prefill_ms", "ms_per_tok", "peak_mb", "peak_delta_mb"):
            row[k] = _mean([s[k] for s in bench_runs]) if bench_runs else None
        merged.append(row)
    return merged


def _fmt(v, spec=".1f"):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return format(v, spec)


def render(stats):
    by_family = defaultdict(list)
    for s in merge_runs(stats):
        by_family[s["family"]].append(s)

    out = []
    for family, group in sorted(by_family.items()):
        vanilla = next((s for s in group if s["method"] in ("vanilla", "llava")), None)
        out.append(f"\n=== family: transformers {family} ===")
        if vanilla is None:
            out.append("  ! no vanilla row: FLOPs reduction and speedup cannot be computed here")
        header = (
            f"{'method':<22} {'n':>9} {'prompt':>7} {'prefill':>8} {'total':>6} "
            f"{'TFLOPs':>7} {'vs van':>7} {'TTFT':>8} {'vision':>7} {'prefill':>8} "
            f"{'speedup':>8} {'ms/tok':>7} {'KV MB':>7} {'peakΔMB':>8}"
        )
        out.append(header)
        out.append("-" * len(header))
        for s in sorted(group, key=lambda x: x["method"] or ""):
            flops_ratio = (
                f"{s['prefill_tflops'] / vanilla['prefill_tflops']:.2f}x"
                if vanilla and vanilla["prefill_tflops"] else "-"
            )
            speedup = (
                f"{vanilla['ttft_ms'] / s['ttft_ms']:.2f}x"
                if vanilla and vanilla.get("ttft_ms") and s.get("ttft_ms") else "-"
            )
            n = f"{s['n_count']}c/{s['n_bench']}b"
            out.append(
                f"{(s['method'] or '?'):<22} {n:>9} "
                f"{_fmt(s['prompt'], '.0f'):>7} {_fmt(s['prefill_eq'], '.0f'):>8} "
                f"{_fmt(s['total'], '.0f'):>6} {_fmt(s['prefill_tflops'], '.2f'):>7} "
                f"{flops_ratio:>7} {_fmt(s['ttft_ms']):>8} {_fmt(s['vision_ms']):>7} "
                f"{_fmt(s['prefill_ms']):>8} {speedup:>8} {_fmt(s['ms_per_tok']):>7} "
                f"{_fmt(s['kv_mb'], '.0f'):>7} {_fmt(s['peak_delta_mb'], '.0f'):>8}"
            )
        if group and group[0]["vision_tflops"]:
            out.append(
                f"\n  vision tower: {group[0]['vision_tflops']:.2f} TFLOPs (constant across "
                f"methods; bounds achievable TTFT speedup)"
            )
    out.append(
        "\nNotes: n is <count-mode samples>c/<bench-mode samples>b.\n"
        "'prefill' is mean tokens per layer (methods prune at different depths, so there is no "
        "single prefill length).\n"
        "TFLOPs are prefill-only, LLM-only, MAC=1, SwiGLU 3ndm -- not comparable with published "
        "numbers.\nLatency compares only within a family; use the speedup column across families."
    )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RUNS_DIR)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        print(f"no runs found in {args.dir}")
        return
    stats = [summarise(r) for r in runs]
    print(render(stats))

    if args.csv:
        import csv

        rows = merge_runs(stats)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

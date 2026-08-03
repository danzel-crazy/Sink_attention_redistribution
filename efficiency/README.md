# Running the efficiency counter

Instrumentation is **off unless `EFFICIENCY_MODE` is set**. With it unset, every eval runs exactly
as before — no hooks are registered and no files are written. So you turn it on by prefixing env
vars onto the commands you already run; no script edits, no new CLI flags.

## Env vars

| var | meaning |
| --- | --- |
| `EFFICIENCY_MODE` | `count` (tokens + FLOPs, ~free, safe during accuracy runs) or `bench` (adds latency + peak memory) |
| `EFFICIENCY_OUT` | output `.jsonl` path (default `efficiency/runs/<run_id>.jsonl`) |
| `EFFICIENCY_DATASET` | dataset label for the table, e.g. `textvqa` |
| `EFFICIENCY_METHOD` | override the auto-detected method label |
| `EFFICIENCY_LIMIT` | stop after N recorded samples (`bench` subsets) |
| `EFFICIENCY_WARMUP` | discard first N samples (default 3; CUDA context + autotune land there) |

## Scripts

One script per row, each taking the GPU id as `$1`. They are standalone — they do **not** use the
`scripts/{FastV,PyramidDrop,SparseVLMs}/...` eval scripts, which currently have their inference
command commented out and a stale `REPO_ROOT=/tmp2/danzel/attention-bias`.

| script | row | family |
| --- | --- | --- |
| `scripts/efficiency/vanilla.sh` | unpruned LLaVA | A |
| `scripts/efficiency/pdrop.sh` | PyramidDrop | A |
| `scripts/efficiency/pdrop_cross.sh` | PyramidDrop + redistribution | A |
| `scripts/efficiency/sparsevlm.sh` | SparseVLM | A |
| `scripts/efficiency/sparsevlm_cross.sh` | SparseVLM + redistribution | A |
| `scripts/efficiency/fastv_vanilla.sh` | unpruned LLaVA on FastV's HF path | B |
| `scripts/efficiency/fastv.sh` | FastV | B |
| `scripts/efficiency/fastv_cross.sh` | FastV + redistribution | B |

The two families need **different conda envs** (family A: transformers 4.37.x; family B: FastV's
vendored 4.39). Each script preflight-checks the active env and exits with instructions rather
than failing halfway through a model load.

```bash
conda activate V2Drop                                  # family A (transformers 4.37.2)
bash scripts/efficiency/run_family_a.sh 0              # vanilla + pdrop + sparsevlm + both cross

conda activate fastv                                   # family B (see FastV/README.md:146)
bash scripts/efficiency/run_family_b.sh 0              # fastv_vanilla + fastv + fastv_cross
```

Or one at a time:

```bash
bash scripts/efficiency/pdrop.sh 3
RETAINED_TOKENS=64 bash scripts/efficiency/sparsevlm_cross.sh 3
```

## Two-pass workflow

**Pass 1 — `count` (default), on the full set.** Deterministic, no syncs, no measurable overhead;
safe to run as your normal accuracy eval. Gives prompt/prefill/total tokens, TFLOPs, KV-cache size.

**Pass 2 — `bench`, on a subset, alone on the GPU.** Gives TTFT, vision, prefill, ms/tok, peak mem.

```bash
EFFICIENCY_MODE=bench bash scripts/efficiency/run_family_a.sh 0     # defaults: limit 200, warmup 5
```

Output paths already include the mode (`textvqa-pdrop-count.jsonl` vs `textvqa-pdrop-bench.jsonl`),
so the two passes never overwrite each other, and the aggregator merges them into one row per
method automatically.

## Knobs

All overridable per run: `DATASET_DIR`, `QUESTION_FILE`, `IMAGE_FOLDER`, `CKPT_A`, `CKPT_B`,
`MAX_NEW_TOKENS`, `LAYER_LIST`, `RATIO_LIST`, `RETAINED_TOKENS`, `FASTV_K`, `FASTV_R`,
`RECEIVER_TOKEN_COUNT`, `REDISTRIBUTION_STRATEGY`, `EFFICIENCY_LIMIT`, `EFFICIENCY_WARMUP`.

## The table

```bash
python -m efficiency.aggregate                 # reads efficiency/runs/
python -m efficiency.aggregate --csv table.csv
```

```
=== family: transformers 4.37.2/sdpa ===
method            n  prompt  prefill  total  TFLOPs  vs van    TTFT  vision  prefill  speedup  ms/tok  KV MB  peakΔMB
pdrop         5000c/200b   612      187    616    1.42   0.31x    91.2    28.4     61.1    1.83x    22.7    148      892
vanilla       5000c/200b   612      612    616    4.58   1.00x   167.0    28.4    137.2    1.00x    24.1    482     2140
```

## Reading it without fooling yourself

- **`prefill` is mean tokens per layer**, not a sequence length. Each method prunes at a different
  depth, so there is no single prefill length; the full per-layer trace is in `layer_token_trace`
  on every row if you need to see the schedule.
- **Compare latency only within a family.** The forks vendor transformers 4.31.0 / 4.37.2 /
  4.39.0.dev0. FastV additionally runs **eager attention on all layers** (hardcoded in
  `inference.py`, plus `output_attentions=True`), while PDrop and SparseVLM run SDPA. FastV's
  absolute TTFT is therefore mostly a measurement of eager attention, not of FastV. Its FLOPs are
  still globally comparable — those are analytic.
- **`vs van` (FLOPs) is the honest cross-method number.** `speedup` is only meaningful against a
  vanilla row in the same family — which is why family B has its own (`--no-use-fastv`). Because
  that row keeps FastV's eager attention, family B's speedup cleanly isolates pruning; do not read
  family A's and family B's absolute milliseconds against each other.
- **Use `peakΔMB`, not `peak`.** Absolute peak is ~14GB of fp16 weights and barely moves.
  `KV MB` is the analytic, noise-free version of the same story.
- **`vision` is constant across methods** (~all encode 576 tokens) and bounds achievable TTFT
  speedup — it is why a 3× LLM FLOPs cut never gives a 3× TTFT cut.
- **ms/tok is confounded by answer length**, which differs per method. Pin `--max_new_tokens` when
  comparing.

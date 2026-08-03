# Global efficiency counter — plan

Goal: one file format + one aggregator so FastV, PyramidDrop, SparseVLM, the `_cross`
(sink-redistribution) variants, and vanilla LLaVA all report FLOPs and latency in the same table.

Target columns: `prompt | prefill | total | TTFT = vision + prefill | ms/tok | peak`, plus TFLOPs.

## 1. The enabling fact

All three methods **physically truncate** `hidden_states` rather than masking:

| method | site |
| --- | --- |
| FastV | `FastV/src/FastV/llava-hf/transformers/.../modeling_llama.py:1171` |
| PyramidDrop | `PyramidDrop/llava/model/modeling_llama_pdrop.py:1237-1277` |
| SparseVLM | `modelling_sparse_llama.py`, at `pruning_loc=[2,6,15]` |

So a `forward_pre_hook` on each decoder layer reading `args[0].shape[1]` recovers the true
per-layer token count for **any** method, with zero edits to any fork's modeling code.
One instrumentation, not four.

## 2. What already exists

`SparseVLMs/llava/model/language_model/modelling_sparse_llama.py:234` already accumulates the
FastV-paper formula `4nd² + 2n²d + 3ndm` and CUDA-event-times the decoder stack, then `print`s
to stdout at `:391`. FastV and PyramidDrop have nothing. This plan lifts that idea out of the
fork, makes it method-agnostic, and gives it a file format.

## 3. Fork matrix — what is and isn't comparable

| pipeline | transformers | weights | latency family |
| --- | --- | --- | --- |
| PyramidDrop | 4.37.2 | `llava-v1.5-7b` (liuhaotian) | **A** |
| SparseVLM | 4.37.2 | `llava-v1.5-7b` | **A** (see caveat) |
| FastV (`inference.py`, HF path) | 4.39.0.dev0 | `llava-hf/llava-1.5-7b-hf` | **B** — `attn_implementation="eager"`, all layers |
| FastV (`src/LLaVA` loader) | 4.31.0 | `llava-v1.5-7b` | **C** — pre-SDPA, eager only |

**Consequence — the central design constraint:**

- **FLOPs / token counts are comparable across everything.** They are analytic: a function of the
  per-layer token trace and `(d, m, n_layers)`, all identical for LLaVA-1.5-7B. One global table. ✅
- **Wall-clock latency is only comparable *within* a family.** Different transformers versions mean
  different attention kernels (4.31 has no SDPA for Llama at all). Absolute ms across families is
  meaningless. ⚠️

**Therefore: report latency as speedup relative to that fork's own vanilla baseline.** Every family
gets its own vanilla row; the comparable number is the ratio, not the millisecond.

Vanilla baselines per family:
- A: `PyramidDrop/llava/eval/model_vqa_loader.py` with `--layer_list` omitted → plain LLaVA.
- B: `FastV/src/FastV/inference/eval/inference.py` with `--no-use-fastv` → `build_fastv_config`
  returns None → `modeling_llava.py:483` takes the stock `language_model(...)` branch, unpruned.
  Keeps `attn_implementation="eager"` and `output_attentions=True`, i.e. **identical to the FastV
  run in everything except pruning** — which is exactly what makes it the right control: family B's
  speedup column then isolates pruning rather than re-measuring eager attention.
- SparseVLM has no clean vanilla (its `LlamaModel` is always the sparse one) → borrow family A's.

Caveat on A: `SparseVLMs/llava/model/llava_arch.py:95` casts `mm_projector` to **bfloat16**;
PyramidDrop leaves it fp16. Affects the `vision_ms` / projector leg. Record dtype; don't compare
that leg across the two without noting it.

## 4. Module layout

Top-level `efficiency/` package, imported by every fork via PYTHONPATH — matching the existing
precedent of `cross_attention_sink_redistribution*`, which the forks already import this way.

- `hooks.py` — `EfficiencyRecorder`: pre-hooks on `model.model.layers[*]` (per-layer token trace),
  hooks on vision tower + `mm_projector` (vision leg), decode-step counter. CUDA events, never
  `time.time()`.
- `flops.py` — analytic FLOPs from the trace + config. No profiler.
- `writer.py` — one JSONL row per sample; one summary JSON per run.
- `aggregate.py` — globs runs, emits the comparison table.

Two modes:
- **`count`** (default, always on): token trace + FLOPs + KV-cache bytes only. Deterministic,
  ~zero overhead, safe to leave on during normal accuracy evals.
- **`bench`** (`--bench`): adds CUDA-event latency + peak memory. Fixed subset (~200 samples),
  warmup discarded, exclusive GPU. Run separately.

Splitting these is deliberate: FLOPs are exact and free, latency is noisy and needs a quiet GPU.
Collecting them in one pass would let dataloader stalls and JSON writes leak into TTFT.

## 5. Per-sample schema (JSONL)

```json
{"run_id": "...", "method": "fastv", "family": "B", "mode": "bench",
 "config": {"k": 2, "r": 0.77},
 "prompt_tokens": 612, "gen_tokens": 4, "total_tokens": 616,
 "layer_token_trace": [612, 612, 168, 168, "..."],
 "equiv_prefill_tokens": 187.3,
 "prefill_tflops": 1.42, "vision_tflops": 4.31,
 "ttft_ms": 91.2, "vision_ms": 28.4, "prefill_ms": 61.1,
 "decode_ms_per_tok": 22.7,
 "peak_mem_mb": 15204, "peak_mem_delta_mb": 892, "kv_cache_mb": 148}
```

`layer_token_trace` is the load-bearing field. It is the only honest answer to "prefill tokens" —
each method prunes at a different depth, so there is no scalar. Everything else derives from it,
and keeping it makes a wrong FLOPs number debuggable months later.

## 6. Integration points (~4 lines each)

1. `SparseVLMs/llava/eval/model_vqa_loader.py`
2. `SparseVLMs/llava/eval/model_vqa_loader_cross.py`
3. `PyramidDrop/llava/eval/model_vqa_loader.py`  ← also the family-A vanilla row
4. `PyramidDrop/llava/eval/model_vqa_loader_cross.py`
5. `FastV/src/FastV/inference/eval/inference.py`  ← FastV + family-B vanilla row

Out of scope for the first cut: HiMAP, TokenCarve.

## 7. Traps to design around

- **Peak memory looks identical across methods** unless weights are subtracted — 7B fp16 is ~14 GB
  and swamps activation savings. Record `peak - post_load_baseline`. Add analytic `kv_cache_mb`
  from the trace, which shows the benefit deterministically and without measurement noise.
- **FastV is structurally handicapped on wall-clock, worse than expected.** It needs
  `output_attentions=True` at layer K, which alone would force eager attention on that one layer.
  But `inference.py` hardcodes `attn_implementation="eager"` at load and `gen_kwargs` sets
  `output_attentions: True` for the whole run, so **all 32 layers run eager** while PyramidDrop and
  SparseVLM run SDPA. FastV's absolute TTFT is therefore not a statement about FastV — it is mostly
  a statement about eager attention. This is why family B is quarantined from family A, and it is
  the main reason the FLOPs and latency columns will disagree. `attn_impl` is recorded on every run
  so the table can never hide it.
- **Don't mix our FLOPs with published ones.** The in-repo formula uses `3ndm` (correct for LLaMA's
  SwiGLU); some papers use `2ndm`, and the whole formula counts a multiply-add as 1. Recompute all
  methods ourselves; never paste a paper's number into the same column.
- **Vision tower FLOPs are constant across methods** (all encode 576 tokens). Keep as a separate
  column — it is the Amdahl bound on achievable speedup and explains why a 45% LLM FLOPs cut does
  not yield a 45% TTFT cut.
- **decode ms/tok is confounded by answer length** differing per method. Pin `max_new_tokens`.
- **`scripts/FastV/textvqa.sh` is stale** — it targets `FastV.src.FastV.inference.eval.model_vqa_loader`,
  which does not exist. Confirm the live FastV path before wiring.

## 8. Status

Built and unit-tested against a real (small) Llama stack: `count` mode, `bench` mode, the
truncation-tracking trace, warmup/limit, and `aggregate.py`. See `efficiency/README.md` to run it.

Both families now have a vanilla row: family A via omitted `--layer_list`, family B via
`--no-use-fastv`. The vanilla branch already existed in `build_fastv_config` +
`modeling_llava.py:472`; only argparse could not reach it (`--use_fastv` is `store_true` and
`set_defaults` forced it True), so a single `store_false` flag was added. The `model.enable_fastV`
/ `enable_cdpruner` / `fastv_info` / `visual_token_num` attributes set in `inference.py` are
vestigial — nothing in the FastV transformers tree reads them, and `config.fastv_config` is the
only real switch.

Known gaps:

- **Validation against real checkpoints is still pending** — the traces below have only been
  proven on a synthetic stack. First real run should confirm each method's trace matches its
  documented schedule (PDrop's `[2,10,20]` with ratios `[0.32,0.16,0.08]`, FastV's drop at layer K,
  SparseVLM's `pruning_loc=[2,6,15]`).
- Some run scripts carry a stale `REPO_ROOT="/tmp2/danzel/attention-bias"` and have their inference
  command commented out; they need fixing independently of this work.

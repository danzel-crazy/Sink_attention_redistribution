
## save snapshots

1. pre_visual
```
BENCHMARK=pope CASES=1-10 \
./visualization/script/run_fastv_multiple_cases_pre_visual.sh 0

```

2. cross
```
BENCHMARK=pope CASES=1-10 \
./visualization/script/run_fastv_multiple_cases.sh 0

```

## visualize only`
```
python -m visualization.combined_fastv_collection \
  --snapshot visualization/snapshots/textvqa/pre_visual \
  --out-dir visualization/output/textvqa/pre_visual
```

# PyramidDrop

Captures the FIRST pruning stage only (`layer_list[0]`, e.g. layer 2 0-based). At that
stage the visual block is still the full 576-token grid, so local token ids are image-patch
ids and the FastV renderer applies unchanged. Later stages (10, 20) still run normally —
they are just not recorded, since their survivors are renumbered and would need a
survivor->grid map.

Run in the `pdrop` conda env.

## save snapshots
```
BENCHMARK=pope CASES=1-10 \
./visualization/script/run_pdrop_multiple_cases.sh 0
```
Knobs: `LAYER_LIST` (default `[2,10,20]`), `IMAGE_TOKEN_RATIO_LIST` / `RATIO_LIST`
(default `[0.32,0.16,0.08]`), plus the usual `REDISTRIBUTION_*` / `RECEIVER_TOKEN_COUNT`.

## visualize
Same renderer as FastV — the snapshot is a `PrefillDebugSnapshot`, with `fastv_k` holding
the prune layer and `fastv_r` the fraction dropped at that stage (`1 - ratio[0]`).
```
python -m visualization.combined_fastv_collection \
  --snapshot visualization/snapshots/pdrop/pope \
  --out-dir visualization/output/pdrop/pope
```

`kept_tokens.png` in each collection shows the actual keep/drop decision at the prune
layer, read from the recorded `kept_visual_local_ids` (FastV snapshots get it too).

Or via the wrapper, which takes a case number and resolves the paths for you:

```
./visualization/script/run_combined_collection.sh            # every captured snapshot
./visualization/script/run_combined_collection.sh 3          # case 3 only
./visualization/script/run_combined_collection.sh 1-5        # cases 1 through 5
./visualization/script/run_combined_collection.sh 3-         # case 3 to the end
BENCHMARK=pope ./visualization/script/run_combined_collection.sh 2
VARIANT=cross ./visualization/script/run_combined_collection.sh 1-3
```

Case numbers are the same 1-based question-file lines the capture scripts take (`CASES=1-10`),
not positions in the snapshot folder — snapshot files are named by `question_id`, which is an
opaque id on textvqa/gqa/mme. Cases you have not captured are reported and skipped.
`BENCHMARK` defaults to `textvqa`, `VARIANT` to `pre_visual` (use `cross` for the plain
`snapshots/<benchmark>/` folder); `DATASET_DIR`, `QUESTION_FILE`, `SNAPSHOT_DIR`, and `OUT_DIR`
all override.

Each case lands in `<out-dir>/<question_id>/combined_collection/`:

- `baseline_scores_*` / `redistributed_scores_*` — top-8/16/32/64/128 token overlays
- `receiver_weights_on_image.png`, `runtime_sink_positions.png`, `sink_budget_summary.png`
- `redistributed_top_hidden_states/` — hidden states at the prune layer of the top-5 tokens by
  `redistributed_scores`, i.e. the ones FastV keeps after sink redistribution. Lines are colored
  by role (receiver / native) and the two largest `|dim|` per token are marked.
  - `top_5_redistributed_hidden_states_3d.png` — the 5 vectors stacked in 3D
  - `top_5_redistributed_hidden_states_lines.png` — the same vectors as flat per-token line plots
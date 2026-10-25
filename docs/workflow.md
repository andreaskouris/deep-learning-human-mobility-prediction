# Reproduction workflow

These commands cover preprocessing, selected-checkpoint evaluation and analysis
for the six canonical experiments. Use new output locations and the separate
environments described in [installation.md](installation.md).

## Integration prerequisite

Use [installation.md](installation.md) to obtain the official pinned checkout and
run `setup/integrate_bigcity.py`. All eight adaptations are generated locally.
Run model-dependent steps
from `ADAPTED_ROOT/runtime`, using external data/weights/assets. Only controlled
dynamic mode and static ablation none are supported. Legacy modes are excluded.

## Common inputs and support caches

`DATASET_ROOT` contains `xa/`; it is not the city directory itself. The external
artifact manifest is the authoritative list of raw files, generated logical
names and hashes. Full metadata counts are computed from the full trajectory
source and structural/dynamic caches, never copied from an existing metadata JSON.

Within the locally integrated runtime, the shared preprocessing order is:

1. Public `scripts/build_dynamic_cache.py --dynamic-mode controlled --time-slots 2928
   --dataset-root DATASET_ROOT`: decode `.dyna` and construct the controlled cache.
2. `scripts/build_structural_caches.py --city xa --dynamic-mode controlled
   --split full`: build relation/static support caches through FileLoader.
3. `scripts/build_full_metadata.py`: supply `--relation-file`, `--static-file`,
   `--trajectory-file`, `--relation-cache`, `--static-cache`, `--dynamic-cache`,
   `--output`. The trajectory source is `traj_xa_11.csv`.
4. NH split caches: `scripts/build_next_hop_cache.py --city xa --split SPLIT
   --seq-len 64 --sample-rate 1 --dynamic-mode controlled`, once each for train,
   validation and test.
5. TTE split caches: `scripts/verify_tte_dataset.py --trajectory_file FILE
   --split SPLIT --output NEW_CACHE --seq_len 64 --road_count 5269 --seed 2021`.
   Dates, 1,800-second intervals and 2,928 time slots retain their canonical defaults.

**Layout constraint:** the structural and NH builders do not accept arbitrary
`--dataset-root` bindings. Their imported configuration still defaults to
`../dataset` relative to the runtime working directory. Supply that layout
explicitly in the integrated setup. This remains a portability limitation. Their canonical
seed is the shared parser default 2021; neither CLI exposes an independent seed flag.

Use `--verify-only` where supported to validate existing caches. Do not use
`--force` against canonical references. The TTE writer refuses an existing file
and can independently reconstruct it in verify-only mode. New caches intended
for runners must use the manifest's logical names under the configured XA root;
standalone cache-builder output paths do not automatically rebind a runner.

## Semantic branches (independent of trajectory-cache construction)

NH TF-IDF:

```sh
python -B -m scripts.build_semantic_cache --dataset-path "$DATASET_ROOT" --city xa
```

Inputs include road geometry, POI TYPEs, road-to-POI mapping and the `.sem` reference
for validation. This command writes the logical TF-IDF cache beside the XA data;
existing-output handling and `--verify-only` are explicit.

NH Qwen TYPE prototypes, using the separate Qwen environment and an existing
local model snapshot:

```sh
python -B -m scripts.build_language_semantic_cache \
  --dataset-path "$DATASET_ROOT" --city xa --hf-cache "$HF_CACHE" \
  --poi-file "$DATASET_ROOT/xa/POI_xa.csv" --model-path "$QWEN_SNAPSHOT" \
  --output-cache "$CACHE_ROOT/type_prototypes.pth"
python -B -m scripts.build_language_road_semantic_cache \
  --dataset-path "$DATASET_ROOT" --city xa \
  --tfidf-cache "$TFIDF_CACHE" --prototype-cache "$CACHE_ROOT/type_prototypes.pth" \
  --output-cache "$CACHE_ROOT/qwen_poi_roads.pth"
```

NH uses revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, unique NAME embedding
normalization, per-TYPE unique-name means and TF-IDF-weighted raw road sums. The
road-cache builder does not replace those sums with a different pooling rule.

TTE structured build uses only `road_xa.csv`, `POI_xa.csv` and `road2poi_xa.json`.
See the README command. TTE Qwen uses the same binary fact matrix, ten fixed
concept descriptions and a locally supplied Qwen model:

```sh
python -B -m scripts.build_tte_qwen_semantic_cache \
  --structured-cache "$STRUCTURED" --structured-metadata "$STRUCTURED_META" \
  --output-cache "$CACHE_ROOT/tte_qwen.pth" \
  --output-metadata "$CACHE_ROOT/tte_qwen.meta.json" \
  --model-path "$QWEN_SNAPSHOT" --hf-cache "$HF_CACHE" --profile fresh \
  --verify-source-only
```

The displayed form stops before model loading/encoding. Actual generation would
omit `--verify-source-only`. Supply
`--model-revision` only when known; historical TTE revision remains unconfirmed.
Fresh source verification enforces the canonical tensor hash/schema/counts and
source provenance. Historical mode retains the full original structured-file hash.

## Model training and selected-checkpoint evaluation - integrated runtime

The six JSON specifications retain the scientific settings. The public command
resolver prints an argv list with explicit dataset, pretrained, GPT-2, checkpoint
and log roots. It does not launch a subprocess. Review the resulting argv before
separately executing it in the integrated runtime.

NH uses strict validation-accuracy improvement; TTE uses strict validation-MAE
improvement. Both preserve patience 3 and at most 20 epochs. The included runner code
loads the selected checkpoint before final test evaluation. New training is not
required to run the source/static tests.

Historical verification requires external access to the original six checkpoints.
It retains the original SHA256 and result checks. Fresh export uses a manifest:

```json
{
  "schema_version": 1,
  "task": "tte",
  "experiments": {
    "baseline": {
      "checkpoint": "runs/baseline/best_tte_none_seed2021.pth",
      "sha256": "REPLACE_WITH_YOUR_CHECKPOINT_SHA256",
      "semantic_mode": "none",
      "selection_metric": "mae_seconds",
      "validation_selected": true
    }
  }
}
```

This illustrates one entry; the manifest must contain all three task entries.
TTE keys are `baseline`, `structured`, `qwen`; NH keys are `nh1`, `nh2`, `nh3`, with
task `next_hop` and selection metric `accuracy`. Paths resolve relative to the
manifest. Hashes and epochs may differ from historical runs. Checkpoint metadata
must retain the canonical scientific settings and TTE training-only normalization.
The NH runner does not record semantic mode/static ablation, so the manifest
attests mode and strict state loading checks shape/key compatibility. Neither a
manifest nor matching metadata proves the globally best epoch without run history.

Run model-dependent export from the integrated runtime directory:

```sh
python -B -m scripts.export_final_next_hop_per_sample \
  --experiment nh1 --profile fresh --checkpoints-manifest "$NH_MANIFEST" \
  --dataset-path "$DATASET_ROOT" --pretrained-checkpoint "$PRETRAINED" \
  --gpt2-path "$GPT2_DIR" --output-dir "$ANALYSIS_ROOT/next_hop/per_sample"
python -B -m scripts.export_final_tte_per_sample \
  --profile fresh --checkpoints-manifest "$TTE_MANIFEST" \
  --dataset-path "$DATASET_ROOT" --pretrained-checkpoint "$PRETRAINED" \
  --gpt2-path "$GPT2_DIR" --output-dir "$ANALYSIS_ROOT/tte/per_sample"
```

NH needs three invocations (`nh1`, `nh2`, `nh3`). After the first, explicitly use
`--overwrite` to allow the remaining files in the shared output directory. These
commands perform real model evaluation. For historical
mode omit the fresh manifest and bind `--checkpoint-root` to the original logical
`final_results` tree. The upstream pretrained hash remains strict in both profiles.

## Statistics and figures from external predictions

Use the analysis environment. These read per-sample exports, which are not
included. Each task has its own analysis root.

```sh
python -B -m scripts.analyze_final_next_hop_statistics --analysis-root "$ANALYSIS_ROOT/next_hop" --profile fresh
python -B -m scripts.analyze_final_tte_statistics --analysis-root "$ANALYSIS_ROOT/tte" --profile fresh
python -B -m scripts.analyze_final_tte_stratified --analysis-root "$ANALYSIS_ROOT/tte" --profile fresh
python -B -m scripts.plot_final_next_hop_figures_5_1_5_2 --analysis-root "$ANALYSIS_ROOT/next_hop" --profile fresh
```

NH stratification requires the locally generated upstream loader and raw test/cache/TF-IDF inputs:
`analyze_final_next_hop_stratified --analysis-root ... --dataset-path ... --profile fresh`.
Then `plot_final_next_hop_figures_5_3_5_4` consumes the resulting stratified table.
`plot_final_tte_figures` consumes TTE statistics/stratification/predictions and also
creates the duration-bootstrap CSV. Its seed/repetitions/formulas remain unchanged.
Use a fresh output directory or explicit `--overwrite` for an existing directory.
The included figures retain the final thesis output bytes. Coverage and limits of
the reproduction tests are documented in [verification.md](verification.md).

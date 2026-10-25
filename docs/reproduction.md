# Reproduction from the integrated runtime

The intended workflow is:

**Clone thesis source → obtain pinned official BIGCity → integrate locally →
prepare external XA data/caches → train/evaluate the six configurations.**

Read [installation.md](installation.md) first. Source integration was executed and
verified. The model-dependent commands below describe the reproduction workflow;
fresh training, Qwen generation and full-test re-evaluation have not been tested.

## Inputs

Obtain XA sources and the upstream pretrained checkpoint from the providers linked
in the [artifact manifest](../reproducibility/external_artifacts.json). Acquire GPT-2
and Qwen assets separately. DATASET_ROOT is the parent of `xa/`. Match raw input
hashes; the archive/split provenance is incomplete, so newly generated splits are
not silently interchangeable with the canonical sources.

The six fine-tuned checkpoints are not supplied. Historical verification requires
the archived six checkpoints and keeps strict hashes/results. Fresh reproduction
uses validation-selected new checkpoints and their own manifest hashes; historical
fine-tuned bytes are not required. Upstream pretrained identity stays strict.

For the two layout-constrained cache wrappers, set
`DATASET_ROOT="$ADAPTED_ROOT/dataset"` and acquire XA under its `xa/` child after
integration. Dataset/cache contents are outside the owned runtime source inventory.
Keep run checkpoints/logs in a separate RUN_ROOT.

## Preprocess

Run the following in `ADAPTED_ROOT/runtime`, with the tensor environment. Use new
cache destinations; never overwrite canonical references.

```sh
python -B -m scripts.build_dynamic_cache --dataset-root "$DATASET_ROOT" --city xa --dynamic-mode controlled --time-slots 2928
python -B -m scripts.build_structural_caches --city xa --dynamic-mode controlled --split full
python -B -m scripts.build_full_metadata \
  --relation-file "$DATASET_ROOT/xa/roadmap_xa/roadmap_xa.rel" \
  --static-file "$DATASET_ROOT/xa/roadmap_xa/road_features_xa.csv" \
  --trajectory-file "$DATASET_ROOT/xa/traj_xa_11.csv" \
  --relation-cache "$DATASET_ROOT/xa/cached_xa_relation.pth" \
  --static-cache "$DATASET_ROOT/xa/cached_xa_static.pth" \
  --dynamic-cache "$DATASET_ROOT/xa/cached_xa_dynamic_controlled_v1.pth" \
  --output "$DATASET_ROOT/xa/cached_meta_xa_full_seq64_sr1_seed2021_v2.json"
for SPLIT in train validation test; do
  python -B -m scripts.build_next_hop_cache --city xa --split "$SPLIT" --seq-len 64 --sample-rate 1 --dynamic-mode controlled
  python -B -m scripts.verify_tte_dataset --trajectory_file "$DATASET_ROOT/xa/traj_xa_$SPLIT.csv" --split "$SPLIT" \
    --output "$DATASET_ROOT/xa/cached_tte_dataset_${SPLIT}_seq64_sr1_seed2021_v1.pth" --seq_len 64 --road_count 5269 --seed 2021
done
```

The full trajectory metadata source is `traj_xa_11.csv`; counts are computed through
FileLoader, never copied from the existing metadata JSON. The structural/NH cache
wrappers retain the `../dataset` working-directory layout constraint. Bind that
layout explicitly; those two CLIs do not expose arbitrary dataset-root flags.

## Semantic branches

- NH baseline: no semantic cache.
- NH TF-IDF: `build_semantic_cache`, from road geometry, POI TYPEs, road-to-POI links and `.sem` validation reference.
- NH Qwen: the same TF-IDF cache plus `build_language_semantic_cache` TYPE prototypes and `build_language_road_semantic_cache` weighted road vectors.
- TTE baseline: no semantic cache.
- TTE structured: `build_tte_structured_semantic_cache`, from `road_xa.csv`, `POI_xa.csv` and `road2poi_xa.json`.
- TTE Qwen: `build_tte_qwen_semantic_cache`, from the same ten facts and locally acquired Qwen assets.

Use the manifest's logical names for caches consumed by runners, rather than an
unbound scratch basename. NH prototypes use
`cached_xa_language_type_prototypes_qwen3_0_6b_v1.pth`; NH road vectors use
`cached_xa_language_road_semantic_qwen3_0_6b_v1.pth`. TTE uses
`cached_xa_tte_semantics_binary_v1.pth` / `.meta.json` and
`cached_xa_tte_semantics_qwen_v1.pth` / `.meta.json`, under `DATASET_ROOT/xa/`.
A standalone builder's custom output does not automatically rebind a runner.

Exact semantic CLI recipes and their output safeguards are retained in
[workflow.md](workflow.md). NH Qwen snapshot provenance is pinned; the historical
TTE Qwen revision remains unconfirmed. Tensor-level structured-cache identity is
separate from whole `.pth` serialization bytes.

## Train and selected-checkpoint evaluation

Resolve a canonical argv list from the thesis clone:

```sh
python -B -m scripts.portable_runtime_support \
  --experiment experiments/nh_baseline.json --runtime-root "$ADAPTED_ROOT/runtime" \
  --dataset-root "$DATASET_ROOT" --pretrained-checkpoint "$PRETRAINED" \
  --gpt2-path "$GPT2_DIR" --run-root "$RUN_ROOT"
```

Repeat for the other five JSONs. This command only prints arguments. Separately
execute the printed runner argv from the integrated runtime when you intend real
training.

All six retain seed 2021, batch 32, LR 0.0002, weight decay 0, maximum 20 epochs,
patience 3, controlled dynamic mode and static ablation none. NH selects strict
validation-accuracy improvement; TTE selects strict validation-MAE improvement.
TTE normalization is train-only, loss is SmoothL1(beta=1), and predictions are not
clipped before MAE/RMSE/MAPE. NH keeps its original loss and top-5 metric definitions.

The runners reload the selected checkpoint before final test evaluation. Historical
and fresh per-sample exporters, manifest schema, paired statistics and plotting
commands are documented in [workflow.md](workflow.md). They evaluate selected
checkpoints or process external prediction CSVs. The README figures and results
are the final thesis outputs.

## Reproduction limits

All required authored preprocessing/task code is included. Generated source imports,
configurations, source fingerprints and selected synthetic contracts pass. The TTE
hook matches canonical tokens/gradients on small CPU fixtures; real-weight/GPU
numerical equivalence, fresh convergence, cross-platform and clean installation
have not been tested. Historical source comparisons are optional and are not an
installation dependency. See [reproducibility.md](reproducibility.md) for remaining
provenance limits and [LICENSE_STATUS.md](../LICENSE_STATUS.md) for generated-source
redistribution restrictions.

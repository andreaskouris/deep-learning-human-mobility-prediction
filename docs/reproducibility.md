# Dependency maps and reproducibility boundary

## Next-hop

```text
XA .dyna + static road count
  -> controlled dynamic builder/preprocessor [PUBLIC]
XA relation/static sources + controlled cache
  -> structural wrappers -> FileLoader/config [LOCAL UPSTREAM] -> support caches
full XA trajectory table + support caches
  -> full-metadata wrapper -> FileLoader [LOCAL UPSTREAM] -> five computed counts
train/validation/test trajectories
  -> NH cache wrapper -> DatasetNextHop [LOCALLY DERIVED UPSTREAM, integer timestamp subtraction]
  -> read-only CachedNextHopDataset
road geometry + POI TYPEs + links + .sem
  -> TF-IDF matrix (5269 x 204)
POI unique names -> local Qwen -> normalized TYPE prototypes (204 x 1024)
  -> TF-IDF-weighted raw road vectors (5269 x 1024)
selected semantic cache + support/split caches + upstream pretrained + GPT-2
  -> NH runner [PUBLIC] -> locally derived model/tokenizer [graph-batching correction]
  -> validation selection -> selected checkpoint -> test metrics
  -> per-sample exporter -> paired/stratified statistics -> thesis figures
```

## Travel time

```text
shared support caches + full metadata (same common branch as NH)
train/validation/test trajectories + utils.timefeatures [OFFICIAL UPSTREAM]
  -> original TTE preparation -> departure-only inputs + duration targets
  -> split caches; training targets alone determine normalization
road attributes + POI TYPEs + links
  -> independently reconstructed 5269 x 10 binary structured tensor
same ten concepts -> local Qwen concept embeddings (10 x 1024)
  -> sum active concepts per road -> normalize active roads, retain zero roads
support/split/semantic caches + pretrained + GPT-2
  -> TTE runner/task model [PUBLIC] + upstream parent/tokenizer adapter [LOCAL UPSTREAM] -> validation MAE selection -> selected checkpoint
  -> test metrics -> per-sample exporter -> paired/stratified statistics/figures
```

The semantic branches do not depend on trajectory-cache construction. Qwen cache
generation is preprocessing, not the prediction backbone. Semantic projections,
masking, batching and timestamp precision are tested in the locally integrated runtime.

## Source dependencies

The task-specific models, runners, preprocessing and analysis are supplied in this
repository. `setup/integrate_bigcity.py` derives the required upstream modules
from thirteen pinned official source files: eight adapted and five unchanged.
The TTE attention adapter reuses upstream tokenization with the task's PAD-masked
attention callback. The supported profile uses controlled preprocessing and
`static_ablation=none`.

The [source inventory](file_selection.md) records component roles and provenance;
[dependency_chains.json](../reproducibility/dependency_chains.json) maps shared
preprocessing, task caches, semantic branches, runners, export, statistics and
figures for each of the six experiments.

## Verification boundary

Source integration, deterministic bytes, backups, repeated execution and selected
CPU contracts are VERIFIED. The eight original preimages remain in the output's
backup directory, and the supplied checkout is unchanged. Integration uses this
repository and the explicit official checkout.

Six unchanged configurations, 120 required source/cache/time bindings, sixteen
AST/source comparisons and synthetic model-helper/metric/cache tests were checked.
The TTE attention adapter matches canonical outputs and helper gradients in five
small CPU fixtures. Full real-weight/GPU
numerical equality and fresh model convergence remain NOT TESTED. Source fingerprints
or import success alone do not establish those properties.

The optional historical source reference is used only for source comparison,
not assembly or normal reproduction. It may be unavailable publicly; the core
installation/preprocessing/training/evaluation path has no such dependency.

## Remaining limitations

- Generated upstream-derived runtime/backup redistribution rights remain unresolved;
  keep that output local. The public tool contains no full upstream replacement.
- The thesis code has no selected license, and the full thesis citation record is
  incomplete; see [LICENSE_STATUS.md](../LICENSE_STATUS.md) and [CITATION.md](../CITATION.md).
- Original XA archive/split-generation provenance and historical TTE Qwen snapshot
  revision are incomplete. Existing source hashes remain unchanged.
- Structural/NH cache wrappers retain the explicit `../dataset` layout constraint.
- Clean install, other platforms, real-weight/GPU execution, Qwen generation,
  full-test re-evaluation and fresh training have not been tested here.

Historical fine-tuned checkpoint integrity remains strict. Fresh runs record their
own validation-selected checkpoint hashes. [Installation](installation.md) and
[reproduction](reproduction.md) give the commands for each step.

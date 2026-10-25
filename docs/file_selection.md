# Repository and source inventory

The repository contains the thesis implementation for six canonical NH/TTE
experiments. The [machine-readable inventory](../reproducibility/file_selection.json)
records exact paths, roles, dependency evidence and distributed-byte SHA256 values.
Historical source fingerprints identify reference implementations; they are
separate from current distributed-file hashes.

| Location | Role |
| --- | --- |
| `train_next_hop.py`, `train_tte.py` | Training, validation checkpoint selection and selected-checkpoint evaluation |
| `models/` | Semantic enrichment, checkpoint compatibility, TTE regression and masked attention adapter |
| `data_provider/` | Controlled dynamic preprocessing and NH/TTE trajectory-cache handling |
| `config/thesis_paths.py` | Canonical split, seed and semantic-cache bindings |
| `scripts/build_*.py`, `scripts/verify_tte_dataset.py` | Shared support caches, task split caches, metadata and semantic representations |
| `scripts/export_final_*` | Historical or fresh selected-checkpoint prediction export |
| `scripts/analyze_final_*`, `scripts/plot_final_*` | Paired/stratified statistics and thesis figures |
| `scripts/portable_runtime_support.py` | Explicit execution paths and historical/fresh artifact policies |
| `setup/` | Pinned-checkout validation and deterministic BIGCity source adaptation |
| `experiments/` | Six canonical configurations with per-field provenance |
| `environments/` | Separate observed BIGCity, Qwen and analysis dependency snapshots |
| `reproducibility/` | Source/artifact fingerprints, dependency maps and scientific contracts |
| `tests/` | Integration safety and synthetic cache, batching, masking, semantic and metric checks |
| `results/`, `assets/figures/` | Final aggregate results and four thesis figures |
| `docs/` | Installation, reproduction commands, verification and upstream changes |

## External BIGCity source

The integrator reads thirteen modules from the pinned official checkout. Eight
are adapted locally; five are retained byte-for-byte. Their paths and both input
and adapted fingerprints are in
[integration_inputs.json](../reproducibility/integration_inputs.json). The changes
and their scientific purpose are documented in
[UPSTREAM_CHANGES.md](UPSTREAM_CHANGES.md).

The inventory retains the field name `external_private_runtime_only` for schema
compatibility. Its entries describe these external BIGCity modules, not additional
source required from the thesis author.

## Reference metadata and excluded artifacts

`standalone_runtime_plan.json` locks a historical source snapshot used only by the
optional `reference` tests. It does not select sources for ordinary integration.
Historical path/provider labels are retained as provenance.

Datasets, per-sample predictions, weights, generated caches and generated BIGCity
runtime/backups are excluded. The implemented scientific profile is controlled
dynamic preprocessing with `static_ablation=none`; unrelated exploratory and
ablation experiments are outside the repository's scope.

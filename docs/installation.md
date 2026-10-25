# Installation and source integration

Use Python **3.12**. The fingerprinted AST profile was verified with 3.12.10 on
Windows; a clean installation and other platforms have not been tested.
The thesis repository contains the implementation and transformations, not BIGCity
source, datasets, weights or generated caches.

## Thesis checkout byte policy

The repository's `.gitattributes` pins source, JSON and documentation to LF,
independent of `core.autocrlf`. Archived `results/**`, `assets/figures/**`, CSV
and SVG files use `-text` to preserve their exact scientific artifact bytes.
The two source manifests validate raw SHA256 bytes; no newline normalization or
fallback is performed by the integrator. Root `.git` metadata is outside the
source inventory; all declared source files remain strictly fingerprinted.

## Obtain official BIGCity separately

Obtain the pinned source in a separate directory:

```sh
git -c core.autocrlf=false clone --filter=blob:none --no-checkout https://github.com/bigscity/BIGCity.git "$BIGCITY_ROOT"
git -C "$BIGCITY_ROOT" -c core.autocrlf=false checkout --detach 5f7168bf7a43c3a40e19e4a80cdb6d77b080edec
```

Use a separate normal checkout, not a linked worktree or an alternate/shared object
store. HEAD must be the exact commit; tree identity is also checked. Thirteen
required source files must match their pinned LF byte hashes. A previously
obtained verified sparse checkout containing those files is supported. The tool
never fetches, switches branches, changes remotes or repairs the supplied checkout.

## Separate environments

```sh
python -m venv .venv-bigcity
.venv-bigcity/Scripts/python -m pip install -r environments/bigcity.txt
python -m venv .venv-qwen
.venv-qwen/Scripts/python -m pip install -r environments/qwen.txt
python -m venv .venv-analysis
.venv-analysis/Scripts/python -m pip install -r environments/analysis.txt
```

These recipes reflect the observed Windows package snapshots, not a tested clean
install. Do not mix BIGCity Transformers 4.47.1 and Qwen Transformers 4.51.3.
Shell examples use POSIX syntax; adapt variables/continuations for PowerShell.
The integration tool itself requires only the standard library and Git.

## Integrate without changing the checkout

From the thesis clone, choose an existing parent and a new sibling output path:

```sh
python -B setup/integrate_bigcity.py --bigcity-root "$BIGCITY_ROOT" --output "$ADAPTED_ROOT" --dry-run
python -B setup/integrate_bigcity.py --bigcity-root "$BIGCITY_ROOT" --output "$ADAPTED_ROOT"
python -B setup/integrate_bigcity.py --bigcity-root "$BIGCITY_ROOT" --output "$ADAPTED_ROOT" --verify-only
```

Expected statuses are `DRY_RUN_VERIFIED`, `INTEGRATED_VERIFIED`, then
`ALREADY_APPLIED_VERIFIED`. A repeat validates all owned source/output/backup hashes and
performs no writes. Different or damaged existing outputs fail; there is no force
or in-place mode. Keep the output separate from both source directories.

The output contains:

```text
ADAPTED_ROOT/
  runtime/                    86 source/support files
  upstream-backup/             eight exact original preimages
  integration_manifest.json    deterministic source/output fingerprints
```

The tool validates all inputs before writing; backups precede adapted copies. It
compiles Python without importing model code, verifies a fresh temporary directory
and publishes it atomically. A failed publication never leaves a successful-looking
partial target. The original checkout remains unchanged.

All eight transformations operate on the user's verified source AST. The
FineTune prompt functions are derived from that source, not embedded replacements.
Original thesis checkpoint/semantic processing is called through the published
mixin. TTE uses the corrected upstream tokenizer with an independent attention
hook supplied in this repository. Only controlled dynamic mode and
`static_ablation=none` are supported.

## Validate source and selected contracts

From the thesis clone:

```sh
python -B -m scripts.verify_public_candidate --runtime-root "$ADAPTED_ROOT/runtime"
python -B -m scripts.run_selected_tests --group tooling --staging-root "$NEW_TOOLING_TEST_ROOT"
python -B -m scripts.run_selected_tests --group science --staging-root "$NEW_SCIENCE_TEST_ROOT"
python -B -m scripts.run_selected_tests --group runtime --runtime-root "$ADAPTED_ROOT/runtime" --staging-root "$NEW_RUNTIME_TEST_ROOT"
```

Use the BIGCity tensor environment for science/runtime groups. Runtime tests block
real model forward/GPU/network/optimizer calls and use small synthetic caches.
The optional `reference` group also needs an explicit hash-locked historical
source reference. It is not required by integration, preprocessing, training or
fresh evaluation. See [verification.md](verification.md) for tested scope.

Keep generated runtime/backups local while upstream redistribution rights remain
unresolved; see [LICENSE_STATUS.md](../LICENSE_STATUS.md). Source integration has
been tested; real-weight reproduction and clean installation have not.

`ADAPTED_ROOT/dataset` is an allowed external-data sidecar. It is not traversed or
hashed by source verification; data integrity is checked by the artifact manifest
and builders. Other unexpected files in owned runtime/backup trees fail validation.
Use Python `-B` or `PYTHONDONTWRITEBYTECODE=1` for immutable-source verification.

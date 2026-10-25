# Testing and Verification

The selected tests cover source integration and small CPU synthetic cases.
They do not run model training, real-weight inference or full-test evaluation.

| Test group | Verified result |
| --- | --- |
| Tooling and integration safety | 40 passed |
| Preprocessing, caches and metrics | 47 passed |
| Integrated runtime regressions | 76 passed |
| Optional historical source comparisons | 6 passed |
| Total | 169 tests, no failures, errors or skips |

The 163 core tests require only this repository, the pinned official BIGCity
checkout and the recorded environment. The six optional `reference` tests also
require an explicitly supplied, hash-locked historical source snapshot. That
snapshot is not an input to integration, training or fresh evaluation.

## Verified properties

- Thirteen required BIGCity source files match the pinned commit/tree and raw
  LF SHA256 values. Eight adapted outputs are also fingerprinted; five remain
  byte-identical to upstream.
- Integration preserves eight original preimages, writes a deterministic manifest
  and publishes a verified temporary output atomically. Repeated integration
  performs no writes. Modified/missing source, invalid HEAD and damaged output
  are rejected. The unsupported-valid-revision condition is unit-tested; a second
  real valid revision was not acquired.
- Six configurations agree with the actual parser and runners. Checks cover
  120 path/time/cache bindings and 16 source/AST comparisons. Controlled/static-none
  specialization is explicit; other profiles are outside the comparison.
- Synthetic tests cover graph batching and sample isolation, integer timestamp
  precision, padding masks, semantic projections, train-only normalization,
  checkpoint policies, preprocessing invariants and metric definitions.
- Source, JSON and documentation use LF under `.gitattributes`, independent of
  `core.autocrlf`. Archived results, CSVs and SVGs retain exact bytes with `-text`.
  Both source/inventory manifests validate raw bytes without normalization.
  Root `.git` metadata is excluded from the source inventory; changed or extra
  source files still fail validation.

## Run the selected tests

After [installation and integration](installation.md), use fresh output paths:

```sh
python -B -m scripts.verify_public_candidate --runtime-root "$ADAPTED_ROOT/runtime"
python -B -m scripts.run_selected_tests --group tooling --staging-root "$NEW_TOOLING_TEST_ROOT"
python -B -m scripts.run_selected_tests --group science --staging-root "$NEW_SCIENCE_TEST_ROOT"
python -B -m scripts.run_selected_tests --group runtime --runtime-root "$ADAPTED_ROOT/runtime" --staging-root "$NEW_RUNTIME_TEST_ROOT"
```

Use the BIGCity environment for science/runtime tests. The test driver blocks
network, optimizer and real-model forward calls. Tensor-helper gradient checks
use synthetic data and perform no optimizer step. The optional `reference` group
also needs `--reference-root` pointing to its historical source snapshot.

## Limits of the verification

The TTE attention adapter matches canonical tokens and helper gradients on five
small CPU fixtures. This does not establish complete pretrained-weight/GPU
numerical equivalence. Source hashes and imports alone do not establish it either.

Clean installation, other platforms, fresh training convergence, Qwen cache
generation and full-test re-evaluation have not been tested for this implementation.
The displayed results are the archived thesis measurements. See
[reproducibility.md](reproducibility.md) for data/model provenance limitations and
[LICENSE_STATUS.md](../LICENSE_STATUS.md) for redistribution restrictions.

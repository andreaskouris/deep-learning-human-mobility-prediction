# BIGCity adaptations

BIGCity is obtained from its [official repository](https://github.com/bigscity/BIGCity)
at commit `5f7168bf7a43c3a40e19e4a80cdb6d77b080edec`, tree
`e6d2eb9e0b969a45589e4ebb26a1ae0e1b598275`.

`setup/integrate_bigcity.py` validates that checkout and applies eight targeted
AST transformations. Original and adapted fingerprints are recorded in
[integration_inputs.json](../reproducibility/integration_inputs.json).
The scientific profile is controlled dynamic preprocessing and
`static_ablation=none` for the six canonical configurations.

| Upstream module | Adaptation and purpose | Used by | Verification |
| --- | --- | --- | --- |
| `config/__init__.py` | Export selected arguments and initialize the configured seed | All six experiments | Imports, seed and configuration tests |
| `config/args_config.py` | Add split, seed, pretrained/GPT-2 paths and semantic/profile arguments | All six experiments and builders | Six configurations checked against 22 parser fields |
| `config/global_vars.py` | Bind split/seed-specific caches and semantic paths | All six experiments | 120 canonical path/time/cache bindings |
| `data_provider/file_loader.py` | Preserve observed zeros, impute absent roads and apply controlled normalization | All six experiments, structural caches and metadata | Controlled-profile class AST comparison and cache/metadata tests |
| `data_provider/data_loader.py` | Retain int64 timestamps and subtract the integer epoch origin | NH experiments and split caches | NH dataset AST comparison and time-boundary tests |
| `models/bigcity4finetune.py` | Add checkpoint-key compatibility, semantic enrichment and recomputed prompts while retaining the canonical freeze policy | All six experiments | Forward/prompt AST comparisons and synthetic initialization tests |
| `models/st_tokenizer.py` | Use graph stride `N+1` to account for the PAD road | All six experiments | Static-none AST comparison and batch/sample-isolation tests |
| `models/backbone.py` | Read GPT-2 configuration and weights from an explicit local path | All six experiments | Complete portable-backbone AST comparison |

The first seven adaptations support the canonical research implementation. The
backbone adaptation makes GPT-2 asset location explicit. These changes preserve
the canonical losses, hyperparameters, metrics, cache representations and
checkpoint-selection rules.

## TTE attention and preprocessing

`models/tte_tokenization_adapter.py` supplies the TTE adapter included in this
repository. It calls the corrected upstream `StTokenizer.forward` through an
attribute view, using the TTE model's PAD-masked attention callback. It registers
no additional parameters or layers. Five CPU fixtures match canonical tokens and
helper gradients; full real-weight/GPU numerical equality has not been tested.

The generated FileLoader calls `preprocess_controlled` from the thesis
preprocessing module. The legacy `original` dynamic mode and functional static
ablations are unsupported by this canonical integration profile.

## Unchanged modules and local output

Five modules are used unchanged: `config/logging_config.py`,
`config/random_seed.py`, `models/bigcity.py`, `models/layers.py` and
`utils/timefeatures.py`. Package markers supplied by this repository are
independently authored.

The integrator backs up eight original preimages before writing adapted files to
a new output directory. It verifies the result before an atomic rename. Repeated
integration validates existing bytes without rewriting them; mismatched sources
or outputs fail explicitly. The supplied BIGCity checkout is never modified.

Generated runtime files and backups contain upstream code. Keep them local while
redistribution rights are unresolved; see
[LICENSE_STATUS.md](../LICENSE_STATUS.md) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Installation and tested scope
are described in [installation.md](installation.md) and
[verification.md](verification.md).

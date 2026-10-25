# Upstream adaptations and source-only integration

Official BIGCity is external: [source](https://github.com/bigscity/BIGCity), commit
`5f7168bf7a43c3a40e19e4a80cdb6d77b080edec`, tree
`e6d2eb9e0b969a45589e4ebb26a1ae0e1b598275`. The pinned tree has no standalone
LICENSE/COPYING file. Public source does not imply redistribution permission.

`setup/integrate_bigcity.py` reads the user's independent checkout and this thesis
clone only. Eight transparent AST transformations preserve upstream code locally;
no upstream replacement, encoded payload or private-source fallback is published.
The exact original and adapted hashes are in `reproducibility/integration_inputs.json`.
Full canonical research/portable references from earlier phases remain private and
unchanged. They are not integration inputs.

| Original upstream path | Required canonical change / reason | Canonical users | Application / verification | Material in generated full file | Rights |
| --- | --- | --- | --- | --- | --- |
| `config/__init__.py` | Export selected args and seed with args.seed; no import-time logging | All six | Edit import/seed AST; actual imports and seed/config tests | Contains upstream code | Generated locally; redistribution unresolved |
| `config/args_config.py` | Add seed, split, checkpoint/GPT-2 and canonical semantic/profile arguments | All six and builders | Insert literal declarations; actual six JSON/22-field parser validation | Contains upstream code | Local only |
| `config/global_vars.py` | Split/seed-specific caches, complete trajectories and semantic paths | All six | Append independent thesis binding hook; 120 required path/time values match canonical | Contains upstream code | Local only |
| `data_provider/file_loader.py` | Preserve observed zeros, impute absent roads, controlled normalization | All six, structural/metadata | Replace legacy processor region with public controlled function; complete class AST matches under controlled specialization; cache/metadata tests | Contains upstream code | Local only |
| `data_provider/data_loader.py` | NH int64 timestamps and integer origin subtraction | All NH / NH split preprocessing | Exact targeted AST edits; complete NH class AST equality and boundary tests | Contains upstream code | Local only |
| `models/bigcity4finetune.py` | Pretrained-key compatibility, freeze policy, semantics and recomputed prompts | All six | Public checkpoint/semantic mixin; derive prompt functions from user's source; forward/prompt AST equality and synthetic constructor tests | Contains upstream code | Local only |
| `models/st_tokenizer.py` | Correct graph stride to N+1 including PAD road | All six | Single guarded AST expression change; static-none AST normalization and sample-isolation tests | Contains upstream code | Local only |
| `models/backbone.py` | Explicit local GPT-2 config/weights | All six | Change source path expressions, require local_files_only; full canonical module AST equality | Contains upstream code | Local only |

The first seven are the original research adaptations. The eighth is the existing
PHASE4 portability change. No new task loss, layer, metric, hyperparameter, cache
representation or checkpoint-selection algorithm is introduced.

## Previously private reused bodies

- **TTE tokenize_tte:** the old complete body mirrors upstream tokenization and
  remains in private historical references. PHASE6 does not copy it. A newly
  authored adapter calls the locally corrected `StTokenizer.forward` through a
  lightweight attribute view whose attention callback uses the existing TTE PAD
  mask. It registers no parameters/layers. Five CPU fixtures produce exactly the
  canonical tokens and helper gradients. Full real-weight/GPU equivalence remains
  PARTIALLY VERIFIED, not established by source fingerprints alone.
- **Legacy preprocess_original:** none of the six runs uses this mode. The locally
  generated loader calls only the already canonical public `preprocess_controlled`.
  Original mode is explicitly unsupported; its old body remains privately archived,
  not silently imported or reconstructed from a replacement bundle.

The supported parser profile limits dynamic mode to controlled and static ablation
to none; their defaults now match that profile. BGE/functional/original modes are
not claimed as reproduced. The six unchanged configurations explicitly select the
same canonical modes/hyperparameters as before.

## Inputs, backups and status

Exactly thirteen original source modules are needed: eight adaptations above plus
five unchanged modules (`config/logging_config.py`, `config/random_seed.py`,
`models/bigcity.py`, `models/layers.py`, `utils/timefeatures.py`). Three package
markers are independently authored. No whole BIGCity tree is copied into thesis
source or integrated runtime.

All eight transformations and the independent TTE hook are automated from public
thesis code plus the pinned checkout; **no authorized private payload is required
for these six source chains**. Eight original files are backed up in the local output
before transformed copies are written. Wrong inputs fail before publication; a
verified temporary directory is atomically renamed into a new output. Repeat
integration verifies bytes and performs no writes.

This establishes VERIFIED source integration and selected synthetic behavior.
Real training/weights, clean installation, historical XA/TTE-Qwen provenance and
legal redistribution of generated upstream-derived output remain open. The tool
is independently authored, but no legal certification or author-approved license
is invented. See [installation.md](installation.md) and [verification.md](verification.md).

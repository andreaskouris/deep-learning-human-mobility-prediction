# PHASE6 verification

Only source/static checks, isolated integration and small CPU synthetic tests ran.
There was no training, optimizer step, actual model inference, Qwen encoding,
full-test measurement, model/data download or package change.

| Check | Actual result |
| --- | --- |
| Tooling/integration-safety tests | 40 passed; zero failures/errors/skips |
| Public preprocessing/cache/metric tests | 47 passed; zero failures/errors/skips |
| Integrated-runtime regression tests | 76 passed; zero failures/errors/skips |
| Optional historical reference comparisons | Six passed; zero failures/errors/skips |
| Total distinct selected tests | 169 (163 core + six reference-only) |
| Source/AST comparisons | 16 passed; explicit profile normalization where required |
| Canonical path/time/cache bindings | 120 equal values across six experiments |
| Canonical JSON configurations | Six byte-identical files |
| Real official-source integration | Dry-run + deterministic outputs + verify-only + unchanged-source backups |
| Repeated integration | Entire output bytes deterministic; existing file mtimes unchanged |
| Negative real-checkout cases | Modified preimage, missing file and invalid/nonexistent HEAD rejected before target creation |
| Unsupported valid revision condition | Unit-tested Git-boundary response; another real valid revision was not downloaded |
| Atomic failure handling | Injected verification failure leaves no published/partial target |

## VERIFIED

Thirteen pinned official source files are checked against commit/tree and raw LF
SHA256. All eight adaptations run from the user's AST and public thesis code;
eight original preimages are preserved byte-identically. The output has 86
runtime/support files plus eight backups and a deterministic manifest. No private
payload or thesis-history object is read. The existing checkout remains unchanged.

The independent thesis copy contains no `.git`; guarded integration and the core
163 tests run without archive/private-source access. Existing package environments
are reused. This proves source-provider independence, not clean installation.

FineTune forward/prompt methods, NH integer loader and portable backbone match
canonical executable ASTs. FileLoader and graph tokenizer comparisons explicitly
specialize controlled/static-none behavior; no equivalence is claimed for legacy
modes. Original semantic, checkpoint and TTE task functions retain their scientific
bodies. Detailed checks are in `../reproducibility/source_equivalence.json`.

## PARTIALLY VERIFIED

The independent TTE hook calls the corrected upstream tokenizer through a small
attention view rather than publishing the copied historical body. Five tiny CPU
fixtures match tokens and helper gradients exactly; existing batching/isolation,
PAD masking, semantic projection, REG packing and initialization tests also pass.
That does not prove full pretrained-weight/GPU numerical equality.

## BLOCKED

Redistribution of generated upstream-derived source/backups remains unresolved.
The author's release license and complete thesis citation record are not selected.
No actual private input blocks the six canonical source-integration chains.

## NOT TESTED

Clean installation, cross-platform execution, full real-weight/GPU model behavior,
new training convergence, Qwen embedding generation and new full-test metrics.
Provider acquisition commands are documented recipes; no unnecessary download was
performed. The archived README results/figures are not new measurements.

Use the actual commands in [installation.md](installation.md). Optional `reference`
tests require an explicitly supplied historical source snapshot; the core path
and ordinary reproduction do not. All tests use fresh staging outputs and block
optimizer/network/real-model forward calls. Synthetic tensor-helper autograd tests
perform no optimizer step or model fine-tuning.

## Git checkout integrity correction

The initial patch reconstruction reproduced 28/73 runtime-source mismatches and
54/99 populated public-inventory mismatches. All were line-ending differences:
22 uniform CRLF plus six mixed LF/CRLF runtime files; 46 uniform CRLF plus eight
mixed LF/CRLF inventory entries. No different executable or artifact content was
found. Source LF bytes are now pinned by `.gitattributes`; archived results,
figures, CSV and SVG bytes remain untouched under `-text` attributes.

Two independent ordinary Git checkouts, with `core.autocrlf=true` and `false`,
match 73/73 runtime-source and 100/100 populated public-inventory SHA256 values.
All 101 public files match their Git blobs. `git fsck --full --strict` succeeds.
Dry-run, integration, verify-only, no-write repeat and all 169 selected tests
pass in each checkout (163 core plus six explicit historical-oracle tests).
The core checks deny research/private-history source reads; the existing Python
environment is reused. Only root `.git` metadata is outside source inventory;
changed or extra source files still fail strict raw-byte integrity checks.

The scientific Python ASTs, six configuration files, all canonical result/figure
bytes and historical external artifact reference hashes are unchanged. The two
executable edits affect only inventory verification and its focused tests.

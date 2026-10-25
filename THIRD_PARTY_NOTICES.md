# Third-party attribution and distribution boundary

- **BIGCity**, Xie Yu, Jingyuan Wang, Yifan Yang, Qian Huang and Ke Qu:
  [paper](https://arxiv.org/abs/2412.00953),
  [official source](https://github.com/bigscity/BIGCity), pinned commit
  `5f7168bf7a43c3a40e19e4a80cdb6d77b080edec`.
  The pinned source inventory has no standalone LICENSE/COPYING/NOTICE file.
  Permission to redistribute upstream source, adapted modules or reused
  implementation bodies is not assumed. No upstream implementation is vendored here.
- **BIGCity XA data and pretrained checkpoint**: obtain externally using the
  providers linked in `reproducibility/external_artifacts.json`. Download
  availability is not treated as permission to redistribute trajectories or weights.
- **GPT-2 assets**: required by the BIGCity backbone and tokenizer, acquired
  separately. The manifest records canonical local asset identities; no GPT-2
  asset is copied into this candidate.
- **Qwen3-Embedding-0.6B**: [official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
  Weights remain external. NH uses the documented pinned snapshot; historical
  TTE snapshot provenance is incomplete. Review the model's terms directly.
- PyTorch, Transformers, PEFT, PyTorch Geometric, Sentence Transformers, NumPy,
  pandas, SciPy and Matplotlib remain external packages under their own terms.
  Environment snapshots identify observed versions; they do not relicense
  dependencies or guarantee a clean installation.

The included figures were generated for the thesis's six canonical experiments,
not copied from BIGCity's article illustrations. Required notices are preserved.
Technical source review is documented in `reproducibility/file_selection.json`;
it is not legal certification.

Attribution, a citation, a patch or a replacement file does not itself establish
redistribution rights. Generated upstream-derived code stays local; no private input is required for integration.


The source-only integration boundary is documented in
[UPSTREAM_CHANGES.md](docs/UPSTREAM_CHANGES.md). Official source is read at integration
time. No private replacement bundle is required. Generated upstream-derived output
and backups must remain local while redistribution rights are unresolved.

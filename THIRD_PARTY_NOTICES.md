# Third-party notices

- **BIGCity**, Xie Yu, Jingyuan Wang, Yifan Yang, Qian Huang and Ke Qu:
  [paper](https://arxiv.org/abs/2412.00953),
  [official source](https://github.com/bigscity/BIGCity), pinned commit
  `5f7168bf7a43c3a40e19e4a80cdb6d77b080edec`.
  The pinned source inventory has no standalone LICENSE/COPYING/NOTICE file.
  BIGCity source is obtained externally and adapted locally; it is not vendored
  in this repository. Redistribution rights for generated source and backups
  remain unresolved. Attribution and source transformations do not establish
  permission to redistribute upstream material.
- **BIGCity XA data and pretrained checkpoint:** acquire them from the providers
  linked in [external_artifacts.json](reproducibility/external_artifacts.json).
  Download availability is not permission to redistribute trajectories or weights.
- **GPT-2 assets:** required by the BIGCity backbone and tokenizer; acquired
  separately. Canonical asset identities are recorded in the artifact manifest.
- **Qwen3-Embedding-0.6B:** see the
  [official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
  Weights remain external. The NH snapshot is pinned; historical TTE snapshot
  provenance is incomplete. The provider's terms apply.
- **Software dependencies:** PyTorch, Transformers, PEFT, PyTorch Geometric,
  Sentence Transformers, NumPy, pandas, SciPy and Matplotlib retain their own
  licenses. Environment snapshots record observed versions.

The included figures were generated for the thesis experiments. They are not
BIGCity article illustrations. Source provenance and dependency roles are recorded
in [file_selection.json](reproducibility/file_selection.json); integration changes
are described in [UPSTREAM_CHANGES.md](docs/UPSTREAM_CHANGES.md).

The thesis code's current licensing position is in
[LICENSE_STATUS.md](LICENSE_STATUS.md).

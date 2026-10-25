"""Build Qwen TYPE prototypes from unique POI names."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer


CACHE_VERSION = "v1"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = (
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
)

EXPECTED_TYPE_COUNT = 204
EXPECTED_EMBEDDING_DIM = 1024

EXPECTED_XA_POI_ROWS = 19107
EXPECTED_XA_UNIQUE_NAMES = 15005
EXPECTED_XA_NAME_TYPE_PAIRS = 15455

MAX_SEQ_LENGTH = 32


def sha256_file(path: Path) -> str:
    

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def require_file(
    path: Path,
    label: str,
) -> None:
    """Require one source file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )


def require_directory(
    path: Path,
    label: str,
) -> None:
    """Require one source directory."""

    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )


def load_poi_pairs(
    *,
    city: str,
    poi_path: Path,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
]:
    """Load and validate unique NAME-TYPE memberships."""

    poi = pd.read_csv(
        poi_path,
        dtype={
            "id": str,
            "NAME": str,
            "TYPE": str,
        },
        keep_default_na=False,
    )

    required_columns = {
        "id",
        "NAME",
        "TYPE",
    }

    missing_columns = (
        required_columns
        -
        set(poi.columns)
    )

    if missing_columns:
        raise ValueError(
            "POI file is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    if city == "xa":
        if len(poi) != EXPECTED_XA_POI_ROWS:
            raise ValueError(
                "Unexpected Xi'an POI row count: "
                f"{len(poi)}"
            )

    blank_names = (
        poi["NAME"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if bool(blank_names.any()):
        raise ValueError(
            "Blank POI NAME values detected."
        )

    blank_types = (
        poi["TYPE"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if bool(blank_types.any()):
        raise ValueError(
            "Blank POI TYPE values detected."
        )

    poi["NAME"] = (
        poi["NAME"].astype(str)
    )

    poi["TYPE"] = (
        poi["TYPE"].astype(str)
    )

    pairs = (
        poi[["NAME", "TYPE"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    unique_names = sorted(
        pairs["NAME"]
        .unique()
        .tolist()
    )

    type_vocabulary = sorted(
        pairs["TYPE"]
        .unique()
        .tolist()
    )

    if city == "xa":
        if (
            len(type_vocabulary)
            != EXPECTED_TYPE_COUNT
        ):
            raise ValueError(
                "Unexpected Xi'an TYPE count: "
                f"{len(type_vocabulary)}"
            )

        if (
            len(unique_names)
            != EXPECTED_XA_UNIQUE_NAMES
        ):
            raise ValueError(
                "Unexpected Xi'an unique NAME "
                f"count: {len(unique_names)}"
            )

        if (
            len(pairs)
            != EXPECTED_XA_NAME_TYPE_PAIRS
        ):
            raise ValueError(
                "Unexpected Xi'an unique "
                "NAME-TYPE pair count: "
                f"{len(pairs)}"
            )

    return (
        pairs,
        unique_names,
        type_vocabulary,
    )


def load_embedding_model(
    *,
    snapshot_path: Path,
    device: str,
) -> SentenceTransformer:
    """Load and validate the frozen Qwen encoder."""

    require_directory(
        snapshot_path,
        "Qwen snapshot",
    )

    if (
        device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not "
            "available."
        )

    model = SentenceTransformer(
        str(snapshot_path),
        device=device,
    )

    model.eval()

    embedding_dim = (
        model.get_sentence_embedding_dimension()
    )

    if (
        embedding_dim
        != EXPECTED_EMBEDDING_DIM
    ):
        raise ValueError(
            "Unexpected Qwen embedding "
            f"dimension: {embedding_dim}"
        )

    if model.default_prompt_name is not None:
        raise ValueError(
            "Unexpected default prompt: "
            f"{model.default_prompt_name}"
        )

    if (
        model.tokenizer.padding_side
        != "right"
    ):
        raise ValueError(
            "Unexpected tokenizer padding side: "
            f"{model.tokenizer.padding_side}"
        )

    if len(model) < 3:
        raise ValueError(
            "Unexpected SentenceTransformer "
            "module structure."
        )

    pooling = model[1]
    normalize = model[2]

    if (
        pooling.__class__.__name__
        != "Pooling"
    ):
        raise ValueError(
            "Expected Pooling as module 1."
        )

    if not bool(
        getattr(
            pooling,
            "pooling_mode_lasttoken",
            False,
        )
    ):
        raise ValueError(
            "Qwen SentenceTransformer is not "
            "configured for last-token pooling."
        )

    if (
        normalize.__class__.__name__
        != "Normalize"
    ):
        raise ValueError(
            "Expected Normalize as module 2."
        )

    model.max_seq_length = (
        MAX_SEQ_LENGTH
    )

    return model


def audit_token_lengths(
    *,
    model: SentenceTransformer,
    names: list[str],
) -> dict[str, float | int]:
    """Verify that no POI NAME needs truncation."""

    encoded = model.tokenizer(
        names,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    lengths = np.asarray(
        [
            len(input_ids)
            for input_ids
            in encoded["input_ids"]
        ],
        dtype=np.int64,
    )

    if lengths.size != len(names):
        raise ValueError(
            "Tokenizer output count does not "
            "match NAME count."
        )

    maximum = int(
        lengths.max()
    )

    if maximum > MAX_SEQ_LENGTH:
        raise ValueError(
            "Observed POI NAME exceeds "
            f"max_seq_length={MAX_SEQ_LENGTH}: "
            f"{maximum}"
        )

    return {
        "min": int(lengths.min()),
        "median": float(
            np.median(lengths)
        ),
        "p95": float(
            np.percentile(
                lengths,
                95,
            )
        ),
        "p99": float(
            np.percentile(
                lengths,
                99,
            )
        ),
        "max": maximum,
        "over_max_seq_length": int(
            np.count_nonzero(
                lengths
                >
                MAX_SEQ_LENGTH
            )
        ),
    }


def encode_unique_names(
    *,
    model: SentenceTransformer,
    names: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode every globally unique POI NAME once."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    embeddings = model.encode(
        names,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    expected_shape = (
        len(names),
        EXPECTED_EMBEDDING_DIM,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            "Unexpected NAME embedding shape: "
            f"{embeddings.shape}"
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "NAME embeddings contain "
            "non-finite values."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if np.any(norms <= 0):
        raise ValueError(
            "Zero-norm NAME embedding "
            "detected."
        )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "NAME embeddings are not "
            "L2-normalized."
        )

    return embeddings


def build_type_prototypes(
    *,
    pairs: pd.DataFrame,
    unique_names: list[str],
    type_vocabulary: list[str],
    name_embeddings: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Mean unique NAME embeddings within TYPE, then L2-normalize."""

    name_to_index = {
        name: index
        for index, name
        in enumerate(unique_names)
    }

    type_prototypes = np.zeros(
        (
            len(type_vocabulary),
            EXPECTED_EMBEDDING_DIM,
        ),
        dtype=np.float32,
    )

    type_name_counts = np.zeros(
        len(type_vocabulary),
        dtype=np.int64,
    )

    pre_normalization_norms = np.zeros(
        len(type_vocabulary),
        dtype=np.float64,
    )

    for (
        type_index,
        poi_type,
    ) in enumerate(type_vocabulary):

        type_names = sorted(
            pairs.loc[
                pairs["TYPE"] == poi_type,
                "NAME",
            ]
            .unique()
            .tolist()
        )

        if not type_names:
            raise ValueError(
                "TYPE has no unique POI "
                f"names: {poi_type}"
            )

        name_indices = np.asarray(
            [
                name_to_index[name]
                for name
                in type_names
            ],
            dtype=np.int64,
        )

        mean_embedding = (
            name_embeddings[
                name_indices
            ]
            .astype(
                np.float64,
                copy=False,
            )
            .mean(
                axis=0,
            )
        )

        mean_norm = float(
            np.linalg.norm(
                mean_embedding
            )
        )

        if (
            not np.isfinite(mean_norm)
            or mean_norm <= 0
        ):
            raise ValueError(
                "Invalid pre-normalization "
                f"TYPE norm: {poi_type}"
            )

        normalized = (
            mean_embedding
            /
            mean_norm
        )

        type_prototypes[
            type_index
        ] = normalized.astype(
            np.float32
        )

        type_name_counts[
            type_index
        ] = len(type_names)

        pre_normalization_norms[
            type_index
        ] = mean_norm

    return (
        type_prototypes,
        type_name_counts,
        pre_normalization_norms,
    )


def build_payload(
    *,
    city: str,
    poi_path: Path,
    snapshot_path: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Build Qwen TYPE prototypes only."""

    (
        pairs,
        unique_names,
        type_vocabulary,
    ) = load_poi_pairs(
        city=city,
        poi_path=poi_path,
    )

    model = load_embedding_model(
        snapshot_path=snapshot_path,
        device=device,
    )

    token_stats = audit_token_lengths(
        model=model,
        names=unique_names,
    )

    name_embeddings = (
        encode_unique_names(
            model=model,
            names=unique_names,
            batch_size=batch_size,
        )
    )

    (
        type_prototypes,
        type_name_counts,
        pre_normalization_norms,
    ) = build_type_prototypes(
        pairs=pairs,
        unique_names=unique_names,
        type_vocabulary=type_vocabulary,
        name_embeddings=name_embeddings,
    )

    payload = {
        "type_prototypes": torch.from_numpy(
            type_prototypes
        ),
        "type_name_counts": torch.from_numpy(
            type_name_counts
        ),
        "type_pre_normalization_norms": (
            torch.from_numpy(
                pre_normalization_norms
            )
        ),
        "type_vocabulary": type_vocabulary,
        "metadata": {
            "version": CACHE_VERSION,
            "city": city,
            "representation": (
                "Qwen language-semantic "
                "TYPE prototypes"
            ),
            "embedding_model": MODEL_NAME,
            "embedding_model_revision": (
                MODEL_REVISION
            ),
            "embedding_dim": (
                EXPECTED_EMBEDDING_DIM
            ),
            "sentence_transformers_version": (
                sentence_transformers.__version__
            ),
            "transformers_version": (
                transformers.__version__
            ),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "device": device,
            "batch_size": batch_size,
            "max_seq_length": (
                MAX_SEQ_LENGTH
            ),
            "prompt": None,
            "padding_side": (
                model.tokenizer.padding_side
            ),
            "pooling": "last-token",
            "name_normalization": "L2",
            "type_aggregation": (
                "arithmetic mean of unique "
                "NAME embeddings within TYPE"
            ),
            "type_normalization": "L2",
            "poi_rows": int(
                len(
                    pd.read_csv(
                        poi_path,
                        usecols=["NAME"],
                    )
                )
            ),
            "global_unique_names": (
                len(unique_names)
            ),
            "unique_name_type_pairs": (
                len(pairs)
            ),
            "type_count": (
                len(type_vocabulary)
            ),
            "sum_unique_names_per_type": int(
                type_name_counts.sum()
            ),
            "token_length_min": (
                token_stats["min"]
            ),
            "token_length_median": (
                token_stats["median"]
            ),
            "token_length_p95": (
                token_stats["p95"]
            ),
            "token_length_p99": (
                token_stats["p99"]
            ),
            "token_length_max": (
                token_stats["max"]
            ),
            "truncated_name_count": (
                token_stats[
                    "over_max_seq_length"
                ]
            ),
            "poi_sha256": sha256_file(
                poi_path
            ),
        },
    }

    validate_payload(
        payload=payload,
    )

    return payload


def validate_payload(
    *,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate one TYPE-prototype payload."""

    required_keys = {
        "type_prototypes",
        "type_name_counts",
        "type_pre_normalization_norms",
        "type_vocabulary",
        "metadata",
    }

    if set(payload.keys()) != required_keys:
        raise ValueError(
            "Unexpected prototype cache keys: "
            f"{sorted(payload.keys())}"
        )

    prototypes = payload[
        "type_prototypes"
    ]

    counts = payload[
        "type_name_counts"
    ]

    pre_norms = payload[
        "type_pre_normalization_norms"
    ]

    vocabulary = payload[
        "type_vocabulary"
    ]

    metadata = payload[
        "metadata"
    ]

    type_count = int(
        metadata["type_count"]
    )

    embedding_dim = int(
        metadata["embedding_dim"]
    )

    if tuple(
        prototypes.shape
    ) != (
        type_count,
        embedding_dim,
    ):
        raise ValueError(
            "Unexpected TYPE prototype shape: "
            f"{tuple(prototypes.shape)}"
        )

    if tuple(counts.shape) != (
        type_count,
    ):
        raise ValueError(
            "Unexpected TYPE count shape."
        )

    if tuple(pre_norms.shape) != (
        type_count,
    ):
        raise ValueError(
            "Unexpected pre-normalization "
            "norm shape."
        )

    if len(vocabulary) != type_count:
        raise ValueError(
            "TYPE vocabulary size mismatch."
        )

    if (
        prototypes.dtype
        != torch.float32
    ):
        raise ValueError(
            "TYPE prototypes must be "
            "float32."
        )

    if counts.dtype != torch.int64:
        raise ValueError(
            "TYPE name counts must be "
            "int64."
        )

    if (
        pre_norms.dtype
        != torch.float64
    ):
        raise ValueError(
            "Pre-normalization TYPE norms "
            "must be float64."
        )

    if not torch.isfinite(
        prototypes
    ).all():
        raise ValueError(
            "TYPE prototypes contain "
            "non-finite values."
        )

    if not torch.isfinite(
        pre_norms
    ).all():
        raise ValueError(
            "TYPE pre-normalization norms "
            "contain non-finite values."
        )

    if torch.any(counts <= 0):
        raise ValueError(
            "A TYPE has zero unique names."
        )

    if torch.any(pre_norms <= 0):
        raise ValueError(
            "A TYPE has a non-positive "
            "pre-normalization norm."
        )

    norms = torch.linalg.vector_norm(
        prototypes,
        dim=1,
    )

    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "TYPE prototypes are not "
            "L2-normalized."
        )

    if (
        metadata["city"] == "xa"
        and int(counts.sum().item())
        != EXPECTED_XA_NAME_TYPE_PAIRS
    ):
        raise ValueError(
            "Xi'an TYPE membership total "
            "does not match the audited "
            "NAME-TYPE pair count."
        )

    cosine = (
        prototypes
        @
        prototypes.T
    )

    off_diagonal = cosine[
        ~torch.eye(
            type_count,
            dtype=torch.bool,
        )
    ]

    unique_rows = int(
        torch.unique(
            prototypes,
            dim=0,
        ).shape[0]
    )

    return {
        "shape": tuple(
            prototypes.shape
        ),
        "prototype_dtype": str(
            prototypes.dtype
        ),
        "type_count": type_count,
        "embedding_dim": embedding_dim,
        "type_name_count_min": int(
            counts.min().item()
        ),
        "type_name_count_median": float(
            counts.double().median().item()
        ),
        "type_name_count_max": int(
            counts.max().item()
        ),
        "pre_norm_min": float(
            pre_norms.min().item()
        ),
        "pre_norm_median": float(
            pre_norms.median().item()
        ),
        "pre_norm_max": float(
            pre_norms.max().item()
        ),
        "prototype_norm_min": float(
            norms.min().item()
        ),
        "prototype_norm_max": float(
            norms.max().item()
        ),
        "cosine_offdiag_min": float(
            off_diagonal.min().item()
        ),
        "cosine_offdiag_mean": float(
            off_diagonal.mean().item()
        ),
        "cosine_offdiag_max": float(
            off_diagonal.max().item()
        ),
        "unique_prototype_rows": (
            unique_rows
        ),
        "global_unique_names": int(
            metadata[
                "global_unique_names"
            ]
        ),
        "unique_name_type_pairs": int(
            metadata[
                "unique_name_type_pairs"
            ]
        ),
        "token_length_max": int(
            metadata[
                "token_length_max"
            ]
        ),
        "truncated_name_count": int(
            metadata[
                "truncated_name_count"
            ]
        ),
    }


def validate_cache(
    *,
    cache_path: Path,
) -> dict[str, Any]:
    """Load and validate one prototype cache."""

    payload = torch.load(
        cache_path,
        weights_only=True,
        map_location="cpu",
    )

    report = validate_payload(
        payload=payload,
    )

    report["size"] = (
        cache_path.stat().st_size
    )

    report["sha256"] = (
        sha256_file(cache_path)
    )

    return report


def build_or_verify(
    *,
    city: str,
    poi_path: Path,
    snapshot_path: Path,
    target_cache: Path,
    device: str,
    batch_size: int,
    verify_only: bool,
    force: bool,
) -> dict[str, Any]:
    """Build atomically or verify TYPE prototypes."""

    if verify_only and force:
        raise ValueError(
            "--verify-only and --force "
            "cannot be combined."
        )

    require_file(
        poi_path,
        "POI source",
    )

    if verify_only:
        require_file(
            target_cache,
            "TYPE prototype cache",
        )

        return validate_cache(
            cache_path=target_cache,
        )

    require_directory(
        snapshot_path,
        "Qwen snapshot",
    )

    if (
        target_cache.exists()
        and not force
    ):
        raise FileExistsError(
            "TYPE prototype cache already "
            "exists. Use --verify-only or "
            f"--force: {target_cache}"
        )

    target_cache.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_cache = (
        target_cache.with_suffix(
            target_cache.suffix
            +
            ".tmp"
        )
    )

    if temporary_cache.exists():
        if not force:
            raise FileExistsError(
                "Temporary TYPE prototype "
                "cache already exists: "
                f"{temporary_cache}"
            )

        temporary_cache.unlink()

    try:
        payload = build_payload(
            city=city,
            poi_path=poi_path,
            snapshot_path=snapshot_path,
            device=device,
            batch_size=batch_size,
        )

        torch.save(
            payload,
            temporary_cache,
        )

        temporary_report = (
            validate_cache(
                cache_path=temporary_cache,
            )
        )

        os.replace(
            temporary_cache,
            target_cache,
        )

        final_report = validate_cache(
            cache_path=target_cache,
        )

        if (
            temporary_report["sha256"]
            != final_report["sha256"]
        ):
            raise RuntimeError(
                "Final TYPE prototype cache "
                "SHA-256 differs from the "
                "validated temporary cache."
            )

        return final_report

    finally:
        temporary_cache.unlink(
            missing_ok=True
        )


def parse_args() -> argparse.Namespace:
    

    parser = argparse.ArgumentParser(
        description=(
            "Build or verify Exp3 Qwen "
            "language-semantic TYPE "
            "prototypes."
        )
    )

    parser.add_argument(
        "--dataset-path",
        default=str(Path("..") / "dataset"),
    )

    parser.add_argument(
        "--city",
        default="xa",
    )

    parser.add_argument(
        "--hf-cache",
        default=str(Path("..") / "hf_cache_exp3"),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument("--output-cache", type=Path)
    parser.add_argument("--poi-file", type=Path)
    parser.add_argument("--model-path", type=Path)
    return parser.parse_args()


def main() -> None:
    

    args = parse_args()

    city_root = (
        Path(args.dataset_path)
        /
        args.city
    )

    poi_path = (
        city_root
        /
        f"POI_{args.city}.csv"
    )

    snapshot_path = (
        Path(args.hf_cache)
        /
        "models--Qwen--"
        "Qwen3-Embedding-0.6B"
        /
        "snapshots"
        /
        MODEL_REVISION
    )

    target_cache = (
        city_root
        /
        (
            f"cached_{args.city}_"
            "language_type_prototypes_"
            "qwen3_0_6b_"
            f"{CACHE_VERSION}.pth"
        )
    )

    if args.output_cache:
        target_cache = args.output_cache.resolve()
    if args.poi_file:
        poi_path = args.poi_file.resolve()
    if args.model_path:
        snapshot_path = args.model_path.resolve()

    action = (
        "VERIFICATION"
        if args.verify_only
        else "BUILD"
    )

    report = build_or_verify(
        city=args.city,
        poi_path=poi_path,
        snapshot_path=snapshot_path,
        target_cache=target_cache,
        device=args.device,
        batch_size=args.batch_size,
        verify_only=args.verify_only,
        force=args.force,
    )

    print(
        f"=== EXP3 TYPE PROTOTYPE "
        f"{action} ==="
    )

    print("City:", args.city)
    print(
        "Prototype cache:",
        target_cache,
    )

    print()
    print(
        "Prototype shape:",
        report["shape"],
    )
    print(
        "Prototype dtype:",
        report[
            "prototype_dtype"
        ],
    )
    print(
        "TYPE categories:",
        report["type_count"],
    )
    print(
        "Embedding dim:",
        report["embedding_dim"],
    )

    print()
    print(
        "Global unique NAMEs:",
        report[
            "global_unique_names"
        ],
    )
    print(
        "Unique NAME-TYPE pairs:",
        report[
            "unique_name_type_pairs"
        ],
    )
    print(
        "Names per TYPE min/median/max:",
        report[
            "type_name_count_min"
        ],
        report[
            "type_name_count_median"
        ],
        report[
            "type_name_count_max"
        ],
    )

    print()
    print(
        "Max observed token length:",
        report[
            "token_length_max"
        ],
    )
    print(
        "Truncated NAMEs:",
        report[
            "truncated_name_count"
        ],
    )

    print()
    print(
        "Pre-normalization TYPE norm "
        "min/median/max:",
        report["pre_norm_min"],
        report["pre_norm_median"],
        report["pre_norm_max"],
    )
    print(
        "Final prototype norm min/max:",
        report[
            "prototype_norm_min"
        ],
        report[
            "prototype_norm_max"
        ],
    )

    print()
    print(
        "Off-diagonal cosine min/mean/max:",
        report[
            "cosine_offdiag_min"
        ],
        report[
            "cosine_offdiag_mean"
        ],
        report[
            "cosine_offdiag_max"
        ],
    )
    print(
        "Unique prototype rows:",
        report[
            "unique_prototype_rows"
        ],
    )

    print()
    print(
        "Cache size:",
        f'{report["size"]:,} bytes',
    )
    print(
        "SHA-256:",
        report["sha256"],
    )

    print()
    print(
        "Exp3 TYPE prototype build and "
        "validation: PASS"
    )
    print(
        "No road-level semantic cache, "
        "model training, or Exp2 artifact "
        "was modified."
    )


if __name__ == "__main__":
    main()
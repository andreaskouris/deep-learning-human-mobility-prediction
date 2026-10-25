"""Build road-level Qwen vectors with canonical POI TF-IDF weights."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import torch


CACHE_VERSION = "v1"

EXPECTED_ROAD_COUNT = 5269
EXPECTED_TYPE_COUNT = 204
EXPECTED_EMBEDDING_DIM = 1024

EXPECTED_ROADS_WITH_SEMANTICS = 2675
EXPECTED_ROADS_WITHOUT_SEMANTICS = 2594


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


def load_source_caches(
    *,
    city: str,
    tfidf_cache_path: Path,
    prototype_cache_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """Load and validate the two verified source artifacts."""

    require_file(
        tfidf_cache_path,
        "Exp2 TF-IDF cache",
    )

    require_file(
        prototype_cache_path,
        "Exp3 TYPE prototype cache",
    )

    tfidf_cache = torch.load(
        tfidf_cache_path,
        weights_only=True,
        map_location="cpu",
    )

    prototype_cache = torch.load(
        prototype_cache_path,
        weights_only=True,
        map_location="cpu",
    )

    required_tfidf_keys = {
        "semantic_features",
        "semantic_mask",
        "idf",
        "type_vocabulary",
        "metadata",
    }

    if set(tfidf_cache.keys()) != required_tfidf_keys:
        raise ValueError(
            "Unexpected Exp2 TF-IDF cache keys: "
            f"{sorted(tfidf_cache.keys())}"
        )

    required_prototype_keys = {
        "type_prototypes",
        "type_name_counts",
        "type_pre_normalization_norms",
        "type_vocabulary",
        "metadata",
    }

    if (
        set(prototype_cache.keys())
        != required_prototype_keys
    ):
        raise ValueError(
            "Unexpected Exp3 prototype cache keys: "
            f"{sorted(prototype_cache.keys())}"
        )

    tfidf_features = tfidf_cache[
        "semantic_features"
    ]

    tfidf_mask = tfidf_cache[
        "semantic_mask"
    ]

    tfidf_vocabulary = tfidf_cache[
        "type_vocabulary"
    ]

    prototypes = prototype_cache[
        "type_prototypes"
    ]

    prototype_vocabulary = prototype_cache[
        "type_vocabulary"
    ]

    if tuple(tfidf_features.shape) != (
        EXPECTED_ROAD_COUNT,
        EXPECTED_TYPE_COUNT,
    ):
        raise ValueError(
            "Unexpected Exp2 TF-IDF shape: "
            f"{tuple(tfidf_features.shape)}"
        )

    if tuple(tfidf_mask.shape) != (
        EXPECTED_ROAD_COUNT,
    ):
        raise ValueError(
            "Unexpected Exp2 semantic mask shape: "
            f"{tuple(tfidf_mask.shape)}"
        )

    if tuple(prototypes.shape) != (
        EXPECTED_TYPE_COUNT,
        EXPECTED_EMBEDDING_DIM,
    ):
        raise ValueError(
            "Unexpected TYPE prototype shape: "
            f"{tuple(prototypes.shape)}"
        )

    if tfidf_features.dtype != torch.float64:
        raise ValueError(
            "Exp2 TF-IDF features must be "
            "float64."
        )

    if tfidf_mask.dtype != torch.bool:
        raise ValueError(
            "Exp2 semantic mask must be bool."
        )

    if prototypes.dtype != torch.float32:
        raise ValueError(
            "Qwen TYPE prototypes must be "
            "float32."
        )

    if (
        tfidf_vocabulary
        != prototype_vocabulary
    ):
        raise ValueError(
            "Exp2 and Exp3 TYPE vocabularies "
            "do not match exactly."
        )

    if len(tfidf_vocabulary) != EXPECTED_TYPE_COUNT:
        raise ValueError(
            "Unexpected TYPE vocabulary size: "
            f"{len(tfidf_vocabulary)}"
        )

    if not torch.isfinite(
        tfidf_features
    ).all():
        raise ValueError(
            "Exp2 TF-IDF features contain "
            "non-finite values."
        )

    if not torch.isfinite(
        prototypes
    ).all():
        raise ValueError(
            "Qwen TYPE prototypes contain "
            "non-finite values."
        )

    prototype_norms = (
        torch.linalg.vector_norm(
            prototypes,
            dim=1,
        )
    )

    if not torch.allclose(
        prototype_norms,
        torch.ones_like(
            prototype_norms
        ),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Qwen TYPE prototypes are not "
            "L2-normalized."
        )

    roads_with_semantics = int(
        tfidf_mask.sum().item()
    )

    roads_without_semantics = int(
        (~tfidf_mask).sum().item()
    )

    if city == "xa":
        if (
            roads_with_semantics
            != EXPECTED_ROADS_WITH_SEMANTICS
        ):
            raise ValueError(
                "Unexpected Xi'an semantic-road "
                f"count: {roads_with_semantics}"
            )

        if (
            roads_without_semantics
            != EXPECTED_ROADS_WITHOUT_SEMANTICS
        ):
            raise ValueError(
                "Unexpected Xi'an missing-road "
                f"count: {roads_without_semantics}"
            )

    return (
        tfidf_cache,
        prototype_cache,
    )


def build_road_features(
    *,
    tfidf_cache: dict[str, Any],
    prototype_cache: dict[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Build raw road-level language-semantic weighted sums."""

    tfidf_features = tfidf_cache[
        "semantic_features"
    ].to(
        dtype=torch.float64
    )

    prototypes = prototype_cache[
        "type_prototypes"
    ].to(
        dtype=torch.float64
    )

    semantic_mask = tfidf_cache[
        "semantic_mask"
    ].to(
        dtype=torch.bool
    )

    road_features = (
        tfidf_features
        @
        prototypes
    )

    expected_shape = (
        EXPECTED_ROAD_COUNT,
        EXPECTED_EMBEDDING_DIM,
    )

    if tuple(road_features.shape) != expected_shape:
        raise ValueError(
            "Unexpected road language-semantic "
            f"shape: {tuple(road_features.shape)}"
        )

    if road_features.dtype != torch.float64:
        raise ValueError(
            "Road language-semantic features "
            "must be float64."
        )

    if not torch.isfinite(
        road_features
    ).all():
        raise ValueError(
            "Road language-semantic features "
            "contain non-finite values."
        )

    road_norms = torch.linalg.vector_norm(
        road_features,
        dim=1,
    )

    inferred_mask = (
        road_norms > 0
    )

    if not torch.equal(
        inferred_mask,
        semantic_mask,
    ):
        raise ValueError(
            "Road language-semantic non-zero "
            "mask does not exactly match the "
            "locked Exp2 semantic mask."
        )

    if bool(
        torch.any(
            road_features[
                ~semantic_mask
            ]
            != 0
        )
    ):
        raise ValueError(
            "A road without semantics has "
            "non-zero language features."
        )

    return (
        road_features,
        semantic_mask,
    )


def build_payload(
    *,
    city: str,
    tfidf_cache_path: Path,
    prototype_cache_path: Path,
) -> dict[str, Any]:
    """Build the Exp3 road-level semantic payload."""

    (
        tfidf_cache,
        prototype_cache,
    ) = load_source_caches(
        city=city,
        tfidf_cache_path=tfidf_cache_path,
        prototype_cache_path=prototype_cache_path,
    )

    (
        road_features,
        semantic_mask,
    ) = build_road_features(
        tfidf_cache=tfidf_cache,
        prototype_cache=prototype_cache,
    )

    type_vocabulary = tfidf_cache[
        "type_vocabulary"
    ]

    prototype_metadata = prototype_cache[
        "metadata"
    ]

    payload = {
        "semantic_features": road_features,
        "semantic_mask": semantic_mask.clone(),
        "type_vocabulary": list(
            type_vocabulary
        ),
        "metadata": {
            "version": CACHE_VERSION,
            "city": city,
            "representation": (
                "raw TF-IDF-weighted Qwen "
                "TYPE prototype road vectors"
            ),
            "road_count": (
                EXPECTED_ROAD_COUNT
            ),
            "type_count": (
                EXPECTED_TYPE_COUNT
            ),
            "embedding_dim": (
                EXPECTED_EMBEDDING_DIM
            ),
            "roads_with_semantics": int(
                semantic_mask.sum().item()
            ),
            "roads_without_semantics": int(
                (~semantic_mask).sum().item()
            ),
            "aggregation": (
                "Exp2 TF-IDF matrix @ "
                "L2-normalized Qwen TYPE "
                "prototype matrix"
            ),
            "road_normalization": (
                "none in cache; L2 at "
                "BIGCity model load"
            ),
            "feature_dtype": "torch.float64",
            "embedding_model": (
                prototype_metadata.get(
                    "embedding_model"
                )
            ),
            "embedding_model_revision": (
                prototype_metadata.get(
                    "embedding_model_revision"
                )
            ),
            "source_tfidf_sha256": (
                sha256_file(
                    tfidf_cache_path
                )
            ),
            "source_prototype_sha256": (
                sha256_file(
                    prototype_cache_path
                )
            ),
        },
    }

    validate_payload(
        payload=payload,
        tfidf_cache=tfidf_cache,
        prototype_cache=prototype_cache,
    )

    return payload


def validate_payload(
    *,
    payload: dict[str, Any],
    tfidf_cache: dict[str, Any],
    prototype_cache: dict[str, Any],
) -> dict[str, Any]:
    """Validate the road semantic payload against both sources."""

    required_keys = {
        "semantic_features",
        "semantic_mask",
        "type_vocabulary",
        "metadata",
    }

    if set(payload.keys()) != required_keys:
        raise ValueError(
            "Unexpected road semantic cache keys: "
            f"{sorted(payload.keys())}"
        )

    features = payload[
        "semantic_features"
    ]

    mask = payload[
        "semantic_mask"
    ]

    vocabulary = payload[
        "type_vocabulary"
    ]

    metadata = payload[
        "metadata"
    ]

    if tuple(features.shape) != (
        EXPECTED_ROAD_COUNT,
        EXPECTED_EMBEDDING_DIM,
    ):
        raise ValueError(
            "Unexpected road semantic shape: "
            f"{tuple(features.shape)}"
        )

    if tuple(mask.shape) != (
        EXPECTED_ROAD_COUNT,
    ):
        raise ValueError(
            "Unexpected road semantic mask "
            f"shape: {tuple(mask.shape)}"
        )

    if features.dtype != torch.float64:
        raise ValueError(
            "Road semantic features must be "
            "float64."
        )

    if mask.dtype != torch.bool:
        raise ValueError(
            "Road semantic mask must be bool."
        )

    if (
        vocabulary
        != tfidf_cache[
            "type_vocabulary"
        ]
    ):
        raise ValueError(
            "Road cache TYPE vocabulary does "
            "not match Exp2."
        )

    if (
        vocabulary
        != prototype_cache[
            "type_vocabulary"
        ]
    ):
        raise ValueError(
            "Road cache TYPE vocabulary does "
            "not match Qwen prototypes."
        )

    if not torch.isfinite(
        features
    ).all():
        raise ValueError(
            "Road semantic features contain "
            "non-finite values."
        )

    source_mask = tfidf_cache[
        "semantic_mask"
    ].to(
        dtype=torch.bool
    )

    if not torch.equal(
        mask,
        source_mask,
    ):
        raise ValueError(
            "Road semantic mask differs from "
            "the locked Exp2 semantic mask."
        )

    norms = torch.linalg.vector_norm(
        features,
        dim=1,
    )

    inferred_mask = (
        norms > 0
    )

    if not torch.equal(
        inferred_mask,
        mask,
    ):
        raise ValueError(
            "Stored road feature norms do not "
            "match semantic mask."
        )

    if bool(
        torch.any(
            features[
                ~mask
            ]
            != 0
        )
    ):
        raise ValueError(
            "A missing-semantic road has "
            "non-zero stored features."
        )

    tfidf = tfidf_cache[
        "semantic_features"
    ].to(
        dtype=torch.float64
    )

    prototypes = prototype_cache[
        "type_prototypes"
    ].to(
        dtype=torch.float64
    )

    reconstructed = (
        tfidf
        @
        prototypes
    )

    if not torch.equal(
        reconstructed,
        features,
    ):
        max_error = float(
            torch.max(
                torch.abs(
                    reconstructed
                    -
                    features
                )
            ).item()
        )

        raise ValueError(
            "Road semantic cache does not "
            "exactly reconstruct from the "
            "verified source artifacts. "
            f"Max error: {max_error}"
        )

    semantic_norms = norms[
        mask
    ]

    if semantic_norms.numel() == 0:
        raise ValueError(
            "No roads with semantics found."
        )

    normalized = (
        features[
            mask
        ]
        /
        semantic_norms.unsqueeze(1)
    )

    float32_features = (
        features.to(
            dtype=torch.float32
        )
    )

    float32_semantic = (
        float32_features[
            mask
        ]
    )

    float32_norms = (
        torch.linalg.vector_norm(
            float32_semantic,
            dim=1,
            keepdim=True,
        )
    )

    normalized_float32 = (
        float32_semantic
        /
        float32_norms
    )

    normalized_difference = (
        normalized
        -
        normalized_float32.to(
            dtype=torch.float64
        )
    ).abs()

    cosine_float64_vs_float32 = (
        normalized
        *
        normalized_float32.to(
            dtype=torch.float64
        )
    ).sum(
        dim=1
    )

    return {
        "shape": tuple(
            features.shape
        ),
        "feature_dtype": str(
            features.dtype
        ),
        "mask_dtype": str(
            mask.dtype
        ),
        "road_count": (
            EXPECTED_ROAD_COUNT
        ),
        "type_count": (
            EXPECTED_TYPE_COUNT
        ),
        "embedding_dim": (
            EXPECTED_EMBEDDING_DIM
        ),
        "roads_with_semantics": int(
            mask.sum().item()
        ),
        "roads_without_semantics": int(
            (~mask).sum().item()
        ),
        "missing_road_max_abs": float(
            features[
                ~mask
            ].abs().max().item()
        ),
        "semantic_norm_min": float(
            semantic_norms.min().item()
        ),
        "semantic_norm_median": float(
            semantic_norms.median().item()
        ),
        "semantic_norm_max": float(
            semantic_norms.max().item()
        ),
        "feature_min": float(
            features.min().item()
        ),
        "feature_max": float(
            features.max().item()
        ),
        "float32_normalized_max_abs_diff": float(
            normalized_difference.max().item()
        ),
        "float32_normalized_mean_abs_diff": float(
            normalized_difference.mean().item()
        ),
        "float32_cosine_min": float(
            cosine_float64_vs_float32.min().item()
        ),
        "float32_cosine_mean": float(
            cosine_float64_vs_float32.mean().item()
        ),
        "float32_cosine_max": float(
            cosine_float64_vs_float32.max().item()
        ),
        "source_tfidf_sha256": metadata[
            "source_tfidf_sha256"
        ],
        "source_prototype_sha256": metadata[
            "source_prototype_sha256"
        ],
    }


def validate_cache(
    *,
    cache_path: Path,
    city: str,
    tfidf_cache_path: Path,
    prototype_cache_path: Path,
) -> dict[str, Any]:
    """Load and independently validate one road cache."""

    (
        tfidf_cache,
        prototype_cache,
    ) = load_source_caches(
        city=city,
        tfidf_cache_path=tfidf_cache_path,
        prototype_cache_path=prototype_cache_path,
    )

    payload = torch.load(
        cache_path,
        weights_only=True,
        map_location="cpu",
    )

    report = validate_payload(
        payload=payload,
        tfidf_cache=tfidf_cache,
        prototype_cache=prototype_cache,
    )

    expected_tfidf_hash = (
        sha256_file(
            tfidf_cache_path
        )
    )

    expected_prototype_hash = (
        sha256_file(
            prototype_cache_path
        )
    )

    if (
        report[
            "source_tfidf_sha256"
        ]
        != expected_tfidf_hash
    ):
        raise ValueError(
            "Stored Exp2 source SHA-256 does "
            "not match the current source."
        )

    if (
        report[
            "source_prototype_sha256"
        ]
        != expected_prototype_hash
    ):
        raise ValueError(
            "Stored prototype source SHA-256 "
            "does not match the current source."
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
    tfidf_cache_path: Path,
    prototype_cache_path: Path,
    target_cache: Path,
    verify_only: bool,
    force: bool,
) -> dict[str, Any]:
    """Build atomically or verify the road semantic cache."""

    if verify_only and force:
        raise ValueError(
            "--verify-only and --force "
            "cannot be combined."
        )

    require_file(
        tfidf_cache_path,
        "Exp2 TF-IDF cache",
    )

    require_file(
        prototype_cache_path,
        "Exp3 TYPE prototype cache",
    )

    if verify_only:
        require_file(
            target_cache,
            "Exp3 road semantic cache",
        )

        return validate_cache(
            cache_path=target_cache,
            city=city,
            tfidf_cache_path=tfidf_cache_path,
            prototype_cache_path=prototype_cache_path,
        )

    if (
        target_cache.exists()
        and not force
    ):
        raise FileExistsError(
            "Road semantic cache already "
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
                "Temporary road semantic "
                "cache already exists: "
                f"{temporary_cache}"
            )

        temporary_cache.unlink()

    try:
        payload = build_payload(
            city=city,
            tfidf_cache_path=tfidf_cache_path,
            prototype_cache_path=prototype_cache_path,
        )

        torch.save(
            payload,
            temporary_cache,
        )

        temporary_report = (
            validate_cache(
                cache_path=temporary_cache,
                city=city,
                tfidf_cache_path=tfidf_cache_path,
                prototype_cache_path=prototype_cache_path,
            )
        )

        os.replace(
            temporary_cache,
            target_cache,
        )

        final_report = validate_cache(
            cache_path=target_cache,
            city=city,
            tfidf_cache_path=tfidf_cache_path,
            prototype_cache_path=prototype_cache_path,
        )

        if (
            temporary_report["sha256"]
            != final_report["sha256"]
        ):
            raise RuntimeError(
                "Final road semantic cache "
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
            "Build or verify the Exp3 "
            "road-level Qwen language-semantic "
            "cache."
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
        "--verify-only",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument("--output-cache", type=Path)
    parser.add_argument("--tfidf-cache", type=Path)
    parser.add_argument("--prototype-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    

    args = parse_args()

    city_root = (
        Path(args.dataset_path)
        /
        args.city
    )

    tfidf_cache_path = (
        city_root
        /
        (
            f"cached_{args.city}_"
            "semantic_tfidf_v1.pth"
        )
    )

    prototype_cache_path = (
        city_root
        /
        (
            f"cached_{args.city}_"
            "language_type_prototypes_"
            "qwen3_0_6b_v1.pth"
        )
    )

    target_cache = (
        city_root
        /
        (
            f"cached_{args.city}_"
            "language_road_semantic_"
            "qwen3_0_6b_"
            f"{CACHE_VERSION}.pth"
        )
    )

    if args.output_cache:
        target_cache = args.output_cache.resolve()
    if args.tfidf_cache:
        tfidf_cache_path = args.tfidf_cache.resolve()
    if args.prototype_cache:
        prototype_cache_path = args.prototype_cache.resolve()

    action = (
        "VERIFICATION"
        if args.verify_only
        else "BUILD"
    )

    report = build_or_verify(
        city=args.city,
        tfidf_cache_path=tfidf_cache_path,
        prototype_cache_path=prototype_cache_path,
        target_cache=target_cache,
        verify_only=args.verify_only,
        force=args.force,
    )

    print(
        f"=== EXP3 ROAD SEMANTIC "
        f"{action} ==="
    )

    print("City:", args.city)
    print(
        "Road semantic cache:",
        target_cache,
    )

    print()
    print(
        "Semantic shape:",
        report["shape"],
    )
    print(
        "Feature dtype:",
        report[
            "feature_dtype"
        ],
    )
    print(
        "Mask dtype:",
        report[
            "mask_dtype"
        ],
    )
    print(
        "Roads:",
        report["road_count"],
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
        "Roads with semantics:",
        report[
            "roads_with_semantics"
        ],
    )
    print(
        "Roads without semantics:",
        report[
            "roads_without_semantics"
        ],
    )
    print(
        "Missing-road max abs:",
        report[
            "missing_road_max_abs"
        ],
    )

    print()
    print(
        "Semantic road norm min/median/max:",
        report[
            "semantic_norm_min"
        ],
        report[
            "semantic_norm_median"
        ],
        report[
            "semantic_norm_max"
        ],
    )

    print(
        "Feature min/max:",
        report["feature_min"],
        report["feature_max"],
    )

    print()
    print(
        "Float32 normalized max abs diff:",
        report[
            "float32_normalized_max_abs_diff"
        ],
    )
    print(
        "Float32 normalized mean abs diff:",
        report[
            "float32_normalized_mean_abs_diff"
        ],
    )
    print(
        "Float32 cosine min/mean/max:",
        report[
            "float32_cosine_min"
        ],
        report[
            "float32_cosine_mean"
        ],
        report[
            "float32_cosine_max"
        ],
    )

    print()
    print(
        "Source Exp2 TF-IDF SHA-256:",
        report[
            "source_tfidf_sha256"
        ],
    )
    print(
        "Source Qwen prototype SHA-256:",
        report[
            "source_prototype_sha256"
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
        "Exp3 road semantic cache build "
        "and validation: PASS"
    )

    print(
        "No BIGCity model, training code, "
        "Exp2 artifact, or Qwen prototype "
        "artifact was modified."
    )


if __name__ == "__main__":
    main()
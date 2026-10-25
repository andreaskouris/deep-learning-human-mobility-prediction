"""Build and verify the BIGCity road semantic TF-IDF cache."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


CACHE_VERSION = "v1"
EXPECTED_TYPE_COUNT = 204
SEM_ATOL = 1e-12


def sha256_file(path: Path) -> str:
    

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def require_file(path: Path, label: str) -> None:
    """Require one source file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )


def load_road_count(geo_path: Path) -> int:
    """Validate contiguous road geo IDs and return road count."""

    geo = pd.read_csv(
        geo_path,
        usecols=["geo_id"],
    )

    road_ids = geo["geo_id"].to_numpy(
        dtype=np.int64
    )

    road_count = len(road_ids)

    expected = np.arange(
        road_count,
        dtype=np.int64,
    )

    if not np.array_equal(
        road_ids,
        expected,
    ):
        raise ValueError(
            "Road geo_id values are not contiguous "
            "0..N-1."
        )

    return road_count


def build_payload(
    *,
    city: str,
    poi_path: Path,
    road2poi_path: Path,
    geo_path: Path,
    sem_path: Path,
) -> dict[str, Any]:
    """Build the road-level semantic TF-IDF representation."""

    road_count = load_road_count(
        geo_path
    )

    poi = pd.read_csv(
        poi_path,
        dtype={
            "id": str,
            "TYPE": str,
        },
    )

    poi["id"] = poi["id"].astype(str)
    poi["TYPE"] = poi["TYPE"].astype(str)

    poi_to_type = dict(
        zip(
            poi["id"],
            poi["TYPE"],
        )
    )

    type_vocabulary = sorted(
        poi["TYPE"].unique().tolist()
    )

    if (
        city == "xa"
        and len(type_vocabulary)
        != EXPECTED_TYPE_COUNT
    ):
        raise ValueError(
            "Unexpected Xi'an TYPE count: "
            f"{len(type_vocabulary)}"
        )

    type_to_index = {
        poi_type: index
        for index, poi_type
        in enumerate(type_vocabulary)
    }

    import json

    with road2poi_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        road2poi = json.load(file)

    counts = np.zeros(
        (
            road_count,
            len(type_vocabulary),
        ),
        dtype=np.float64,
    )

    missing_poi_references = 0

    for (
        road_id_text,
        poi_ids,
    ) in road2poi.items():

        road_id = int(road_id_text)

        if not (
            0 <= road_id < road_count
        ):
            raise ValueError(
                "road2poi contains an out-of-range "
                f"road ID: {road_id}"
            )

        for poi_id in poi_ids:

            poi_type = poi_to_type.get(
                str(poi_id)
            )

            if poi_type is None:
                missing_poi_references += 1
                continue

            counts[
                road_id,
                type_to_index[poi_type],
            ] += 1.0

    if missing_poi_references != 0:
        raise ValueError(
            "Missing POI references detected: "
            f"{missing_poi_references}"
        )

    document_frequency = np.count_nonzero(
        counts > 0,
        axis=0,
    )

    idf = (
        np.log(
            (road_count + 1)
            /
            (document_frequency + 1)
        )
        + 1.0
    )

    tfidf = counts * idf

    semantic_mask = (
        counts.sum(axis=1) > 0
    )

    payload = {
        "semantic_features": torch.from_numpy(
            tfidf
        ),
        "semantic_mask": torch.from_numpy(
            semantic_mask
        ),
        "idf": torch.from_numpy(
            idf
        ),
        "type_vocabulary": type_vocabulary,
        "metadata": {
            "version": CACHE_VERSION,
            "city": city,
            "road_count": road_count,
            "poi_count": len(poi),
            "type_count": len(
                type_vocabulary
            ),
            "roads_with_semantics": int(
                semantic_mask.sum()
            ),
            "roads_without_semantics": int(
                (~semantic_mask).sum()
            ),
            "missing_poi_references": (
                missing_poi_references
            ),
            "tfidf_formula": (
                "count * "
                "(log((N+1)/(df+1)) + 1)"
            ),
            "idf_network_size": road_count,
            "semantic_similarity": (
                "cosine similarity"
            ),
        },
    }

    validate_payload_against_sem(
        payload=payload,
        sem_path=sem_path,
    )

    return payload


def validate_payload_against_sem(
    *,
    payload: dict[str, Any],
    sem_path: Path,
) -> dict[str, Any]:
    """Validate cache structure and reconstruct .sem weights."""

    required_keys = {
        "semantic_features",
        "semantic_mask",
        "idf",
        "type_vocabulary",
        "metadata",
    }

    if set(payload.keys()) != required_keys:
        raise ValueError(
            "Unexpected semantic cache keys: "
            f"{sorted(payload.keys())}"
        )

    features = payload[
        "semantic_features"
    ]

    mask = payload[
        "semantic_mask"
    ]

    idf = payload["idf"]
    vocabulary = payload[
        "type_vocabulary"
    ]
    metadata = payload["metadata"]

    road_count = int(
        metadata["road_count"]
    )

    type_count = int(
        metadata["type_count"]
    )

    if tuple(features.shape) != (
        road_count,
        type_count,
    ):
        raise ValueError(
            "Unexpected semantic feature shape: "
            f"{tuple(features.shape)}"
        )

    if tuple(mask.shape) != (
        road_count,
    ):
        raise ValueError(
            "Unexpected semantic mask shape: "
            f"{tuple(mask.shape)}"
        )

    if tuple(idf.shape) != (
        type_count,
    ):
        raise ValueError(
            "Unexpected IDF shape: "
            f"{tuple(idf.shape)}"
        )

    if len(vocabulary) != type_count:
        raise ValueError(
            "TYPE vocabulary size does not match "
            "semantic feature width."
        )

    if features.dtype != torch.float64:
        raise ValueError(
            "Semantic features must be float64 "
            "inside the reproducibility cache."
        )

    if idf.dtype != torch.float64:
        raise ValueError(
            "IDF values must be float64."
        )

    if mask.dtype != torch.bool:
        raise ValueError(
            "Semantic mask must be boolean."
        )

    if not torch.isfinite(
        features
    ).all():
        raise ValueError(
            "Semantic features contain "
            "non-finite values."
        )

    if torch.any(features < 0):
        raise ValueError(
            "Semantic features contain "
            "negative values."
        )

    inferred_mask = (
        features.sum(dim=1) > 0
    )

    if not torch.equal(
        inferred_mask,
        mask,
    ):
        raise ValueError(
            "Semantic mask does not match "
            "non-zero semantic rows."
        )

    sem = pd.read_csv(
        sem_path
    )

    origins = torch.tensor(
        sem[
            "origin_id"
        ].to_numpy(
            dtype=np.int64
        ),
        dtype=torch.long,
    )

    destinations = torch.tensor(
        sem[
            "destination_id"
        ].to_numpy(
            dtype=np.int64
        ),
        dtype=torch.long,
    )

    stored_weights = torch.tensor(
        sem[
            "semantic_weight"
        ].to_numpy(
            dtype=np.float64
        ),
        dtype=torch.float64,
    )

    norms = torch.linalg.vector_norm(
        features,
        dim=1,
    )

    normalized = torch.zeros_like(
        features
    )

    nonzero = norms > 0

    normalized[nonzero] = (
        features[nonzero]
        /
        norms[nonzero].unsqueeze(1)
    )

    reconstructed = (
        normalized[origins]
        *
        normalized[destinations]
    ).sum(dim=1)

    errors = torch.abs(
        reconstructed
        -
        stored_weights
    )

    all_weights_match = bool(
        torch.allclose(
            reconstructed,
            stored_weights,
            atol=SEM_ATOL,
            rtol=0,
        )
    )

    if not all_weights_match:
        raise ValueError(
            "Semantic TF-IDF cache does not "
            "reconstruct roadmap .sem."
        )

    x = stored_weights.numpy()
    y = reconstructed.numpy()

    correlation = float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
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
        "road_count": road_count,
        "type_count": type_count,
        "roads_with_semantics": int(
            mask.sum().item()
        ),
        "roads_without_semantics": int(
            (~mask).sum().item()
        ),
        "sem_edges": len(sem),
        "sem_mae": float(
            errors.mean().item()
        ),
        "sem_max_error": float(
            errors.max().item()
        ),
        "sem_correlation": correlation,
        "sem_allclose_1e_12": (
            all_weights_match
        ),
    }


def validate_cache(
    *,
    cache_path: Path,
    sem_path: Path,
) -> dict[str, Any]:
    """Load and validate one semantic cache."""

    payload = torch.load(
        cache_path,
        weights_only=True,
        map_location="cpu",
    )

    report = (
        validate_payload_against_sem(
            payload=payload,
            sem_path=sem_path,
        )
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
    road2poi_path: Path,
    geo_path: Path,
    sem_path: Path,
    target_cache: Path,
    verify_only: bool,
    force: bool,
) -> dict[str, Any]:
    """Build atomically or verify the semantic cache."""

    if verify_only and force:
        raise ValueError(
            "--verify-only and --force "
            "cannot be combined."
        )

    required_sources = {
        "POI source": poi_path,
        "road2poi source": road2poi_path,
        "road geo source": geo_path,
        "semantic graph": sem_path,
    }

    for label, path in (
        required_sources.items()
    ):
        require_file(
            path,
            label,
        )

    if verify_only:

        require_file(
            target_cache,
            "semantic cache",
        )

        return validate_cache(
            cache_path=target_cache,
            sem_path=sem_path,
        )

    if (
        target_cache.exists()
        and not force
    ):
        raise FileExistsError(
            "Semantic cache already exists. "
            "Use --verify-only or --force: "
            f"{target_cache}"
        )

    target_cache.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_cache = (
        target_cache.with_suffix(
            target_cache.suffix
            + ".tmp"
        )
    )

    if temporary_cache.exists():

        if not force:
            raise FileExistsError(
                "Temporary semantic cache "
                "already exists: "
                f"{temporary_cache}"
            )

        temporary_cache.unlink()

    try:

        payload = build_payload(
            city=city,
            poi_path=poi_path,
            road2poi_path=road2poi_path,
            geo_path=geo_path,
            sem_path=sem_path,
        )

        torch.save(
            payload,
            temporary_cache,
        )

        temporary_report = (
            validate_cache(
                cache_path=temporary_cache,
                sem_path=sem_path,
            )
        )

        os.replace(
            temporary_cache,
            target_cache,
        )

        final_report = validate_cache(
            cache_path=target_cache,
            sem_path=sem_path,
        )

        if (
            temporary_report["sha256"]
            != final_report["sha256"]
        ):
            raise RuntimeError(
                "Final semantic cache SHA-256 "
                "differs from validated "
                "temporary cache."
            )

        return final_report

    finally:

        temporary_cache.unlink(
            missing_ok=True
        )


def parse_args() -> argparse.Namespace:
    

    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the road-level "
            "POI TF-IDF semantic cache."
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

    return parser.parse_args()


def main() -> None:
    

    args = parse_args()

    city_root = (
        Path(args.dataset_path)
        /
        args.city
    )

    roadmap_root = (
        city_root
        /
        f"roadmap_{args.city}"
    )

    poi_path = (
        city_root
        /
        f"POI_{args.city}.csv"
    )

    road2poi_path = (
        city_root
        /
        f"road2poi_{args.city}.json"
    )

    geo_path = (
        roadmap_root
        /
        f"roadmap_{args.city}.geo"
    )

    sem_path = (
        roadmap_root
        /
        f"roadmap_{args.city}.sem"
    )

    target_cache = (
        city_root
        /
        (
            f"cached_{args.city}_"
            f"semantic_tfidf_"
            f"{CACHE_VERSION}.pth"
        )
    )

    action = (
        "VERIFICATION"
        if args.verify_only
        else "BUILD"
    )

    report = build_or_verify(
        city=args.city,
        poi_path=poi_path,
        road2poi_path=road2poi_path,
        geo_path=geo_path,
        sem_path=sem_path,
        target_cache=target_cache,
        verify_only=args.verify_only,
        force=args.force,
    )

    print(
        f"=== SEMANTIC CACHE "
        f"{action} ==="
    )

    print("City:", args.city)
    print(
        "Semantic cache:",
        target_cache,
    )

    print()
    print(
        "Semantic shape:",
        report["shape"],
    )
    print(
        "Feature dtype:",
        report["feature_dtype"],
    )
    print(
        "Mask dtype:",
        report["mask_dtype"],
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

    print()
    print(
        ".sem edges:",
        report["sem_edges"],
    )
    print(
        ".sem MAE:",
        report["sem_mae"],
    )
    print(
        ".sem max error:",
        report["sem_max_error"],
    )
    print(
        ".sem correlation:",
        report[
            "sem_correlation"
        ],
    )
    print(
        ".sem allclose (1e-12):",
        report[
            "sem_allclose_1e_12"
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
        "Semantic cache build and "
        "validation: PASS"
    )
    print(
        "No model, training, or "
        "trajectory cache was modified."
    )


if __name__ == "__main__":
    main()

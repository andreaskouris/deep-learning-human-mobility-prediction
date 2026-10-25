from __future__ import annotations

# Support direct script execution without depending on the caller PYTHONPATH.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib

import json

from pathlib import Path

import numpy as np

import torch

import torch.nn.functional as F

import argparse

EXPECTED_STRUCTURED_SHA256 = (
    "081952F5ABD0A5E005430113EACC7E516A76CF2210E8FF346039508982416717"
)

MODEL_NAME = (
    "Qwen/Qwen3-Embedding-0.6B"
)

EXPECTED_FEATURE_NAMES = [
    "tunnel",
    "bridge",
    "roundabout",
    "oneway",
    "fuel_station",
    "parking",
    "transport_hub",
    "taxi_stand",
    "transit_stop",
    "logistics_freight",
]

CONCEPT_DESCRIPTIONS = {
    "tunnel":
        "road segment in a tunnel",

    "bridge":
        "road segment on a bridge",

    "roundabout":
        "road segment forming part of a roundabout",

    "oneway":
        "one-way road segment",

    "fuel_station":
        "road segment near a fuel station",

    "parking":
        "road segment near a parking facility",

    "transport_hub":
        "road segment near a transport hub",

    "taxi_stand":
        "road segment near a taxi stand",

    "transit_stop":
        "road segment near a public transit stop",

    "logistics_freight":
        "road segment near a logistics or freight facility",
}

EXPECTED_ROAD_COUNT = 5269

EXPECTED_CONCEPT_COUNT = 10

EXPECTED_ACTIVE_ROADS = 2437

EXPECTED_ZERO_ROADS = 2832

EXPECTED_COUNTS = [
    61,
    151,
    33,
    2043,
    27,
    300,
    7,
    238,
    831,
    5,
]

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return (
        digest.hexdigest()
        .upper()
    )

def main(argv=None):
    global CANONICAL_SOURCE_SHA256, HF_CACHE, HISTORICAL, MODEL_PATH, MODEL_REVISION, OUTPUT_CACHE, OUTPUT_META, STRUCTURED_CACHE, STRUCTURED_META, SentenceTransformer, active_count, active_mask, active_norms, artifact, binary_features, concept_counts, concept_embeddings, concept_norms, concept_np, derived_mask, description, descriptions, device, embedding_dim, feature_names, handle, index, mask_match, meta, meta_sha, model, name, normalized_road_embeddings, output_sha, qwen_active_count, reload_exact, reload_features, reloaded, road_embeddings, source_metadata, stream, structured, structured_sha, unique_binary_values, validate_payload, zero_count, zero_rows_exact
    parser = argparse.ArgumentParser(description="TTE Qwen with explicit local sources")
    for option in ("structured-cache", "output-cache", "output-metadata", "model-path", "hf-cache"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--structured-metadata", type=Path)
    parser.add_argument("--model-revision", default=None, help="Known snapshot revision; historical revision remains unconfirmed")
    parser.add_argument("--profile", choices=("historical", "fresh"), default="historical")
    parser.add_argument("--verify-source-only", action="store_true")
    options = parser.parse_args(argv)
    STRUCTURED_CACHE, STRUCTURED_META = options.structured_cache, options.structured_metadata
    OUTPUT_CACHE, OUTPUT_META = options.output_cache, options.output_metadata
    MODEL_PATH, MODEL_REVISION, HF_CACHE = options.model_path, options.model_revision, options.hf_cache
    HISTORICAL = options.profile == "historical"
    from scripts.build_tte_structured_semantic_cache import require_distinct
    require_distinct([STRUCTURED_CACHE, *([STRUCTURED_META] if STRUCTURED_META else []), OUTPUT_CACHE, OUTPUT_META])
    if not HISTORICAL and STRUCTURED_META is None:
        raise ValueError("Fresh profile requires --structured-metadata")
    if OUTPUT_CACHE.exists() or OUTPUT_META.exists():
        raise FileExistsError("Output cache or metadata exists; refusing replacement")

    print(
        "========================================"
    )

    print(
        "TTE QWEN SEMANTIC CACHE BUILD"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "Structured source:",
        STRUCTURED_CACHE,
    )

    print()

    if not STRUCTURED_CACHE.is_file():

        raise FileNotFoundError(
            STRUCTURED_CACHE
        )

    structured_sha = (
        sha256_file(
            STRUCTURED_CACHE
        )
    )

    print(
        "Structured cache SHA-256:",
        structured_sha,
    )

    if (
        HISTORICAL and (structured_sha
        != EXPECTED_STRUCTURED_SHA256)
    ):

        raise RuntimeError(
            "Structured semantic cache hash mismatch."
        )

    structured = torch.load(
        STRUCTURED_CACHE,
        map_location="cpu",
        weights_only=True,
    )

    if (
        "semantic_features"
        not in structured
    ):

        raise RuntimeError(
            "semantic_features missing "
            "from structured cache."
        )

    if not HISTORICAL:
        from scripts.build_tte_structured_semantic_cache import validate_payload, CANONICAL_SOURCE_SHA256
        validate_payload(structured, canonical=True)
        source_metadata = json.loads(STRUCTURED_META.read_text(encoding="utf-8"))
        if source_metadata.get("source_sha256") != CANONICAL_SOURCE_SHA256:
            raise ValueError("Structured source provenance mismatch")

    binary_features = (
        structured[
            "semantic_features"
        ]
        .detach()
        .cpu()
        .to(
            torch.float32
        )
    )

    feature_names = (
        structured.get(
            "feature_names"
        )
    )

    if feature_names is None:

        raise RuntimeError(
            "feature_names missing "
            "from structured cache."
        )

    feature_names = list(
        feature_names
    )

    print()

    print(
        "Structured shape:",
        tuple(
            binary_features.shape
        ),
    )

    print(
        "Feature names:",
        feature_names,
    )

    if tuple(
        binary_features.shape
    ) != (
        EXPECTED_ROAD_COUNT,
        EXPECTED_CONCEPT_COUNT,
    ):

        raise RuntimeError(
            "Unexpected structured shape."
        )

    if (
        feature_names
        != EXPECTED_FEATURE_NAMES
    ):

        raise RuntimeError(
            "Feature-name/order mismatch."
        )

    unique_binary_values = (
        torch.unique(
            binary_features
        )
        .tolist()
    )

    print(
        "Structured unique values:",
        unique_binary_values,
    )

    if not set(
        unique_binary_values
    ).issubset(
        {
            0.0,
            1.0,
        }
    ):

        raise RuntimeError(
            "Structured source is not binary."
        )

    concept_counts = (
        binary_features
        .sum(
            dim=0
        )
        .to(
            torch.int64
        )
        .tolist()
    )

    print(
        "Concept counts:",
        concept_counts,
    )

    if (
        concept_counts
        != EXPECTED_COUNTS
    ):

        raise RuntimeError(
            "Concept support counts changed."
        )

    active_mask = (
        binary_features
        .sum(
            dim=1
        )
        > 0
    )

    active_count = int(
        active_mask
        .sum()
        .item()
    )

    zero_count = int(
        (
            ~active_mask
        )
        .sum()
        .item()
    )

    print(
        "Active roads:",
        active_count,
    )

    print(
        "Zero-semantic roads:",
        zero_count,
    )

    if (
        active_count
        != EXPECTED_ACTIVE_ROADS
        or
        zero_count
        != EXPECTED_ZERO_ROADS
    ):

        raise RuntimeError(
            "Structured active/zero-road count mismatch."
        )

    print()

    print(
        "LOCKED STRUCTURED SOURCE: PASS"
    )

    print()

    print(
        "========================================"
    )

    print(
        "LOADING QWEN"
    )

    print(
        "========================================"
    )

    if options.verify_source_only:
        return {"structured_tensor_verified": True, "qwen_loaded": False}
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)
    OUTPUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_META.parent.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "HF cache:",
        HF_CACHE,
    )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(MODEL_PATH),
        revision=MODEL_REVISION,
        cache_folder=str(
            HF_CACHE
        ),
        device=device,
        local_files_only=True,
    )

    print(
        "Model loaded: PASS"
    )

    descriptions = [
        CONCEPT_DESCRIPTIONS[
            name
        ]
        for name
        in feature_names
    ]

    print()

    print(
        "Concept descriptions:"
    )

    for index, (
        name,
        description,
    ) in enumerate(
        zip(
            feature_names,
            descriptions,
        )
    ):

        print(
            f"{index:02d} "
            f"{name:<20} "
            f"=> {description}"
        )

    concept_np = model.encode(
        descriptions,
        batch_size=10,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    concept_embeddings = (
        torch.from_numpy(
            np.asarray(
                concept_np,
                dtype=np.float32,
            )
        )
    )

    print()

    print(
        "Concept embedding shape:",
        tuple(
            concept_embeddings.shape
        ),
    )

    if (
        concept_embeddings.ndim
        != 2
        or
        concept_embeddings.shape[0]
        != EXPECTED_CONCEPT_COUNT
    ):

        raise RuntimeError(
            "Unexpected concept embedding shape."
        )

    embedding_dim = int(
        concept_embeddings.shape[1]
    )

    print(
        "Embedding dimension:",
        embedding_dim,
    )

    if embedding_dim != 1024:

        raise RuntimeError(
            "Expected Qwen embedding dimension 1024, "
            f"found {embedding_dim}."
        )

    if not torch.isfinite(
        concept_embeddings
    ).all():

        raise RuntimeError(
            "Non-finite concept embeddings."
        )

    concept_norms = torch.linalg.vector_norm(
        concept_embeddings,
        ord=2,
        dim=1,
    )

    print(
        "Concept norm min/max:",
        (
            float(
                concept_norms.min()
            ),
            float(
                concept_norms.max()
            ),
        ),
    )

    if not torch.allclose(
        concept_norms,
        torch.ones_like(
            concept_norms
        ),
        atol=1e-5,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Concept embeddings are not unit normalized."
        )

    print(
        "Qwen concept embeddings: PASS"
    )

    road_embeddings = (
        binary_features
        @
        concept_embeddings
    )

    normalized_road_embeddings = (
        torch.zeros_like(
            road_embeddings,
            dtype=torch.float32,
        )
    )

    if active_mask.any():

        normalized_road_embeddings[
            active_mask
        ] = F.normalize(
            road_embeddings[
                active_mask
            ],
            p=2,
            dim=1,
            eps=1e-12,
        )

    road_embeddings = (
        normalized_road_embeddings
    )

    print()

    print(
        "========================================"
    )

    print(
        "ROAD EMBEDDING VERIFICATION"
    )

    print(
        "========================================"
    )

    print(
        "Road embedding shape:",
        tuple(
            road_embeddings.shape
        ),
    )

    if tuple(
        road_embeddings.shape
    ) != (
        EXPECTED_ROAD_COUNT,
        1024,
    ):

        raise RuntimeError(
            "Unexpected road embedding shape."
        )

    if (
        road_embeddings.dtype
        != torch.float32
    ):

        raise RuntimeError(
            "Road embeddings are not float32."
        )

    if not torch.isfinite(
        road_embeddings
    ).all():

        raise RuntimeError(
            "Non-finite road embeddings."
        )

    zero_rows_exact = bool(
        torch.equal(
            road_embeddings[
                ~active_mask
            ],
            torch.zeros_like(
                road_embeddings[
                    ~active_mask
                ]
            ),
        )
    )

    print(
        "Zero roads exact zero:",
        zero_rows_exact,
    )

    if not zero_rows_exact:

        raise RuntimeError(
            "Missing-semantic roads "
            "are not exact zeros."
        )

    active_norms = torch.linalg.vector_norm(
        road_embeddings[
            active_mask
        ],
        ord=2,
        dim=1,
    )

    print(
        "Active-road norm min:",
        float(
            active_norms.min()
        ),
    )

    print(
        "Active-road norm max:",
        float(
            active_norms.max()
        ),
    )

    print(
        "Active-road norm mean:",
        float(
            active_norms.mean()
        ),
    )

    if not torch.allclose(
        active_norms,
        torch.ones_like(
            active_norms
        ),
        atol=1e-5,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Active road embeddings "
            "are not unit normalized."
        )

    derived_mask = (
        road_embeddings
        .abs()
        .sum(
            dim=1
        )
        > 0
    )

    mask_match = bool(
        torch.equal(
            derived_mask,
            active_mask,
        )
    )

    print(
        "Qwen support mask matches "
        "structured support:",
        mask_match,
    )

    if not mask_match:

        raise RuntimeError(
            "Qwen road support differs "
            "from structured semantics."
        )

    qwen_active_count = int(
        derived_mask
        .sum()
        .item()
    )

    print(
        "Qwen active roads:",
        qwen_active_count,
    )

    print(
        "Qwen zero roads:",
        int(
            (
                ~derived_mask
            )
            .sum()
            .item()
        ),
    )

    artifact = {
        "semantic_features":
            road_embeddings.contiguous(),

        "feature_names":
            feature_names,

        "concept_descriptions":
            descriptions,

        "concept_embeddings":
            concept_embeddings.contiguous(),

        "model_name":
            MODEL_NAME,
        "model_revision": MODEL_REVISION,

        "source_structured_cache_sha256":
            structured_sha,

        "construction":
            (
                "Unit-normalized Qwen embedding "
                "for each of the same 10 structured "
                "TTE concepts; active concept embeddings "
                "are summed per road and the resulting "
                "road vector is L2-normalized. "
                "Roads with no active concepts remain "
                "exactly zero."
            ),
    }

    with OUTPUT_CACHE.open("xb") as stream:
        torch.save(artifact, stream)

    output_sha = (
        sha256_file(
            OUTPUT_CACHE
        )
    )

    reloaded = torch.load(
        OUTPUT_CACHE,
        map_location="cpu",
        weights_only=True,
    )

    reload_features = (
        reloaded[
            "semantic_features"
        ]
    )

    reload_exact = bool(
        torch.equal(
            reload_features,
            road_embeddings,
        )
    )

    if not reload_exact:

        raise RuntimeError(
            "Saved Qwen semantic cache "
            "does not reload exactly."
        )

    meta = {
        "version":
            "tte_qwen_v1",

        "model_name":
            MODEL_NAME,
        "model_revision": MODEL_REVISION,

        "embedding_dimension":
            1024,

        "road_count":
            EXPECTED_ROAD_COUNT,

        "concept_count":
            EXPECTED_CONCEPT_COUNT,

        "feature_names":
            feature_names,

        "concept_descriptions": {
            name:
                CONCEPT_DESCRIPTIONS[
                    name
                ]
            for name
            in feature_names
        },

        "concept_support_counts":
            concept_counts,

        "active_road_count":
            active_count,

        "zero_semantic_road_count":
            zero_count,

        "structured_source_path":
            str(
                STRUCTURED_CACHE.resolve()
            ),

        "structured_source_sha256":
            structured_sha,

        "output_cache_path":
            str(
                OUTPUT_CACHE.resolve()
            ),

        "output_cache_sha256":
            output_sha,

        "qwen_cache_root":
            str(
                HF_CACHE.resolve()
            ),

        "normalization": {
            "concept_embeddings":
                "L2 unit norm",

            "road_embedding":
                (
                    "sum active concept embeddings, "
                    "then L2-normalize active roads"
                ),

            "missing_semantics":
                "exact zero vector",
        },

        "information_control":
            (
                "Uses exactly the same 10 binary "
                "road-context facts as "
                "cached_xa_tte_semantics_binary_v1.pth. "
                "No external data or additional "
                "road facts are introduced."
            ),
    }

    with OUTPUT_META.open(
        "x",
        encoding="utf-8",
    ) as handle:

        json.dump(
            meta,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    meta_sha = (
        sha256_file(
            OUTPUT_META
        )
    )

    print()

    print(
        "========================================"
    )

    print(
        "TTE QWEN CACHE SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "Roads:",
        EXPECTED_ROAD_COUNT,
    )

    print(
        "Concepts:",
        EXPECTED_CONCEPT_COUNT,
    )

    print(
        "Embedding dim:",
        embedding_dim,
    )

    print(
        "Active roads:",
        active_count,
    )

    print(
        "Zero-semantic roads:",
        zero_count,
    )

    print(
        "Support identical to structured:",
        mask_match,
    )

    print(
        "Zero roads exact zero:",
        zero_rows_exact,
    )

    print(
        "Reload exact:",
        reload_exact,
    )

    print()

    print(
        "Structured source SHA-256:",
        structured_sha,
    )

    print()

    print(
        "Output cache:"
    )

    print(
        OUTPUT_CACHE.resolve()
    )

    print(
        "Output cache SHA-256:",
        output_sha,
    )

    print()

    print(
        "Metadata:"
    )

    print(
        OUTPUT_META.resolve()
    )

    print(
        "Metadata SHA-256:",
        meta_sha,
    )

    print()

    print(
        "========================================"
    )

    print(
        "TTE QWEN SEMANTIC CACHE BUILD: PASS"
    )

    print(
        "========================================"
    )

    print(
        "No training was performed."
    )

    print(
        "No checkpoint was modified."
    )

    print(
        "No external data were used."
    )

    print(
        "Information content is identical "
        "to TTE-Exp1 structured semantics."
    )


if __name__ == "__main__":
    main()

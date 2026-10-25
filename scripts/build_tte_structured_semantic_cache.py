"""Build binary TTE road-context features from road attributes and linked POIs.

Reference artifacts are used only for independent verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


FEATURE_NAMES = [
    "tunnel", "bridge", "roundabout", "oneway", "fuel_station", "parking",
    "transport_hub", "taxi_stand", "transit_stop", "logistics_freight",
]
POI_GROUPS = {
    "fuel_station": ["4080", "4081", "4082", "40A0"],
    "parking": ["4101", "4102"],
    "transport_hub": ["8081", "8083", "8088", "808D", "8092", "8093"],
    "taxi_stand": ["808C"],
    "transit_stop": ["808B", "808F"],
    "logistics_freight": ["A400"],
}
VERSION = "tte_semantics_binary_v1"
CANONICAL_FILE_SHA256 = "081952F5ABD0A5E005430113EACC7E516A76CF2210E8FF346039508982416717"
CANONICAL_TENSOR_SHA256 = "52EAE4C2E30FF4A91858645A37AFE59BC0E1882D1DBA8543FAE1E0E95EF7F189"
CANONICAL_SOURCE_SHA256 = {
    "POI_xa.csv": "DCC0F0F693528953A907135199DA7B27397E76C9F9639A1D5B177A798F35345D",
    "road2poi_xa.json": "C17F3796AE849EE16573D5DED3EB8C0DE5AF30F6C050CB4BC2D9FD2CFBE49BEC",
    "road_xa.csv": "D154AD2691A19C44303DC33E64EDA4A3817C667915D993BE9A06AAD3E0FB6CE7",
}
CANONICAL_COUNTS = [61, 151, 33, 2043, 27, 300, 7, 238, 831, 5]


def sha256_file(path: Path) -> str:
    
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest().upper()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash little-endian float32 tensor values in row-major feature order."""
    raw = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(raw.tobytes(order="C")).hexdigest().upper()


def require_distinct(paths: list[Path]) -> None:
    """Reject resolved aliases and existing hard links among protected paths."""
    for index, path in enumerate(paths):
        for other in paths[:index]:
            if path.resolve() == other.resolve() or (
                path.exists() and other.exists() and os.path.samefile(path, other)
            ):
                raise ValueError(f"Paths must be distinct: {path} and {other}")


def source_hashes(road_file: Path, poi_file: Path, mapping_file: Path) -> dict[str, str]:
    """Use canonical logical names even when inputs have portable custom names."""
    return {
        "POI_xa.csv": sha256_file(poi_file),
        "road2poi_xa.json": sha256_file(mapping_file),
        "road_xa.csv": sha256_file(road_file),
    }


def build_payload(
    road_file: Path, poi_file: Path, mapping_file: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute all facts from sources; no reference artifact is accepted here."""
    road = pd.read_csv(road_file, usecols=["id", *FEATURE_NAMES[:4]])
    ids = road["id"].to_numpy()
    if len(road) == 0 or not pd.api.types.is_integer_dtype(road["id"]):
        raise ValueError("Road IDs must be nonempty integers.")
    if not road["id"].is_unique or not np.array_equal(np.sort(ids), np.arange(len(road))):
        raise ValueError("Road IDs must be unique and contiguous from zero.")
    road = road.set_index("id").loc[np.arange(len(road))]
    attributes = road[FEATURE_NAMES[:4]].to_numpy()
    if not np.isin(attributes, [0, 1]).all():
        raise ValueError("Road attributes must already be binary, without missing values.")
    poi = pd.read_csv(poi_file, usecols=["id", "TYPE"], dtype={"id": str, "TYPE": str})
    if poi[["id", "TYPE"]].isna().any().any() or not poi["id"].is_unique:
        raise ValueError("POI IDs/types must be present and POI IDs unique.")
    poi_types = dict(zip(poi["id"], poi["TYPE"]))
    with Path(mapping_file).open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError("Road-to-POI mapping must be an object.")
    features = torch.zeros((len(road), len(FEATURE_NAMES)), dtype=torch.float32)
    features[:, :4] = torch.from_numpy(attributes.astype(np.float32))
    seen_roads: set[int] = set()
    for key, references in mapping.items():
        road_id = int(key)
        if str(road_id) != key or road_id in seen_roads or not 0 <= road_id < len(road):
            raise ValueError(f"Invalid or duplicated mapped road ID: {key}")
        seen_roads.add(road_id)
        if not isinstance(references, list):
            raise ValueError("Each road mapping must contain a list of POI references.")
        types = set()
        for reference in references:
            if isinstance(reference, bool) or not isinstance(reference, (str, int)):
                raise ValueError("POI references must be string or integer IDs.")
            try:
                types.add(poi_types[str(reference)])
            except KeyError as exc:
                raise ValueError(f"Unknown POI reference: {reference}") from exc
        for column, name in enumerate(FEATURE_NAMES[4:], start=4):
            features[road_id, column] = bool(types.intersection(POI_GROUPS[name]))
    payload = {
        "semantic_features": features,
        "feature_names": list(FEATURE_NAMES),
        "road_count": len(road),
        "semantic_dim": len(FEATURE_NAMES),
        "version": VERSION,
    }
    counts = features.sum(dim=0).to(torch.int64).tolist()
    active = features.sum(dim=1)
    metadata = {
        "all_zero_roads": int((active == 0).sum()),
        "feature_names": list(FEATURE_NAMES),
        "max_active_concepts_per_road": int(active.max()),
        "poi_groups": POI_GROUPS,
        "positive_road_counts": dict(zip(FEATURE_NAMES, counts)),
        "road_count": len(road),
        "roads_with_any_semantic": int((active > 0).sum()),
        "semantic_dim": len(FEATURE_NAMES),
        "source_sha256": source_hashes(road_file, poi_file, mapping_file),
        "version": VERSION,
    }
    return payload, metadata


def validate_payload(payload: dict[str, Any], *, canonical: bool = False) -> str:
    """Validate schema and binary facts; optionally enforce the locked XA profile."""
    keys = {"semantic_features", "feature_names", "road_count", "semantic_dim", "version"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Structured cache schema mismatch.")
    tensor = payload["semantic_features"]
    if type(payload["road_count"]) is not int or payload["road_count"] <= 0:
        raise ValueError("Invalid road count.")
    if (payload["feature_names"] != FEATURE_NAMES or type(payload["semantic_dim"]) is not int
            or payload["semantic_dim"] != 10 or payload["version"] != VERSION):
        raise ValueError("Structured cache feature specification mismatch.")
    if (not torch.is_tensor(tensor) or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (payload["road_count"], 10)
            or tensor.device.type != "cpu" or not tensor.is_contiguous()):
        raise ValueError("Expected a contiguous CPU float32 road-by-10 tensor.")
    if not torch.isfinite(tensor).all() or not ((tensor == 0) | (tensor == 1)).all():
        raise ValueError("Structured semantic values must be finite and binary.")
    digest = tensor_sha256(tensor)
    if canonical:
        active = tensor.sum(1)
        if (payload["road_count"] != 5269 or tensor.sum(0).tolist() != CANONICAL_COUNTS
                or int((active > 0).sum()) != 2437 or int((active == 0).sum()) != 2832
                or int(active.max()) != 5 or digest != CANONICAL_TENSOR_SHA256):
            raise ValueError("Canonical XA tensor hash/support invariants mismatch.")
    return digest


def compare_payloads(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare the entire tensor and every non-tensor field, without tolerance."""
    validate_payload(actual)
    validate_payload(expected)
    for key in actual:
        equal = (torch.equal(actual[key], expected[key]) if key == "semantic_features"
                 else actual[key] == expected[key])
        if not equal:
            raise ValueError(f"Structured cache differs in {key}.")


def build_or_verify(
    *, road_file: Path, poi_file: Path, mapping_file: Path, output: Path,
    metadata_output: Path, reference: Path | None = None,
    reference_metadata: Path | None = None, verify_only: bool = False,
    canonical: bool = True,
) -> dict[str, Any]:
    """Reconstruct independently, verify references, and refuse existing outputs.

    ``canonical=False`` is for synthetic fixtures. The CLI always enforces XA.
    Output publication uses exclusive links so concurrent existing files are not
    replaced. The staging filesystem must support hard links.
    """
    protected = [road_file, poi_file, mapping_file]
    protected += [p for p in [reference, reference_metadata] if p is not None]
    require_distinct([*protected, output, metadata_output])
    for path in protected:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(p): sha256_file(p) for p in protected}
    if not verify_only and (output.exists() or metadata_output.exists()):
        raise FileExistsError("Output or metadata already exists; use --verify-only.")
    payload, metadata = build_payload(road_file, poi_file, mapping_file)
    digest = validate_payload(payload, canonical=canonical)
    if canonical and metadata["source_sha256"] != CANONICAL_SOURCE_SHA256:
        raise ValueError("Canonical XA input source hashes mismatch.")
    if reference is not None:
        compare_payloads(payload, torch.load(reference, map_location="cpu", weights_only=True))
    if reference_metadata is not None:
        if metadata != json.loads(reference_metadata.read_text(encoding="utf-8")):
            raise ValueError("Reference structured metadata mismatch.")
    if verify_only:
        compare_payloads(torch.load(output, map_location="cpu", weights_only=True), payload)
        if metadata != json.loads(metadata_output.read_text(encoding="utf-8")):
            raise ValueError("Output structured metadata mismatch.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".structured-", dir=output.parent) as directory:
            temporary = Path(directory) / (output.name + ".tmp")
            torch.save(payload, temporary)
            compare_payloads(torch.load(temporary, map_location="cpu", weights_only=True), payload)
            with tempfile.TemporaryDirectory(prefix=".structured-meta-", dir=metadata_output.parent) as md:
                temporary_meta = Path(md) / "metadata.json"
                with temporary_meta.open("w", encoding="utf-8", newline="\n") as handle:
                    json.dump(metadata, handle, indent=2, sort_keys=True)
                if before != {str(p): sha256_file(p) for p in protected}:
                    raise RuntimeError("A protected input/reference changed during construction.")
                os.link(temporary, output)
                try:
                    os.link(temporary_meta, metadata_output)
                except BaseException:
                    output.unlink()
                    raise
    if before != {str(p): sha256_file(p) for p in protected}:
        raise RuntimeError("A protected input/reference changed during verification.")
    file_digest = sha256_file(output)
    return {
        "mode": "verify-only" if verify_only else "build",
        "shape": list(payload["semantic_features"].shape),
        "element_count": payload["semantic_features"].numel(),
        "tensor_sha256": digest, "file_sha256": file_digest,
        "canonical_file_sha256": CANONICAL_FILE_SHA256,
        "canonical_file_bytes_equal": file_digest == CANONICAL_FILE_SHA256,
        "reference_tensor_equal": reference is not None,
        "reference_metadata_equal": reference_metadata is not None,
        "reference_file_bytes_equal": file_digest == sha256_file(reference) if reference else None,
        "metadata_file_sha256": sha256_file(metadata_output),
        "metadata_reference_bytes_equal": (metadata_output.read_bytes() == reference_metadata.read_bytes()
                                           if reference_metadata else None),
        "support_counts": list(metadata["positive_road_counts"].values()),
        "metadata": metadata, "protected_files_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["road-file", "poi-file", "mapping-file", "output", "metadata-output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-metadata", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    cli = parser.parse_args()
    print(json.dumps(build_or_verify(**vars(cli)), indent=2))


if __name__ == "__main__":
    main()

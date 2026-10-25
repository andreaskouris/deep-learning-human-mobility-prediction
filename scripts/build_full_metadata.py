"""Build BIGCity full metadata through FileLoader without creating task caches.

Requires prebuilt relation, static and controlled-dynamic caches. Existing
metadata is never an input to reconstruction. CLI runs enforce the XA profile;
the Python API also supports small synthetic fixtures for unit tests.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator
import uuid

import numpy as np
import pandas as pd
import torch

from scripts.build_tte_structured_semantic_cache import require_distinct, sha256_file


FIELDS = ["edge_cnt", "road_cnt", "time_slots_cnt", "traj_cnt", "traj_category_cnt"]
CANONICAL_METADATA = dict(zip(FIELDS, [13032, 5269, 2928, 384618, 26787]))
CANONICAL_FILE_SHA256 = "CEAE086A96DB0439AB3662ACC30AD1973179E2DBDE30EAFC36C052561DC7658E"
CANONICAL_SOURCE_SHA256 = {
    "relation_file": "145063C82CF1606ED14D91C0ACBECC71AE97B7345EB956A752B1074FAB1ED6E1",
    "static_file": "216946652825506CE7B7751CF6795ED1D7759484D2C9B115BD2FD434A3B1F28E",
    "trajectory_file": "0F1F85574B0B4C6BAD6605C2B130213774FD361CFFB2E7068661F0849582A6EB",
}


@dataclass(frozen=True)
class MetadataInputs:
    """Explicit read-only source and cache paths; no dataset layout assumption."""

    relation_file: Path
    static_file: Path
    trajectory_file: Path
    relation_cache: Path
    static_cache: Path
    dynamic_cache: Path

    def paths(self) -> dict[str, Path]:
        return dict(vars(self))


def validate_metadata(metadata: Any, *, canonical: bool = False) -> None:
    """Require all five positive integer counts, optionally matching locked XA."""
    if not isinstance(metadata, dict) or set(metadata) != set(FIELDS):
        raise ValueError("Full metadata must contain exactly the five count fields.")
    if any(type(metadata[key]) is not int or metadata[key] <= 0 for key in FIELDS):
        raise ValueError("Metadata counts must be positive integers.")
    if canonical and metadata != CANONICAL_METADATA:
        raise ValueError("Computed metadata differs from the canonical XA profile.")


@contextmanager
def temporary_attributes(obj: Any, values: dict[str, Any]) -> Iterator[None]:
    """Restore module/class attributes even when an imported loader fails."""
    missing = object()
    original = {name: getattr(obj, name, missing) for name in values}
    try:
        for name, value in values.items():
            setattr(obj, name, value)
        yield
    finally:
        for name, value in original.items():
            if value is missing:
                delattr(obj, name)
            else:
                setattr(obj, name, value)


class ReadOnlyTorch:
    """Keep FileLoader's real torch reads but reject cache creation fallbacks."""

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    def save(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("FileLoader attempted to write a support cache.")


@contextmanager
def loader_session(inputs: MetadataInputs, absent_metadata: Path) -> Iterator[Any]:
    """Use the existing class with isolated paths, singleton and NumPy state.

    The process-global BIGCity configuration is restored on exit. This helper
    is intentionally single-threaded, as is the CLI. No model is constructed.
    """
    if absent_metadata.exists():
        raise FileExistsError("Reconstruction metadata path must not exist.")
    original_argv = sys.argv[:]
    numpy_state = np.random.get_state()
    try:
        sys.argv = [original_argv[0], "--city", "xa", "--split", "full",
                    "--seq_len", "64", "--sample_rate", "1", "--seed", "2021",
                    "--dynamic_mode", "controlled"]
        module = importlib.import_module("data_provider.file_loader")
        globals_module = module.global_vars
        overrides = {
            "road_relation_file": str(inputs.relation_file),
            "road_static_file": str(inputs.static_file),
            "road_relation_tensor_file": str(inputs.relation_cache),
            "road_static_tensor_file": str(inputs.static_cache),
            "road_dynamic_tensor_file": str(inputs.dynamic_cache),
            # If a cache disappears, fallback must fail rather than read .dyna.
            "road_dynamic_file": str(absent_metadata.parent / (uuid.uuid4().hex + ".dyna")),
            "cur_traj_file": str(inputs.trajectory_file),
            "traj_file": str(inputs.trajectory_file),
            "dataset_meta_file": str(absent_metadata),
        }
        settings = {"city": "xa", "split": "full", "seq_len": 64,
                    "sample_rate": 1.0, "seed": 2021, "dynamic_mode": "controlled"}
        with (temporary_attributes(module.args, settings),
              temporary_attributes(globals_module, overrides),
              temporary_attributes(module, {"torch": ReadOnlyTorch()}),
              temporary_attributes(module.FileLoader, {"_instance": None})):
            np.random.seed(2021)
            yield module.FileLoader()
    finally:
        sys.argv = original_argv
        np.random.set_state(numpy_state)


def compute_metadata(loader: Any, inputs: MetadataInputs) -> dict[str, int]:
    """Run the real loader read methods and cross-check sources and tensors."""
    loader.load_road_relation_file()
    loader.load_road_features_file()
    loader.load_traj_dataset_file()
    metadata = loader.get_meta_info()
    validate_metadata(metadata)
    relation = pd.read_csv(inputs.relation_file,
                           usecols=["origin_id", "destination_id", "geographical_weight"])
    raw_ids = relation[["origin_id", "destination_id"]].to_numpy()
    if not np.isfinite(raw_ids).all() or not (raw_ids == np.floor(raw_ids)).all():
        raise ValueError("Relation road IDs must be finite integers.")
    edges = torch.tensor(raw_ids.astype("int64")).T
    weights = torch.tensor(relation["geographical_weight"].to_numpy(dtype="float32"))
    static = pd.read_csv(inputs.static_file).to_numpy()
    if not np.isfinite(static).all() or not (static == np.floor(static)).all():
        raise ValueError("Static feature values must be finite integers.")
    static_tensor = torch.tensor(static, dtype=torch.int64)
    if (loader.edges.dtype != torch.int64 or not torch.equal(edges, loader.edges)
            or loader.edge_weight.dtype != torch.float32
            or not torch.equal(weights, loader.edge_weight)
            or not torch.isfinite(loader.edge_weight).all()
            or metadata["edge_cnt"] != len(relation)):
        raise ValueError("Relation cache differs from the raw source or count.")
    if (loader.static_features.dtype != torch.int64
            or not torch.equal(static_tensor, loader.static_features)
            or metadata["road_cnt"] != len(static)):
        raise ValueError("Static cache differs from the raw source or count.")
    if edges.min() < 0 or edges.max() >= metadata["road_cnt"]:
        raise ValueError("Relation cache has out-of-range road IDs.")
    dynamic = loader.dynamic_features
    if (dynamic.dtype != torch.float32 or tuple(dynamic.shape)
            != (metadata["road_cnt"], metadata["time_slots_cnt"])
            or not torch.isfinite(dynamic).all()):
        raise ValueError("Dynamic cache shape/dtype/finite-value mismatch.")
    if "usr_id" not in loader.traj_data or loader.traj_data["usr_id"].isna().any():
        raise ValueError("Full trajectory source must have nonmissing usr_id values.")
    if metadata["traj_cnt"] != len(loader.traj_data):
        raise ValueError("Full trajectory count mismatch.")
    if metadata["traj_category_cnt"] != len(set(loader.traj_data["usr_id"])):
        raise ValueError("User/category count was not reconstructed from full data.")
    return metadata


def serialize_metadata(metadata: dict[str, int], line_endings: str) -> bytes:
    """Stable field order and no final newline; LF/CRLF is reported separately."""
    text = json.dumps({key: metadata[key] for key in FIELDS}, indent=4)
    if line_endings == "crlf":
        text = text.replace("\n", "\r\n")
    elif line_endings != "lf":
        raise ValueError("line_endings must be lf or crlf.")
    return text.encode("utf-8")


def build_or_verify(
    *, inputs: MetadataInputs, output: Path, reference: Path | None = None,
    verify_only: bool = False, line_endings: str = "lf", canonical: bool = True,
) -> dict[str, Any]:
    """Compute independently with absent metadata; never build support/task caches."""
    protected = list(inputs.paths().values()) + ([reference] if reference else [])
    require_distinct([*protected, output])
    for path in protected:
        if not path.is_file():
            raise FileNotFoundError(path)
    if verify_only and not output.is_file():
        raise FileNotFoundError(output)
    if not verify_only and output.exists():
        raise FileExistsError("Metadata output already exists; use --verify-only.")
    serialize_metadata(CANONICAL_METADATA, line_endings)  # Validate the format option only.
    before = {str(path): sha256_file(path) for path in protected}
    source_hashes = {name: before[str(path)] for name, path in inputs.paths().items()}
    if canonical and any(source_hashes[name] != expected
                         for name, expected in CANONICAL_SOURCE_SHA256.items()):
        raise ValueError("Canonical XA raw source hashes mismatch.")
    reference_value = json.loads(reference.read_text(encoding="utf-8")) if reference else None
    if reference is not None:
        validate_metadata(reference_value, canonical=canonical)

    def reconstruct(candidate: Path, *, save: bool) -> dict[str, int]:
        with loader_session(inputs, candidate) as loader:
            value = compute_metadata(loader, inputs)
            validate_metadata(value, canonical=canonical)
            if reference is not None and value != reference_value:
                raise ValueError("Reconstructed metadata differs from reference.")
            if save:
                loader.save_meta_info(str(candidate))
                saved = json.loads(candidate.read_text(encoding="utf-8"))
                validate_metadata(saved)
                if saved != value:
                    raise ValueError("FileLoader metadata serialization changed content.")
                candidate.write_bytes(serialize_metadata(value, line_endings))
            return value

    if verify_only:
        absent = output.parent / (".metadata-" + uuid.uuid4().hex + ".json")
        metadata = reconstruct(absent, save=False)
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_metadata(existing, canonical=canonical)
        if existing != metadata:
            raise ValueError("Existing output differs from independently reconstructed metadata.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".full-metadata-", dir=output.parent) as directory:
            temporary = Path(directory) / "metadata.json"
            metadata = reconstruct(temporary, save=True)
            if before != {str(path): sha256_file(path) for path in protected}:
                raise RuntimeError("A protected metadata source/reference changed.")
            os.link(temporary, output)
    if before != {str(path): sha256_file(path) for path in protected}:
        raise RuntimeError("A protected metadata source/reference changed.")
    data = output.read_bytes()
    file_digest = sha256_file(output)
    return {
        "mode": "verify-only" if verify_only else "build", "metadata": metadata,
        "reference_dictionary_equal": reference is not None,
        "file_sha256": file_digest, "canonical_file_sha256": CANONICAL_FILE_SHA256,
        "canonical_file_bytes_equal": file_digest == CANONICAL_FILE_SHA256,
        "reference_file_bytes_equal": data == reference.read_bytes() if reference else None,
        "line_endings": "crlf" if b"\r\n" in data else "lf",
        "source_sha256": source_hashes, "protected_files_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in MetadataInputs.__dataclass_fields__:
        parser.add_argument("--" + field.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--line-endings", choices=["lf", "crlf"], default="lf")
    cli = vars(parser.parse_args())
    inputs = MetadataInputs(**{key: cli.pop(key) for key in MetadataInputs.__dataclass_fields__})
    print(json.dumps(build_or_verify(inputs=inputs, **cli), indent=2))


if __name__ == "__main__":
    main()

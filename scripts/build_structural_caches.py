"""Build and validate road-relation and static-feature caches through FileLoader."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class StructuralPaths:
    """Input and output paths required by the structural cache builder."""

    relation_source: Path
    static_source: Path
    relation_cache: Path
    static_cache: Path
    dynamic_cache: Path

    @classmethod
    def from_global_vars(cls, global_vars: Any) -> "StructuralPaths":
        """Create resolved paths from BIGCity's global configuration."""

        return cls(
            relation_source=Path(
                global_vars.road_relation_file
            ).resolve(),
            static_source=Path(
                global_vars.road_static_file
            ).resolve(),
            relation_cache=Path(
                global_vars.road_relation_tensor_file
            ).resolve(),
            static_cache=Path(
                global_vars.road_static_tensor_file
            ).resolve(),
            dynamic_cache=Path(
                global_vars.road_dynamic_tensor_file
            ).resolve(),
        )


@dataclass(frozen=True)
class CacheExpectations:
    """Expected cache dimensions for a known dataset."""

    road_count: int
    edge_count: int
    static_width: int
    time_slots: int


@dataclass(frozen=True)
class FileFingerprint:
    """Properties used to detect whether a file changed."""

    size: int
    modified_ns: int
    sha256: str

    @classmethod
    def from_path(cls, path: Path) -> "FileFingerprint":
        stat = path.stat()

        return cls(
            size=stat.st_size,
            modified_ns=stat.st_mtime_ns,
            sha256=sha256_file(path),
        )


@dataclass(frozen=True)
class StructuralCacheReport:
    """Validated information about the generated caches."""

    relation: dict[str, Any]
    static: dict[str, Any]
    dynamic: dict[str, Any]


CITY_EXPECTATIONS: dict[str, CacheExpectations] = {
    "xa": CacheExpectations(
        road_count=5269,
        edge_count=13032,
        static_width=8,
        time_slots=2928,
    ),
}


def sha256_file(path: Path) -> str:
    

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def require_input_files(paths: StructuralPaths) -> None:
    """Verify that all required source and dynamic-cache files exist."""

    required = {
        "relation source": paths.relation_source,
        "static source": paths.static_source,
        "dynamic cache": paths.dynamic_cache,
    }

    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n"
            + "\n".join(missing)
        )


def prepare_output_files(
    paths: StructuralPaths,
    *,
    verify_only: bool,
    force: bool,
) -> None:
    """Apply overwrite protection before building or verifying caches."""

    if verify_only and force:
        raise ValueError(
            "--verify-only and --force cannot be used together."
        )

    outputs = (
        paths.relation_cache,
        paths.static_cache,
    )

    if verify_only:
        missing = [
            str(path)
            for path in outputs
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Cannot verify missing structural caches:\n"
                + "\n".join(missing)
            )

        return

    existing = [
        str(path)
        for path in outputs
        if path.exists()
    ]

    if existing and not force:
        raise FileExistsError(
            "Structural cache files already exist.\n"
            "Use --verify-only to inspect them or --force "
            "to rebuild them:\n"
            + "\n".join(existing)
        )

    if force:
        for path in outputs:
            path.unlink(missing_ok=True)


def load_cache(path: Path) -> dict[str, Any]:
    """Load a PyTorch cache using the safer weights-only mode."""

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(cache, dict):
        raise TypeError(
            f"Cache must contain a dictionary: {path}"
        )

    return cache


def validate_relation_cache(
    path: Path,
    expectations: CacheExpectations | None,
) -> dict[str, Any]:
    """Validate the road-relation graph cache."""

    cache = load_cache(path)

    expected_keys = {
        "edges",
        "edge_weight",
        "edge_cnt",
    }

    if set(cache.keys()) != expected_keys:
        raise ValueError(
            "Unexpected relation-cache keys: "
            f"{sorted(cache.keys())}"
        )

    edges = cache["edges"]
    weights = cache["edge_weight"]
    edge_count = int(cache["edge_cnt"])

    if not isinstance(edges, torch.Tensor):
        raise TypeError("Relation edges must be a tensor.")

    if not isinstance(weights, torch.Tensor):
        raise TypeError("Relation weights must be a tensor.")

    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(
            "Relation edges must have shape (2, edge_count)."
        )

    if weights.ndim != 1:
        raise ValueError(
            "Relation weights must be one-dimensional."
        )

    if edges.dtype != torch.int64:
        raise TypeError(
            f"Unexpected edges dtype: {edges.dtype}"
        )

    if weights.dtype != torch.float32:
        raise TypeError(
            f"Unexpected edge-weight dtype: {weights.dtype}"
        )

    if edges.shape[1] != edge_count:
        raise ValueError(
            "edge_cnt does not match the edges tensor."
        )

    if weights.shape[0] != edge_count:
        raise ValueError(
            "edge_cnt does not match the weights tensor."
        )

    if not bool(torch.isfinite(weights).all()):
        raise ValueError(
            "Relation weights contain NaN or infinite values."
        )

    if expectations is not None:
        if edge_count != expectations.edge_count:
            raise ValueError(
                "Unexpected number of relation edges: "
                f"{edge_count}"
            )

    return {
        "keys": sorted(cache.keys()),
        "edges_shape": tuple(edges.shape),
        "edges_dtype": str(edges.dtype),
        "weights_shape": tuple(weights.shape),
        "weights_dtype": str(weights.dtype),
        "edge_count": edge_count,
        "finite_weights": True,
        "non_positive_weights": int(
            (weights <= 0).sum().item()
        ),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_static_cache(
    path: Path,
    expectations: CacheExpectations | None,
) -> dict[str, Any]:
    """Validate the road-level static-feature cache."""

    cache = load_cache(path)

    expected_keys = {
        "static_features",
        "road_cnt",
    }

    if set(cache.keys()) != expected_keys:
        raise ValueError(
            "Unexpected static-cache keys: "
            f"{sorted(cache.keys())}"
        )

    features = cache["static_features"]
    road_count = int(cache["road_cnt"])

    if not isinstance(features, torch.Tensor):
        raise TypeError(
            "Static features must be a tensor."
        )

    if features.ndim != 2:
        raise ValueError(
            "Static features must be two-dimensional."
        )

    if features.dtype != torch.int64:
        raise TypeError(
            f"Unexpected static-feature dtype: "
            f"{features.dtype}"
        )

    if features.shape[0] != road_count:
        raise ValueError(
            "road_cnt does not match the static tensor."
        )

    if expectations is not None:
        expected_shape = (
            expectations.road_count,
            expectations.static_width,
        )

        if tuple(features.shape) != expected_shape:
            raise ValueError(
                "Unexpected static-feature shape: "
                f"{tuple(features.shape)}"
            )

        if road_count != expectations.road_count:
            raise ValueError(
                f"Unexpected road count: {road_count}"
            )

    return {
        "keys": sorted(cache.keys()),
        "shape": tuple(features.shape),
        "dtype": str(features.dtype),
        "road_count": road_count,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_dynamic_cache(
    path: Path,
    expectations: CacheExpectations | None,
) -> dict[str, Any]:
    """Validate the existing dynamic cache without modifying it."""

    cache = load_cache(path)

    expected_keys = {
        "dynamic_features",
        "time_slots_cnt",
    }

    if set(cache.keys()) != expected_keys:
        raise ValueError(
            "Unexpected dynamic-cache keys: "
            f"{sorted(cache.keys())}"
        )

    features = cache["dynamic_features"]
    time_slots = int(cache["time_slots_cnt"])

    if not isinstance(features, torch.Tensor):
        raise TypeError(
            "Dynamic features must be a tensor."
        )

    if features.ndim != 2:
        raise ValueError(
            "Dynamic features must be two-dimensional."
        )

    if features.dtype != torch.float32:
        raise TypeError(
            f"Unexpected dynamic-feature dtype: "
            f"{features.dtype}"
        )

    if not bool(torch.isfinite(features).all()):
        raise ValueError(
            "Dynamic features contain NaN or infinite values."
        )

    minimum = float(features.min().item())
    maximum = float(features.max().item())

    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            "Dynamic features are outside the [0, 1] range."
        )

    if features.shape[1] != time_slots:
        raise ValueError(
            "time_slots_cnt does not match the dynamic tensor."
        )

    if expectations is not None:
        expected_shape = (
            expectations.road_count,
            expectations.time_slots,
        )

        if tuple(features.shape) != expected_shape:
            raise ValueError(
                "Unexpected dynamic-feature shape: "
                f"{tuple(features.shape)}"
            )

        if time_slots != expectations.time_slots:
            raise ValueError(
                f"Unexpected time-slot count: {time_slots}"
            )

    return {
        "keys": sorted(cache.keys()),
        "shape": tuple(features.shape),
        "dtype": str(features.dtype),
        "time_slots": time_slots,
        "finite": True,
        "minimum": minimum,
        "maximum": maximum,
        "zero_values": int(
            (features == 0).sum().item()
        ),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_or_verify_structural_caches(
    *,
    paths: StructuralPaths,
    loader: Any,
    expectations: CacheExpectations | None,
    verify_only: bool,
    force: bool,
) -> StructuralCacheReport:
    """Build or verify relation and static caches.

    The existing dynamic cache is fingerprinted before and after the
    operation to prove that it was reused without modification.
    """

    require_input_files(paths)

    prepare_output_files(
        paths,
        verify_only=verify_only,
        force=force,
    )

    dynamic_before = FileFingerprint.from_path(
        paths.dynamic_cache
    )

    if not verify_only:
        loader.load_road_relation_file()
        loader.load_road_features_file()

    if not paths.relation_cache.is_file():
        raise FileNotFoundError(
            f"Relation cache was not created: "
            f"{paths.relation_cache}"
        )

    if not paths.static_cache.is_file():
        raise FileNotFoundError(
            f"Static cache was not created: "
            f"{paths.static_cache}"
        )

    dynamic_after = FileFingerprint.from_path(
        paths.dynamic_cache
    )

    if dynamic_before != dynamic_after:
        raise RuntimeError(
            "The existing dynamic cache changed during "
            "structural-cache processing."
        )

    relation_report = validate_relation_cache(
        paths.relation_cache,
        expectations,
    )

    static_report = validate_static_cache(
        paths.static_cache,
        expectations,
    )

    dynamic_report = validate_dynamic_cache(
        paths.dynamic_cache,
        expectations,
    )

    loaded_dynamic = getattr(
        loader,
        "dynamic_features",
        None,
    )

    if not verify_only and loaded_dynamic is not None:
        expected_shape = dynamic_report["shape"]

        if tuple(loaded_dynamic.shape) != expected_shape:
            raise ValueError(
                "The dynamic tensor loaded by file_loader "
                "has an unexpected shape."
            )

    return StructuralCacheReport(
        relation=relation_report,
        static=static_report,
        dynamic=dynamic_report,
    )


def load_bigcity_runtime(
    dynamic_mode: str,
    split: str,
) -> tuple[Any, Any, Any]:
    """Import BIGCity after supplying only supported repository flags."""

    original_argv = sys.argv[:]

    try:
        sys.argv = [
            original_argv[0],
            "--dynamic_mode",
            dynamic_mode,
            "--split",
            split,
        ]

        from config.args_config import args
        from config import global_vars
        from data_provider.file_loader import file_loader

    finally:
        sys.argv = original_argv

    args.dynamic_mode = dynamic_mode
    args.split = split

    return args, global_vars, file_loader


def print_report(
    *,
    paths: StructuralPaths,
    report: StructuralCacheReport,
    verify_only: bool,
    dynamic_mode: str,
    split: str,
) -> None:
    """Print the complete build or verification report."""

    action = "VERIFICATION" if verify_only else "BUILD"

    print(f"=== STRUCTURAL CACHE {action} ===")
    print("Dynamic mode:", dynamic_mode)
    print("Split:", split)
    print("Relation source:", paths.relation_source)
    print("Static source:", paths.static_source)
    print("Relation cache:", paths.relation_cache)
    print("Static cache:", paths.static_cache)
    print("Dynamic cache:", paths.dynamic_cache)

    relation = report.relation

    print("\n=== RELATION CACHE ===")
    print("Keys:", relation["keys"])
    print("Edges shape:", relation["edges_shape"])
    print("Edges dtype:", relation["edges_dtype"])
    print("Weights shape:", relation["weights_shape"])
    print("Weights dtype:", relation["weights_dtype"])
    print("Edge count:", relation["edge_count"])
    print("Finite weights:", relation["finite_weights"])
    print(
        "Non-positive weights:",
        relation["non_positive_weights"],
    )
    print(
        "Cache size:",
        f'{relation["size"]:,} bytes',
    )
    print("SHA-256:", relation["sha256"])

    static = report.static

    print("\n=== STATIC CACHE ===")
    print("Keys:", static["keys"])
    print("Static shape:", static["shape"])
    print("Static dtype:", static["dtype"])
    print("Road count:", static["road_count"])
    print(
        "Cache size:",
        f'{static["size"]:,} bytes',
    )
    print("SHA-256:", static["sha256"])

    dynamic = report.dynamic

    print("\n=== DYNAMIC CACHE REUSE ===")
    print("Dynamic cache unchanged: True")
    print("Dynamic shape:", dynamic["shape"])
    print("Dynamic dtype:", dynamic["dtype"])
    print("Time slots:", dynamic["time_slots"])
    print("Finite values:", dynamic["finite"])
    print("Minimum:", dynamic["minimum"])
    print("Maximum:", dynamic["maximum"])
    print("Zero values:", dynamic["zero_values"])
    print("SHA-256:", dynamic["sha256"])

    print(
        "\nStructural cache "
        f'{action.lower()} and validation: PASS'
    )
    print("No trajectory caches were created.")
    print("No training or evaluation was performed.")


def parse_arguments() -> argparse.Namespace:
    

    parser = argparse.ArgumentParser(
        description=(
            "Build or verify BIGCity relation and static caches."
        )
    )

    parser.add_argument(
        "--city",
        default="xa",
        help=(
            "Dataset identifier used for expected dimensions. "
            "Default: xa."
        ),
    )

    parser.add_argument(
        "--dynamic-mode",
        choices=("original", "controlled"),
        default="original",
        help="Dynamic preprocessing mode.",
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "full"),
        default="full",
        help="BIGCity trajectory split configuration.",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify existing caches without rebuilding them."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete and rebuild existing relation and static "
            "caches."
        ),
    )

    return parser.parse_args()


def main() -> None:
    

    cli_args = parse_arguments()

    runtime_args, global_vars, loader = (
        load_bigcity_runtime(
            dynamic_mode=cli_args.dynamic_mode,
            split=cli_args.split,
        )
    )

    if runtime_args.dynamic_mode != cli_args.dynamic_mode:
        raise RuntimeError(
            "BIGCity dynamic mode does not match the "
            "requested mode."
        )

    if runtime_args.split != cli_args.split:
        raise RuntimeError(
            "BIGCity split does not match the requested split."
        )

    paths = StructuralPaths.from_global_vars(
        global_vars
    )

    expectations = CITY_EXPECTATIONS.get(
        cli_args.city
    )

    report = build_or_verify_structural_caches(
        paths=paths,
        loader=loader,
        expectations=expectations,
        verify_only=cli_args.verify_only,
        force=cli_args.force,
    )

    print_report(
        paths=paths,
        report=report,
        verify_only=cli_args.verify_only,
        dynamic_mode=cli_args.dynamic_mode,
        split=cli_args.split,
    )


if __name__ == "__main__":
    main()
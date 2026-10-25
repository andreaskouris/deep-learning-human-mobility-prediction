"""Build or verify the next-hop trajectory cache through DatasetNextHop."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch


CACHE_KEYS = {
    "traj_road_id_lists",
    "traj_time_id_lists",
    "traj_time_features_lists",
    "nexthop_labels",
}

XA_COUNTS = {
    "train": 230_770,
    "validation": 76_923,
    "test": 76_923,
    "full": 384_618,
}


def sha256_file(path: Path) -> str:
    

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def validate_cache(
    path: Path,
    *,
    expected_samples: int | None,
    seq_len: int,
    road_count: int | None,
    time_slots: int | None,
) -> dict[str, Any]:
    """Validate the structure and contents of a next-hop cache."""

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(cache, dict):
        raise TypeError(
            "Next-hop cache must contain a dictionary."
        )

    if set(cache.keys()) != CACHE_KEYS:
        raise ValueError(
            f"Unexpected cache keys: {sorted(cache.keys())}"
        )

    roads = cache["traj_road_id_lists"]
    times = cache["traj_time_id_lists"]
    time_features = cache[
        "traj_time_features_lists"
    ]
    labels = cache["nexthop_labels"]

    tensors = {
        "traj_road_id_lists": roads,
        "traj_time_id_lists": times,
        "traj_time_features_lists": time_features,
        "nexthop_labels": labels,
    }

    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a tensor."
            )

    if roads.ndim != 2:
        raise ValueError(
            "Road tensor must be two-dimensional."
        )

    if times.ndim != 2:
        raise ValueError(
            "Time-ID tensor must be two-dimensional."
        )

    if time_features.ndim != 3:
        raise ValueError(
            "Time-feature tensor must be "
            "three-dimensional."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Label tensor must be one-dimensional."
        )

    if roads.dtype != torch.int64:
        raise TypeError(
            f"Unexpected road dtype: {roads.dtype}"
        )

    if times.dtype != torch.int32:
        raise TypeError(
            f"Unexpected time-ID dtype: {times.dtype}"
        )

    if not time_features.dtype.is_floating_point:
        raise TypeError(
            "Time features must use a "
            "floating-point dtype."
        )

    if labels.dtype != torch.int64:
        raise TypeError(
            f"Unexpected label dtype: {labels.dtype}"
        )

    samples = int(roads.shape[0])

    expected_road_shape = (
        samples,
        seq_len,
    )

    if tuple(roads.shape) != expected_road_shape:
        raise ValueError(
            "Unexpected road shape: "
            f"{tuple(roads.shape)}"
        )

    if tuple(times.shape) != expected_road_shape:
        raise ValueError(
            "Unexpected time-ID shape: "
            f"{tuple(times.shape)}"
        )

    if tuple(time_features.shape[:2]) != (
        samples,
        seq_len,
    ):
        raise ValueError(
            "Unexpected time-feature shape: "
            f"{tuple(time_features.shape)}"
        )

    if tuple(labels.shape) != (samples,):
        raise ValueError(
            "Unexpected label shape: "
            f"{tuple(labels.shape)}"
        )

    if samples == 0:
        raise ValueError(
            "Next-hop cache contains no samples."
        )

    if (
        expected_samples is not None
        and samples != expected_samples
    ):
        raise ValueError(
            f"Unexpected sample count: {samples}; "
            f"expected {expected_samples}."
        )

    if time_features.shape[2] == 0:
        raise ValueError(
            "Time-feature width cannot be zero."
        )

    if not bool(
        torch.isfinite(time_features).all()
    ):
        raise ValueError(
            "Time features contain NaN or "
            "infinite values."
        )

    road_min = int(roads.min().item())
    road_max = int(roads.max().item())

    label_min = int(labels.min().item())
    label_max = int(labels.max().item())

    time_min = int(times.min().item())
    time_max = int(times.max().item())

    if road_min < 0:
        raise ValueError(
            "Road IDs must be non-negative."
        )

    if label_min < 0:
        raise ValueError(
            "Next-hop labels must be non-negative."
        )

    if time_min < 0:
        raise ValueError(
            "Time IDs must be non-negative."
        )

    if road_count is not None:
        if road_max >= road_count:
            raise ValueError(
                "Road tensor contains an "
                "out-of-range road ID."
            )

        if label_max >= road_count:
            raise ValueError(
                "Labels contain an "
                "out-of-range road ID."
            )

    if (
        time_slots is not None
        and time_max >= time_slots
    ):
        raise ValueError(
            "Time tensor contains an "
            "out-of-range time ID."
        )

    return {
        "keys": sorted(cache.keys()),
        "samples": samples,
        "road_shape": tuple(roads.shape),
        "road_dtype": str(roads.dtype),
        "time_shape": tuple(times.shape),
        "time_dtype": str(times.dtype),
        "time_feature_shape": tuple(
            time_features.shape
        ),
        "time_feature_dtype": str(
            time_features.dtype
        ),
        "time_feature_width": int(
            time_features.shape[2]
        ),
        "label_shape": tuple(labels.shape),
        "label_dtype": str(labels.dtype),
        "road_range": (
            road_min,
            road_max,
        ),
        "label_range": (
            label_min,
            label_max,
        ),
        "time_range": (
            time_min,
            time_max,
        ),
        "finite_time_features": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_or_verify(
    *,
    trajectory_source: Path,
    target_cache: Path,
    cache_owner: Any,
    prepare_trajectory_data: Callable[[], None],
    dataset_factory: Callable[[], Any],
    expected_samples: int | None,
    seq_len: int,
    road_count: int | None,
    time_slots: int | None,
    verify_only: bool,
    force: bool,
) -> dict[str, Any]:
    """Build atomically or verify one next-hop cache."""

    if verify_only and force:
        raise ValueError(
            "--verify-only and --force "
            "cannot be combined."
        )

    if not trajectory_source.is_file():
        raise FileNotFoundError(
            "Missing trajectory file: "
            f"{trajectory_source}"
        )

    validation_args = {
        "expected_samples": expected_samples,
        "seq_len": seq_len,
        "road_count": road_count,
        "time_slots": time_slots,
    }

    if verify_only:
        if not target_cache.is_file():
            raise FileNotFoundError(
                "Missing next-hop cache: "
                f"{target_cache}"
            )

        return validate_cache(
            target_cache,
            **validation_args,
        )

    if target_cache.exists() and not force:
        raise FileExistsError(
            "Next-hop cache already exists. "
            "Use --verify-only or --force: "
            f"{target_cache}"
        )

    target_cache.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_cache = target_cache.with_suffix(
        target_cache.suffix + ".tmp"
    )

    if temporary_cache.exists():
        if not force:
            raise FileExistsError(
                "Temporary cache already exists: "
                f"{temporary_cache}"
            )

        temporary_cache.unlink()

    original_target = (
        cache_owner.cached_next_hop_dataset
    )

    dataset = None

    try:
        cache_owner.cached_next_hop_dataset = str(
            temporary_cache
        )

        # Load only trajectory data.
        # Do not call file_loader.load_all().
        prepare_trajectory_data()

        # DatasetNextHop creates the requested temporary cache.
        dataset = dataset_factory()

        if not temporary_cache.is_file():
            raise RuntimeError(
                "DatasetNextHop did not create "
                "the temporary cache: "
                f"{temporary_cache}"
            )

        # Release the large in-memory dataset before
        # loading the cache again for validation.
        del dataset
        dataset = None
        gc.collect()

        temporary_report = validate_cache(
            temporary_cache,
            **validation_args,
        )

        # Replace the final cache only after the
        # temporary cache has passed validation.
        os.replace(
            temporary_cache,
            target_cache,
        )

        final_report = validate_cache(
            target_cache,
            **validation_args,
        )

        if (
            final_report["sha256"]
            != temporary_report["sha256"]
        ):
            raise RuntimeError(
                "Cache changed during finalization."
            )

        return final_report

    finally:
        cache_owner.cached_next_hop_dataset = (
            original_target
        )

        if dataset is not None:
            del dataset
            gc.collect()

        temporary_cache.unlink(
            missing_ok=True
        )


def load_runtime(
    *,
    city: str,
    split: str,
    seq_len: int,
    sample_rate: float,
    dynamic_mode: str,
) -> tuple[Any, Any, Any, Any]:
    """Import BIGCity with explicit runtime arguments."""

    original_argv = sys.argv[:]

    try:
        sys.argv = [
            original_argv[0],
            "--city",
            city,
            "--split",
            split,
            "--seq_len",
            str(seq_len),
            "--sample_rate",
            str(sample_rate),
            "--dynamic_mode",
            dynamic_mode,
        ]

        from config.args_config import args
        from config import global_vars
        from data_provider.file_loader import file_loader
        from data_provider.data_loader import (
            DatasetNextHop,
        )

    finally:
        sys.argv = original_argv

    return (
        args,
        global_vars,
        file_loader,
        DatasetNextHop,
    )


def parse_arguments() -> argparse.Namespace:
    

    parser = argparse.ArgumentParser(
        description=(
            "Build or verify one BIGCity "
            "next-hop trajectory cache."
        )
    )

    parser.add_argument(
        "--city",
        default="xa",
        help="Dataset city identifier. Default: xa.",
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "test",
            "full",
        ),
        required=True,
        help="Trajectory split to process.",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length. Default: 64.",
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help=(
            "Fraction of trajectories to load. "
            "Default: 1.0."
        ),
    )

    parser.add_argument(
        "--dynamic-mode",
        choices=(
            "original",
            "controlled",
        ),
        default="original",
        help=(
            "BIGCity dynamic configuration. "
            "The next-hop cache itself contains "
            "trajectory data, not dynamic features."
        ),
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Validate an existing cache without "
            "rebuilding it."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace an existing final cache only "
            "after a new temporary cache passes "
            "validation."
        ),
    )

    return parser.parse_args()


def main() -> None:
    

    cli = parse_arguments()

    if cli.seq_len < 2:
        raise ValueError(
            "--seq-len must be at least 2."
        )

    if not 0.0 < cli.sample_rate <= 1.0:
        raise ValueError(
            "--sample-rate must be in "
            "the interval (0, 1]."
        )

    (
        args,
        global_vars,
        file_loader,
        dataset_class,
    ) = load_runtime(
        city=cli.city,
        split=cli.split,
        seq_len=cli.seq_len,
        sample_rate=cli.sample_rate,
        dynamic_mode=cli.dynamic_mode,
    )

    configuration_matches = (
        args.city == cli.city
        and args.split == cli.split
        and args.seq_len == cli.seq_len
        and float(args.sample_rate)
        == cli.sample_rate
        and args.dynamic_mode
        == cli.dynamic_mode
    )

    if not configuration_matches:
        raise RuntimeError(
            "BIGCity runtime configuration "
            "does not match the CLI."
        )

    trajectory_source = Path(
        global_vars.cur_traj_file
    ).resolve()

    target_cache = Path(
        global_vars.cached_next_hop_dataset
    ).resolve()

    expected_samples = None
    road_count = None
    time_slots = None

    if cli.city == "xa":
        road_count = 5269
        time_slots = 2928

        if cli.sample_rate == 1.0:
            expected_samples = (
                XA_COUNTS[cli.split]
            )

    action = (
        "verification"
        if cli.verify_only
        else "build"
    )

    report = build_or_verify(
        trajectory_source=trajectory_source,
        target_cache=target_cache,
        cache_owner=global_vars,
        prepare_trajectory_data=(
            file_loader.load_traj_dataset_file
        ),
        dataset_factory=dataset_class,
        expected_samples=expected_samples,
        seq_len=cli.seq_len,
        road_count=road_count,
        time_slots=time_slots,
        verify_only=cli.verify_only,
        force=cli.force,
    )

    print(
        f"=== NEXT-HOP CACHE "
        f"{action.upper()} ==="
    )
    print("City:", cli.city)
    print("Split:", cli.split)
    print("Sequence length:", cli.seq_len)
    print("Sample rate:", cli.sample_rate)
    print("Dynamic mode:", cli.dynamic_mode)
    print(
        "Trajectory source:",
        trajectory_source,
    )
    print("Target cache:", target_cache)

    print("\n=== CACHE CONTENT ===")

    for key, value in report.items():
        print(f"{key}: {value}")

    print(
        f"\nNext-hop cache {action} "
        "and validation: PASS"
    )
    print(
        "No other task cache was requested."
    )
    print(
        "No model training or evaluation "
        "was performed."
    )


if __name__ == "__main__":
    main()
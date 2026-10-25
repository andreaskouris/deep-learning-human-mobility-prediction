from __future__ import annotations

import hashlib
from ast import literal_eval
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.timefeatures import time_features


TTE_CACHE_VERSION = "v1"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def build_tte_payload(
    *,
    trajectory_file: str | Path,
    split: str,
    seq_len: int,
    road_count: int,
    start_timestamp: int,
    interval_seconds: int,
    time_slots_count: int,
    sample_rate: float = 1.0,
    seed: int = 2021,
) -> dict[str, Any]:
    """Build trip-level TTE inputs from the route and departure time.

    Intermediate and arrival timestamps validate targets, never model inputs.
    """

    trajectory_file = Path(trajectory_file)

    if not trajectory_file.is_file():
        raise FileNotFoundError(
            f"Missing trajectory file: {trajectory_file}"
        )

    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    if road_count <= 0:
        raise ValueError("road_count must be positive.")

    if interval_seconds <= 0:
        raise ValueError(
            "interval_seconds must be positive."
        )

    if time_slots_count <= 0:
        raise ValueError(
            "time_slots_count must be positive."
        )

    if not (0 < sample_rate <= 1):
        raise ValueError(
            "sample_rate must be in (0, 1]."
        )

    data = pd.read_csv(
        trajectory_file,
        delimiter=";",
    )

    required_columns = {
        "id",
        "path",
        "tlist",
        "duration",
    }

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required trajectory columns: "
            f"{sorted(missing_columns)}"
        )

    if sample_rate < 1.0:
        data = data.sample(
            frac=sample_rate,
            random_state=seed,
        )

    data = data.reset_index(drop=True)

    sample_count = len(data)

    pad_road_id = road_count
    empty_time_id = time_slots_count

    road_ids = torch.full(
        (sample_count, seq_len),
        fill_value=pad_road_id,
        dtype=torch.int64,
    )

    time_ids = torch.full(
        (sample_count, seq_len),
        fill_value=empty_time_id,
        dtype=torch.int32,
    )

    time_feature_tensor = torch.zeros(
        (sample_count, seq_len, 6),
        dtype=torch.float32,
    )

    valid_mask = torch.zeros(
        (sample_count, seq_len),
        dtype=torch.bool,
    )

    lengths = torch.empty(
        sample_count,
        dtype=torch.int64,
    )

    departure_timestamps = torch.empty(
        sample_count,
        dtype=torch.int64,
    )

    duration_seconds = torch.empty(
        sample_count,
        dtype=torch.float32,
    )

    sample_ids = []

    for row_index, row in data.iterrows():
        path = literal_eval(row["path"])
        timestamps = literal_eval(row["tlist"])

        if not isinstance(path, list):
            raise TypeError(
                f"Row {row_index}: path is not a list."
            )

        if not isinstance(timestamps, list):
            raise TypeError(
                f"Row {row_index}: tlist is not a list."
            )

        if len(path) != len(timestamps):
            raise ValueError(
                f"Row {row_index}: path/tlist length "
                "mismatch."
            )

        if len(path) == 0:
            raise ValueError(
                f"Row {row_index}: empty trajectory."
            )

        if len(path) > seq_len:
            raise ValueError(
                f"Row {row_index}: trajectory length "
                f"{len(path)} exceeds seq_len={seq_len}."
            )

        path_array = np.asarray(
            path,
            dtype=np.int64,
        )

        if (
            path_array.min() < 0
            or path_array.max() >= road_count
        ):
            raise ValueError(
                f"Row {row_index}: road ID outside "
                f"[0, {road_count - 1}]."
            )

        timestamp_array = np.asarray(
            timestamps,
            dtype=np.int64,
        )

        if np.any(
            np.diff(timestamp_array) < 0
        ):
            raise ValueError(
                f"Row {row_index}: timestamps are "
                "not monotonic."
            )

        departure = int(timestamp_array[0])

        computed_duration = int(
            timestamp_array[-1]
            - timestamp_array[0]
        )

        csv_duration = int(row["duration"])

        if computed_duration != csv_duration:
            raise ValueError(
                f"Row {row_index}: duration mismatch "
                f"(CSV={csv_duration}, "
                f"computed={computed_duration})."
            )

        if computed_duration <= 0:
            raise ValueError(
                f"Row {row_index}: non-positive "
                "duration."
            )

        departure_time_id = (
            departure - start_timestamp
        ) // interval_seconds

        if not (
            0
            <= departure_time_id
            < time_slots_count
        ):
            raise ValueError(
                f"Row {row_index}: departure time ID "
                f"{departure_time_id} outside valid "
                f"range [0, {time_slots_count - 1}]."
            )

        departure_features = time_features(
            pd.to_datetime(
                [departure],
                unit="s",
            ),
            freq="s",
        )[:, 0]

        trajectory_length = len(path)

        road_ids[
            row_index,
            :trajectory_length,
        ] = torch.from_numpy(path_array)

        time_ids[
            row_index,
            :trajectory_length,
        ] = int(departure_time_id)

        time_feature_tensor[
            row_index,
            :trajectory_length,
            :,
        ] = torch.from_numpy(
            departure_features
        ).unsqueeze(0).expand(
            trajectory_length,
            -1,
        )

        valid_mask[
            row_index,
            :trajectory_length,
        ] = True

        lengths[row_index] = trajectory_length
        departure_timestamps[row_index] = departure
        duration_seconds[row_index] = float(
            computed_duration
        )

        sample_ids.append(row["id"])

    return {
        "road_ids": road_ids,
        "time_ids": time_ids,
        "time_features": time_feature_tensor,
        "valid_mask": valid_mask,
        "lengths": lengths,
        "departure_timestamps": (
            departure_timestamps
        ),
        "duration_seconds": duration_seconds,
        "sample_ids": sample_ids,
        "metadata": {
            "cache_version": TTE_CACHE_VERSION,
            "task": "tte",
            "split": split,
            "source_file": str(
                trajectory_file.resolve()
            ),
            "source_sha256": sha256_file(
                trajectory_file
            ),
            "sample_count": sample_count,
            "seq_len": seq_len,
            "road_count": road_count,
            "pad_road_id": pad_road_id,
            "time_slots_count": (
                time_slots_count
            ),
            "empty_time_id": empty_time_id,
            "start_timestamp": start_timestamp,
            "interval_seconds": (
                interval_seconds
            ),
            "sample_rate": sample_rate,
            "seed": seed,
            "temporal_input": (
                "departure_only"
            ),
            "target": (
                "trip_duration_seconds"
            ),
        },
    }


def validate_tte_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    required_keys = {
        "road_ids",
        "time_ids",
        "time_features",
        "valid_mask",
        "lengths",
        "departure_timestamps",
        "duration_seconds",
        "sample_ids",
        "metadata",
    }

    if set(payload.keys()) != required_keys:
        raise ValueError(
            "Unexpected TTE payload keys: "
            f"{sorted(payload.keys())}"
        )

    road_ids = payload["road_ids"]
    time_ids = payload["time_ids"]
    time_features_tensor = payload[
        "time_features"
    ]
    valid_mask = payload["valid_mask"]
    lengths = payload["lengths"]
    durations = payload[
        "duration_seconds"
    ]
    metadata = payload["metadata"]

    sample_count = int(
        metadata["sample_count"]
    )
    seq_len = int(metadata["seq_len"])
    road_count = int(
        metadata["road_count"]
    )
    pad_road_id = int(
        metadata["pad_road_id"]
    )
    empty_time_id = int(
        metadata["empty_time_id"]
    )

    expected_2d = (
        sample_count,
        seq_len,
    )

    if tuple(road_ids.shape) != expected_2d:
        raise ValueError(
            "Unexpected road_ids shape."
        )

    if tuple(time_ids.shape) != expected_2d:
        raise ValueError(
            "Unexpected time_ids shape."
        )

    if tuple(valid_mask.shape) != expected_2d:
        raise ValueError(
            "Unexpected valid_mask shape."
        )

    if tuple(
        time_features_tensor.shape
    ) != (
        sample_count,
        seq_len,
        6,
    ):
        raise ValueError(
            "Unexpected time_features shape."
        )

    if tuple(lengths.shape) != (
        sample_count,
    ):
        raise ValueError(
            "Unexpected lengths shape."
        )

    if tuple(durations.shape) != (
        sample_count,
    ):
        raise ValueError(
            "Unexpected duration shape."
        )

    if not torch.isfinite(
        time_features_tensor
    ).all():
        raise ValueError(
            "Non-finite time features."
        )

    if not torch.isfinite(
        durations
    ).all():
        raise ValueError(
            "Non-finite durations."
        )

    if torch.any(durations <= 0):
        raise ValueError(
            "Non-positive durations detected."
        )

    mask_lengths = valid_mask.sum(
        dim=1
    ).to(torch.int64)

    if not torch.equal(
        mask_lengths,
        lengths,
    ):
        raise ValueError(
            "valid_mask does not match lengths."
        )

    position_index = torch.arange(
        seq_len
    ).unsqueeze(0)

    expected_mask = (
        position_index
        < lengths.unsqueeze(1)
    )

    if not torch.equal(
        valid_mask,
        expected_mask,
    ):
        raise ValueError(
            "valid_mask is not left-aligned."
        )

    if torch.any(
        road_ids[valid_mask] < 0
    ) or torch.any(
        road_ids[valid_mask] >= road_count
    ):
        raise ValueError(
            "Invalid real road IDs detected."
        )

    if torch.any(
        road_ids[~valid_mask]
        != pad_road_id
    ):
        raise ValueError(
            "PAD road positions are invalid."
        )

    if torch.any(
        time_ids[~valid_mask]
        != empty_time_id
    ):
        raise ValueError(
            "PAD time positions are invalid."
        )

    if torch.any(
        time_features_tensor[
            ~valid_mask
        ] != 0
    ):
        raise ValueError(
            "PAD time features are not zero."
        )

    first_time_ids = time_ids[:, 0]

    if torch.any(
        first_time_ids < 0
    ) or torch.any(
        first_time_ids
        >= metadata["time_slots_count"]
    ):
        raise ValueError(
            "Invalid departure time IDs."
        )

    for row_index in range(
        sample_count
    ):
        length = int(
            lengths[row_index].item()
        )

        if not torch.all(
            time_ids[
                row_index,
                :length,
            ]
            == time_ids[row_index, 0]
        ):
            raise ValueError(
                "Future time IDs detected in "
                f"row {row_index}."
            )

        first_features = (
            time_features_tensor[
                row_index,
                0,
                :,
            ]
        )

        if not torch.all(
            time_features_tensor[
                row_index,
                :length,
                :,
            ]
            == first_features
        ):
            raise ValueError(
                "Future per-road time features "
                f"detected in row {row_index}."
            )

    return {
        "samples": sample_count,
        "road_shape": tuple(
            road_ids.shape
        ),
        "time_shape": tuple(
            time_ids.shape
        ),
        "time_feature_shape": tuple(
            time_features_tensor.shape
        ),
        "duration_min": float(
            durations.min().item()
        ),
        "duration_median": float(
            durations.median().item()
        ),
        "duration_max": float(
            durations.max().item()
        ),
        "length_min": int(
            lengths.min().item()
        ),
        "length_max": int(
            lengths.max().item()
        ),
        "pad_road_id": pad_road_id,
        "empty_time_id": empty_time_id,
        "temporal_input": metadata[
            "temporal_input"
        ],
        "target": metadata["target"],
    }


class CachedTTEDataset(Dataset):
    def __init__(
        self,
        cache_path: str | Path,
    ):
        self.cache_path = Path(
            cache_path
        )

        payload = torch.load(
            self.cache_path,
            map_location="cpu",
            weights_only=True,
        )

        validate_tte_payload(payload)

        self.road_ids = payload[
            "road_ids"
        ]
        self.time_ids = payload[
            "time_ids"
        ]
        self.time_features = payload[
            "time_features"
        ]
        self.valid_mask = payload[
            "valid_mask"
        ]
        self.lengths = payload[
            "lengths"
        ]
        self.duration_seconds = payload[
            "duration_seconds"
        ]
        self.metadata = payload[
            "metadata"
        ]

    def __len__(self) -> int:
        return len(
            self.duration_seconds
        )

    def __getitem__(
        self,
        index: int,
    ):
        return (
            self.road_ids[index],
            self.time_ids[index],
            self.time_features[index],
            self.valid_mask[index],
            self.lengths[index],
            self.duration_seconds[index],
        )

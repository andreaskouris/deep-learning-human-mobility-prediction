import os

import torch
from torch.utils.data import Dataset


_REQUIRED_KEYS = {
    "traj_road_id_lists",
    "traj_time_id_lists",
    "traj_time_features_lists",
    "nexthop_labels",
}


class CachedNextHopDataset(Dataset):
    """Read-only next-hop dataset backed by a validated BIGCity cache."""

    def __init__(self, cache_path):
        super().__init__()

        self.cache_path = os.path.abspath(
            os.path.expanduser(cache_path)
        )

        if not os.path.isfile(self.cache_path):
            raise FileNotFoundError(
                f"Next-hop cache not found: {self.cache_path}"
            )

        cached_data = torch.load(
            self.cache_path,
            weights_only=True,
        )

        missing_keys = (
            _REQUIRED_KEYS
            - set(cached_data.keys())
        )

        if missing_keys:
            raise ValueError(
                "Next-hop cache is missing required keys: "
                + ", ".join(sorted(missing_keys))
            )

        self.traj_road_id_lists = cached_data[
            "traj_road_id_lists"
        ]

        self.traj_time_id_lists = cached_data[
            "traj_time_id_lists"
        ]

        self.traj_time_features_lists = cached_data[
            "traj_time_features_lists"
        ]

        self.nexthop_labels = cached_data[
            "nexthop_labels"
        ]

        self._validate()

    def _validate(self):
        sample_count = (
            self.traj_road_id_lists.shape[0]
        )

        if sample_count <= 0:
            raise ValueError(
                "Next-hop cache contains no samples."
            )

        tensors = {
            "traj_time_id_lists":
                self.traj_time_id_lists,
            "traj_time_features_lists":
                self.traj_time_features_lists,
            "nexthop_labels":
                self.nexthop_labels,
        }

        for name, tensor in tensors.items():
            if tensor.shape[0] != sample_count:
                raise ValueError(
                    f"{name} contains "
                    f"{tensor.shape[0]} samples, "
                    f"expected {sample_count}."
                )

        if self.traj_road_id_lists.ndim != 2:
            raise ValueError(
                "traj_road_id_lists must be 2-dimensional."
            )

        if self.traj_time_id_lists.ndim != 2:
            raise ValueError(
                "traj_time_id_lists must be 2-dimensional."
            )

        if (
            self.traj_road_id_lists.shape
            != self.traj_time_id_lists.shape
        ):
            raise ValueError(
                "Road and time ID tensors must have identical shapes."
            )

        if self.traj_time_features_lists.ndim != 3:
            raise ValueError(
                "traj_time_features_lists must be 3-dimensional."
            )

        if (
            self.traj_time_features_lists.shape[:2]
            != self.traj_road_id_lists.shape
        ):
            raise ValueError(
                "Time-feature sequence dimensions must match "
                "road sequence dimensions."
            )

        if self.traj_time_features_lists.shape[2] != 6:
            raise ValueError(
                "Expected exactly 6 temporal features per position."
            )

        if self.nexthop_labels.ndim != 1:
            raise ValueError(
                "nexthop_labels must be 1-dimensional."
            )

        if self.traj_road_id_lists.dtype != torch.int64:
            raise ValueError(
                "Road IDs must use torch.int64."
            )

        if self.traj_time_id_lists.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError(
                "Time IDs must use an integer dtype."
            )

        if (
            self.traj_time_features_lists.dtype
            != torch.float32
        ):
            raise ValueError(
                "Time features must use torch.float32."
            )

        if self.nexthop_labels.dtype != torch.int64:
            raise ValueError(
                "Next-hop labels must use torch.int64."
            )

        if not torch.isfinite(
            self.traj_time_features_lists
        ).all():
            raise ValueError(
                "Time features contain non-finite values."
            )

    def __len__(self):
        return self.nexthop_labels.shape[0]

    def __getitem__(self, index):
        return (
            self.traj_road_id_lists[index],
            self.traj_time_id_lists[index],
            self.traj_time_features_lists[index],
            self.nexthop_labels[index],
        )
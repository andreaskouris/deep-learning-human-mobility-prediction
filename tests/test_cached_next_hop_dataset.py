import os
import tempfile
import unittest

import torch

from data_provider.cached_next_hop_dataset import (
    CachedNextHopDataset,
)


class CachedNextHopDatasetTests(unittest.TestCase):

    def make_valid_cache(self, path):
        torch.save(
            {
                "traj_road_id_lists": torch.tensor(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                    ],
                    dtype=torch.int64,
                ),
                "traj_time_id_lists": torch.tensor(
                    [
                        [10, 11, 12],
                        [13, 14, 15],
                    ],
                    dtype=torch.int32,
                ),
                "traj_time_features_lists": torch.zeros(
                    2,
                    3,
                    6,
                    dtype=torch.float32,
                ),
                "nexthop_labels": torch.tensor(
                    [4, 7],
                    dtype=torch.int64,
                ),
            },
            path,
        )

    def test_loads_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = os.path.join(
                tmp_dir,
                "cache.pth",
            )

            self.make_valid_cache(cache_path)

            dataset = CachedNextHopDataset(
                cache_path
            )

            self.assertEqual(len(dataset), 2)

            (
                road_ids,
                time_ids,
                time_features,
                label,
            ) = dataset[0]

            self.assertEqual(
                tuple(road_ids.shape),
                (3,),
            )

            self.assertEqual(
                tuple(time_ids.shape),
                (3,),
            )

            self.assertEqual(
                tuple(time_features.shape),
                (3, 6),
            )

            self.assertEqual(
                label.item(),
                4,
            )

    def test_missing_cache_is_rejected(self):
        with self.assertRaises(
            FileNotFoundError
        ):
            CachedNextHopDataset(
                "this_file_does_not_exist.pth"
            )

    def test_missing_required_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = os.path.join(
                tmp_dir,
                "cache.pth",
            )

            torch.save(
                {
                    "traj_road_id_lists":
                        torch.zeros(
                            2,
                            3,
                            dtype=torch.int64,
                        ),
                },
                cache_path,
            )

            with self.assertRaises(ValueError):
                CachedNextHopDataset(
                    cache_path
                )

    def test_mismatched_sample_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = os.path.join(
                tmp_dir,
                "cache.pth",
            )

            self.make_valid_cache(cache_path)

            data = torch.load(
                cache_path,
                weights_only=True,
            )

            data["nexthop_labels"] = (
                data["nexthop_labels"][:1]
            )

            torch.save(
                data,
                cache_path,
            )

            with self.assertRaises(ValueError):
                CachedNextHopDataset(
                    cache_path
                )

    def test_nonfinite_time_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = os.path.join(
                tmp_dir,
                "cache.pth",
            )

            self.make_valid_cache(cache_path)

            data = torch.load(
                cache_path,
                weights_only=True,
            )

            data[
                "traj_time_features_lists"
            ][0, 0, 0] = float("nan")

            torch.save(
                data,
                cache_path,
            )

            with self.assertRaises(ValueError):
                CachedNextHopDataset(
                    cache_path
                )


if __name__ == "__main__":
    unittest.main()
"""Tests for the permanent BIGCity next-hop cache builder.

The tests use only small synthetic tensors inside temporary folders.
They never access or modify the real Xi'an dataset.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from scripts.build_next_hop_cache import (
    build_or_verify,
    sha256_file,
)


SAMPLES = 3
SEQ_LEN = 4
ROAD_COUNT = 10
TIME_SLOTS = 8


def make_valid_payload(
    *,
    feature_offset: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Create a small valid synthetic next-hop cache."""

    roads = torch.tensor(
        [
            [0, 1, 2, 2],
            [3, 4, 4, 4],
            [5, 6, 7, 7],
        ],
        dtype=torch.int64,
    )

    times = torch.tensor(
        [
            [0, 1, 2, 2],
            [1, 2, 3, 3],
            [2, 3, 4, 4],
        ],
        dtype=torch.int32,
    )

    time_features = (
        torch.arange(
            SAMPLES * SEQ_LEN * 3,
            dtype=torch.float32,
        ).reshape(
            SAMPLES,
            SEQ_LEN,
            3,
        )
        + feature_offset
    )

    labels = torch.tensor(
        [3, 5, 8],
        dtype=torch.int64,
    )

    return {
        "traj_road_id_lists": roads,
        "traj_time_id_lists": times,
        "traj_time_features_lists": time_features,
        "nexthop_labels": labels,
    }


class FakeDatasetFactory:
    """Synthetic replacement for DatasetNextHop."""

    def __init__(
        self,
        cache_owner: Any,
        *,
        feature_offset: float = 0.0,
        invalid_labels: bool = False,
    ) -> None:
        self.cache_owner = cache_owner
        self.feature_offset = feature_offset
        self.invalid_labels = invalid_labels
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1

        payload = make_valid_payload(
            feature_offset=self.feature_offset
        )

        if self.invalid_labels:
            payload["nexthop_labels"] = (
                payload["nexthop_labels"].to(
                    torch.float32
                )
            )

        cache_path = Path(
            self.cache_owner.cached_next_hop_dataset
        )

        torch.save(
            payload,
            cache_path,
        )

        return object()


class NextHopCacheBuilderTests(unittest.TestCase):
    """Tests that never touch the real Xi'an files."""

    def setUp(self) -> None:
        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temp_directory.name
        )

        self.trajectory_source = (
            self.root / "synthetic_trajectories.csv"
        )

        self.trajectory_source.write_text(
            "path,tlist\n",
            encoding="utf-8",
        )

        self.target_cache = (
            self.root / "cached_next_hop.pth"
        )

        self.temporary_cache = (
            self.target_cache.with_suffix(
                self.target_cache.suffix + ".tmp"
            )
        )

        self.original_configured_cache = (
            self.root / "original_configured_cache.pth"
        )

        self.cache_owner = SimpleNamespace(
            cached_next_hop_dataset=str(
                self.original_configured_cache
            )
        )

        self.prepare_calls = 0

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def prepare_trajectory_data(self) -> None:
        """Record preparation without reading real data."""

        self.prepare_calls += 1

    def build(
        self,
        *,
        factory: FakeDatasetFactory,
        verify_only: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run the builder with standard synthetic settings."""

        return build_or_verify(
            trajectory_source=self.trajectory_source,
            target_cache=self.target_cache,
            cache_owner=self.cache_owner,
            prepare_trajectory_data=(
                self.prepare_trajectory_data
            ),
            dataset_factory=factory,
            expected_samples=SAMPLES,
            seq_len=SEQ_LEN,
            road_count=ROAD_COUNT,
            time_slots=TIME_SLOTS,
            verify_only=verify_only,
            force=force,
        )

    def test_build_creates_valid_cache_atomically(
        self,
    ) -> None:
        factory = FakeDatasetFactory(
            self.cache_owner
        )

        report = self.build(
            factory=factory
        )

        self.assertTrue(
            self.target_cache.is_file()
        )

        self.assertFalse(
            self.temporary_cache.exists()
        )

        self.assertEqual(
            self.prepare_calls,
            1,
        )

        self.assertEqual(
            factory.calls,
            1,
        )

        self.assertEqual(
            self.cache_owner.cached_next_hop_dataset,
            str(self.original_configured_cache),
        )

        self.assertEqual(
            report["samples"],
            SAMPLES,
        )

        self.assertEqual(
            report["road_shape"],
            (SAMPLES, SEQ_LEN),
        )

        self.assertEqual(
            report["time_shape"],
            (SAMPLES, SEQ_LEN),
        )

        self.assertEqual(
            report["time_feature_shape"],
            (SAMPLES, SEQ_LEN, 3),
        )

        self.assertEqual(
            report["label_shape"],
            (SAMPLES,),
        )

    def test_verify_only_does_not_prepare_or_build(
        self,
    ) -> None:
        torch.save(
            make_valid_payload(),
            self.target_cache,
        )

        factory = FakeDatasetFactory(
            self.cache_owner
        )

        report = self.build(
            factory=factory,
            verify_only=True,
        )

        self.assertEqual(
            self.prepare_calls,
            0,
        )

        self.assertEqual(
            factory.calls,
            0,
        )

        self.assertEqual(
            report["samples"],
            SAMPLES,
        )

    def test_existing_cache_is_not_overwritten(
        self,
    ) -> None:
        torch.save(
            make_valid_payload(),
            self.target_cache,
        )

        original_hash = sha256_file(
            self.target_cache
        )

        factory = FakeDatasetFactory(
            self.cache_owner,
            feature_offset=100.0,
        )

        with self.assertRaises(
            FileExistsError
        ):
            self.build(
                factory=factory,
                force=False,
            )

        self.assertEqual(
            factory.calls,
            0,
        )

        self.assertEqual(
            self.prepare_calls,
            0,
        )

        self.assertEqual(
            sha256_file(self.target_cache),
            original_hash,
        )

    def test_force_rebuild_replaces_existing_cache(
        self,
    ) -> None:
        torch.save(
            make_valid_payload(),
            self.target_cache,
        )

        original_hash = sha256_file(
            self.target_cache
        )

        factory = FakeDatasetFactory(
            self.cache_owner,
            feature_offset=100.0,
        )

        report = self.build(
            factory=factory,
            force=True,
        )

        rebuilt_hash = sha256_file(
            self.target_cache
        )

        self.assertNotEqual(
            rebuilt_hash,
            original_hash,
        )

        self.assertEqual(
            report["sha256"],
            rebuilt_hash,
        )

        self.assertEqual(
            factory.calls,
            1,
        )

        self.assertEqual(
            self.prepare_calls,
            1,
        )

    def test_invalid_temporary_cache_is_rejected(
        self,
    ) -> None:
        torch.save(
            make_valid_payload(),
            self.target_cache,
        )

        original_hash = sha256_file(
            self.target_cache
        )

        factory = FakeDatasetFactory(
            self.cache_owner,
            invalid_labels=True,
        )

        with self.assertRaises(
            TypeError
        ):
            self.build(
                factory=factory,
                force=True,
            )

        self.assertTrue(
            self.target_cache.is_file()
        )

        self.assertEqual(
            sha256_file(self.target_cache),
            original_hash,
        )

        self.assertFalse(
            self.temporary_cache.exists()
        )

        self.assertEqual(
            self.cache_owner.cached_next_hop_dataset,
            str(self.original_configured_cache),
        )

    def test_out_of_range_road_id_is_rejected(
        self,
    ) -> None:
        payload = make_valid_payload()

        payload["traj_road_id_lists"][
            0,
            0,
        ] = ROAD_COUNT

        torch.save(
            payload,
            self.target_cache,
        )

        factory = FakeDatasetFactory(
            self.cache_owner
        )

        with self.assertRaises(
            ValueError
        ):
            self.build(
                factory=factory,
                verify_only=True,
            )

        self.assertEqual(
            factory.calls,
            0,
        )

        self.assertEqual(
            self.prepare_calls,
            0,
        )

    def test_missing_trajectory_source_is_rejected(
        self,
    ) -> None:
        self.trajectory_source.unlink()

        factory = FakeDatasetFactory(
            self.cache_owner
        )

        with self.assertRaises(
            FileNotFoundError
        ):
            self.build(
                factory=factory
            )

        self.assertEqual(
            factory.calls,
            0,
        )

        self.assertEqual(
            self.prepare_calls,
            0,
        )


if __name__ == "__main__":
    unittest.main()
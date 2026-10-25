"""Tests for the permanent structural-cache builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.build_structural_caches import (
    CacheExpectations,
    StructuralPaths,
    build_or_verify_structural_caches,
)


class FakeStructuralLoader:
    """Small synthetic replacement for BIGCity's file_loader."""

    def __init__(
        self,
        paths: StructuralPaths,
        *,
        mutate_dynamic: bool = False,
    ) -> None:
        self.paths = paths
        self.mutate_dynamic = mutate_dynamic
        self.relation_calls = 0
        self.static_calls = 0
        self.dynamic_features = None

    def load_road_relation_file(self) -> None:
        self.relation_calls += 1

        edges = torch.tensor(
            [
                [0, 1, 1],
                [1, 0, 1],
            ],
            dtype=torch.int64,
        )

        weights = torch.tensor(
            [1.0, 0.0, 2.0],
            dtype=torch.float32,
        )

        torch.save(
            {
                "edges": edges,
                "edge_weight": weights,
                "edge_cnt": 3,
            },
            self.paths.relation_cache,
        )

    def load_road_features_file(self) -> None:
        self.static_calls += 1

        static_features = torch.tensor(
            [
                [1, 2],
                [3, 4],
            ],
            dtype=torch.int64,
        )

        torch.save(
            {
                "static_features": static_features,
                "road_cnt": 2,
            },
            self.paths.static_cache,
        )

        dynamic_cache = torch.load(
            self.paths.dynamic_cache,
            map_location="cpu",
            weights_only=True,
        )

        if self.mutate_dynamic:
            changed_features = (
                dynamic_cache["dynamic_features"].clone()
            )

            changed_features[0, 0] = 0.25

            torch.save(
                {
                    "dynamic_features": changed_features,
                    "time_slots_cnt": dynamic_cache[
                        "time_slots_cnt"
                    ],
                },
                self.paths.dynamic_cache,
            )

            dynamic_cache = torch.load(
                self.paths.dynamic_cache,
                map_location="cpu",
                weights_only=True,
            )

        self.dynamic_features = dynamic_cache[
            "dynamic_features"
        ]


class StructuralCacheBuilderTests(unittest.TestCase):
    """Synthetic tests that never touch the real Xi'an dataset."""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)

        self.paths = StructuralPaths(
            relation_source=root / "synthetic.rel",
            static_source=root / "synthetic_static.csv",
            relation_cache=root / "cached_relation.pth",
            static_cache=root / "cached_static.pth",
            dynamic_cache=root / "cached_dynamic.pth",
        )

        self.paths.relation_source.write_text(
            "origin_id,destination_id\n",
            encoding="utf-8",
        )

        self.paths.static_source.write_text(
            "road_id,feature_1,feature_2\n",
            encoding="utf-8",
        )

        dynamic_features = torch.tensor(
            [
                [0.0, 0.5],
                [1.0, 0.25],
            ],
            dtype=torch.float32,
        )

        torch.save(
            {
                "dynamic_features": dynamic_features,
                "time_slots_cnt": 2,
            },
            self.paths.dynamic_cache,
        )

        self.expectations = CacheExpectations(
            road_count=2,
            edge_count=3,
            static_width=2,
            time_slots=2,
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_build_creates_and_validates_caches(self) -> None:
        loader = FakeStructuralLoader(self.paths)

        report = build_or_verify_structural_caches(
            paths=self.paths,
            loader=loader,
            expectations=self.expectations,
            verify_only=False,
            force=False,
        )

        self.assertTrue(
            self.paths.relation_cache.is_file()
        )

        self.assertTrue(
            self.paths.static_cache.is_file()
        )

        self.assertEqual(loader.relation_calls, 1)
        self.assertEqual(loader.static_calls, 1)

        self.assertEqual(
            report.relation["edges_shape"],
            (2, 3),
        )

        self.assertEqual(
            report.static["shape"],
            (2, 2),
        )

        self.assertEqual(
            report.dynamic["shape"],
            (2, 2),
        )

    def test_existing_caches_are_not_overwritten(self) -> None:
        first_loader = FakeStructuralLoader(self.paths)

        build_or_verify_structural_caches(
            paths=self.paths,
            loader=first_loader,
            expectations=self.expectations,
            verify_only=False,
            force=False,
        )

        second_loader = FakeStructuralLoader(self.paths)

        with self.assertRaises(FileExistsError):
            build_or_verify_structural_caches(
                paths=self.paths,
                loader=second_loader,
                expectations=self.expectations,
                verify_only=False,
                force=False,
            )

        self.assertEqual(
            second_loader.relation_calls,
            0,
        )

        self.assertEqual(
            second_loader.static_calls,
            0,
        )

    def test_verify_only_does_not_call_loader(self) -> None:
        build_loader = FakeStructuralLoader(self.paths)

        build_or_verify_structural_caches(
            paths=self.paths,
            loader=build_loader,
            expectations=self.expectations,
            verify_only=False,
            force=False,
        )

        verification_loader = FakeStructuralLoader(
            self.paths
        )

        report = build_or_verify_structural_caches(
            paths=self.paths,
            loader=verification_loader,
            expectations=self.expectations,
            verify_only=True,
            force=False,
        )

        self.assertEqual(
            verification_loader.relation_calls,
            0,
        )

        self.assertEqual(
            verification_loader.static_calls,
            0,
        )

        self.assertEqual(
            report.relation["edge_count"],
            3,
        )

        self.assertEqual(
            report.static["road_count"],
            2,
        )

    def test_force_rebuild_replaces_existing_caches(
        self,
    ) -> None:
        torch.save(
            {"invalid": True},
            self.paths.relation_cache,
        )

        torch.save(
            {"invalid": True},
            self.paths.static_cache,
        )

        loader = FakeStructuralLoader(self.paths)

        report = build_or_verify_structural_caches(
            paths=self.paths,
            loader=loader,
            expectations=self.expectations,
            verify_only=False,
            force=True,
        )

        self.assertEqual(
            report.relation["edge_count"],
            3,
        )

        self.assertEqual(
            report.static["road_count"],
            2,
        )

        self.assertEqual(loader.relation_calls, 1)
        self.assertEqual(loader.static_calls, 1)

    def test_dynamic_cache_modification_is_detected(
        self,
    ) -> None:
        loader = FakeStructuralLoader(
            self.paths,
            mutate_dynamic=True,
        )

        with self.assertRaises(RuntimeError):
            build_or_verify_structural_caches(
                paths=self.paths,
                loader=loader,
                expectations=self.expectations,
                verify_only=False,
                force=False,
            )


if __name__ == "__main__":
    unittest.main()
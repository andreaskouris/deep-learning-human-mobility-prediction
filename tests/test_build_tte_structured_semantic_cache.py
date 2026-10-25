"""Synthetic facts and integrity tests; no research data or model is loaded."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from scripts import build_tte_structured_semantic_cache as builder


class StructuredCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get("THESIS_TEST_STAGING"))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.road = self.root / "roads.csv"
        self.poi = self.root / "pois.csv"
        self.mapping = self.root / "mapping.json"
        # Deliberately shuffled IDs: output row i must represent road ID i.
        pd.DataFrame({"id": [2, 0, 1], "tunnel": [0, 1, 0], "bridge": [0, 0, 1],
                      "roundabout": [0, 1, 0], "oneway": [0, 0, 1]}).to_csv(self.road, index=False)
        pd.DataFrame({"id": list(range(1, 9)),
                      "TYPE": ["4080", "40A0", "4101", "8081", "808C", "808B", "A400", "other"]}
                     ).to_csv(self.poi, index=False)
        self.mapping.write_text(json.dumps({"0": [1, 2, 3, 4, 5, 6, 7, 7], "1": [8]}))
        self.output = self.root / "result.pth"
        self.metadata = self.root / "result.meta.json"
        self.kwargs = dict(road_file=self.road, poi_file=self.poi, mapping_file=self.mapping,
                           output=self.output, metadata_output=self.metadata, canonical=False)

    def build(self, **kwargs):
        return builder.build_or_verify(**(self.kwargs | kwargs))

    def test_source_facts_all_ten_columns_and_road_order(self):
        payload, metadata = builder.build_payload(self.road, self.poi, self.mapping)
        expected = torch.tensor([[1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                                 [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
        self.assertTrue(torch.equal(payload["semantic_features"], expected))
        self.assertEqual(payload["feature_names"], builder.FEATURE_NAMES)
        self.assertEqual(metadata["positive_road_counts"], dict.fromkeys(builder.FEATURE_NAMES, 1))
        self.assertEqual(metadata["roads_with_any_semantic"], 2)
        self.assertEqual(metadata["all_zero_roads"], 1)
        self.assertEqual(metadata["max_active_concepts_per_road"], 8)

    def test_reference_is_not_used_to_construct(self):
        with patch.object(builder.torch, "load", side_effect=AssertionError("Reference read")):
            payload, _ = builder.build_payload(self.road, self.poi, self.mapping)
        self.assertEqual(payload["semantic_features"].shape, (3, 10))

    def test_build_reference_and_verify_only_do_not_change_sources(self):
        originals = {p: p.read_bytes() for p in [self.road, self.poi, self.mapping]}
        self.build()
        reference, reference_meta = self.root / "reference.pth", self.root / "reference.json"
        reference.write_bytes(self.output.read_bytes())
        reference_meta.write_bytes(self.metadata.read_bytes())
        protected = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                     for p in [self.output, self.metadata, reference, reference_meta]}
        with patch.object(builder.torch, "save", side_effect=AssertionError("verify wrote")):
            report = self.build(verify_only=True, reference=reference, reference_metadata=reference_meta)
        self.assertTrue(report["reference_tensor_equal"])
        self.assertTrue(report["reference_metadata_equal"])
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})
        self.assertEqual(protected, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected})

    def test_existing_outputs_never_overwritten(self):
        for path in [self.output, self.metadata]:
            with self.subTest(path=path):
                path.write_bytes(b"keep")
                with self.assertRaises(FileExistsError):
                    self.build()
                self.assertEqual(path.read_bytes(), b"keep")
                path.unlink()

    def test_unknown_poi_reference_rejected(self):
        self.mapping.write_text('{"0": [999]}')
        with self.assertRaisesRegex(ValueError, "Unknown POI"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_invalid_road_ids_and_binary_attributes_rejected(self):
        original = pd.read_csv(self.road)
        for field, value in [("id", 0), ("tunnel", 2), ("bridge", float("nan"))]:
            with self.subTest(field=field):
                changed = original.copy()
                changed.loc[0, field] = value
                changed.to_csv(self.road, index=False)
                with self.assertRaises(ValueError):
                    self.build()

    def test_duplicate_poi_ids_rejected(self):
        poi = pd.read_csv(self.poi, dtype=str)
        poi.loc[1, "id"] = poi.loc[0, "id"]
        poi.to_csv(self.poi, index=False)
        with self.assertRaises(ValueError):
            self.build()

    def test_out_of_range_mapping_rejected(self):
        self.mapping.write_text('{"3": [1]}')
        with self.assertRaises(ValueError):
            self.build()

    def test_tensor_tampering_detected_beyond_support_counts(self):
        self.build()
        payload = torch.load(self.output, weights_only=True)
        payload["semantic_features"][[0, 2], 0] = torch.tensor([0., 1.])
        torch.save(payload, self.output)
        with self.assertRaisesRegex(ValueError, "semantic_features"):
            self.build(verify_only=True)

    def test_feature_order_and_dtype_tampering_rejected(self):
        payload, _ = builder.build_payload(self.road, self.poi, self.mapping)
        with self.assertRaises(ValueError):
            builder.validate_payload(payload | {"feature_names": list(reversed(builder.FEATURE_NAMES))})
        with self.assertRaises(ValueError):
            builder.validate_payload(payload | {"semantic_features": payload["semantic_features"].double()})

    def test_metadata_tampering_rejected(self):
        self.build()
        metadata = json.loads(self.metadata.read_text())
        metadata["source_sha256"]["road_xa.csv"] = "bad"
        self.metadata.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.build(verify_only=True)

    def test_reference_mismatch_prevents_publication(self):
        payload, _ = builder.build_payload(self.road, self.poi, self.mapping)
        payload["semantic_features"][0, 0] = 0
        reference = self.root / "wrong.pth"
        torch.save(payload, reference)
        with self.assertRaises(ValueError):
            self.build(reference=reference)
        self.assertFalse(self.output.exists())

    def test_source_output_and_hardlink_aliases_rejected(self):
        with self.assertRaises(ValueError):
            self.build(output=self.road)
        alias = self.root / "alias.pth"
        os.link(self.road, alias)
        with self.assertRaises(ValueError):
            self.build(output=alias)

    def test_canonical_profile_does_not_accept_synthetic_inputs(self):
        with self.assertRaises(ValueError):
            self.build(canonical=True)

    def test_missing_verify_outputs_do_not_create_files(self):
        before = set(self.root.iterdir())
        with self.assertRaises(FileNotFoundError):
            self.build(verify_only=True)
        self.assertEqual(before, set(self.root.iterdir()))

    def test_metadata_publication_failure_rolls_back_only_new_output(self):
        original_link = os.link

        def link(source, destination):
            if destination == self.metadata:
                raise OSError("synthetic publication failure")
            return original_link(source, destination)

        with patch.object(builder.os, "link", side_effect=link):
            with self.assertRaises(OSError):
                self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.metadata.exists())

    def test_concurrent_output_is_not_replaced(self):
        original_link = os.link

        def concurrent_link(source, destination):
            if destination == self.output:
                destination.write_bytes(b"concurrent owner")
            return original_link(source, destination)

        with patch.object(builder.os, "link", side_effect=concurrent_link):
            with self.assertRaises(FileExistsError):
                self.build()
        self.assertEqual(self.output.read_bytes(), b"concurrent owner")
        self.assertFalse(self.metadata.exists())


if __name__ == "__main__":
    unittest.main()

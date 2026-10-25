"""Synthetic FileLoader integration, isolation and metadata integrity tests."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from scripts import build_full_metadata as builder


class FullMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get("THESIS_TEST_STAGING"))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.inputs = builder.MetadataInputs(**{name: self.root / (name + (".pth" if "cache" in name else ".csv"))
                                                for name in builder.MetadataInputs.__dataclass_fields__})
        pd.DataFrame({"origin_id": [0, 1, 1], "destination_id": [1, 0, 1],
                      "geographical_weight": [1., 0.5, 2.]}).to_csv(self.inputs.relation_file, index=False)
        pd.DataFrame({"a": [0, 1], "b": [1, 0]}).to_csv(self.inputs.static_file, index=False)
        pd.DataFrame({"id": [0, 1, 2, 3], "usr_id": [11, 11, 22, 22]}).to_csv(
            self.inputs.trajectory_file, sep=";", index=False)
        torch.save({"edges": torch.tensor([[0, 1, 1], [1, 0, 1]]),
                    "edge_weight": torch.tensor([1., 0.5, 2.]), "edge_cnt": 3}, self.inputs.relation_cache)
        torch.save({"static_features": torch.tensor([[0, 1], [1, 0]]), "road_cnt": 2}, self.inputs.static_cache)
        torch.save({"dynamic_features": torch.tensor([[0., 0.5, 1.], [1., 0.5, 0.]]),
                    "time_slots_cnt": 3}, self.inputs.dynamic_cache)
        self.expected = dict(zip(builder.FIELDS, [3, 2, 3, 4, 2]))
        self.output = self.root / "metadata.json"
        self.reference = self.root / "reference.json"
        self.reference.write_bytes(builder.serialize_metadata(self.expected, "crlf"))
        self.kwargs = dict(inputs=self.inputs, output=self.output, reference=self.reference, canonical=False)

    def build(self, **kwargs):
        return builder.build_or_verify(**(self.kwargs | kwargs))

    def snapshot(self):
        return {p: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in [*self.inputs.paths().values(), self.reference]}

    def test_real_loader_computes_all_five_fields(self):
        before = self.snapshot()
        report = self.build()
        self.assertEqual(report["metadata"], self.expected)
        self.assertTrue(report["reference_dictionary_equal"])
        self.assertEqual(before, self.snapshot())
        self.assertEqual(json.loads(self.output.read_text()), self.expected)
        self.assertEqual(set(self.root.iterdir()), {*self.inputs.paths().values(), self.reference, self.output})

    def test_line_endings_reported_separately_from_dictionary_equality(self):
        lf = self.build()
        crlf_path = self.root / "crlf.json"
        crlf = self.build(output=crlf_path, line_endings="crlf")
        self.assertEqual(lf["metadata"], crlf["metadata"])
        self.assertFalse(lf["reference_file_bytes_equal"])
        self.assertTrue(crlf["reference_file_bytes_equal"])
        self.assertNotEqual(lf["file_sha256"], crlf["file_sha256"])

    def test_verify_only_is_read_only_and_reconstructs(self):
        self.build()
        before = self.snapshot() | {self.output: (self.output.read_bytes(), self.output.stat().st_mtime_ns)}
        names = set(self.root.iterdir())
        with patch.object(builder.Path, "write_bytes", side_effect=AssertionError("verify wrote")):
            report = self.build(verify_only=True)
        self.assertTrue(report["reference_dictionary_equal"])
        self.assertEqual(before, self.snapshot() | {self.output: (self.output.read_bytes(), self.output.stat().st_mtime_ns)})
        self.assertEqual(names, set(self.root.iterdir()))

    def test_existing_output_never_used_for_category_count(self):
        wrong = self.expected | {"traj_category_cnt": 999}
        self.output.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "independently reconstructed"):
            self.build(verify_only=True)
        self.assertEqual(json.loads(self.output.read_text()), wrong)

    def test_wrong_reference_rejected_before_publication(self):
        self.reference.write_text(json.dumps(self.expected | {"traj_category_cnt": 999}))
        with self.assertRaisesRegex(ValueError, "differs from reference"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_overwrite_and_alias_rejected(self):
        self.output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.output.read_bytes(), b"keep")
        with self.assertRaises(ValueError):
            self.build(output=self.reference)
        with self.assertRaises(ValueError):
            self.build(output=self.inputs.trajectory_file)

    def test_missing_support_cache_does_not_trigger_creation(self):
        self.inputs.dynamic_cache.unlink()
        before = set(self.root.iterdir())
        with patch.object(torch, "save", side_effect=AssertionError("cache creation")):
            with self.assertRaises(FileNotFoundError):
                self.build()
        self.assertEqual(before, set(self.root.iterdir()))

    def test_cache_disappearing_after_preflight_cannot_be_regenerated(self):
        original_compute = builder.compute_metadata

        def interrupted_compute(loader, inputs):
            inputs.relation_cache.unlink()
            return original_compute(loader, inputs)

        with patch.object(builder, "compute_metadata", side_effect=interrupted_compute):
            with self.assertRaisesRegex(RuntimeError, "attempted to write a support cache"):
                self.build()
        self.assertFalse(self.inputs.relation_cache.exists())
        self.assertFalse(self.output.exists())

    def test_tampered_structural_cache_rejected(self):
        data = torch.load(self.inputs.relation_cache, weights_only=True)
        data["edges"][0, 0] = 1
        torch.save(data, self.inputs.relation_cache)
        with self.assertRaisesRegex(ValueError, "Relation cache"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_dynamic_shape_and_finite_checks(self):
        for tensor in [torch.zeros(2, 2), torch.full((2, 3), float("nan"))]:
            with self.subTest(shape=tensor.shape):
                torch.save({"dynamic_features": tensor, "time_slots_cnt": 3}, self.inputs.dynamic_cache)
                with self.assertRaisesRegex(ValueError, "Dynamic cache"):
                    self.build()

    def test_missing_user_ids_rejected(self):
        pd.DataFrame({"usr_id": [1., float("nan")]}).to_csv(self.inputs.trajectory_file, sep=";", index=False)
        with self.assertRaisesRegex(ValueError, "usr_id"):
            self.build()

    def test_canonical_profile_rejects_synthetic_sources(self):
        with self.assertRaisesRegex(ValueError, "source hashes"):
            self.build(canonical=True)

    def test_singleton_paths_argv_and_numpy_state_are_restored(self):
        import importlib
        import sys
        import numpy as np
        original = sys.argv[:]
        try:
            sys.argv = [original[0]]
            module = importlib.import_module("data_provider.file_loader")
        finally:
            sys.argv = original
        singleton, trajectory = module.FileLoader._instance, module.global_vars.cur_traj_file
        state = np.random.get_state()
        self.build()
        self.assertIs(module.FileLoader._instance, singleton)
        self.assertEqual(module.global_vars.cur_traj_file, trajectory)
        self.assertEqual(sys.argv, original)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(state[1], after[1]))
        self.assertEqual(state[2:], after[2:])

    def test_metadata_schema_rejects_bool_extra_and_zero(self):
        for value in [self.expected | {"road_cnt": True}, self.expected | {"extra": 1},
                      self.expected | {"road_cnt": 0}]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    builder.validate_metadata(value)

    def test_verify_only_missing_output_creates_nothing(self):
        names = set(self.root.iterdir())
        with self.assertRaises(FileNotFoundError):
            self.build(verify_only=True)
        self.assertEqual(names, set(self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()

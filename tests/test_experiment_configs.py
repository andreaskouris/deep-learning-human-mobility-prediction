"""Small, CPU-only contract tests for experiment configuration safety."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import validate_experiment_configs as check

ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = check.read_json(ROOT / "reproducibility/external_artifacts.json")
        cls.index = check.artifact_index(cls.manifest)
        cls.parser = check.parser_from_source(ROOT / "config/args_config.py")

    def config(self, name="nh_baseline"):
        return check.read_json(ROOT / f"experiments/{name}.json")

    def test_all_six_configs_parse_against_actual_runner_declarations(self):
        result = check.validate_all(ROOT)
        self.assertEqual(len(result["experiments"]), 6)
        self.assertTrue(all(r["valid"] for r in result["experiments"]))
        self.assertFalse(result["training_or_generation_executed"])

    def test_ast_parser_does_not_execute_module_or_function_side_effects(self):
        source = """raise RuntimeError('module must never execute')
def get_args_parser():
    raise RuntimeError('function must never execute')
    parser.add_argument('--value', type=int, default=7)
    return parser
"""
        with patch.object(Path, "read_text", return_value=source):
            parser = check.parser_from_source(Path("synthetic_parser.py"))
            self.assertEqual(parser.parse_args(["--value", "11"]).value, 11)

    def test_dynamic_parser_values_are_rejected_without_execution(self):
        source = """def get_args_parser():
    parser.add_argument('--value', default=dangerous_function())
"""
        with patch.object(Path, "read_text", return_value=source):
            with self.assertRaises(ValueError):
                check.parser_from_source(Path("synthetic_parser.py"))

    def test_weight_decay_flag_typo_is_rejected(self):
        cli = self.config()["cli"]
        cli["--weight_decay"] = cli.pop("--weight")
        with self.assertRaisesRegex(ValueError, "Unknown runner argument"):
            check.parse_cli(self.parser, cli)

    def test_max_epochs_cannot_be_replaced_by_historical_best_epoch(self):
        config = self.config()
        config["cli"]["--train_epochs"] = config["historical"]["best_epoch"]["value"]
        with self.assertRaisesRegex(ValueError, "--train_epochs"):
            check.validate_config(config, self.parser, self.index)

    def test_cross_task_semantic_mode_is_rejected(self):
        config = self.config("tte_structured")
        config["semantic_mode"] = config["cli"]["--semantic_mode"] = "poi_tfidf"
        with self.assertRaisesRegex(ValueError, "Task/semantic-mode"):
            check.validate_config(config, self.parser, self.index)

    def test_missing_evidence_is_rejected(self):
        config = self.config()
        del config["provenance"]["cli"]["--seed"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            check.validate_config(config, self.parser, self.index)

    def test_saved_checkpoint_mismatch_is_rejected(self):
        config = self.config("tte_qwen")
        config["historical"]["checkpoint_metadata"]["configuration"]["seq_len"] = 32
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            check.validate_config(config, self.parser, self.index)

    def test_tte_does_not_promise_custom_log_path(self):
        config = self.config("tte_baseline")
        config["cli"]["--log_path"] = "{RUN_ROOT}/logs"
        with self.assertRaisesRegex(ValueError, "ignores"):
            check.validate_config(config, self.parser, self.index)

    def test_path_substitution_preserves_spaces_without_shell_interpretation(self):
        argv = check.argv_from_cli(
            {"--dataset_path": "{DATASET_ROOT}", "--use_gpu": True},
            {"DATASET_ROOT": "/data with spaces/$(literal)"},
        )
        self.assertEqual(argv, ["--dataset_path", "/data with spaces/$(literal)", "--use_gpu"])

    def test_missing_path_binding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "placeholder"):
            check.argv_from_cli({"--dataset_path": "{DATASET_ROOT}"}, {})

    def test_duplicate_json_key_is_rejected(self):
        with patch.object(Path, "read_text", return_value='{"--seed":2021,"--seed":42}'):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                check.read_json(Path("duplicate.json"))

    def test_qwen_dependency_closure_includes_preprocessing_inputs(self):
        nh = check.dependency_closure(["qwen_poi_road_cache"], self.index)
        self.assertTrue({"qwen_type_prototypes", "poi_tfidf_cache", "xa_poi",
                         "xa_geo", "xa_sem", "xa_road2poi", "qwen_model"} <= set(nh))
        tte = check.dependency_closure(["tte_qwen_cache"], self.index)
        self.assertTrue({"tte_structured_cache", "xa_road", "xa_poi",
                         "xa_road2poi", "qwen_model"} <= set(tte))

    def test_artifact_cycle_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["artifacts"][0]["depends_on"] = [manifest["artifacts"][0]["id"]]
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            check.artifact_index(manifest)

    def test_historical_finetuned_hashes_cannot_gate_new_runs(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["historical_finetuned_policy"]["hashes_are_acceptance_criteria"] = True
        with self.assertRaisesRegex(ValueError, "cannot gate"):
            check.artifact_index(manifest)

    def test_unknown_artifact_reference_is_rejected(self):
        config = self.config()
        config["required_runtime_artifacts"].append("unavailable_asset")
        with self.assertRaisesRegex(ValueError, "Unknown artifact"):
            check.validate_config(config, self.parser, self.index)

    def test_invalid_hash_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["artifacts"][0]["canonical_sha256"] = "1234"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            check.artifact_index(manifest)

    def test_runner_selection_contract_detects_accidental_tie_replacement(self):
        source = (ROOT / "train_tte.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "tte_validation_improved")
        comparison = next(n for n in ast.walk(function) if isinstance(n, ast.Compare))
        comparison.ops = [ast.LtE()]
        with patch.object(Path, "read_text", return_value=ast.unparse(tree)):
            with self.assertRaisesRegex(ValueError, "must be strict"):
                check.validate_runner_source(Path("mutated_runner.py"), "tte")

    def test_environment_groups_are_independent_and_exactly_pinned(self):
        result = check.validate_environments(ROOT)
        self.assertGreater(result["bigcity"]["pinned_packages"], 8)
        self.assertGreater(result["qwen"]["pinned_packages"], 5)
        self.assertNotEqual(check.read_pins(ROOT / "environments/bigcity.txt")["transformers"],
                            check.read_pins(ROOT / "environments/qwen.txt")["transformers"])
        self.assertNotIn("transformers", check.read_pins(ROOT / "environments/analysis.txt"))


if __name__ == "__main__":
    unittest.main()

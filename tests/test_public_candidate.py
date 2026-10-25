"""Candidate selection, integrity and dependency-boundary checks."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts import verify_public_candidate as verify
from scripts.run_selected_tests import run

ROOT=Path(os.environ.get('PUBLIC_CANDIDATE_ROOT',Path(__file__).resolve().parents[1]))


class PublicCandidateTests(unittest.TestCase):
    def test_exact_inventory_and_canonical_interfaces(self):
        result=verify.verify(ROOT)
        self.assertEqual(len(result['configurations']),6)
        self.assertEqual(result['full_runtime_status'],'SOURCE_INTEGRATION_AVAILABLE_MODEL_REPRODUCTION_NOT_TESTED')
        self.assertFalse(result['training_or_generation_executed'])

    def test_no_upstream_or_derived_payload_is_included(self):
        data=json.loads((ROOT/'reproducibility/file_selection.json').read_text(encoding='utf-8'))
        self.assertFalse(any(x['publication_category'] in {'UPSTREAM','ADAPTATION','DERIVED_NEW','POTENTIALLY_DERIVED_NEW'} for x in data['included']))
        self.assertFalse((ROOT/'models/bigcity4finetune.py').exists());self.assertFalse((ROOT/'config/args_config.py').exists())
        self.assertTrue((ROOT/'models/bigcity4tte.py').exists())
        self.assertTrue((ROOT/'models/tte_tokenization_adapter.py').exists())
        self.assertTrue((ROOT/'train_next_hop.py').exists());self.assertTrue((ROOT/'train_tte.py').exists())

    def test_unsafe_paths_are_rejected(self):
        for value in ('../outside.py','/root.py','C:/data.py','a\\b.py','a//b.py'):
            with self.subTest(path=value),self.assertRaises(ValueError):verify.safe_path(value)

    def test_unknown_local_import_is_not_silently_treated_as_a_package(self):
        with self.assertRaisesRegex(ValueError,'Unexpected missing'):
            verify.classify_import('models.unreviewed',{},set())
        self.assertEqual(verify.classify_import('models.bigcity4tte',{}, {'models.bigcity4tte'}),'WITHHELD_EXTERNAL')

    def test_docstring_normalization_preserves_executable_differences(self):
        self.assertEqual(verify.normalized_ast('def f():\n    "old"\n    return 2\n'),verify.normalized_ast('def f():\n    "new"\n    return 2\n'))
        self.assertNotEqual(verify.normalized_ast('def f():\n    return 2\n'),verify.normalized_ast('def f():\n    return 3\n'))

    def test_tampered_inventory_hash_is_rejected(self):
        data=json.loads((ROOT/'reproducibility/file_selection.json').read_text(encoding='utf-8'))
        record=next(x for x in data['included'] if x['path']!='reproducibility/file_selection.json');record['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'digest mismatch'):verify.check_inventory(ROOT,data)

    def test_git_metadata_is_separate_from_strict_source_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); data = b'VALUE = 1\n'
            (root/'module.py').write_bytes(data)
            (root/'.git/objects').mkdir(parents=True)
            (root/'.git/objects/private-metadata').write_bytes(b'git object')
            manifest = {'included': [{'path': 'module.py', 'sha256': hashlib.sha256(data).hexdigest().upper(),
                                      'publication_category': 'ORIGINAL_REVIEWED'}]}
            self.assertEqual(verify.check_inventory(root, manifest), 1)
            (root/'module.py').write_bytes(b'VALUE = 2\n')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                verify.check_inventory(root, manifest)
            (root/'module.py').write_bytes(data); (root/'unexpected.py').write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                verify.check_inventory(root, manifest)

    def test_runtime_fingerprints_match_distributed_source_bytes(self):
        spec = json.loads((ROOT/'reproducibility/integration_inputs.json').read_text(encoding='utf-8'))
        self.assertEqual(len(spec['thesis_runtime_files']), 73)
        for record in spec['thesis_runtime_files']:
            with self.subTest(path=record['path']):
                data = (ROOT/record['path']).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest().upper(), record['sha256'])
                self.assertNotIn(b'\r', data)

    def test_duplicate_inventory_paths_are_rejected(self):
        data=json.loads((ROOT/'reproducibility/file_selection.json').read_text(encoding='utf-8'));data['included'].append(deepcopy(data['included'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):verify.check_inventory(ROOT,data)

    def test_noncanonical_semantic_modes_are_outside_public_contract(self):
        parser=verify.interface_parser(ROOT/'reproducibility/runner_interface.json')
        self.assertEqual(parser.parse_args(['--semantic_mode','tte_qwen']).semantic_mode,'tte_qwen')
        self.assertNotIn('poi_bge_tfidf',parser._option_string_actions['--semantic_mode'].choices)

    def test_runtime_tests_require_explicit_integrated_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'fresh'
            with self.assertRaisesRegex(ValueError,'--runtime-root'):run(ROOT,'runtime',target)
            self.assertFalse(target.exists())

    def test_staging_cannot_overwrite_existing_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)
            with self.assertRaisesRegex(ValueError,'fresh staging'):run(ROOT,'tooling',target)


if __name__=='__main__':
    unittest.main()

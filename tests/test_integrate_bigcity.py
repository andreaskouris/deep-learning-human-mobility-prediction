"""Transactional source-only integration safety; no real model/data/network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from setup import integrate_bigcity as tool


class IntegrationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.candidate = self.root/'public'; self.upstream = self.root/'upstream'
        self.candidate.mkdir(); self.upstream.mkdir(); self.output = self.root/'adapted'
        self.spec = b'{}'; self.payload = {'model.py': b'VALUE = 2\n'}; self.backup = {'model.py': b'VALUE = 1\n'}
        self.mock = patch.object(tool, 'source_inputs', return_value=(self.spec, self.payload, self.backup))
        self.mock.start(); self.addCleanup(self.mock.stop)

    def run_tool(self, **kwargs):
        return tool.integrate(self.candidate, self.upstream, self.output, **kwargs)

    def test_dry_run_writes_nothing(self):
        self.assertEqual(self.run_tool(dry_run=True)['status'], 'DRY_RUN_VERIFIED')
        self.assertFalse(self.output.exists())

    def test_backup_and_adapted_source_are_separate(self):
        self.run_tool()
        self.assertEqual((self.output/'upstream-backup/model.py').read_bytes(), self.backup['model.py'])
        self.assertEqual((self.output/'runtime/model.py').read_bytes(), self.payload['model.py'])
        self.assertEqual(list(self.upstream.iterdir()), [])

    def test_repeat_is_read_only(self):
        self.run_tool(); before = {str(p): p.stat().st_mtime_ns for p in self.output.rglob('*')}
        self.assertEqual(self.run_tool()['status'], 'ALREADY_APPLIED_VERIFIED')
        self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in self.output.rglob('*')})

    def test_runtime_tampering_fails_without_repair(self):
        self.run_tool(); path = self.output/'runtime/model.py'; path.write_text('TAMPER = 3')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'): self.run_tool()
        self.assertEqual(path.read_text(), 'TAMPER = 3')

    def test_backup_tampering_fails(self):
        self.run_tool(); (self.output/'upstream-backup/model.py').write_text('BAD = 3')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'): self.run_tool()

    def test_existing_unrelated_output_is_preserved(self):
        self.output.mkdir(); (self.output/'keep').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'manifest missing'): self.run_tool()
        self.assertEqual((self.output/'keep').read_text(), 'keep')

    def test_extra_files_are_not_silently_accepted(self):
        self.run_tool(); (self.output/'extra').write_text('extra')
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'): self.run_tool()

    def test_dataset_sidecar_is_preserved_and_not_source_validated(self):
        self.run_tool(); dataset=self.output/'dataset';dataset.mkdir();data=dataset/'external.bin';data.write_bytes(b'external data')
        before=data.stat().st_mtime_ns
        self.assertEqual(self.run_tool()['status'], 'ALREADY_APPLIED_VERIFIED')
        self.assertEqual(data.read_bytes(),b'external data');self.assertEqual(before,data.stat().st_mtime_ns)

    def test_verification_requires_complete_output(self):
        with self.assertRaisesRegex(ValueError, 'existing complete output'): self.run_tool(verify_only=True)
        self.assertFalse(self.output.exists())

    def test_failure_before_publication_leaves_no_partial_install(self):
        with patch.object(tool, 'verify_output', side_effect=OSError('injected verification failure')):
            with self.assertRaisesRegex(OSError, 'injected'): self.run_tool()
        self.assertFalse(self.output.exists()); self.assertEqual(list(self.root.glob('adapted.partial-*')), [])

    def test_source_overlap_is_rejected(self):
        self.output = self.upstream/'output'
        with self.assertRaisesRegex(ValueError, 'separate'): self.run_tool()

    def test_links_are_rejected(self):
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Symlink'): self.run_tool()

    def test_unsafe_relative_paths_are_rejected(self):
        for path in ('../escape.py', '/escape.py', 'C:/escape.py', 'a\\b.py'):
            with self.subTest(path=path), self.assertRaises(ValueError): tool.relative(path)

    def test_unsupported_revision_is_rejected_before_source_changes(self):
        self.mock.stop()
        (self.candidate/'reproducibility').mkdir(); (self.upstream/'.git').mkdir()
        (self.candidate/'reproducibility/integration_inputs.json').write_text(json.dumps({'upstream': {'commit':tool.PIN,'tree':tool.TREE}}))
        def fake_git(root,*args):
            return str(self.upstream) if args[-1]=='--show-toplevel' else str(self.upstream/'.git') if args[-1]=='--absolute-git-dir' else '0'*40
        with patch.object(tool, 'git', side_effect=fake_git), self.assertRaisesRegex(ValueError, 'Unsupported BIGCity revision'):
            self.run_tool()
        self.assertFalse(self.output.exists())

    def test_missing_or_mismatched_preimage_is_rejected(self):
        self.mock.stop(); (self.candidate/'reproducibility').mkdir(); (self.upstream/'.git').mkdir()
        records=[{'path':p,'sha256':'0'*64,'adaptation':True} for p in sorted(tool.TRANSFORMS)]
        spec={'upstream':{'commit':tool.PIN,'tree':tool.TREE},'required_upstream':records}
        (self.candidate/'reproducibility/integration_inputs.json').write_text(json.dumps(spec))
        def fake_git(root,*args):
            return str(self.upstream) if args[-1]=='--show-toplevel' else str(self.upstream/'.git') if args[-1]=='--absolute-git-dir' else tool.PIN if args[-1]=='HEAD^{commit}' else tool.TREE
        with patch.object(tool,'git',side_effect=fake_git):
            with self.assertRaisesRegex(ValueError, 'Missing upstream source'): self.run_tool()
            first=self.upstream/records[0]['path'];first.parent.mkdir(parents=True);first.write_text('CHANGED = 1')
            with self.assertRaisesRegex(ValueError, 'Upstream source mismatch'): self.run_tool()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()

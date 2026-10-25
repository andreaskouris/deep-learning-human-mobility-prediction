"""Synthetic path, digest and reproduction-profile contracts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from scripts import portable_runtime_support as support


class ProfileTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.checkpoint=self.root/'new_checkpoint.pth';self.checkpoint.write_bytes(b'synthetic-checkpoint')
        self.templates={'baseline': {'semantic_mode':'none','checkpoint':Path('historical.pth'),'checkpoint_sha256':'A'*64,'best_epoch':16}}
        self.path=self.root/'checkpoints.json'
        self.manifest={'schema_version':1,'task':'tte','experiments':{'baseline':{'checkpoint':'new_checkpoint.pth','sha256':support.sha256_file(self.checkpoint),'semantic_mode':'none','selection_metric':'mae_seconds','validation_selected':True}}}
        self.path.write_text(json.dumps(self.manifest))
        self.options=SimpleNamespace(profile='fresh',checkpoints_manifest=self.path)

    def test_fresh_checkpoint_gets_own_hash_and_resolved_path(self):
        result=support.bind_checkpoints(self.options,self.templates,'tte')['baseline']
        self.assertEqual(result['checkpoint'],self.checkpoint)
        self.assertNotEqual(result['checkpoint_sha256'],self.templates['baseline']['checkpoint_sha256'])
        support.require_hash(result['checkpoint'],result['checkpoint_sha256'])
        self.assertEqual(self.templates['baseline']['checkpoint'],Path('historical.pth'))

    def test_historical_checkpoint_root_can_move_without_changing_hash(self):
        self.options.profile='historical';self.options.checkpoints_manifest=None;self.options.checkpoint_root=self.root/'archive'
        self.templates['baseline']['checkpoint']=Path('old/final_results/tte/01_baseline/checkpoint/best.pth')
        result=support.bind_checkpoints(self.options,self.templates,'tte')['baseline']
        self.assertEqual(result['checkpoint'],self.root/'archive/tte/01_baseline/checkpoint/best.pth')
        self.assertEqual(result['checkpoint_sha256'],'A'*64)

    def test_fresh_requires_manifest(self):
        self.options.checkpoints_manifest=None
        with self.assertRaisesRegex(ValueError,'requires'):
            support.bind_checkpoints(self.options,self.templates,'tte')

    def test_historical_preserves_hash(self):
        self.options.profile='historical';self.options.checkpoints_manifest=None
        self.assertEqual(support.bind_checkpoints(self.options,self.templates,'tte'),self.templates)
        with self.assertRaisesRegex(ValueError,'SHA256'):
            support.require_hash(self.checkpoint,self.templates['baseline']['checkpoint_sha256'])

    def test_historical_manifest_cannot_override_hashes(self):
        self.options.profile='historical'
        with self.assertRaisesRegex(ValueError,'fixed canonical'):
            support.bind_checkpoints(self.options,self.templates,'tte')

    def test_fresh_selection_and_mode_are_strict(self):
        for key,value in [('semantic_mode','tte_qwen'),('selection_metric','test_mae'),('validation_selected',False),('sha256','bad')]:
            with self.subTest(key=key):
                manifest=deepcopy(self.manifest);manifest['experiments']['baseline'][key]=value;self.path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):support.bind_checkpoints(self.options,self.templates,'tte')

    def test_unknown_experiment_rejected(self):
        self.manifest['experiments']['legacy']=self.manifest['experiments']['baseline'];self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'every canonical'):support.bind_checkpoints(self.options,self.templates,'tte')

    def test_checkpoint_metadata_selection_invariants(self):
        checkpoint={'configuration':dict(support.CANONICAL_COMMON,selection_metric='accuracy'),'epoch':2,'validation_metrics':{'accuracy':0.5}}
        support.validate_checkpoint_metadata(checkpoint,'none','next_hop')
        checkpoint['configuration']['seed']=42
        with self.assertRaisesRegex(ValueError,'seed'):support.validate_checkpoint_metadata(checkpoint,'none','next_hop')

    def test_tte_normalization_provenance(self):
        config=dict(support.CANONICAL_COMMON,selection_metric='mae_seconds',task='tte',semantic_mode='tte_structured',static_ablation='none',temporal_input='departure_only',prediction_clipping=False,loss='SmoothL1(beta=1.0)',target='trip_duration_seconds')
        checkpoint={'configuration':config,'epoch':3,'validation_metrics':{'mae_seconds':2.0},'target_normalization':{'source':'training_split_only'}}
        support.validate_checkpoint_metadata(checkpoint,'tte_structured','tte')
        checkpoint['target_normalization']['source']='test'
        with self.assertRaisesRegex(ValueError,'training split'):support.validate_checkpoint_metadata(checkpoint,'tte_structured','tte')

    def test_output_requires_empty_directory(self):
        out=self.root/'outputs';out.mkdir();(out/'existing.csv').write_text('keep')
        with self.assertRaises(FileExistsError):support.prepare_output(out)
        self.assertEqual((out/'existing.csv').read_text(),'keep')

    def test_output_cannot_contain_protected_source(self):
        with self.assertRaisesRegex(ValueError,'overlap'):support.prepare_output(self.root,overwrite=True,protected=[self.checkpoint])

    def test_analysis_roots_and_nested_model_inputs(self):
        ns={'ROOT':Path('final_analysis/next_hop'),'INPUT_DIR':Path('final_analysis/next_hop/per_sample'),'OUTPUT_DIR':Path('final_analysis/next_hop/statistics'),'MODELS':{'nh1':{'path':Path('final_analysis/next_hop/per_sample/a.csv')}}}
        analysis=self.root/'analysis';output=self.root/'stats'
        support.analysis_cli(ns,'next_hop',['--analysis-root',str(analysis),'--output-dir',str(output),'--profile','fresh'])
        self.assertEqual(ns['INPUT_DIR'],analysis/'per_sample');self.assertEqual(ns['MODELS']['nh1']['path'],analysis/'per_sample/a.csv')
        self.assertEqual(ns['OUTPUT_DIR'],output);self.assertFalse(ns['HISTORICAL'])

    def test_stratified_dataset_root_is_explicit(self):
        project=Path(support.__file__).resolve().parents[1];old=project/'final_analysis/next_hop';data=project.parent/'dataset/xa'
        ns={'ROOT':project,'DATA_ROOT':data,'MASTER_PATH':old/'per_sample/master.csv','TEST_CACHE':data/'test.pth','OUTPUT_DIR':old/'stratified'}
        dataset=self.root/'sources';analysis=self.root/'results'
        support.analysis_cli(ns,'next_hop',['--analysis-root',str(analysis),'--dataset-path',str(dataset)])
        self.assertEqual(ns['TEST_CACHE'],dataset/'xa/test.pth');self.assertEqual(ns['MASTER_PATH'],analysis/'per_sample/master.csv')


if __name__=='__main__':unittest.main()

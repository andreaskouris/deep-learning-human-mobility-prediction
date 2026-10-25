"""Portable runtime interfaces, tested only on tiny synthetic data."""
import ast
import contextlib
from copy import deepcopy
import importlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

RUNTIME=Path(os.environ['PORTABLE_RUNTIME']).resolve() if os.environ.get('PORTABLE_RUNTIME') else None
SCRIPTS=[
 'export_final_next_hop_per_sample','export_final_tte_per_sample','build_tte_qwen_semantic_cache',
 'analyze_final_next_hop_statistics','analyze_final_next_hop_stratified',
 'analyze_final_tte_statistics','analyze_final_tte_stratified',
 'plot_final_next_hop_figures_5_1_5_2','plot_final_next_hop_figures_5_3_5_4','plot_final_tte_figures']


@unittest.skipUnless(RUNTIME,'Set PORTABLE_RUNTIME to an explicit local-only runtime')
class PortableInterfaces(unittest.TestCase):
    def module(self,name):
        loaders = {'export_final_next_hop_per_sample': lambda: importlib.import_module('scripts.export_final_next_hop_per_sample'),
'export_final_tte_per_sample': lambda: importlib.import_module('scripts.export_final_tte_per_sample'),
'build_tte_qwen_semantic_cache': lambda: importlib.import_module('scripts.build_tte_qwen_semantic_cache'),
'analyze_final_next_hop_statistics': lambda: importlib.import_module('scripts.analyze_final_next_hop_statistics'),
'analyze_final_next_hop_stratified': lambda: importlib.import_module('scripts.analyze_final_next_hop_stratified'),
'analyze_final_tte_statistics': lambda: importlib.import_module('scripts.analyze_final_tte_statistics'),
'analyze_final_tte_stratified': lambda: importlib.import_module('scripts.analyze_final_tte_stratified'),
'plot_final_next_hop_figures_5_1_5_2': lambda: importlib.import_module('scripts.plot_final_next_hop_figures_5_1_5_2'),
'plot_final_next_hop_figures_5_3_5_4': lambda: importlib.import_module('scripts.plot_final_next_hop_figures_5_3_5_4'),
'plot_final_tte_figures': lambda: importlib.import_module('scripts.plot_final_tte_figures'),
'verify_tte_dataset': lambda: importlib.import_module('scripts.verify_tte_dataset'),
'build_tte_structured_semantic_cache': lambda: importlib.import_module('scripts.build_tte_structured_semantic_cache')}
        module=loaders[name]()
        self.assertTrue(Path(module.__file__).resolve().is_relative_to(RUNTIME))
        return module

    def test_imports_never_read_data_or_change_argv(self):
        original=list(sys.argv)
        with patch('pandas.read_csv',side_effect=AssertionError('import-time data read')),patch('torch.load',side_effect=AssertionError('import-time tensor read')),patch.object(Path,'mkdir',side_effect=AssertionError('import-time output creation')):
            for name in SCRIPTS:
                with self.subTest(script=name):importlib.reload(self.module(name))
        self.assertEqual(sys.argv,original)

    def test_all_cli_help_is_side_effect_free(self):
        with patch('pandas.read_csv',side_effect=AssertionError('CLI data read')),patch('torch.load',side_effect=AssertionError('CLI tensor read')),patch.object(Path,'mkdir',side_effect=AssertionError('CLI output creation')),contextlib.redirect_stdout(io.StringIO()):
            for name in SCRIPTS:
                with self.subTest(script=name):
                    with self.assertRaises(SystemExit) as result:
                        self.module(name).main(['--help'])
                    self.assertEqual(result.exception.code,0)

    def test_canonical_json_path_resolver_all_six(self):
        from scripts.portable_runtime_support import resolve_experiment_command
        for config_file in sorted((RUNTIME/'experiments').glob('*.json')):
            command=resolve_experiment_command(config_file,runtime_root=RUNTIME,dataset_root=RUNTIME/'external sources',pretrained_checkpoint=RUNTIME/'external/pretrain.pth',gpt2_path=RUNTIME/'external/gpt2',run_root=RUNTIME/'fresh runs')
            self.assertEqual(command[command.index('--seed')+1],'2021')
            self.assertEqual(command[command.index('--learning_rate')+1],'0.0002')
            self.assertEqual(command[command.index('--batch_size')+1],'32')
            self.assertNotIn('{',' '.join(command));self.assertIn('--gpt2_path',command);self.assertIn('--log_path',command)

    def test_real_parser_accepts_gpt2_and_log_paths(self):
        from scripts.validate_experiment_configs import parser_from_source
        parser=parser_from_source(RUNTIME/'config/args_config.py')
        options=parser.parse_args(['--gpt2_path','assets/gpt2','--log_path','runs/logs'])
        self.assertEqual(options.gpt2_path,'assets/gpt2');self.assertEqual(options.log_path,'runs/logs')

    def test_tte_builder_exclusive_and_verify_only(self):
        module=self.module('verify_tte_dataset')
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);source=root/'trajectory.csv';output=root/'cache.pth'
            pd.DataFrame({'id':[1,2],'path':['[0,1]','[1,2]'],'tlist':['[1538352000,1538352060]','[1538353800,1538353920]'],'duration':[60,120]}).to_csv(source,sep=';',index=False)
            args=['--trajectory_file',str(source),'--split','test','--output',str(output),'--road_count','3']
            module.main(args);before=output.read_bytes()
            with patch.object(module,'build_tte_payload',side_effect=AssertionError('Existing file should be rejected before reconstruction')):
                with self.assertRaises(FileExistsError):module.main(args)
            self.assertEqual(output.read_bytes(),before)
            module.main(args+['--verify-only']);self.assertEqual(output.read_bytes(),before)
            payload=torch.load(output,weights_only=True);payload['duration_seconds'][0]+=1;torch.save(payload,output)
            with self.assertRaisesRegex(ValueError,'content differs'):module.main(args+['--verify-only'])

    def test_tte_builder_rejects_input_alias(self):
        module=self.module('verify_tte_dataset')
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.csv';source.write_text('keep')
            with self.assertRaisesRegex(ValueError,'aliases'):
                module.main(['--trajectory_file',str(source),'--split','test','--output',str(source)])
            self.assertEqual(source.read_text(),'keep')

    def test_tte_qwen_source_only_uses_canonical_tensor_gate(self):
        module=self.module('build_tte_qwen_semantic_cache')
        builder=self.module('build_tte_structured_semantic_cache')
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);cache=root/'structured.pth';metadata=root/'meta.json'
            features=torch.zeros((3,10),dtype=torch.float32);features[0,0]=1;features[1,1]=1
            payload={'semantic_features':features,'feature_names':builder.FEATURE_NAMES,'road_count':3,'semantic_dim':10,'version':builder.VERSION}
            torch.save(payload,cache);metadata.write_text(json.dumps({'source_sha256':builder.CANONICAL_SOURCE_SHA256}))
            args=['--structured-cache',str(cache),'--structured-metadata',str(metadata),'--output-cache',str(root/'output.pth'),'--output-metadata',str(root/'output.json'),'--model-path',str(root/'missing-model'),'--hf-cache',str(root/'hf'),'--profile','fresh','--verify-source-only']
            validate=builder.validate_payload
            with patch.object(builder,'validate_payload',side_effect=lambda p,canonical:validate(p,canonical=False)) as canonical_gate,patch.multiple(module,EXPECTED_ROAD_COUNT=3,EXPECTED_COUNTS=[1,1,0,0,0,0,0,0,0,0],EXPECTED_ACTIVE_ROADS=2,EXPECTED_ZERO_ROADS=1):
                result=module.main(args)
                self.assertTrue(result['structured_tensor_verified']);self.assertFalse(result['qwen_loaded'])
                self.assertTrue(canonical_gate.call_args.kwargs['canonical'])
            self.assertFalse((root/'output.pth').exists());self.assertFalse((root/'output.json').exists())
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):module.main([a if a!='fresh' else 'historical' for a in args])

    def test_qwen_canonical_hash_check_is_not_optional_in_fresh(self):
        module=self.module('build_tte_qwen_semantic_cache');builder=self.module('build_tte_structured_semantic_cache')
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);cache=root/'cache.pth';torch.save({'semantic_features':torch.zeros(1,10),'feature_names':builder.FEATURE_NAMES,'road_count':1,'semantic_dim':10,'version':builder.VERSION},cache)
            metadata=root/'meta.json';metadata.write_text('{}')
            args=['--structured-cache',str(cache),'--structured-metadata',str(metadata),'--output-cache',str(root/'out.pth'),'--output-metadata',str(root/'out.json'),'--model-path',str(root/'model'),'--hf-cache',str(root/'hf'),'--profile','fresh','--verify-source-only']
            with self.assertRaisesRegex(ValueError,'Canonical XA'):module.main(args)

    def test_tte_historical_metric_gate_and_fresh_computation(self):
        module=self.module('export_final_tte_per_sample')
        values={'mae_seconds':1.,'rmse_seconds':2.,'mape_percent':3.}
        with patch.object(module,'HISTORICAL',True):
            with self.assertRaisesRegex(RuntimeError,'reproduction failed'):module.verify_metrics('baseline',values)
        with patch.object(module,'HISTORICAL',False):module.verify_metrics('baseline',values)
        result=module.metric_block(torch.tensor([2.,5.]).numpy(),torch.tensor([1.,3.]).numpy())
        self.assertAlmostEqual(result['mae_seconds'],1.5)

    def test_tte_summary_epoch_comes_from_loaded_checkpoint(self):
        module=self.module('export_final_tte_per_sample')
        from scripts.portable_runtime_support import CANONICAL_COMMON
        configuration=dict(CANONICAL_COMMON,selection_metric='mae_seconds',task='tte',semantic_mode='none',static_ablation='none',temporal_input='departure_only',prediction_clipping=False,loss='SmoothL1(beta=1.0)',target='trip_duration_seconds')
        checkpoint={'configuration':configuration,'epoch':3,'validation_metrics':{'mae_seconds':2.0},'target_normalization':{'source':'training_split_only','mean_seconds':485.50155132816224,'std_seconds':350.44401804624164}}
        epochs={}
        class BeforeModelConstruction(Exception):pass
        with patch.object(module,'args',SimpleNamespace(),create=True),patch.object(module,'CHECKPOINT_EPOCHS',epochs),patch.object(module,'BigCity4TTE',side_effect=BeforeModelConstruction,create=True),patch.object(torch,'load',return_value=checkpoint),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BeforeModelConstruction):
                module.predict_checkpoint(checkpoint_path='synthetic.pth',semantic_mode='none',payload={},device=torch.device('cpu'))
        self.assertEqual(epochs,{'none':3})
        tree=ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig'))
        summary=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='summary' for t in n.targets))
        self.assertNotIn('EXPECTED_HASHES',ast.unparse(summary))

    def test_gpt2_loads_are_explicit_and_local_only(self):
        for name,expected in [('models/backbone.py',2),('models/bigcity4finetune.py',1)]:
            tree=ast.parse((RUNTIME/name).read_text(encoding='utf-8-sig'))
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='from_pretrained' and any(isinstance(a,ast.Attribute) and a.attr=='gpt2_path' for a in n.args)]
            self.assertEqual(len(calls),expected)
            self.assertTrue(all(any(k.arg=='local_files_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in calls))

    def test_tte_runner_uses_configured_log_root(self):
        tree=ast.parse((RUNTIME/'train_tte.py').read_text())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='log_dir' for t in n.targets))
        self.assertEqual(ast.unparse(assignment.value.args[0]),'args.log_path')


if __name__=='__main__':unittest.main()

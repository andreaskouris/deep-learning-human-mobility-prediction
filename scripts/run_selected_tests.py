"""Run selected synthetic tests with explicit, isolated output and source roots."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.verify_public_candidate import normalized_ast

GROUPS = {
    'tooling': ['test_portable_runtime_support.py', 'test_public_candidate.py', 'test_integrate_bigcity.py'],
    'science': ['test_cached_next_hop_dataset.py', 'test_next_hop_metrics.py', 'test_tte_metrics.py',
                'test_build_tte_structured_semantic_cache.py', 'test_build_structural_caches.py', 'test_build_next_hop_cache.py', 'test_build_dynamic_cache.py', 'test_dynamic_preprocessing.py'],
    'runtime': ['test_file_loader_dynamic_modes.py',
                'test_build_full_metadata.py', 'test_experiment_configs.py', 'test_train_next_hop.py',
                'test_canonical_runtime_regressions.py', 'test_portable_runtime_interfaces.py']
    ,'reference': ['test_revised_implementation.py', 'test_tte_tokenizer_hook.py'],
}

CHILD = r'''
import contextlib,io,json,os,pathlib,socket,sys,tempfile,unittest
root=pathlib.Path(sys.argv[1]).resolve();temporary=pathlib.Path(sys.argv[2]).resolve();group=sys.argv[3]
patterns=json.loads(sys.argv[4]);sys.path.insert(0,str(root));sys.argv=['synthetic-only']
temporary.mkdir();tempfile.tempdir=str(temporary);os.environ['THESIS_TEST_STAGING']=str(temporary)
def blocked(*a,**k):raise AssertionError('Training, real model inference, GPU and network are forbidden')
socket.socket.connect=blocked;socket.create_connection=blocked
if group not in ('runtime','reference'):
    import importlib.abc
    class PublicBoundary(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            denied={'train_next_hop','train_tte','data_provider.file_loader','data_provider.data_loader','data_provider.original_dynamic_preprocessing','utils.timefeatures'}
            if fullname in denied or fullname=='config' or fullname.startswith('config.') or fullname=='models' or fullname.startswith('models.'):
                raise ImportError('Runtime dependency requires the explicitly integrated BIGCity source: '+fullname)
    sys.meta_path.insert(0,PublicBoundary())
if group!='tooling':
    import torch
    torch.set_num_threads(1);torch.set_default_device('cpu');torch.cuda.is_available=lambda:False
    torch.cuda._lazy_init=blocked;torch.cuda.init=blocked;torch.Tensor.cuda=blocked;torch.nn.Module.cuda=blocked
    real_load=torch.load
    def synthetic_load(path,*a,**k):
        if isinstance(path,(str,os.PathLike)):
            assert pathlib.Path(path).resolve().is_relative_to(temporary),'Non-synthetic artifact read blocked'
        return real_load(path,*a,**k)
    torch.load=synthetic_load
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):cls.step=blocked
if group in ('runtime','reference'):
    import train_next_hop,train_tte
    from models.bigcity4finetune import BigCity4FineTune
    from models.bigcity4tte import BigCity4TTE
    from models.backbone import Backbone
    from models.bigcity import BigCity
    for module in (train_next_hop,train_tte):
        for name in ('main','run_experiment','train_one_epoch','evaluate'):
            if hasattr(module,name):setattr(module,name,blocked)
    for cls,name in ((BigCity4FineTune,'forward'),(BigCity4TTE,'forward_tte'),(Backbone,'forward'),(BigCity,'forward')):
        setattr(cls,name,blocked)
    assert pathlib.Path(train_next_hop.__file__).resolve().is_relative_to(root)
    assert pathlib.Path(train_tte.__file__).resolve().is_relative_to(root)
suite=unittest.TestSuite()
for pattern in patterns:suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'),pattern=pattern))
stream=io.StringIO()
with contextlib.redirect_stdout(io.StringIO()):result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
print('__RESULT__'+json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'skipped':len(result.skipped),'successful':result.wasSuccessful(),'details':stream.getvalue()}))
'''


def prepare_overlay(candidate, runtime, output):
    manifest = json.loads((runtime.parent / 'integration_manifest.json').read_text())
    expected = {r['path'][8:]: r['sha256'] for r in manifest['files'] if r['path'].startswith('runtime/')}
    actual = {p.relative_to(runtime).as_posix() for p in runtime.rglob('*') if p.is_file()}
    if actual != set(expected):
        raise ValueError('Explicit integrated runtime inventory mismatch')
    for path, digest in expected.items():
        if hashlib.sha256((runtime/path).read_bytes()).hexdigest().upper() != digest:
            raise ValueError('Explicit runtime source digest mismatch: '+path)
    shutil.copytree(runtime, output)
    for path in candidate.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(candidate)
        target = output / relative
        if relative.as_posix() == 'config/__init__.py':
            continue  # The public marker is replaced by locally derived runtime config.
        if path.suffix == '.py' and target.exists():
            if normalized_ast(path.read_text(encoding='utf-8-sig')) != normalized_ast(target.read_text(encoding='utf-8-sig')):
                raise ValueError('Candidate overlay changes executable source: ' + relative.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return output


def run(candidate, group, staging, runtime=None, reference=None):
    candidate, staging = Path(candidate).resolve(), Path(staging).absolute()
    for path in (staging, *staging.parents):
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError('Linked staging path rejected')
    if staging.exists() or staging.resolve().is_relative_to(candidate) or candidate.is_relative_to(staging.resolve()):
        raise ValueError('Use a fresh staging directory outside the candidate')
    if group in ('runtime','reference') and runtime is None:
        raise ValueError('BLOCKED: --runtime-root must explicitly supply the verified integrated runtime')
    if runtime is not None:
        runtime = Path(runtime).resolve()
        if staging.resolve().is_relative_to(runtime) or runtime.is_relative_to(staging.resolve()):
            raise ValueError('Staging overlaps runtime input')
    if group == 'reference':
        if reference is None:
            raise ValueError('Runtime relocation tests require explicit --reference-root')
        reference = Path(reference).resolve()
        original = json.loads((candidate / 'reproducibility/standalone_runtime_plan.json').read_text())
        for item in original['files']:
            if hashlib.sha256((reference/item['path']).read_bytes()).hexdigest().upper() != item['sha256']:
                raise ValueError('Canonical reference digest mismatch: '+item['path'])
    staging.mkdir(parents=True)
    execution_root = prepare_overlay(candidate, runtime, staging / 'integrated-overlay') if group in ('runtime','reference') else candidate
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(execution_root),
        PUBLIC_CANDIDATE_ROOT=str(candidate), PORTABLE_RUNTIME=str(execution_root),
        MPLCONFIGDIR=str(staging/'matplotlib'), CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if reference is not None:
        environment['CANONICAL_REFERENCE_ROOT'] = str(reference)
    (staging / 'matplotlib').mkdir()
    process = subprocess.run([sys.executable, '-B', '-c', CHILD, str(execution_root), str(staging/'synthetic'),
                              group, json.dumps(GROUPS[group])], cwd=execution_root, env=environment,
                             capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
    lines = [line[len('__RESULT__'):] for line in process.stdout.splitlines() if line.startswith('__RESULT__')]
    if process.returncode or len(lines) != 1:
        raise ValueError('Guarded test process failed: ' + process.stderr[-4000:])
    result = json.loads(lines[0]);result['group'] = group
    (staging / 'test-report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if not result['successful'] or result['tests'] == 0 or result['skipped']:
        raise ValueError(result['details'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=tuple(GROUPS), default='tooling')
    parser.add_argument('--staging-root', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--reference-root', type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(Path(__file__).resolve().parents[1], args.group, args.staging_root, args.runtime_root, args.reference_root)
        print(json.dumps({key: result[key] for key in ('group','tests','failures','errors','skipped','successful')}))
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

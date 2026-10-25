"""Derive a local canonical thesis runtime from an explicit pinned BIGCity checkout.

Only the supplied checkout and this candidate are source providers. The original
checkout is never changed; generated upstream-derived files must remain local.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
if __package__ in (None, ''):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from setup.source_transformations import TRANSFORMS, transform

PIN = '5f7168bf7a43c3a40e19e4a80cdb6d77b080edec'
TREE = 'e6d2eb9e0b969a45589e4ebb26a1ae0e1b598275'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest().upper()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')


def no_links(path):
    for item in (Path(path).absolute(), *Path(path).absolute().parents):
        require(not item.is_symlink() and not (hasattr(item, 'is_junction') and item.is_junction()), 'Symlink/junction rejected')


def relative(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value, 'Unsafe source path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(p not in ('', '.', '..') for p in value.split('/')), 'Unsafe source path')
    return path


def git(root, *args):
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    environment.update(GIT_NO_REPLACE_OBJECTS='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_TERMINAL_PROMPT='0', GIT_NO_LAZY_FETCH='1')
    process = subprocess.run(['git', '--no-optional-locks', '-c', 'core.hooksPath=' + os.devnull, '-C', str(root), *args],
                             env=environment, capture_output=True, timeout=30)
    require(process.returncode == 0, 'Explicit upstream Git check failed: ' + process.stderr.decode(errors='replace').strip())
    return process.stdout.decode().strip()


def source_inputs(candidate, upstream):
    candidate, upstream = Path(candidate).resolve(), Path(upstream).resolve()
    no_links(candidate); no_links(upstream)
    require(sys.version_info[:2] == (3, 12), 'The fingerprinted integration profile requires Python 3.12')
    spec_path = candidate / 'reproducibility/integration_inputs.json'
    spec_bytes = spec_path.read_bytes(); spec = json.loads(spec_bytes)
    require(spec['upstream']['commit'] == PIN and spec['upstream']['tree'] == TREE, 'Unsupported integration specification')
    require(Path(git(upstream, 'rev-parse', '--show-toplevel')).resolve() == upstream, 'Use the actual BIGCity checkout root')
    gitdir = Path(git(upstream, 'rev-parse', '--absolute-git-dir')).resolve()
    require(gitdir == upstream / '.git', 'Shared worktree/object database is unsupported')
    require(not (gitdir / 'commondir').exists() and not (gitdir / 'objects/info/alternates').exists(), 'Shared/alternate Git objects rejected')
    require(git(upstream, 'rev-parse', 'HEAD^{commit}') == PIN, 'Unsupported BIGCity revision: exact pinned HEAD required')
    require(git(upstream, 'rev-parse', 'HEAD^{tree}') == TREE, 'Upstream tree identity mismatch')
    payloads, backups = {}, {}
    required = spec['required_upstream']
    require({r['path'] for r in required if r['adaptation']} == set(TRANSFORMS), 'Incomplete adaptation inventory')
    for record in required:
        path = upstream / relative(record['path']); no_links(path)
        require(path.is_file(), 'Missing upstream source: ' + record['path'])
        require(path.stat().st_size < 1024 * 1024, 'Unexpectedly large upstream source')
        data = path.read_bytes()
        require(digest(data) == record['sha256'], 'Upstream source mismatch: ' + record['path'] + ' (LF checkout required)')
        if record['adaptation']:
            backups[record['path']] = data
            data = transform(record['path'], data)
            require(digest(data) == record['adapted_sha256'], 'Generated adaptation fingerprint mismatch: ' + record['path'])
        payloads[record['path']] = data
    for record in spec['thesis_runtime_files']:
        path = candidate / relative(record['path']); no_links(path)
        require(path.is_file() and path.stat().st_size < 1024 * 1024, 'Missing/large thesis source: ' + record['path'])
        data = path.read_bytes()
        require(digest(data) == record['sha256'], 'Thesis source mismatch: ' + record['path'])
        require(record['path'] not in payloads, 'Source-provider path collision')
        payloads[record['path']] = data
    for path, data in payloads.items():
        require(b'\0' not in data, 'Binary source rejected')
        if path.endswith('.py'):
            compile(data.decode('utf-8-sig'), path, 'exec')
    return spec_bytes, payloads, backups


def manifest_for(spec_bytes, payloads, backups):
    inventory = [{'path': 'runtime/' + p, 'sha256': digest(data)} for p, data in sorted(payloads.items())]
    inventory += [{'path': 'upstream-backup/' + p, 'sha256': digest(data)} for p, data in sorted(backups.items())]
    return {'schema_version': 1, 'profile': 'six-canonical-controlled-static-none', 'upstream_commit': PIN, 'upstream_tree': TREE,
            'spec_sha256': digest(spec_bytes), 'files': inventory, 'adapted_files': sorted(backups),
            'private_inputs_used': False, 'thesis_git_history_used': False,
            'rights_status': 'Generated upstream-derived output is LOCAL; redistribution rights unresolved'}


def verify_output(output, expected):
    output = Path(output); no_links(output)
    require(output.is_dir(), 'Output directory is missing')
    path = output / 'integration_manifest.json'
    require(path.is_file(), 'Incomplete output: integration manifest missing')
    require(path.read_bytes() == encode(expected), 'Existing output belongs to another integration specification')
    expected_files = {r['path'] for r in expected['files']} | {'integration_manifest.json'}
    allowed_roots = {'runtime', 'upstream-backup', 'integration_manifest.json', 'dataset'}
    require({p.name for p in output.iterdir()} <= allowed_roots, 'Existing output inventory mismatch; unsupported sidecar')
    # Dataset contents are external inputs; never traverse or hash large data here.
    actual = {'integration_manifest.json'}
    for directory in ('runtime', 'upstream-backup'):
        actual.update(p.relative_to(output).as_posix() for p in (output/directory).rglob('*') if p.is_file())
    require(actual == expected_files, 'Existing output inventory mismatch; no files will be replaced')
    for item in expected['files']:
        path = output / relative(item['path']); no_links(path)
        require(digest(path.read_bytes()) == item['sha256'], 'Output/backup digest mismatch: ' + item['path'])


def integrate(candidate, bigcity_root, output, *, dry_run=False, verify_only=False):
    candidate, upstream, output = Path(candidate).resolve(), Path(bigcity_root).resolve(), Path(output).absolute()
    no_links(output)
    for protected in (candidate, upstream):
        require(not output.resolve().is_relative_to(protected) and not protected.is_relative_to(output.resolve()), 'Output must be separate from source directories')
    spec, payloads, backups = source_inputs(candidate, upstream)
    expected = manifest_for(spec, payloads, backups)
    if output.exists():
        verify_output(output, expected)
        return {'status': 'ALREADY_APPLIED_VERIFIED', 'runtime_files': len(payloads), 'backup_files': len(backups), 'writes': False}
    require(not verify_only, 'Verification requires an existing complete output')
    if dry_run:
        return {'status': 'DRY_RUN_VERIFIED', 'runtime_files': len(payloads), 'backup_files': len(backups), 'writes': False}
    require(output.parent.is_dir(), 'Output parent must already exist')
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent)).resolve()
    published = False
    try:
        # Back up every original before materializing its derived version.
        for prefix, records in (('upstream-backup', backups), ('runtime', payloads)):
            for name, data in sorted(records.items()):
                path = staging / prefix / relative(name); path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(data)
        (staging / 'integration_manifest.json').write_bytes(encode(expected))
        verify_output(staging, expected)
        require(not output.exists(), 'Output appeared during integration; refusing replacement')
        staging.rename(output)
        published = True
    finally:
        if not published and staging.exists():
            require(staging.parent == output.parent.resolve() and staging.name.startswith(output.name + '.partial-'), 'Unsafe cleanup target')
            shutil.rmtree(staging)
    return {'status': 'INTEGRATED_VERIFIED', 'runtime_files': len(payloads), 'backup_files': len(backups), 'writes': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bigcity-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(integrate(Path(__file__).resolve().parents[1], args.bigcity_root, args.output,
                                   dry_run=args.dry_run, verify_only=args.verify_only), indent=2))
        return 0
    except (OSError, ValueError, KeyError, SyntaxError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

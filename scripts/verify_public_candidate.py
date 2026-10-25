"""Check candidate inventory, canonical interfaces and explicitly withheld imports."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import xml.etree.ElementTree as ET

from scripts import validate_experiment_configs as canonical

LOCAL_ROOTS = {'scripts', 'utils', 'data_provider', 'models', 'config', 'train_next_hop', 'train_tte', 'setup'}


def normalized_ast(source):
    tree = deepcopy(ast.parse(source))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def safe_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ':' in value or '\\' in value or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Unsafe candidate path')
    return path


def check_inventory(root, manifest):
    root = Path(root).resolve()
    records = manifest['included']
    names = [item['path'] for item in records]
    if len(names) != len({name.casefold() for name in names}):
        raise ValueError('Duplicate/case-colliding candidate path')
    actual = set()
    for entry in root.iterdir():
        if entry.name == '.git':
            continue  # Root Git metadata is not part of the published source tree.
        paths = entry.rglob('*') if entry.is_dir() else (entry,)
        actual.update(p.relative_to(root).as_posix() for p in paths if p.is_file())
    if actual != set(names):
        raise ValueError('Candidate inventory mismatch')
    for item in records:
        path = root / safe_path(item['path'])
        for parent in (path, *path.parents):
            if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
                raise ValueError('Candidate links/junctions are not allowed')
        data = path.read_bytes()
        if item['path'] != 'reproducibility/file_selection.json':
            if hashlib.sha256(data).hexdigest().upper() != item['sha256']:
                raise ValueError('Candidate digest mismatch: ' + item['path'])
        if path.suffix.lower() in {'.pth', '.pt', '.ckpt', '.safetensors', '.pkl', '.npz'}:
            raise ValueError('Model/cache payload rejected')
        if item['publication_category'] in {'UPSTREAM', 'ADAPTATION', 'DERIVED_NEW', 'POTENTIALLY_DERIVED_NEW'}:
            raise ValueError('Unresolved implementation cannot be published')
    return len(records)


def classify_import(name, modules, external):
    if name in modules:
        return 'LOCAL'
    if name in external:
        return 'WITHHELD_EXTERNAL'
    if name.split('.')[0] in LOCAL_ROOTS:
        raise ValueError('Unexpected missing local dependency: ' + name)
    return 'THIRD_PARTY_OR_STDLIB'


def inspect_imports(root, external):
    paths = {p.relative_to(root).as_posix(): p for p in Path(root).rglob('*.py')}
    modules = {p[:-3].replace('/', '.'): p for p in paths if not p.endswith('/__init__.py')}
    modules.update({p[:-12].replace('/', '.'): p for p in paths if p.endswith('/__init__.py')})
    edges, blocked, third_party = set(), set(), set()
    for relative, path in paths.items():
        trees = [ast.parse(path.read_text(encoding='utf-8-sig'), filename=relative)]
        # Inspect the guarded subprocess program as source, too.
        for node in trees[0].body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CHILD' for t in node.targets):
                trees.append(ast.parse(ast.literal_eval(node.value)))
        for tree in trees:
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        raise ValueError('Relative imports require explicit review')
                    names = [node.module] if node.module else []
                    names.extend(node.module + '.' + a.name for a in node.names if node.module and node.module + '.' + a.name in modules)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module':
                    if not node.args or not isinstance(node.args[0], ast.Constant):
                        raise ValueError('Unresolved dynamic import')
                    names = [node.args[0].value]
                for name in names:
                    kind = classify_import(name, modules, external)
                    if kind == 'LOCAL':
                        edges.add((relative, name))
                    elif kind == 'WITHHELD_EXTERNAL':
                        blocked.add((relative, name))
                    elif name.split('.')[0] not in sys.stdlib_module_names:
                        third_party.add(name.split('.')[0])
    return {'python_files': len(paths), 'resolved_local_edges': len(edges),
            'withheld_imports': [{'source': a, 'module': b} for a, b in sorted(blocked)],
            'unexpected_missing_local_imports': [], 'third_party_roots': sorted(third_party)}


def interface_parser(path):
    facts = canonical.read_json(path)
    parser = argparse.ArgumentParser(add_help=False)
    types = {'str': str, 'int': int, 'float': float}
    for flag, values in facts['options'].items():
        options = dict(values)
        if 'type' in options:
            options['type'] = types[options['type']]
        parser.add_argument(flag, **options)
    return parser


def verify(root, runtime_root=None):
    root = Path(root).resolve()
    selection = canonical.read_json(root / 'reproducibility/file_selection.json')
    count = check_inventory(root, selection)
    external = set(selection['external_runtime_modules'])
    imports = inspect_imports(root, external)
    parser = interface_parser(root / 'reproducibility/runner_interface.json')
    artifacts = canonical.artifact_index(canonical.read_json(root / 'reproducibility/external_artifacts.json'))
    configurations = [canonical.validate_config(canonical.read_json(p), parser, artifacts) for p in sorted((root / 'experiments').glob('*.json'))]
    if len(configurations) != 6 or {x['id'] for x in configurations} != set(canonical.EXPECTED):
        raise ValueError('Expected six canonical experiments')
    figures = []
    for path in sorted((root / 'assets/figures').glob('*.svg')):
        tree = ET.parse(path)
        for element in tree.iter():
            if element.tag.rsplit('}', 1)[-1] in {'script', 'foreignObject', 'image'}:
                raise ValueError('Embedded/executable figure payload rejected')
            for key, value in element.attrib.items():
                if key.endswith('href') and not value.startswith('#'):
                    raise ValueError('External figure resource rejected')
        figures.append(path.name)
    result = {'status': 'VERIFIED_PUBLIC_SUBSET', 'files': count, 'imports': imports,
              'configuration_validation': 'DECLARATIVE_CANONICAL_INTERFACE_ONLY', 'configurations': configurations,
              'environments': canonical.validate_environments(root), 'figures': figures,
              'full_runtime_status': 'SOURCE_INTEGRATION_AVAILABLE_MODEL_REPRODUCTION_NOT_TESTED',
              'training_or_generation_executed': False}
    if runtime_root is not None:
        runtime = Path(runtime_root).resolve()
        actual = canonical.parser_from_source(runtime / 'config/args_config.py')
        facts = canonical.read_json(root / 'reproducibility/runner_interface.json')
        raw = (runtime / 'config/args_config.py').read_bytes()
        if hashlib.sha256(raw).hexdigest().upper() != facts['source_module_sha256']:
            raise ValueError('Supplied runtime parser source mismatch')
        for flag, values in facts['options'].items():
            action = actual._option_string_actions[flag]
            if action.default != values.get('default'):
                raise ValueError('Runtime interface default mismatch: ' + flag)
            if values.get('choices') and action.choices is not None and not set(values['choices']) <= set(action.choices):
                raise ValueError('Runtime interface choice mismatch: ' + flag)
        result['explicit_private_runtime_validation'] = canonical.validate_all(runtime)
        result['actual_parser_fields_compared'] = len(facts['options'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runtime-root', type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.root, args.runtime_root), indent=2))
        return 0
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

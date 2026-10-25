"""Transparent edits to hash-verified, user-supplied BIGCity syntax trees.

Upstream functions and prompt literals are read from that source, never embedded
as replacement payloads. Only the six controlled, static-none experiments are supported.
"""
import ast
from copy import deepcopy


def require(value, message):
    if not value:
        raise ValueError(message)


def one(items, label):
    require(len(items) == 1, 'Unsupported source structure: ' + label)
    return items[0]


def method(tree, owner, name):
    cls = one([x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == owner], owner)
    return one([x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name], owner + '.' + name)


def statements(source):
    return ast.parse(source).body


def attr(node, name):
    return isinstance(node, ast.Attribute) and node.attr == name


def targets(node, name):
    return isinstance(node, ast.Assign) and any(attr(x, name) or isinstance(x, ast.Name) and x.id == name for x in node.targets)


def args_config(tree):
    factory = one([x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'get_args_parser'], 'parser factory')
    extra = '''
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--split', type=str, choices=['train','validation','test','full'], default='train')
parser.add_argument('--pretrained_checkpoint', type=str, default=None)
parser.add_argument('--gpt2_path', type=str, default='./models/gpt2')
parser.add_argument('--dynamic_mode', choices=['controlled'], default='controlled')
parser.add_argument('--static_ablation', choices=['none'], default='none')
parser.add_argument('--semantic_mode', choices=['none','poi_tfidf','poi_qwen_tfidf','tte_structured','tte_qwen'], default='none')
'''
    end = one([i for i, x in enumerate(factory.body) if isinstance(x, ast.Return)], 'parser return')
    factory.body[end:end] = statements(extra)
    return tree


def config_init(tree):
    tree.body.insert(0, ast.ImportFrom('args_config', [ast.alias('args')], 1))
    seed = one([n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'set_random_seeds'], 'seed initialization')
    seed.args = [ast.Attribute(ast.Name('args', ast.Load()), 'seed', ast.Load())]
    return tree


def global_vars(tree):
    tree.body += statements('from config.thesis_paths import apply_paths\napply_paths(globals(), args)')
    return tree


def file_loader(tree):
    fn = method(tree, 'FileLoader', 'load_road_features_file')
    block = one([n for n in ast.walk(fn) if isinstance(n, ast.If) and any(targets(s, 'dynamic_data') for s in n.orelse)], 'dynamic source branch')
    start = one([i for i, s in enumerate(block.orelse) if targets(s, 'missing_ids')], 'legacy preprocessing start')
    stop = one([i for i, s in enumerate(block.orelse) if i > start and isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and attr(s.value.func, 'save')], 'dynamic cache publication')
    block.orelse[start:stop] = statements('''
if args.dynamic_mode != 'controlled':
    raise ValueError('This thesis runtime supports controlled dynamic mode only.')
observed_roads = torch.zeros(self.road_cnt, dtype=torch.bool)
observed_roads[torch.unique(road_ids)] = True
self.dynamic_features = preprocess_controlled(self.dynamic_features, observed_roads)
''')
    tree.body.insert(0, ast.ImportFrom('data_provider.dynamic_preprocessing', [ast.alias('preprocess_controlled')], 0))
    return tree


def data_loader(tree):
    fn = method(tree, 'DatasetNextHop', 'read_traj_data')
    times = one([n for n in fn.body if targets(n, 'traj_time_lists')], 'NH timestamps')
    dtype = one([k for k in times.value.keywords if k.arg == 'dtype'], 'NH timestamp dtype')
    dtype.value = ast.Attribute(ast.Name('torch', ast.Load()), 'int64', ast.Load())
    slot = one([n for n in fn.body if targets(n, 'traj_time_id_lists')], 'NH time slots')
    start = one([n for n in ast.walk(slot) if isinstance(n, ast.Call) and attr(n.func, 'tensor')], 'timestamp origin tensor')
    class IntegerOrigin(ast.NodeTransformer):
        def visit_Call(self, n):
            return ast.Name('start_timestamp', ast.Load()) if n is start else self.generic_visit(n)
    IntegerOrigin().visit(slot)
    index = fn.body.index(slot)
    fn.body[index:index] = statements('start_timestamp = int(global_vars.start_time.timestamp())')
    return tree


def st_tokenizer(tree):
    fn = method(tree, 'StTokenizer', 'forward')
    offset = one([n for n in ast.walk(fn) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)
                  and isinstance(n.right, ast.Name) and n.right.id == 'N'
                  and isinstance(n.left, ast.Call) and attr(n.left.func, 'arange')], 'graph stride')
    offset.right = ast.BinOp(ast.Name('N', ast.Load()), ast.Add(), ast.Constant(1))
    return tree


def local_gpt2(tree):
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and attr(n.func, 'from_pretrained')]
    require(len(calls) in (1, 2), 'Unexpected GPT-2 loader count')
    for call in calls:
        require(call.args, 'GPT-2 asset source is missing')
        call.args[0] = ast.Attribute(ast.Name('args', ast.Load()), 'gpt2_path', ast.Load())
        require(not any(k.arg == 'local_files_only' for k in call.keywords), 'Unexpected existing loader option')
        call.keywords.append(ast.keyword('local_files_only', ast.Constant(True)))
    return tree


def fine_tune(tree):
    cls = one([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BigCity4FineTune'], 'fine-tune class')
    cls.bases.insert(0, ast.Name('SemanticEnrichmentMixin', ast.Load()))
    tree.body.insert(0, ast.ImportFrom('models.thesis_semantics', [ast.alias('SemanticEnrichmentMixin'), ast.alias('_prepare_pretrained_state_dict')], 0))
    init = method(tree, 'BigCity4FineTune', '__init__')
    load = one([i for i, n in enumerate(init.body) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == 'checkpoint'], 'checkpoint branch')
    stop = one([i for i, n in enumerate(init.body) if isinstance(n, ast.For)], 'tokenizer freezing')
    init.body[load:stop] = statements('self.load_thesis_pretrained(checkpoint)')
    freeze = one([n for n in init.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and attr(n.value.func, 'requires_grad_')], 'special token freezing')
    index = init.body.index(freeze)
    init.body[index:index+1] = [ast.If(ast.UnaryOp(ast.Not(), ast.Attribute(ast.Name('self', ast.Load()), 'special_token_was_synthesized', ast.Load())), [freeze], [])]
    init.body[index+1:index+1] = statements('self.initialize_semantics()')
    prompt_count = 0
    for n in init.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and attr(n.value.func, 'encode_prompt'):
            require(len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute), 'Unexpected prompt target')
            n.targets[0].attr += '_IDS'
            n.value.func.attr = 'encode_prompt_ids'; n.value.args = n.value.args[:1]; prompt_count += 1
    require(prompt_count == 5, 'Unexpected prompt initialization count')
    # Derive prompt builders from the obtained upstream function, not a copied payload.
    original = method(tree, 'BigCity4FineTune', 'encode_prompt')
    encoded = deepcopy(original); encoded.name = 'encode_prompt_ids'; encoded.args.args = encoded.args.args[:2]
    first = original.body[0]
    require(isinstance(first, ast.Assign), 'Unexpected prompt encoding structure')
    encoded.body = [ast.Return(deepcopy(first.value))]
    builder = deepcopy(original); builder.name = 'build_prompt'; builder.args.args[1].arg = 'encoded_prompt'
    body = deepcopy(original.body[1:])
    special = one([s for s in body if targets(s, 'special_token')], 'prompt special token')
    token_call = one([n for n in ast.walk(special) if isinstance(n, ast.Call) and attr(n.func, 'special_token')], 'special token lookup')
    token_call.args = [ast.Name('special_token_ids', ast.Load())]
    special_index = body.index(special)
    body[special_index:special_index] = statements('special_token_ids = torch.tensor([special_token_id], dtype=torch.long, device=self.device)')
    joined = one([s for s in body if targets(s, 'prompt_with_special_token')], 'prompt concatenation')
    builder.body = [s for s in body if s is not joined and not isinstance(s, ast.Return)] + [ast.Return(joined.value)]
    cls.body.remove(original); cls.body.extend([encoded, builder])
    forward = method(tree, 'BigCity4FineTune', 'forward')
    branches = [n for n in ast.walk(forward) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == 'task_name']
    require(len(branches) == 5, 'Unexpected original task branches')
    for branch in branches:
        task = branch.test.comparators[0].value
        concat = one([n for n in branch.body if targets(n, 'batch_psm_tokens')], 'task prompt concatenation')
        prompt = one([n for n in ast.walk(concat) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                      and n.value.id == 'self' and n.attr.endswith('_PROMPT')], 'task prompt attribute')
        prompt_name = prompt.attr + '_IDS'
        class ReplacePrompt(ast.NodeTransformer):
            def visit_Attribute(self, n):
                return ast.Name('task_prompt', ast.Load()) if n is prompt else self.generic_visit(n)
        ReplacePrompt().visit(concat)
        index = branch.body.index(concat)
        branch.body[index:index] = statements('task_prompt = self.build_prompt(self.' + prompt_name + ', ' + ('0' if task in ('next_hop','traj_recover') else '1') + ')')
        if task == 'next_hop':
            branch.body[:0] = statements('batch_tokens = self.apply_semantic_enrichment(batch_tokens, batch_road_id)')
    return local_gpt2(tree)


TRANSFORMS = {'config/__init__.py': config_init, 'config/args_config.py': args_config,
              'config/global_vars.py': global_vars, 'data_provider/file_loader.py': file_loader,
              'data_provider/data_loader.py': data_loader, 'models/st_tokenizer.py': st_tokenizer,
              'models/backbone.py': local_gpt2, 'models/bigcity4finetune.py': fine_tune}


def transform(path, source):
    tree = TRANSFORMS[path](ast.parse(source.decode('utf-8-sig'), filename=path))
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree) + '\n'
    compile(result, path, 'exec')
    return result.encode('utf-8')

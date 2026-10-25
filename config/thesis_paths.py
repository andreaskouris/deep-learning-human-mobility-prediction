"""Canonical XA source/cache bindings; no data is read on import."""
import os


def apply_paths(namespace, args):
    root = namespace['city_root_path']
    tag = lambda value: format(value, 'g').replace('.', 'p').replace('-', 'm')
    scope = 'develop' if args.develop else args.split
    signature = f'{scope}_seq{args.seq_len}_sr{tag(args.sample_rate)}_seed{str(args.seed).replace("-", "m")}_v2'
    splits = {name: os.path.join(root, f'traj_{args.city}_{name}.csv') for name in ('train', 'validation', 'test')}
    splits['full'] = namespace['traj_file']
    namespace.update(split_traj_files=splits, cache_scope=scope, base_cache_signature=signature, cache_signature=signature,
                     sample_rate_tag=tag(args.sample_rate), seed_tag=str(args.seed).replace('-', 'm'), mask_rate_tag=tag(args.mask_rate))
    namespace['cur_traj_file'] = namespace['traj_file_short'] if args.develop else splits[args.split]
    namespace['dataset_meta_file'] = os.path.join(root, f'cached_meta_{args.city}_{signature}.json')
    namespace['road_dynamic_tensor_file'] = os.path.join(root, f'cached_{args.city}_dynamic_{args.dynamic_mode}_v1.pth')
    semantics = {'road_semantic_tensor_file': 'semantic_tfidf', 'road_language_semantic_tensor_file': 'language_road_semantic_qwen3_0_6b',
                 'road_tte_structured_semantic_tensor_file': 'tte_semantics_binary', 'road_tte_qwen_semantic_tensor_file': 'tte_semantics_qwen'}
    for field, suffix in semantics.items():
        namespace[field] = os.path.join(root, f'cached_{args.city}_{suffix}_v1.pth')
    def cached_name(dataset, signature=signature):
        return os.path.join(root, f'cached_{dataset}_dataset_{signature}.pth')
    namespace['generate_cached_file_name'] = cached_name
    for dataset in namespace['datasets']:
        namespace['cached_' + dataset + '_dataset'] = cached_name(dataset)
    namespace['cached_files'] = {name: namespace['cached_' + name + '_dataset'] for name in namespace['datasets']}

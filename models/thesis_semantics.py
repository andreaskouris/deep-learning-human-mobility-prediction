"""Original thesis semantic projection and checkpoint compatibility extensions."""
from collections import OrderedDict
import torch
import torch.nn as nn
from config.args_config import args
from config import global_vars

def _prepare_pretrained_state_dict(checkpoint, special_token):
    """Adapt published and locally generated BIGCity checkpoints."""
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model_state_dict'), dict):
        source_state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and all((isinstance(key, str) for key in checkpoint)) and all((torch.is_tensor(value) for value in checkpoint.values())):
        source_state_dict = checkpoint
    else:
        raise TypeError('Unsupported checkpoint structure. Expected either a raw state_dict or a dictionary containing model_state_dict.')
    prefix_mapping = {'backbone.mlp_c.': 'backbone.tasks_mlp.road_clas.', 'backbone.mlp_t.': 'backbone.tasks_mlp.time_reg.', 'backbone.mlp_r.': 'backbone.tasks_mlp.state_reg.'}
    adapted_state_dict = OrderedDict()
    for old_key, tensor in source_state_dict.items():
        new_key = old_key
        for old_prefix, new_prefix in prefix_mapping.items():
            if old_key.startswith(old_prefix):
                new_key = new_prefix + old_key[len(old_prefix):]
                break
        if new_key in adapted_state_dict:
            raise RuntimeError(f'Checkpoint compatibility mapping collision: {new_key}')
        adapted_state_dict[new_key] = tensor
    synthesized_special_token = False
    if 'special_token.weight' not in adapted_state_dict:
        expected_shape = (2, args.d_model)
        actual_shape = tuple(special_token.weight.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(f'Unexpected special-token shape. Expected {expected_shape}, found {actual_shape}.')
        width = special_token.weight.shape[1]
        dtype = special_token.weight.dtype
        adapted_state_dict['special_token.weight'] = torch.stack([torch.ones(width, dtype=dtype), torch.full((width,), 2.0, dtype=dtype)])
        synthesized_special_token = True
    return (adapted_state_dict, synthesized_special_token)

class SemanticEnrichmentMixin:

    def initialize_semantics(self):
        """Normalize nonzero cache rows and create the zero-initialized, bias-free projection."""
        self.semantic_mode = args.semantic_mode
        if self.semantic_mode in {'poi_tfidf', 'poi_qwen_tfidf', 'tte_structured', 'tte_qwen'}:
            if self.semantic_mode == 'poi_tfidf':
                semantic_cache_path = global_vars.road_semantic_tensor_file
                semantic_feature_dim = 204
            elif self.semantic_mode == 'poi_qwen_tfidf':
                semantic_cache_path = global_vars.road_language_semantic_tensor_file
                semantic_feature_dim = 1024
            elif self.semantic_mode == 'tte_structured':
                semantic_cache_path = global_vars.road_tte_structured_semantic_tensor_file
                semantic_feature_dim = 10
            elif self.semantic_mode == 'tte_qwen':
                semantic_cache_path = global_vars.road_tte_qwen_semantic_tensor_file
                semantic_feature_dim = 1024
            else:
                raise RuntimeError(f'Unexpected semantic mode inside semantic-cache setup: {self.semantic_mode}')
            semantic_cache = torch.load(semantic_cache_path, map_location='cpu', weights_only=True)
            semantic_features = semantic_cache['semantic_features'].to(dtype=torch.float32)
            if self.semantic_mode in {'tte_structured', 'tte_qwen'}:
                semantic_mask = semantic_features.abs().sum(dim=1) > 0
            else:
                semantic_mask = semantic_cache['semantic_mask'].to(dtype=torch.bool)
            road_count = int(self.tokenizer.road_cnt)
            expected_feature_shape = (road_count, semantic_feature_dim)
            if tuple(semantic_features.shape) != expected_feature_shape:
                raise RuntimeError(f'Unexpected semantic feature shape. Expected {expected_feature_shape}, found {tuple(semantic_features.shape)}.')
            if tuple(semantic_mask.shape) != (road_count,):
                raise RuntimeError(f'Unexpected semantic mask shape: {tuple(semantic_mask.shape)}.')
            norms = torch.linalg.vector_norm(semantic_features, dim=1, keepdim=True)
            normalized_features = torch.zeros_like(semantic_features)
            nonzero = norms.squeeze(1) > 0
            normalized_features[nonzero] = semantic_features[nonzero] / norms[nonzero]
            if not torch.equal(nonzero, semantic_mask):
                raise RuntimeError('Semantic cache mask does not match the non-zero semantic rows.')
            self.register_buffer('semantic_features', normalized_features, persistent=False)
            self.register_buffer('semantic_mask', semantic_mask, persistent=False)
            self.semantic_projection = nn.Linear(semantic_feature_dim, args.d_model, bias=False)
            nn.init.zeros_(self.semantic_projection.weight)
        elif self.semantic_mode != 'none':
            raise ValueError(f'Unsupported semantic mode: {self.semantic_mode}')

    def apply_semantic_enrichment(self, batch_tokens, batch_road_id):
        if self.semantic_mode == 'none':
            return batch_tokens
        semantic_inputs = self.semantic_features[batch_road_id]
        semantic_available = self.semantic_mask[batch_road_id].unsqueeze(-1)
        semantic_delta = self.semantic_projection(semantic_inputs)
        semantic_delta = semantic_delta * semantic_available.to(dtype=semantic_delta.dtype)
        return batch_tokens + semantic_delta

    def load_thesis_pretrained(self, checkpoint):
        checkpoint_path = f'./checkpoints/{args.city}_pretrain_best.pth' if checkpoint is None else checkpoint
        checkpoint_object = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict, synthesized_special_token = _prepare_pretrained_state_dict(checkpoint_object, self.special_token)
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {'backbone.tasks_mlp.tul_clas.fc1.weight', 'backbone.tasks_mlp.tul_clas.fc1.bias', 'backbone.tasks_mlp.tul_clas.fc2.weight', 'backbone.tasks_mlp.tul_clas.fc2.bias'}
        missing_keys = set(incompatible.missing_keys)
        unexpected_keys = set(incompatible.unexpected_keys)
        disallowed_missing = missing_keys - allowed_missing
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(f'Checkpoint compatibility validation failed. Disallowed missing keys: {sorted(disallowed_missing)}; unexpected keys: {sorted(unexpected_keys)}')
        self.pretrained_checkpoint_path = str(checkpoint_path)
        self.special_token_was_synthesized = synthesized_special_token

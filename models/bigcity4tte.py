from __future__ import annotations
from models.tte_tokenization_adapter import TTETokenizationAdapter
import torch
import torch.nn as nn
from config.args_config import args
from models.bigcity4finetune import BigCity4FineTune
from models.layers import MLP
TTE_PROMPT = 'Predict the travel time of the trajectory, regress the total trip duration'

class BigCity4TTE(TTETokenizationAdapter, BigCity4FineTune):
    """Trip-level TTE task extension with route and departure-time inputs.

The mirrored tokenizer body is supplied by an explicit private adapter."""

    def __init__(self, device, checkpoint=None):
        """Initialize the TTE head under an isolated experiment RNG state."""
        super().__init__(device, checkpoint=checkpoint)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            tte_head = MLP(args.d_model, args.d_model, 1)
        self.backbone.tasks_mlp['tte_reg'] = tte_head.to(device)
        self.TTE_PROMPT_IDS = self.encode_prompt_ids(TTE_PROMPT)
        self.tte_loss = nn.SmoothL1Loss(beta=1.0)

    def masked_tte_cross_attention(self, x, valid_mask):
        """
        BIGCity CrossAttention with TTE padding
        excluded from keys/values.

        The same pretrained W_q parameter is used.
        Only artificial PAD positions are masked.
        """
        if x.ndim != 3:
            raise ValueError('Expected TTE attention input with shape (B, L, D).')
        B, L, _ = x.shape
        valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
        if tuple(valid_mask.shape) != (B, L):
            raise ValueError('Unexpected TTE valid_mask shape for cross-attention.')
        if torch.any(valid_mask.sum(dim=1) <= 0):
            raise ValueError('Every TTE trajectory must contain at least one real road.')
        cross_attention = self.tokenizer.cross_attention
        Q = torch.matmul(x, cross_attention.W_q)
        K = x
        V = x
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / (2 * cross_attention.D) ** 0.5
        key_mask = valid_mask.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~key_mask, torch.finfo(attention_scores.dtype).min)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(~key_mask, 0.0)
        output = torch.matmul(attention_weights, V)
        output = output * valid_mask.unsqueeze(-1).to(dtype=output.dtype)
        return output

    def apply_tte_semantic_enrichment(self, batch_tokens, batch_road_id, valid_mask):
        """
        Apply semantics only to real road tokens.

        PAD road id == road_count and therefore
        must never index the semantic cache, whose
        valid rows are 0 ... road_count - 1.
        """
        if self.semantic_mode == 'none':
            return batch_tokens
        if batch_road_id.shape != valid_mask.shape:
            raise ValueError('batch_road_id and valid_mask must have identical shapes.')
        road_count = int(self.semantic_features.shape[0])
        real_road_ids = batch_road_id[valid_mask]
        if real_road_ids.numel() == 0:
            raise ValueError('TTE batch contains no real road tokens.')
        if torch.any(real_road_ids < 0) or torch.any(real_road_ids >= road_count):
            raise ValueError('Real TTE road ID is outside the semantic-cache range.')
        safe_road_ids = batch_road_id.clone()
        safe_road_ids[~valid_mask] = 0
        semantic_inputs = self.semantic_features[safe_road_ids]
        semantic_available = self.semantic_mask[safe_road_ids].unsqueeze(-1)
        semantic_delta = self.semantic_projection(semantic_inputs)
        semantic_gate = semantic_available & valid_mask.unsqueeze(-1)
        semantic_delta = semantic_delta * semantic_gate.to(dtype=semantic_delta.dtype)
        return batch_tokens + semantic_delta

    @staticmethod
    def pack_tte_tokens(batch_tokens, valid_mask, reg_token):
        """
        Transform

            real roads -> PAD

        into

            real roads -> REG -> PAD

        independently for every trajectory.

        Because BIGCity uses causal GPT-2 blocks,
        the REG token cannot attend to trailing
        PAD tokens.
        """
        B, L, D = batch_tokens.shape
        if tuple(valid_mask.shape) != (B, L):
            raise ValueError('Unexpected TTE valid_mask shape.')
        lengths = valid_mask.sum(dim=1).to(torch.long)
        if torch.any(lengths <= 0):
            raise ValueError('TTE trajectory has no valid road tokens.')
        if torch.any(lengths > L):
            raise ValueError('TTE valid length exceeds token sequence length.')
        road_positions = torch.arange(L, device=batch_tokens.device, dtype=torch.long).unsqueeze(0)
        expected_mask = road_positions < lengths.unsqueeze(1)
        if not torch.equal(valid_mask, expected_mask):
            raise ValueError('TTE valid_mask must be left-aligned.')
        if tuple(reg_token.shape) != (B, 1, D):
            raise ValueError('Unexpected REG token shape.')
        source_tokens = torch.cat([batch_tokens, reg_token], dim=1)
        positions = torch.arange(L + 1, device=batch_tokens.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        reg_positions = lengths.unsqueeze(1)
        reg_source_index = torch.full_like(positions, L)
        source_indices = torch.where(positions < reg_positions, positions, torch.where(positions == reg_positions, reg_source_index, positions - 1))
        packed_tokens = torch.gather(source_tokens, dim=1, index=source_indices.unsqueeze(-1).expand(-1, -1, D))
        return (packed_tokens, lengths)

    def forward_tte(self, batch_road_id, batch_time_id, batch_time_features, valid_mask, batch_target_normalized=None):
        """
        Forward pass for trip-level TTE.

        The returned prediction is in normalized
        target space. Conversion back to seconds
        is performed by the training/evaluation
        runner using train-only statistics.
        """
        batch_tokens = self.tokenize_tte(batch_road_id, batch_time_id, batch_time_features, valid_mask)
        B, L, D = batch_tokens.shape
        valid_mask = valid_mask.to(device=batch_tokens.device, dtype=torch.bool)
        if tuple(batch_road_id.shape) != (B, L):
            raise ValueError('Unexpected TTE road-id shape.')
        batch_tokens = self.apply_tte_semantic_enrichment(batch_tokens, batch_road_id, valid_mask)
        reg_token_id = torch.tensor([1], dtype=torch.long, device=batch_tokens.device)
        reg_token = self.special_token(reg_token_id).view(1, 1, D).expand(B, 1, D)
        packed_tokens, lengths = self.pack_tte_tokens(batch_tokens, valid_mask, reg_token)
        task_prompt = self.build_prompt(self.TTE_PROMPT_IDS, 1)
        prompt_length = int(task_prompt.shape[1])
        batch_psm_tokens = torch.cat([task_prompt.expand(B, -1, -1), packed_tokens], dim=1)
        tte_output = self.backbone(batch_psm_tokens, ['tte_reg'])['tte_reg']
        reg_output_positions = prompt_length + lengths
        batch_indices = torch.arange(B, device=batch_tokens.device)
        prediction_normalized = tte_output[batch_indices, reg_output_positions, 0]
        if batch_target_normalized is None:
            return prediction_normalized
        target = batch_target_normalized.to(device=prediction_normalized.device, dtype=prediction_normalized.dtype).view(-1)
        if target.shape != (B,):
            raise ValueError('Unexpected normalized TTE target shape.')
        loss = self.tte_loss(prediction_normalized, target)
        return (loss, prediction_normalized)

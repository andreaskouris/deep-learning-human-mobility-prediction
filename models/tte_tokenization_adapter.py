"""Use the locally derived BIGCity tokenizer with the canonical TTE attention mask."""
import torch
from config.args_config import args
from models.st_tokenizer import StTokenizer


class _TokenizerView:
    def __init__(self, tokenizer, attention):
        self._tokenizer = tokenizer
        self.cross_attention = attention

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


class TTETokenizationAdapter:
    def tokenize_tte(self, batch_road_id, batch_time_id, batch_time_features, valid_mask):
        """Reuse upstream token operations; exclude artificial PAD keys/values."""
        if batch_road_id.ndim != 2:
            raise ValueError('TTE road IDs must have shape (B, L).')
        batch, length = batch_road_id.shape
        if tuple(batch_time_id.shape) != (batch, length):
            raise ValueError('Unexpected TTE time-id shape.')
        if tuple(batch_time_features.shape[:2]) != (batch, length):
            raise ValueError('Unexpected TTE time-feature shape.')
        if length != args.seq_len:
            raise ValueError('TTE sequence length does not match args.seq_len.')
        valid_mask = valid_mask.to(device=batch_road_id.device, dtype=torch.bool)
        if tuple(valid_mask.shape) != (batch, length):
            raise ValueError('Unexpected TTE valid_mask shape.')
        view = _TokenizerView(self.tokenizer, lambda values: self.masked_tte_cross_attention(values, valid_mask))
        return StTokenizer.forward(view, batch_road_id, batch_time_id, batch_time_features)

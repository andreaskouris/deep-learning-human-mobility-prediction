"""Synthetic checks for source relocation; no model forward or pretrained load."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from config.args_config import args
from config import global_vars
from models.bigcity4finetune import BigCity4FineTune
from models.bigcity4tte import BigCity4TTE
from models.thesis_semantics import SemanticEnrichmentMixin, _prepare_pretrained_state_dict
from data_provider.dynamic_preprocessing import preprocess_controlled


class RelocatedImplementationTests(unittest.TestCase):
    def test_semantic_modes_keep_zero_rows_and_projection(self):
        fields = {'poi_tfidf': ('road_semantic_tensor_file', 204), 'poi_qwen_tfidf': ('road_language_semantic_tensor_file', 1024),
                  'tte_structured': ('road_tte_structured_semantic_tensor_file', 10), 'tte_qwen': ('road_tte_qwen_semantic_tensor_file', 1024)}
        class Fixture(SemanticEnrichmentMixin, nn.Module):
            pass
        for mode, (field, width) in fields.items():
            with self.subTest(mode=mode), patch.multiple(args, semantic_mode=mode, d_model=8):
                values = torch.zeros(3, width); values[0, 0] = 3; values[0, 1] = 4
                payload = {'semantic_features': values, 'semantic_mask': torch.tensor([True, False, False])}
                with patch('torch.load', return_value=payload):
                    model = Fixture(); model.tokenizer = SimpleNamespace(road_cnt=3); model.initialize_semantics()
                self.assertEqual(tuple(model.semantic_projection.weight.shape), (8, width))
                self.assertIsNone(model.semantic_projection.bias)
                self.assertEqual(torch.count_nonzero(model.semantic_projection.weight), 0)
                torch.testing.assert_close(model.semantic_features[0, :2], torch.tensor([.6, .8]))
                self.assertEqual(torch.count_nonzero(model.semantic_features[1:]), 0)

    def test_checkpoint_compatibility_keeps_deterministic_tokens(self):
        with patch.object(args, 'd_model', 8):
            result, synthesized = _prepare_pretrained_state_dict({'backbone.mlp_c.weight': torch.ones(1)}, nn.Embedding(2, 8))
        self.assertTrue(synthesized)
        self.assertIn('backbone.tasks_mlp.road_clas.weight', result)
        torch.testing.assert_close(result['special_token.weight'], torch.stack([torch.ones(8), torch.full((8,), 2.)]))

    def test_tte_mixin_keeps_initialization_and_state_dict(self):
        reference = Path(os.environ['CANONICAL_REFERENCE_ROOT'])
        namespace = {'__name__': 'verified_canonical_tte_reference'}
        exec(compile((reference / 'models/bigcity4tte.py').read_text(), '<canonical-tte-reference>', 'exec'), namespace)
        original_class = namespace['BigCity4TTE']
        def fake_parent(self, device, checkpoint=None):
            nn.Module.__init__(self); self.backbone = nn.Module(); self.backbone.tasks_mlp = nn.ModuleDict()
        with patch.object(BigCity4FineTune, '__init__', fake_parent), patch.object(BigCity4FineTune, 'encode_prompt_ids', return_value=torch.tensor([1])), patch.multiple(args, d_model=8, seed=2021):
            before = torch.random.get_rng_state().clone()
            actual = BigCity4TTE(torch.device('cpu')); expected = original_class(torch.device('cpu'))
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(set(actual.state_dict()), set(expected.state_dict()))
        for key in actual.state_dict():
            self.assertTrue(torch.equal(actual.state_dict()[key], expected.state_dict()[key]), key)
        self.assertEqual(actual.tte_loss.beta, expected.tte_loss.beta)

    def test_controlled_function_matches_canonical_source(self):
        namespace = {'__name__': 'verified_canonical_preprocessing_reference'}
        exec(compile((Path(os.environ['CANONICAL_REFERENCE_ROOT']) / 'data_provider/dynamic_preprocessing.py').read_text(), '<canonical-controlled-reference>', 'exec'), namespace)
        for dtype in (torch.float32, torch.float64):
            matrix = torch.tensor([[0, 2], [2, 4], [0, 0]], dtype=dtype); mask = torch.tensor([True, True, False])
            self.assertTrue(torch.equal(preprocess_controlled(matrix, mask), namespace['preprocess_controlled'](matrix, mask)))


if __name__ == '__main__':
    unittest.main()

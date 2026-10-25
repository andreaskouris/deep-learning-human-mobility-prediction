"""Numerical TTE hook comparison against explicit read-only canonical source."""
from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch import nn
from config.args_config import args
from models.bigcity4tte import BigCity4TTE
from test_canonical_runtime_regressions import tiny_tokenizer


class TTEHookReferenceTests(unittest.TestCase):
    def canonical_class(self):
        namespace={'__name__':'canonical_reference_only'}
        path=Path(os.environ['CANONICAL_REFERENCE_ROOT'])/'models/bigcity4tte.py'
        exec(compile(path.read_text(), '<canonical-source-reference>', 'exec'), namespace)
        return namespace['BigCity4TTE']

    def compare(self, gradient):
        canonical=self.canonical_class()
        with patch.multiple(args,seq_len=3,pre_dyna=False,static_ablation='none'):
            for seed in range(5):
                torch.manual_seed(seed)
                first=BigCity4TTE.__new__(BigCity4TTE); second=canonical.__new__(canonical)
                nn.Module.__init__(first);nn.Module.__init__(second)
                first.tokenizer=tiny_tokenizer();second.tokenizer=deepcopy(first.tokenizer)
                roads=torch.tensor([[0,1,3],[2,0,1]])
                slots=torch.tensor([[seed%3]*3,[(seed+1)%3]*3])
                features=torch.randn(2,3,6)
                mask=torch.tensor([[True,True,False],[True,True,True]])
                actual=first.tokenize_tte(roads,slots,features,mask);expected=second.tokenize_tte(roads,slots,features,mask)
                self.assertTrue(torch.equal(actual,expected),seed)
                if gradient:
                    actual.sum().backward();expected.sum().backward()
                    for (_,a),(_,b) in zip(first.tokenizer.named_parameters(),second.tokenizer.named_parameters()):
                        if a.grad is None or b.grad is None:self.assertIs(a.grad,b.grad)
                        else:self.assertTrue(torch.equal(a.grad,b.grad),seed)

    def test_masked_hook_matches_canonical_tokens_exactly(self):
        self.compare(False)

    def test_tiny_helper_gradients_match_without_optimizer(self):
        self.compare(True)


if __name__=='__main__':
    unittest.main()

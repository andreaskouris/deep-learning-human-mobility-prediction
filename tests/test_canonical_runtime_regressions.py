"""CPU-only synthetic contracts; no pretrained model or real trajectory is used."""
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn

with patch.object(sys, "argv", ["synthetic-regressions"]):
    from config.args_config import args
    from config import global_vars
    from data_provider.data_loader import DatasetNextHop
    from data_provider.tte_dataset import build_tte_payload, validate_tte_payload
    from models.st_tokenizer import StTokenizer
    from models.bigcity4finetune import BigCity4FineTune
    from models.bigcity4tte import BigCity4TTE
    from models.layers import CrossAttention
    from train_tte import tte_validation_improved, normalize_target, denormalize_prediction
    from utils.tte_metrics import TTEMetricAccumulator


class TinyGraph(nn.Module):
    """Deterministic neighbor aggregation exposes graph/sample misalignment."""
    def __init__(self):
        super().__init__()
        self.input_rows = None
        self.edge_index = None

    def forward(self, values, edges, weights):
        self.input_rows = values.detach().clone()
        self.edge_index = edges.detach().clone()
        result = values.clone()
        result.index_add_(0, edges[1], values[edges[0]] * weights[:, None])
        return result


def tiny_tokenizer():
    token = StTokenizer.__new__(StTokenizer)
    nn.Module.__init__(token)
    token.device = torch.device("cpu")
    token.road_cnt, token.edge_cnt = 3, 2
    token.edges = torch.tensor([[0, 1], [1, 2]])
    token.edge_weight = torch.tensor([1., 2.])
    token.static_features = torch.zeros(4, 1, dtype=torch.long)
    token.static_origin_embedding = nn.ModuleList([nn.Embedding(1, 2)])
    nn.init.zeros_(token.static_origin_embedding[0].weight)
    token.static_embedding_layers = nn.ModuleList([nn.Identity(), TinyGraph(), nn.Identity()])
    # Every (road, slot) pair has a unique signature, including the PAD row.
    token.dynamic_features = torch.tensor(
        [[[100 * slot + 10 * road, 1000 * road + slot] for slot in range(3)]
         for road in range(4)], dtype=torch.float32)
    token.dynamic_embedding_layers = nn.ModuleList([nn.Identity(), TinyGraph(), nn.Identity()])
    token.cross_attention = CrossAttention(4)
    with torch.no_grad():
        token.cross_attention.W_q.zero_()
    token.time_embedding_layers = nn.ModuleList([nn.Identity()])
    token.final_mlp = nn.Identity()
    return token


class GraphBatchingTests(unittest.TestCase):
    def check_graph(self, tte):
        token = tiny_tokenizer()
        roads = torch.tensor([[0, 1, 2], [2, 0, 1]])
        slots = torch.tensor([[0, 0, 0], [2, 2, 2]])
        features = torch.zeros(2, 3, 6)
        mask = torch.ones(2, 3, dtype=torch.bool)
        if tte:
            model = BigCity4TTE.__new__(BigCity4TTE)
            nn.Module.__init__(model)
            model.tokenizer = token
            run = lambda r, s, f: model.tokenize_tte(r, s, f, mask)
        else:
            run = token
        with patch.multiple(args, seq_len=3, pre_dyna=False, static_ablation="none"), torch.no_grad():
            output = run(roads, slots, features)
            recorded = token.dynamic_embedding_layers[1].input_rows
            expected_rows = torch.cat([token.dynamic_features[:, 0],
                                       token.dynamic_features[:, 2]], dim=0)
            self.assertTrue(torch.equal(recorded, expected_rows))
            self.assertTrue(torch.equal(token.dynamic_embedding_layers[1].edge_index,
                                        torch.tensor([[0, 1, 4, 5], [1, 2, 5, 6]])))
            # Numerical behavior must be independent of batch ordering.
            reverse = run(roads.flip(0), slots.flip(0), features.flip(0)).flip(0)
            torch.testing.assert_close(output, reverse)
            # Changing the second sample cannot alter the first sample.
            changed_slots = slots.clone()
            changed_slots[1] = 1
            isolated = run(roads, changed_slots, features)
            torch.testing.assert_close(output[0], isolated[0])

    def test_next_hop_graph_rows_edges_and_sample_isolation(self):
        self.check_graph(False)

    def test_tte_graph_rows_edges_and_sample_isolation(self):
        self.check_graph(True)


class TimeAndDatasetTests(unittest.TestCase):
    def test_next_hop_integer_timestamp_boundaries(self):
        start = int(global_vars.start_time.timestamp())
        times = [start + 1799, start + 1800, start + 1801, start + 3600]
        data = pd.DataFrame({"path": [str([0, 1, 2, 3])], "tlist": [str(times)]})
        dataset = DatasetNextHop.__new__(DatasetNextHop)
        dataset.dataset_len = 1
        with patch.object(args, "seq_len", 4), patch(
            "data_provider.data_loader.file_loader.get_traj_data", return_value=data
        ):
            dataset.read_traj_data()
        self.assertEqual(dataset.traj_time_id_lists.dtype, torch.int32)
        self.assertEqual(dataset.traj_time_id_lists.tolist(), [[0, 1, 1, 1]])
        self.assertEqual(dataset.nexthop_labels.tolist(), [3])

    def build_tte(self, root, rows):
        path = root / "synthetic.csv"
        pd.DataFrame(rows).to_csv(path, sep=";", index=False)
        return build_tte_payload(
            trajectory_file=path, split="train", seq_len=4, road_count=4,
            start_timestamp=1538352000, interval_seconds=1800,
            time_slots_count=8, sample_rate=1., seed=2021)

    def test_tte_departure_only_padding_and_duration(self):
        start = 1538352000
        rows = [{"id": "synthetic-a", "path": "[0, 1, 2]",
                 "tlist": str([start + 1799, start + 1800, start + 1860]), "duration": 61},
                {"id": "synthetic-b", "path": "[2, 3]",
                 "tlist": str([start + 1800, start + 1890]), "duration": 90}]
        with tempfile.TemporaryDirectory() as folder:
            payload = self.build_tte(Path(folder), rows)
            validate_tte_payload(payload)
        self.assertEqual(payload["time_ids"].tolist(), [[0, 0, 0, 8], [1, 1, 8, 8]])
        self.assertEqual(payload["road_ids"].tolist(), [[0, 1, 2, 4], [2, 3, 4, 4]])
        self.assertEqual(payload["duration_seconds"].tolist(), [61., 90.])
        self.assertEqual(payload["valid_mask"].tolist(), [[True, True, True, False],
                                                        [True, True, False, False]])
        torch.testing.assert_close(payload["time_features"][0, 0],
                                   payload["time_features"][0, 2])
        self.assertTrue(torch.equal(payload["time_features"][~payload["valid_mask"]],
                                    torch.zeros(3, 6)))

    def test_tte_duration_mismatch_is_rejected(self):
        rows = [{"id": "synthetic", "path": "[0, 1]",
                 "tlist": "[1538352000,1538352060]", "duration": 59}]
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "duration mismatch"):
            self.build_tte(Path(folder), rows)

    def test_tte_nonmonotonic_timestamps_are_rejected(self):
        rows = [{"id": "synthetic", "path": "[0, 1]",
                 "tlist": "[1538352060,1538352000]", "duration": 60}]
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "not monotonic"):
            self.build_tte(Path(folder), rows)


class TTEPaddingTests(unittest.TestCase):
    def model(self):
        model = BigCity4TTE.__new__(BigCity4TTE)
        nn.Module.__init__(model)
        model.tokenizer = tiny_tokenizer()
        return model

    def test_padding_values_cannot_affect_real_attention_outputs(self):
        model = self.model()
        values = torch.tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.], [0., 0., 0., 0.]]])
        valid = torch.tensor([[True, True, False]])
        mutated = values.clone()
        mutated[0, 2] = 100000
        with torch.no_grad():
            first = model.masked_tte_cross_attention(values, valid)
            second = model.masked_tte_cross_attention(mutated, valid)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(first[0, 2], torch.zeros(4)))

    def test_all_padding_attention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one real road"):
            self.model().masked_tte_cross_attention(torch.zeros(1, 3, 4),
                                                   torch.zeros(1, 3, dtype=torch.bool))

    def test_reg_token_is_inserted_immediately_after_real_roads(self):
        tokens = torch.tensor([[[1.], [2.], [99.]], [[3.], [4.], [5.]]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        packed, lengths = BigCity4TTE.pack_tte_tokens(tokens, mask, torch.tensor([[[8.]], [[9.]]]))
        self.assertEqual(packed.squeeze(-1).tolist(), [[1., 2., 8., 99.], [3., 4., 5., 9.]])
        self.assertEqual(lengths.tolist(), [2, 3])

    def test_nonleft_aligned_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "left-aligned"):
            BigCity4TTE.pack_tte_tokens(torch.zeros(1, 3, 2),
                                       torch.tensor([[True, False, True]]), torch.zeros(1, 1, 2))


class SemanticProjectionTests(unittest.TestCase):
    def make_model(self, mode, dimension):
        def fake_base(model, device):
            nn.Module.__init__(model)
            model.device = device
            model.tokenizer = nn.Linear(1, 1)
            model.tokenizer.road_cnt = 3
            model.special_token = nn.Embedding(2, 768)
            model.backbone = nn.Module()
            model.backbone.tasks_mlp = nn.ModuleDict()
        features = torch.zeros(3, dimension)
        features[1, 0], features[2, 1] = 3., 4.
        payload = {"semantic_features": features, "semantic_mask": torch.tensor([False, True, True])}
        tokenizer = SimpleNamespace(encode=lambda *a, **k: torch.tensor([[1, 2]]))
        cls = BigCity4TTE if mode.startswith("tte_") else BigCity4FineTune
        with (patch.multiple(args, semantic_mode=mode, d_model=768, seed=2021),
              patch("models.bigcity4finetune.BigCity.__init__", new=fake_base),
              patch("models.bigcity4finetune.torch.load", side_effect=[{}, payload]),
              patch("models.thesis_semantics._prepare_pretrained_state_dict", return_value=({}, False)),
              patch.object(BigCity4FineTune, "load_state_dict",
                           return_value=SimpleNamespace(missing_keys=[], unexpected_keys=[])),
              patch("models.bigcity4finetune.GPT2Tokenizer.from_pretrained", return_value=tokenizer)):
            return cls(torch.device("cpu"), checkpoint="synthetic-not-read.pth")

    def test_actual_constructor_projection_dimensions_zero_init_and_normalization(self):
        for mode, dimension in [("poi_tfidf", 204), ("poi_qwen_tfidf", 1024),
                                ("tte_structured", 10), ("tte_qwen", 1024)]:
            with self.subTest(mode=mode):
                model = self.make_model(mode, dimension)
                self.assertEqual(tuple(model.semantic_projection.weight.shape), (768, dimension))
                self.assertEqual(int(torch.count_nonzero(model.semantic_projection.weight)), 0)
                self.assertEqual(model.semantic_mask.tolist(), [False, True, True])
                torch.testing.assert_close(torch.linalg.vector_norm(model.semantic_features, dim=1),
                                           torch.tensor([0., 1., 1.]))

    def test_tte_pad_id_never_indexes_semantics_and_zero_roads_stay_zero(self):
        model = self.make_model("tte_structured", 10)
        with torch.no_grad():
            model.semantic_projection.weight.fill_(1.)
            base = torch.zeros(1, 3, 768)
            output = model.apply_tte_semantic_enrichment(
                base, torch.tensor([[1, 0, 3]]), torch.tensor([[True, True, False]]))
        self.assertTrue(torch.equal(output[0, 0], torch.ones(768)))
        self.assertTrue(torch.equal(output[0, 1:], torch.zeros(2, 768)))

    def test_next_hop_semantic_gate(self):
        model = self.make_model("poi_qwen_tfidf", 1024)
        with torch.no_grad():
            model.semantic_projection.weight.fill_(2.)
            output = model.apply_semantic_enrichment(torch.zeros(1, 2, 768),
                                                     torch.tensor([[0, 1]]))
        self.assertTrue(torch.equal(output[0, 0], torch.zeros(768)))
        self.assertTrue(torch.equal(output[0, 1], torch.full((768,), 2.)))

    def test_tte_head_initialization_is_identical_across_semantic_modes(self):
        first = self.make_model("tte_structured", 10)
        second = self.make_model("tte_qwen", 1024)
        for key, tensor in first.backbone.tasks_mlp["tte_reg"].state_dict().items():
            self.assertTrue(torch.equal(tensor, second.backbone.tasks_mlp["tte_reg"].state_dict()[key]))

    def test_baseline_semantics_return_same_tensor(self):
        for cls, method in [(BigCity4FineTune, "apply_semantic_enrichment"),
                            (BigCity4TTE, "apply_tte_semantic_enrichment")]:
            model = cls.__new__(cls)
            nn.Module.__init__(model)
            model.semantic_mode = "none"
            values, roads = torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long)
            extra = [torch.ones(1, 2, dtype=torch.bool)] if cls is BigCity4TTE else []
            self.assertIs(getattr(model, method)(values, roads, *extra), values)


class TTEMetricsAndSelectionTests(unittest.TestCase):
    def test_selection_is_strict_and_normalization_round_trips(self):
        self.assertTrue(tte_validation_improved(9., 10.))
        self.assertFalse(tte_validation_improved(10., 10.))
        self.assertFalse(tte_validation_improved(11., 10.))
        values = torch.tensor([30., 60., 90.])
        torch.testing.assert_close(denormalize_prediction(normalize_target(values, mean_seconds=60., std_seconds=30.),
                                                               mean_seconds=60., std_seconds=30.),
                                   values)

    def test_metrics_use_seconds_and_streaming_matches_one_batch(self):
        predictions, targets = torch.tensor([12., 18., 40.]), torch.tensor([10., 20., 30.])
        one, stream = TTEMetricAccumulator(), TTEMetricAccumulator()
        one.update(predictions, targets)
        stream.update(predictions[:1], targets[:1])
        stream.update(predictions[1:], targets[1:])
        expected = one.compute()
        self.assertEqual(stream.compute(), expected)
        self.assertAlmostEqual(expected["mae_seconds"], 14 / 3)
        self.assertAlmostEqual(expected["rmse_seconds"], math.sqrt(108 / 3))
        self.assertAlmostEqual(expected["mape_percent"], (0.2 + 0.1 + 1 / 3) / 3 * 100)

    def test_metrics_reject_nonpositive_targets_and_nonfinite_predictions(self):
        for prediction, target in [([1.], [0.]), ([float("nan")], [1.])]:
            with self.assertRaises(ValueError):
                TTEMetricAccumulator().update(prediction, target)


if __name__ == "__main__":
    unittest.main()

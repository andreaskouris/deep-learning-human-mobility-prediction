import math
import unittest

import torch

from utils.next_hop_metrics import (
    NextHopMetricAccumulator,
)


class TestNextHopMetricAccumulator(unittest.TestCase):

    def test_known_ranks(self):
        # Four samples.
        #
        # True-label ranks:
        # sample 1 -> rank 1
        # sample 2 -> rank 2
        # sample 3 -> rank 5
        # sample 4 -> outside top 5

        logits = torch.tensor(
            [
                [10.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
                [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
                [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            ],
            dtype=torch.float32,
        )

        labels = torch.tensor(
            [
                0,  # rank 1
                1,  # rank 2
                4,  # rank 5
                5,  # rank 6 -> outside top 5
            ],
            dtype=torch.long,
        )

        metrics = NextHopMetricAccumulator(k=5)
        metrics.update(logits, labels)

        result = metrics.compute()

        expected_accuracy = 1.0 / 4.0

        expected_mrr = (
            1.0
            + 1.0 / 2.0
            + 1.0 / 5.0
            + 0.0
        ) / 4.0

        expected_ndcg = (
            1.0
            + 1.0 / math.log2(3.0)
            + 1.0 / math.log2(6.0)
            + 0.0
        ) / 4.0

        self.assertAlmostEqual(
            result["accuracy"],
            expected_accuracy,
            places=7,
        )

        self.assertAlmostEqual(
            result["mrr@5"],
            expected_mrr,
            places=7,
        )

        self.assertAlmostEqual(
            result["ndcg@5"],
            expected_ndcg,
            places=7,
        )

        self.assertEqual(
            result["num_samples"],
            4,
        )

    def test_multiple_updates_are_aggregated_by_sample(self):
        metrics = NextHopMetricAccumulator(k=5)

        logits_batch_1 = torch.tensor(
            [
                [5.0, 1.0, 0.0],
                [5.0, 4.0, 0.0],
            ]
        )

        labels_batch_1 = torch.tensor(
            [0, 1]
        )

        logits_batch_2 = torch.tensor(
            [
                [1.0, 2.0, 3.0],
            ]
        )

        labels_batch_2 = torch.tensor(
            [2]
        )

        metrics.update(
            logits_batch_1,
            labels_batch_1,
        )

        metrics.update(
            logits_batch_2,
            labels_batch_2,
        )

        result = metrics.compute()

        self.assertEqual(
            result["num_samples"],
            3,
        )

        self.assertAlmostEqual(
            result["accuracy"],
            2.0 / 3.0,
            places=7,
        )

        expected_mrr = (
            1.0
            + 1.0 / 2.0
            + 1.0
        ) / 3.0

        self.assertAlmostEqual(
            result["mrr@5"],
            expected_mrr,
            places=7,
        )

    def test_k_larger_than_number_of_classes(self):
        metrics = NextHopMetricAccumulator(k=5)

        logits = torch.tensor(
            [
                [3.0, 2.0, 1.0],
            ]
        )

        labels = torch.tensor(
            [2]
        )

        metrics.update(logits, labels)

        result = metrics.compute()

        self.assertAlmostEqual(
            result["mrr@5"],
            1.0 / 3.0,
            places=7,
        )

    def test_invalid_shapes_raise(self):
        metrics = NextHopMetricAccumulator(k=5)

        with self.assertRaises(ValueError):
            metrics.update(
                torch.zeros(2, 3, 4),
                torch.zeros(2, dtype=torch.long),
            )

        with self.assertRaises(ValueError):
            metrics.update(
                torch.zeros(2, 3),
                torch.zeros(2, 1, dtype=torch.long),
            )

        with self.assertRaises(ValueError):
            metrics.update(
                torch.zeros(2, 3),
                torch.zeros(3, dtype=torch.long),
            )

    def test_compute_before_update_raises(self):
        metrics = NextHopMetricAccumulator(k=5)

        with self.assertRaises(RuntimeError):
            metrics.compute()


if __name__ == "__main__":
    unittest.main()
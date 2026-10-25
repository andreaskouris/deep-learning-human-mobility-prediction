import math

import torch


class NextHopMetricAccumulator:
    """Streaming metrics for next-hop road-segment prediction."""

    def __init__(self, k=5):
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        self.k = k
        self.reset()

    def reset(self):
        self.num_samples = 0
        self.correct_top1 = 0
        self.reciprocal_rank_sum = 0.0
        self.ndcg_sum = 0.0

    @torch.no_grad()
    def update(self, logits, labels):
        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape (batch_size, num_classes)."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape (batch_size,)."
            )

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "logits and labels must contain the same number of samples."
            )

        if logits.shape[0] == 0:
            return

        num_classes = logits.shape[1]
        effective_k = min(self.k, num_classes)

        topk_indices = torch.topk(
            logits,
            k=effective_k,
            dim=1,
        ).indices

        top1_predictions = topk_indices[:, 0]

        self.correct_top1 += (
            top1_predictions == labels
        ).sum().item()

        matches = (
            topk_indices
            == labels.unsqueeze(1)
        )

        matched_rows = matches.any(dim=1)

        if matched_rows.any():
            matched_positions = (
                matches[matched_rows]
                .float()
                .argmax(dim=1)
                + 1
            )

            reciprocal_ranks = (
                1.0
                / matched_positions.float()
            )

            ndcg_values = (
                1.0
                / torch.log2(
                    matched_positions.float()
                    + 1.0
                )
            )

            self.reciprocal_rank_sum += (
                reciprocal_ranks.sum().item()
            )

            self.ndcg_sum += (
                ndcg_values.sum().item()
            )

        self.num_samples += labels.shape[0]

    def compute(self):
        if self.num_samples == 0:
            raise RuntimeError(
                "No samples have been added to the metric accumulator."
            )

        return {
            "accuracy": (
                self.correct_top1
                / self.num_samples
            ),
            f"mrr@{self.k}": (
                self.reciprocal_rank_sum
                / self.num_samples
            ),
            f"ndcg@{self.k}": (
                self.ndcg_sum
                / self.num_samples
            ),
            "num_samples": self.num_samples,
        }
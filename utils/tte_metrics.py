from __future__ import annotations

import math

import torch


class TTEMetricAccumulator:
    """Accumulate MAE/RMSE in seconds and MAPE in percent.

    Predictions are evaluated without clipping; targets must be positive.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sample_count = 0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.absolute_percentage_error_sum = 0.0

    def update(
        self,
        prediction_seconds,
        target_seconds,
    ):
        prediction_seconds = (
            torch.as_tensor(
                prediction_seconds,
                dtype=torch.float64,
            )
            .detach()
            .cpu()
            .view(-1)
        )

        target_seconds = (
            torch.as_tensor(
                target_seconds,
                dtype=torch.float64,
            )
            .detach()
            .cpu()
            .view(-1)
        )

        if (
            prediction_seconds.shape
            != target_seconds.shape
        ):
            raise ValueError(
                "Prediction and target shapes "
                "must match."
            )

        if prediction_seconds.numel() == 0:
            return

        if not torch.isfinite(
            prediction_seconds
        ).all():
            raise ValueError(
                "Non-finite TTE predictions."
            )

        if not torch.isfinite(
            target_seconds
        ).all():
            raise ValueError(
                "Non-finite TTE targets."
            )

        if torch.any(
            target_seconds <= 0
        ):
            raise ValueError(
                "MAPE requires strictly "
                "positive TTE targets."
            )

        error = (
            prediction_seconds
            - target_seconds
        )

        absolute_error = error.abs()

        self.absolute_error_sum += float(
            absolute_error.sum().item()
        )

        self.squared_error_sum += float(
            error.square().sum().item()
        )

        self.absolute_percentage_error_sum += (
            float(
                (
                    absolute_error
                    / target_seconds
                )
                .sum()
                .item()
            )
        )

        self.sample_count += int(
            prediction_seconds.numel()
        )

    def compute(self):
        if self.sample_count == 0:
            raise RuntimeError(
                "Cannot compute TTE metrics "
                "without samples."
            )

        mae = (
            self.absolute_error_sum
            / self.sample_count
        )

        mse = (
            self.squared_error_sum
            / self.sample_count
        )

        rmse = math.sqrt(mse)

        mape = (
            self.absolute_percentage_error_sum
            / self.sample_count
            * 100.0
        )

        return {
            "mae_seconds": mae,
            "rmse_seconds": rmse,
            "mape_percent": mape,
            "sample_count": (
                self.sample_count
            ),
        }

import sys
import unittest
from unittest.mock import patch


# BIGCity parses command-line arguments during module import.
# Hide unittest's own CLI arguments while importing the runner.
with patch.object(
    sys,
    "argv",
    ["train_next_hop.py"],
):
    from train_next_hop import (
        early_stopping_reached,
        load_test_dataset_after_model_selection,
        validation_improved,
    )


class TrainNextHopSelectionTests(
    unittest.TestCase
):

    def test_higher_validation_accuracy_improves(self):
        self.assertTrue(
            validation_improved(
                0.81,
                0.80,
            )
        )

    def test_equal_validation_accuracy_does_not_improve(self):
        self.assertFalse(
            validation_improved(
                0.80,
                0.80,
            )
        )

    def test_lower_validation_accuracy_does_not_improve(self):
        self.assertFalse(
            validation_improved(
                0.79,
                0.80,
            )
        )

    def test_first_validation_result_is_improvement(self):
        self.assertTrue(
            validation_improved(
                0.10,
                float("-inf"),
            )
        )

    def test_early_stopping_before_patience_is_false(self):
        self.assertFalse(
            early_stopping_reached(
                2,
                3,
            )
        )

    def test_early_stopping_at_patience_is_true(self):
        self.assertTrue(
            early_stopping_reached(
                3,
                3,
            )
        )

    def test_early_stopping_above_patience_is_true(self):
        self.assertTrue(
            early_stopping_reached(
                4,
                3,
            )
        )

    def test_invalid_patience_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            early_stopping_reached(
                0,
                0,
            )


class TrainNextHopTestIsolationTests(
    unittest.TestCase
):

    @patch(
        "train_next_hop.CachedNextHopDataset"
    )
    def test_test_dataset_rejected_before_model_selection(
        self,
        dataset_class,
    ):
        with self.assertRaises(
            RuntimeError
        ):
            load_test_dataset_after_model_selection(
                "test_cache.pth",
                best_epoch=None,
                checkpoint_reloaded=False,
            )

        dataset_class.assert_not_called()

    @patch(
        "train_next_hop.CachedNextHopDataset"
    )
    def test_test_dataset_rejected_before_checkpoint_reload(
        self,
        dataset_class,
    ):
        with self.assertRaises(
            RuntimeError
        ):
            load_test_dataset_after_model_selection(
                "test_cache.pth",
                best_epoch=4,
                checkpoint_reloaded=False,
            )

        dataset_class.assert_not_called()

    @patch(
        "train_next_hop.CachedNextHopDataset"
    )
    def test_test_dataset_loaded_after_model_selection(
        self,
        dataset_class,
    ):
        expected_dataset = object()

        dataset_class.return_value = (
            expected_dataset
        )

        result = (
            load_test_dataset_after_model_selection(
                "test_cache.pth",
                best_epoch=4,
                checkpoint_reloaded=True,
            )
        )

        dataset_class.assert_called_once_with(
            "test_cache.pth"
        )

        self.assertIs(
            result,
            expected_dataset,
        )


if __name__ == "__main__":
    unittest.main()
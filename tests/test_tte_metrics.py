"""TTE metric contracts independent of BIGCity model source."""
import math
import unittest

from utils.tte_metrics import TTEMetricAccumulator


class TTEMetricTests(unittest.TestCase):
    def test_negative_predictions_are_not_clipped(self):
        metrics=TTEMetricAccumulator();metrics.update([-5.,15.],[5.,10.]);result=metrics.compute()
        self.assertEqual(result['mae_seconds'],7.5)
        self.assertAlmostEqual(result['rmse_seconds'],math.sqrt(62.5))
        self.assertEqual(result['mape_percent'],125.)

    def test_streaming_results_are_independent_of_batch_boundaries(self):
        together=TTEMetricAccumulator();together.update([105.,190.,-2.],[100.,200.,3.])
        split=TTEMetricAccumulator();split.update([105.],[100.]);split.update([190.,-2.],[200.,3.])
        self.assertEqual(together.compute(),split.compute())
        self.assertEqual(split.compute()['sample_count'],3)

    def test_invalid_percentage_targets_and_nonfinite_predictions_are_rejected(self):
        for prediction,target in [([1.],[0.]),([1.],[-1.]),([float('nan')],[1.])]:
            with self.subTest(target=target),self.assertRaises(ValueError):
                TTEMetricAccumulator().update(prediction,target)


if __name__=='__main__':
    unittest.main()

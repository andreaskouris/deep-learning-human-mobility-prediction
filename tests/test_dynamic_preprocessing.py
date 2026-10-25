import unittest
import torch
from data_provider.dynamic_preprocessing import preprocess_controlled

class DynamicPreprocessingTests(unittest.TestCase):

    def test_controlled_preserves_observed_zeros(self) -> None:
        source = torch.tensor([[0.0, 2.0], [2.0, 4.0], [0.0, 0.0]], dtype=torch.float32)
        observed_roads = torch.tensor([True, True, False])
        result = preprocess_controlled(source, observed_roads)
        expected = torch.tensor([[0.0, 0.5], [0.5, 1.0], [0.25, 0.75]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)
        self.assertEqual(float(result[0, 0]), 0.0)

    def test_controlled_rejects_empty_observations(self) -> None:
        source = torch.zeros((3, 2), dtype=torch.float32)
        with self.assertRaises(ValueError):
            preprocess_controlled(source, torch.tensor([False, False, False]))
if __name__ == '__main__':
    unittest.main()

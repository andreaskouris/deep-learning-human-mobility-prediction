import tempfile
import unittest
from pathlib import Path
import pandas as pd
import torch
from scripts.build_dynamic_cache import build_dynamic_cache

class BuildDynamicCacheTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temporary_directory.name)
        self.city_root = self.dataset_root / 'xa'
        roadmap_root = self.city_root / 'roadmap_xa'
        roadmap_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'feature': [0, 1, 2, 3]}).to_csv(roadmap_root / 'road_features_xa.csv', index=False)
        pd.DataFrame({'dyna_id': [0, 1, 2, 3, 4, 5], 'entity_id': [0, 0, 2, 2, 3, 3], 'traffic_speed': [0.0, 2.0, 2.0, 4.0, 4.0, 6.0]}).to_csv(self.city_root / 'xa.dyna', index=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_controlled_cache(self) -> None:
        cache_path = build_dynamic_cache(dataset_root=self.dataset_root, city='xa', dynamic_mode='controlled', time_slots=2, chunk_size=2)
        cache = torch.load(cache_path, weights_only=True)
        expected = torch.tensor([[0.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0]], dtype=torch.float32)
        torch.testing.assert_close(cache['dynamic_features'], expected)
        self.assertEqual(cache['time_slots_cnt'], 2)

    def test_refuses_accidental_overwrite(self) -> None:
        build_dynamic_cache(dataset_root=self.dataset_root, city='xa', dynamic_mode='controlled', time_slots=2, chunk_size=2)
        with self.assertRaises(FileExistsError):
            build_dynamic_cache(dataset_root=self.dataset_root, city='xa', dynamic_mode='controlled', time_slots=2, chunk_size=2)

    def test_noncanonical_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'controlled'):
            build_dynamic_cache(self.dataset_root, 'xa', 'original', 2, chunk_size=2)
        self.assertFalse((self.city_root / 'cached_xa_dynamic_original_v1.pth').exists())
if __name__ == '__main__':
    unittest.main()

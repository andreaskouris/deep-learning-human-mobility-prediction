import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import torch
_original_argv = sys.argv[:]
try:
    sys.argv[:] = [sys.argv[0]]
    from config.args_config import args
    from config import global_vars
    from data_provider.file_loader import FileLoader
finally:
    sys.argv[:] = _original_argv

class FileLoaderDynamicModeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.static_file = self.root / 'static.csv'
        self.dynamic_file = self.root / 'dynamic.csv'
        pd.DataFrame({'feature': [0, 1, 2, 3]}).to_csv(self.static_file, index=False)
        pd.DataFrame({'dyna_id': [0, 1, 2, 3, 4, 5], 'entity_id': [0, 0, 2, 2, 3, 3], 'traffic_speed': [0.0, 2.0, 2.0, 4.0, 4.0, 6.0]}).to_csv(self.dynamic_file, index=False)

    def tearDown(self) -> None:
        FileLoader._instance = None
        self.temporary_directory.cleanup()

    def load_mode(self, mode: str) -> tuple[torch.Tensor, dict]:
        static_cache = self.root / f'static_{mode}.pth'
        dynamic_cache = self.root / f'dynamic_{mode}.pth'
        FileLoader._instance = None
        start_time = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 1, 0, 30, tzinfo=timezone.utc)
        with patch.object(global_vars, 'road_static_file', str(self.static_file)), patch.object(global_vars, 'road_static_tensor_file', str(static_cache)), patch.object(global_vars, 'road_dynamic_file', str(self.dynamic_file)), patch.object(global_vars, 'road_dynamic_tensor_file', str(dynamic_cache)), patch.object(global_vars, 'start_time', start_time), patch.object(global_vars, 'end_time', end_time), patch.object(global_vars, 'interval', 1800), patch.object(args, 'dynamic_mode', mode):
            loader = FileLoader()
            loader.load_road_features_file()
            result = loader.dynamic_features.clone()
        cache = torch.load(dynamic_cache, weights_only=True)
        return (result, cache)

    def test_controlled_mode_preserves_zero(self) -> None:
        result, cache = self.load_mode('controlled')
        expected = torch.tensor([[0.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(cache['dynamic_features'], expected)
        self.assertEqual(cache['time_slots_cnt'], 2)
        self.assertEqual(float(result[0, 0]), 0.0)
if __name__ == '__main__':
    unittest.main()

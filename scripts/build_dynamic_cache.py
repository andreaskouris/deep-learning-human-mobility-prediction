from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import pandas as pd
import torch
from data_provider.dynamic_preprocessing import preprocess_controlled

def calculate_sha256(path: Path, chunk_size: int=1024 * 1024) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()

def load_raw_dynamic_features(static_file: Path, dynamic_file: Path, time_slots: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Read dynamic observations into a road-by-time tensor."""
    if time_slots <= 0:
        raise ValueError('Time-slot count must be positive.')
    if chunk_size <= 0:
        raise ValueError('Chunk size must be positive.')
    if not static_file.is_file():
        raise FileNotFoundError(f'Static feature file not found: {static_file}')
    if not dynamic_file.is_file():
        raise FileNotFoundError(f'Dynamic feature file not found: {dynamic_file}')
    road_count = len(pd.read_csv(static_file))
    if road_count <= 0:
        raise ValueError('Static feature file contains no roads.')
    dynamic_features = torch.zeros((road_count, time_slots), dtype=torch.float32)
    observed_roads = torch.zeros(road_count, dtype=torch.bool)
    total_rows = 0
    for chunk_number, chunk in enumerate(pd.read_csv(dynamic_file, usecols=['dyna_id', 'entity_id', 'traffic_speed'], chunksize=chunk_size), start=1):
        dyna_ids = torch.from_numpy(chunk['dyna_id'].to_numpy(dtype='int64', copy=True))
        road_ids = torch.from_numpy(chunk['entity_id'].to_numpy(dtype='int64', copy=True))
        traffic_speeds = torch.from_numpy(chunk['traffic_speed'].to_numpy(dtype='float32', copy=True))
        if bool((dyna_ids < 0).any()):
            raise ValueError('Negative dynamic ID detected.')
        if bool((road_ids < 0).any() or (road_ids >= road_count).any()):
            raise ValueError('Out-of-range road ID detected.')
        if not bool(torch.isfinite(traffic_speeds).all()):
            raise ValueError('Non-finite traffic speed detected.')
        time_ids = torch.remainder(dyna_ids, time_slots)
        dynamic_features[road_ids, time_ids] = traffic_speeds
        observed_roads[torch.unique(road_ids)] = True
        total_rows += len(chunk)
        if chunk_number % 5 == 0:
            print(f'  chunks={chunk_number}, rows={total_rows:,}, observed_roads={int(observed_roads.sum()):,}')
    if total_rows == 0:
        raise ValueError('Dynamic feature file contains no rows.')
    return (dynamic_features, observed_roads, total_rows)

def build_dynamic_cache(dataset_root: Path, city: str, dynamic_mode: str, time_slots: int, chunk_size: int=500000, force: bool=False) -> Path:
    """Build and verify one BIGCity dynamic-feature cache."""
    if dynamic_mode not in {'controlled'}:
        raise ValueError('The canonical builder requires controlled dynamic mode.')
    dataset_root = Path(dataset_root).resolve()
    city_root = dataset_root / city
    static_file = city_root / f'roadmap_{city}' / f'road_features_{city}.csv'
    dynamic_file = city_root / f'{city}.dyna'
    cache_file = city_root / f'cached_{city}_dynamic_{dynamic_mode}_v1.pth'
    if cache_file.exists() and (not force):
        raise FileExistsError(f'Dynamic cache already exists. Use --force only for an intentional rebuild: {cache_file}')
    print('=== DYNAMIC CACHE BUILD ===')
    print('Mode:', dynamic_mode)
    print('City:', city)
    print('Static source:', static_file)
    print('Dynamic source:', dynamic_file)
    print('Target cache:', cache_file)
    print('Time slots:', time_slots)
    print('Chunk size:', f'{chunk_size:,}')
    print()
    print('Reading source data...')
    raw_features, observed_roads, total_rows = load_raw_dynamic_features(static_file=static_file, dynamic_file=dynamic_file, time_slots=time_slots, chunk_size=chunk_size)
    print()
    print('Applying preprocessing...')
    processed_features = preprocess_controlled(raw_features, observed_roads)
    expected_shape = (raw_features.shape[0], time_slots)
    if tuple(processed_features.shape) != expected_shape:
        raise RuntimeError(f'Unexpected processed tensor shape: {tuple(processed_features.shape)}')
    if not bool(torch.isfinite(processed_features).all()):
        raise ValueError('Processed features contain non-finite values.')
    minimum = float(processed_features.min())
    maximum = float(processed_features.max())
    tolerance = 1e-07
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(f'Processed values are outside [0, 1]: min={minimum}, max={maximum}')
    temporary_file = cache_file.with_suffix(cache_file.suffix + '.tmp')
    if temporary_file.exists():
        temporary_file.unlink()
    try:
        torch.save({'dynamic_features': processed_features, 'time_slots_cnt': time_slots}, temporary_file)
        os.replace(temporary_file, cache_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()
    reloaded = torch.load(cache_file, weights_only=True)
    if set(reloaded) != {'dynamic_features', 'time_slots_cnt'}:
        raise RuntimeError('Reloaded cache has unexpected keys.')
    if reloaded['time_slots_cnt'] != time_slots:
        raise RuntimeError('Reloaded time-slot count does not match.')
    if not torch.equal(reloaded['dynamic_features'], processed_features):
        raise RuntimeError('Reloaded tensor differs from saved tensor.')
    cache_hash = calculate_sha256(cache_file)
    print()
    print('=== BUILD RESULTS ===')
    print('Rows read:', f'{total_rows:,}')
    print('Observed roads:', f'{int(observed_roads.sum()):,}')
    print('Missing roads:', f'{int((~observed_roads).sum()):,}')
    print('Tensor shape:', tuple(processed_features.shape))
    print('Tensor dtype:', processed_features.dtype)
    print('Finite values:', bool(torch.isfinite(processed_features).all()))
    print('Minimum:', minimum)
    print('Maximum:', maximum)
    print('Zero values:', f'{int((processed_features == 0).sum()):,}')
    print('Cache size:', f'{cache_file.stat().st_size:,} bytes')
    print('Cache SHA-256:', cache_hash)
    print('Reload verification: PASS')
    print('Cache created:', cache_file)
    return cache_file

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build only the BIGCity dynamic-feature cache.')
    parser.add_argument('--dataset-root', type=Path, default=Path('../dataset'))
    parser.add_argument('--city', type=str, default='xa')
    parser.add_argument('--dynamic-mode', choices=['controlled'], required=True)
    parser.add_argument('--time-slots', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, default=500000)
    parser.add_argument('--force', action='store_true', help='Replace an existing cache intentionally.')
    return parser.parse_args()

def main() -> None:
    arguments = parse_arguments()
    build_dynamic_cache(dataset_root=arguments.dataset_root, city=arguments.city, dynamic_mode=arguments.dynamic_mode, time_slots=arguments.time_slots, chunk_size=arguments.chunk_size, force=arguments.force)
if __name__ == '__main__':
    main()

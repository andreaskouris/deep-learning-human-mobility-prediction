from __future__ import annotations
import csv
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from config.args_config import args
from config import global_vars
from data_provider.tte_dataset import CachedTTEDataset
from models.bigcity4tte import BigCity4TTE
from utils.tte_metrics import TTEMetricAccumulator
from train_next_hop import load_model_support_data, resolve_device, resolve_pretrained_checkpoint, save_checkpoint_atomic
TASK_NAME = 'tte'
PRIMARY_METRIC = 'mae_seconds'
TTE_CACHE_VERSION = 'v1'
LOCKED_TRAIN_MEAN_SECONDS = 485.50155132816224
LOCKED_TRAIN_STD_SECONDS = 350.44401804624164

def set_global_seed(seed):
    """
    Seed Python, NumPy and PyTorch.

    This controls the experimental RNG state
    across the four TTE semantic conditions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tte_validation_improved(current_mae, best_mae):
    """
    Lower validation MAE is better.

    Equal MAE does not replace the existing
    best checkpoint.
    """
    return current_mae < best_mae

def build_tte_cache_path(split):
    """
    Resolve one canonical TTE cache.
    """
    if split not in {'train', 'validation', 'test'}:
        raise ValueError(f'Unsupported TTE split: {split}')
    sample_rate_tag = global_vars.format_cache_value(args.sample_rate)
    seed_tag = global_vars.format_cache_value(args.seed)
    filename = f'cached_tte_dataset_{split}_seq{args.seq_len}_sr{sample_rate_tag}_seed{seed_tag}_{TTE_CACHE_VERSION}.pth'
    return os.path.join(global_vars.city_root_path, filename)

def validate_tte_cache_metadata(dataset, *, split, model_metadata):
    metadata = dataset.metadata
    expected = {'task': 'tte', 'split': split, 'seq_len': args.seq_len, 'road_count': int(model_metadata['road_cnt']), 'time_slots_count': int(model_metadata['time_slots_cnt']), 'sample_rate': float(args.sample_rate), 'seed': int(args.seed), 'temporal_input': 'departure_only', 'target': 'trip_duration_seconds'}
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise RuntimeError(f'{split} TTE cache metadata mismatch for {key}: expected {expected_value!r}, found {actual_value!r}.')

def compute_train_target_statistics(train_dataset):
    """
    Compute z-normalization parameters from
    the training target only.
    """
    y = train_dataset.duration_seconds.to(torch.float64)
    if y.numel() == 0:
        raise RuntimeError('Training TTE cache is empty.')
    if not torch.isfinite(y).all():
        raise RuntimeError('Training TTE targets contain non-finite values.')
    if torch.any(y <= 0):
        raise RuntimeError('Training TTE targets must be strictly positive.')
    mean = float(y.mean().item())
    std = float(y.std(unbiased=False).item())
    if not np.isfinite(mean):
        raise RuntimeError('Non-finite TTE train mean.')
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError('Invalid TTE train standard deviation.')
    if abs(mean - LOCKED_TRAIN_MEAN_SECONDS) > 1e-10:
        raise RuntimeError('Training target mean differs from the independently verified locked value.')
    if abs(std - LOCKED_TRAIN_STD_SECONDS) > 1e-10:
        raise RuntimeError('Training target std differs from the independently verified locked value.')
    return (mean, std)

def normalize_target(duration_seconds, *, mean_seconds, std_seconds):
    return (duration_seconds - mean_seconds) / std_seconds

def denormalize_prediction(prediction_normalized, *, mean_seconds, std_seconds):
    return prediction_normalized * std_seconds + mean_seconds

def create_dataloader(dataset, *, shuffle, device, generator=None):
    """
    Match the controlled next-hop loader:
    single process, optional deterministic
    shuffle generator.
    """
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == 'cuda', generator=generator)

def move_batch_to_device(batch, device):
    road_ids, time_ids, time_features, valid_mask, lengths, duration_seconds = batch
    non_blocking = device.type == 'cuda'
    return (road_ids.to(device, non_blocking=non_blocking), time_ids.to(device, non_blocking=non_blocking), time_features.to(device, non_blocking=non_blocking), valid_mask.to(device, non_blocking=non_blocking), lengths.to(device, non_blocking=non_blocking), duration_seconds.to(device, non_blocking=non_blocking))

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, *, mean_seconds, std_seconds):
    """
    Run one complete TTE training epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    start_time = time.perf_counter()
    progress_bar = tqdm(dataloader, total=len(dataloader), unit='batch', desc=f'Epoch {epoch}/{args.train_epochs} - train')
    for batch in progress_bar:
        road_ids, time_ids, time_features, valid_mask, _lengths, duration_seconds = move_batch_to_device(batch, device)
        target_normalized = normalize_target(duration_seconds, mean_seconds=mean_seconds, std_seconds=std_seconds)
        if not torch.isfinite(target_normalized).all():
            raise RuntimeError('Non-finite normalized training target detected.')
        optimizer.zero_grad(set_to_none=True)
        loss, _prediction_normalized = model.forward_tte(road_ids, time_ids, time_features, valid_mask, target_normalized)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite TTE training loss detected.')
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_size = duration_seconds.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress_bar.set_postfix(loss=f'{loss.item():.5f}', lr=f"{optimizer.param_groups[0]['lr']:.3e}")
    elapsed = time.perf_counter() - start_time
    if total_samples == 0:
        raise RuntimeError('TTE training DataLoader produced no samples.')
    return {'loss': total_loss / total_samples, 'samples': total_samples, 'seconds': elapsed, 'batches_per_second': len(dataloader) / elapsed if elapsed > 0 else 0.0}

@torch.no_grad()
def evaluate(model, dataloader, device, split_name, *, mean_seconds, std_seconds):
    """
    Evaluate normalized SmoothL1 plus
    MAE/RMSE/MAPE in physical seconds.
    """
    model.eval()
    metric_accumulator = TTEMetricAccumulator()
    total_loss = 0.0
    total_samples = 0
    start_time = time.perf_counter()
    progress_bar = tqdm(dataloader, total=len(dataloader), unit='batch', desc=split_name)
    for batch in progress_bar:
        road_ids, time_ids, time_features, valid_mask, _lengths, duration_seconds = move_batch_to_device(batch, device)
        target_normalized = normalize_target(duration_seconds, mean_seconds=mean_seconds, std_seconds=std_seconds)
        loss, prediction_normalized = model.forward_tte(road_ids, time_ids, time_features, valid_mask, target_normalized)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite {split_name} TTE loss detected.')
        if not torch.isfinite(prediction_normalized).all():
            raise RuntimeError(f'Non-finite {split_name} TTE prediction detected.')
        prediction_seconds = denormalize_prediction(prediction_normalized, mean_seconds=mean_seconds, std_seconds=std_seconds)
        if not torch.isfinite(prediction_seconds).all():
            raise RuntimeError(f'Non-finite {split_name} prediction in seconds.')
        batch_size = duration_seconds.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        metric_accumulator.update(prediction_seconds, duration_seconds)
    if total_samples == 0:
        raise RuntimeError(f'{split_name} TTE DataLoader produced no samples.')
    metrics = metric_accumulator.compute()
    if metrics['sample_count'] != total_samples:
        raise RuntimeError(f'{split_name} metric sample count mismatch.')
    elapsed = time.perf_counter() - start_time
    metrics['loss'] = total_loss / total_samples
    metrics['seconds'] = elapsed
    metrics['batches_per_second'] = len(dataloader) / elapsed if elapsed > 0 else 0.0
    return metrics

def save_history_csv(history, output_path):
    if not history:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

def load_best_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def run_experiment(device, pretrained_checkpoint):
    """
    Controlled BIGCity trip-level TTE
    train/validation/test experiment.
    """
    set_global_seed(args.seed)
    metadata_path, model_metadata = load_model_support_data()
    train_cache_path = build_tte_cache_path('train')
    validation_cache_path = build_tte_cache_path('validation')
    test_cache_path = build_tte_cache_path('test')
    for required_path in (train_cache_path, validation_cache_path):
        if not os.path.isfile(required_path):
            raise FileNotFoundError(f'Required TTE cache not found: {required_path}')
    if not os.path.isfile(test_cache_path):
        raise FileNotFoundError(f'Required TTE test cache not found: {test_cache_path}')
    train_dataset = CachedTTEDataset(train_cache_path)
    validation_dataset = CachedTTEDataset(validation_cache_path)
    validate_tte_cache_metadata(train_dataset, split='train', model_metadata=model_metadata)
    validate_tte_cache_metadata(validation_dataset, split='validation', model_metadata=model_metadata)
    train_mean_seconds, train_std_seconds = compute_train_target_statistics(train_dataset)
    logging.info(f'TTE train target mean seconds: {train_mean_seconds:.12f}')
    logging.info(f'TTE train target population std seconds: {train_std_seconds:.12f}')
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = create_dataloader(train_dataset, shuffle=True, device=device, generator=train_generator)
    validation_loader = create_dataloader(validation_dataset, shuffle=False, device=device)
    logging.info(f'Train TTE samples: {len(train_dataset)}')
    logging.info(f'Validation TTE samples: {len(validation_dataset)}')
    logging.info('Test dataset remains uninstantiated during training and model selection.')
    model = BigCity4TTE(device, checkpoint=pretrained_checkpoint).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)
    total_training_steps = len(train_loader) * args.train_epochs
    if total_training_steps <= 0:
        raise RuntimeError('Total TTE training steps must be positive.')
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    best_mae = float('inf')
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    experiment_name = f'tte_{args.semantic_mode}_seed{args.seed}'
    log_dir = os.path.join(args.log_path, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_path, f'best_{experiment_name}.pth')
    for epoch in range(1, args.train_epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, mean_seconds=train_mean_seconds, std_seconds=train_std_seconds)
        validation_metrics = evaluate(model, validation_loader, device, 'validation', mean_seconds=train_mean_seconds, std_seconds=train_std_seconds)
        record = {'epoch': epoch, 'train_loss': train_stats['loss'], 'train_seconds': train_stats['seconds'], 'train_batches_per_second': train_stats['batches_per_second'], 'validation_loss': validation_metrics['loss'], 'validation_mae_seconds': validation_metrics['mae_seconds'], 'validation_rmse_seconds': validation_metrics['rmse_seconds'], 'validation_mape_percent': validation_metrics['mape_percent'], 'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(record)
        logging.info(f'TTE epoch result: {record}')
        current_mae = validation_metrics[PRIMARY_METRIC]
        if tte_validation_improved(current_mae, best_mae):
            best_mae = current_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_payload = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'validation_metrics': validation_metrics, 'target_normalization': {'mean_seconds': train_mean_seconds, 'std_seconds': train_std_seconds, 'source': 'training_split_only'}, 'configuration': {'task': TASK_NAME, 'city': args.city, 'semantic_mode': args.semantic_mode, 'seq_len': args.seq_len, 'sample_rate': args.sample_rate, 'dynamic_mode': args.dynamic_mode, 'static_ablation': args.static_ablation, 'seed': args.seed, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'weight_decay': args.weight, 'max_epochs': args.train_epochs, 'patience': args.patience, 'loss': 'SmoothL1(beta=1.0)', 'selection_metric': PRIMARY_METRIC, 'prediction_clipping': False, 'temporal_input': 'departure_only', 'target': 'trip_duration_seconds'}, 'artifacts': {'pretrained_checkpoint': os.path.abspath(pretrained_checkpoint), 'metadata': os.path.abspath(metadata_path), 'train_cache': os.path.abspath(train_cache_path), 'validation_cache': os.path.abspath(validation_cache_path)}}
            save_checkpoint_atomic(checkpoint_path, checkpoint_payload)
            logging.info(f'New best TTE checkpoint: epoch={epoch}, validation_MAE={best_mae:.6f} sec')
        else:
            epochs_without_improvement += 1
            logging.info(f'No validation MAE improvement. Patience: {epochs_without_improvement}/{args.patience}')
        if epochs_without_improvement >= args.patience:
            logging.info('TTE early stopping triggered.')
            break
    history_path = os.path.join(log_dir, 'tte_history.csv')
    save_history_csv(history, history_path)
    if best_epoch is None:
        raise RuntimeError('No TTE best checkpoint was created.')
    logging.info(f'Reloading best TTE checkpoint from epoch {best_epoch}.')
    best_checkpoint = load_best_checkpoint(checkpoint_path, model, device)
    test_dataset = CachedTTEDataset(test_cache_path)
    validate_tte_cache_metadata(test_dataset, split='test', model_metadata=model_metadata)
    test_loader = create_dataloader(test_dataset, shuffle=False, device=device)
    test_metrics = evaluate(model, test_loader, device, 'test', mean_seconds=train_mean_seconds, std_seconds=train_std_seconds)
    checkpoint_hash = sha256_file(checkpoint_path)
    results = {'task': TASK_NAME, 'semantic_mode': args.semantic_mode, 'seed': args.seed, 'best_epoch': best_epoch, 'best_validation_metrics': best_checkpoint['validation_metrics'], 'test_metrics': test_metrics, 'target_normalization': {'mean_seconds': train_mean_seconds, 'std_seconds': train_std_seconds, 'source': 'training_split_only'}, 'checkpoint_path': os.path.abspath(checkpoint_path), 'checkpoint_sha256': checkpoint_hash, 'cache_source_sha256': {'train': train_dataset.metadata['source_sha256'], 'validation': validation_dataset.metadata['source_sha256'], 'test': test_dataset.metadata['source_sha256']}}
    results_path = os.path.join(log_dir, 'tte_results.json')
    with open(results_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    logging.info(f'FINAL TTE TEST RESULTS: {test_metrics}')
    logging.info(f'TTE best checkpoint SHA256: {checkpoint_hash}')
    return results

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.patience <= 0:
        raise ValueError('--patience must be positive.')
    if args.train_epochs <= 0:
        raise ValueError('--train_epochs must be positive.')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive.')
    device = resolve_device()
    pretrained_checkpoint = resolve_pretrained_checkpoint()
    logging.info(f'Using device: {device}')
    logging.info('Controlled BIGCity TTE experiment.')
    logging.info(f'Configuration: city={args.city}, semantic_mode={args.semantic_mode}, seq_len={args.seq_len}, sample_rate={args.sample_rate}, seed={args.seed}, batch_size={args.batch_size}, lr={args.learning_rate}, epochs={args.train_epochs}, patience={args.patience}')
    results = run_experiment(device, pretrained_checkpoint)
    print('\n=== FINAL TTE RESULTS ===')
    print(json.dumps(results, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()

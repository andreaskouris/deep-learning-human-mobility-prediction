import csv
import hashlib
import json
import logging
import os
import time
import traceback
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from config import global_vars
from config.args_config import args
from config.logging_config import init_logger, make_log_dir
from data_provider.cached_next_hop_dataset import CachedNextHopDataset
from data_provider.file_loader import file_loader
from models.bigcity4finetune import BigCity4FineTune
from utils.next_hop_metrics import NextHopMetricAccumulator
TASK_NAME = 'next_hop'
METRIC_K = 5
PRIMARY_METRIC = 'accuracy'

def validation_improved(current_accuracy, best_accuracy):
    """
    Return True only when validation Accuracy strictly improves.

    Equal Accuracy does not replace the existing best checkpoint.
    """
    return current_accuracy > best_accuracy

def early_stopping_reached(epochs_without_improvement, patience):
    """
    Return True when the configured early-stopping patience
    has been exhausted.
    """
    if patience <= 0:
        raise ValueError('patience must be a positive integer.')
    return epochs_without_improvement >= patience

def load_test_dataset_after_model_selection(test_cache_path, *, best_epoch, checkpoint_reloaded):
    """
    Load the test dataset only after validation-based model
    selection and successful reload of the selected checkpoint.
    """
    if best_epoch is None:
        raise RuntimeError('Test dataset cannot be loaded before validation-based model selection.')
    if not checkpoint_reloaded:
        raise RuntimeError('Test dataset cannot be loaded before the best checkpoint is reloaded.')
    return CachedNextHopDataset(test_cache_path)

def sha256_file(file_path, chunk_size=1024 * 1024):
    """
    Calculate SHA-256 without loading the entire file into memory.
    """
    digest = hashlib.sha256()
    with open(file_path, 'rb') as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()

def resolve_pretrained_checkpoint():
    """
    Resolve and validate the pretrained BIGCity checkpoint.
    """
    if not args.pretrained_checkpoint:
        raise ValueError('--pretrained_checkpoint is required. Example: --pretrained_checkpoint ./checkpoints/xa_pretrain.pth')
    checkpoint_path = os.path.abspath(os.path.expanduser(args.pretrained_checkpoint))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Pretrained checkpoint not found: {checkpoint_path}')
    return checkpoint_path

def resolve_device():
    """
    Resolve the single CPU or CUDA device used for the experiment.
    """
    if args.use_multi_gpu:
        raise ValueError('train_next_hop.py supports one device only.')
    if not args.use_gpu:
        if args.device != '-1':
            raise ValueError('--device may only be specified when --use_gpu is enabled.')
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('--use_gpu was requested, but CUDA is unavailable.')
    if args.device == '-1':
        device_index = 0
    else:
        try:
            device_index = int(args.device)
        except ValueError as exc:
            raise ValueError('--device must be a non-negative CUDA device index.') from exc
    if device_index < 0:
        raise ValueError('CUDA device index must be non-negative.')
    if device_index >= torch.cuda.device_count():
        raise ValueError(f'CUDA device {device_index} does not exist.')
    return torch.device(f'cuda:{device_index}')

def build_cache_signature(split):
    """
    Build the parameter-aware trajectory-cache signature.
    """
    sample_rate_tag = global_vars.format_cache_value(args.sample_rate)
    seed_tag = global_vars.format_cache_value(args.seed)
    return f'{split}_seq{args.seq_len}_sr{sample_rate_tag}_seed{seed_tag}_v2'

def get_next_hop_cache_path(split):
    """
    Return the explicit next-hop cache path for a data split.
    """
    signature = build_cache_signature(split)
    return os.path.join(global_vars.city_root_path, f'cached_next_hop_dataset_{signature}.pth')

def get_full_metadata_path():
    """
    Return the full-dataset metadata path required by BIGCity.
    """
    signature = build_cache_signature('full')
    return os.path.join(global_vars.city_root_path, f'cached_meta_{args.city}_{signature}.json')

def load_model_support_data():
    """
    Load only the road-level information required by BIGCity.

    Raw trajectory CSV files are intentionally not loaded.
    """
    file_loader.load_road_relation_file()
    file_loader.load_road_features_file()
    metadata_path = get_full_metadata_path()
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f'Required full metadata file not found: {metadata_path}')
    with open(metadata_path, 'r', encoding='utf-8') as file_handle:
        metadata = json.load(file_handle)
    required_keys = {'edge_cnt', 'road_cnt', 'time_slots_cnt', 'traj_category_cnt'}
    missing_keys = required_keys - set(metadata.keys())
    if missing_keys:
        raise ValueError('Metadata is missing required keys: ' + ', '.join(sorted(missing_keys)))
    if int(metadata['edge_cnt']) != int(file_loader.edge_cnt):
        raise ValueError('Metadata edge count does not match loaded structural data.')
    if int(metadata['road_cnt']) != int(file_loader.road_cnt):
        raise ValueError('Metadata road count does not match loaded static data.')
    if int(metadata['time_slots_cnt']) != int(file_loader.time_slots_cnt):
        raise ValueError('Metadata time-slot count does not match loaded dynamic data.')
    file_loader.traj_category_cnt = int(metadata['traj_category_cnt'])
    if file_loader.traj_data is not None:
        raise RuntimeError('Raw trajectory dataframe was unexpectedly loaded.')
    return (metadata_path, metadata)

def create_dataloader(dataset, *, shuffle, device, generator=None):
    """
    Create a deterministic single-process DataLoader.
    """
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, pin_memory=device.type == 'cuda', generator=generator)

def move_batch_to_device(batch, device):
    """
    Move one next-hop batch to the selected device.
    """
    road_ids, time_ids, time_features, labels = batch
    non_blocking = device.type == 'cuda'
    return (road_ids.to(device, non_blocking=non_blocking), time_ids.to(device, non_blocking=non_blocking), time_features.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking))

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Run one complete next-hop training epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    start_time = time.perf_counter()
    progress_bar = tqdm(dataloader, total=len(dataloader), unit='batch', desc=f'Epoch {epoch}/{args.train_epochs} - train')
    for batch in progress_bar:
        road_ids, time_ids, time_features, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(TASK_NAME, road_ids, time_ids, time_features, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss detected.')
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress_bar.set_postfix(loss=f'{loss.item():.5f}', lr=f"{optimizer.param_groups[0]['lr']:.3e}")
    elapsed = time.perf_counter() - start_time
    if total_samples == 0:
        raise RuntimeError('Training DataLoader produced no samples.')
    average_loss = total_loss / total_samples
    batches_per_second = len(dataloader) / elapsed if elapsed > 0 else 0.0
    return {'loss': average_loss, 'samples': total_samples, 'seconds': elapsed, 'batches_per_second': batches_per_second}

@torch.no_grad()
def evaluate(model, dataloader, device, split_name):
    """
    Evaluate loss and ranking metrics on one split.
    """
    model.eval()
    metric_accumulator = NextHopMetricAccumulator(k=METRIC_K)
    total_loss = 0.0
    total_samples = 0
    start_time = time.perf_counter()
    progress_bar = tqdm(dataloader, total=len(dataloader), unit='batch', desc=split_name)
    for batch in progress_bar:
        road_ids, time_ids, time_features, labels = move_batch_to_device(batch, device)
        loss, logits = model(TASK_NAME, road_ids, time_ids, time_features, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite {split_name} loss detected.')
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        metric_accumulator.update(logits, labels)
    if total_samples == 0:
        raise RuntimeError(f'{split_name} DataLoader produced no samples.')
    metrics = metric_accumulator.compute()
    elapsed = time.perf_counter() - start_time
    metrics['loss'] = total_loss / total_samples
    metrics['seconds'] = elapsed
    metrics['batches_per_second'] = len(dataloader) / elapsed if elapsed > 0 else 0.0
    return metrics

def save_checkpoint_atomic(checkpoint_path, payload):
    """
    Atomically save the selected best checkpoint.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temporary_path = checkpoint_path + '.tmp'
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)

def load_best_checkpoint(checkpoint_path, model, device):
    """
    Reload the validation-selected checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def save_history_csv(history, output_path):
    """
    Save per-epoch training and validation statistics.
    """
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

def save_json(payload, output_path):
    """
    Save a JSON experiment artifact.
    """
    with open(output_path, 'w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, indent=2)

def run_experiment(device, pretrained_checkpoint, log_dir):
    """
    Run controlled BIGCity next-hop
    train/validation/test evaluation.
    """
    metadata_path, metadata = load_model_support_data()
    train_cache_path = get_next_hop_cache_path('train')
    validation_cache_path = get_next_hop_cache_path('validation')
    test_cache_path = get_next_hop_cache_path('test')
    if not os.path.isfile(train_cache_path):
        raise FileNotFoundError(f'Train cache not found: {train_cache_path}')
    if not os.path.isfile(validation_cache_path):
        raise FileNotFoundError(f'Validation cache not found: {validation_cache_path}')
    if not os.path.isfile(test_cache_path):
        raise FileNotFoundError(f'Test cache not found: {test_cache_path}')
    logging.info('Loading training dataset.')
    train_dataset = CachedNextHopDataset(train_cache_path)
    logging.info('Loading validation dataset.')
    validation_dataset = CachedNextHopDataset(validation_cache_path)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = create_dataloader(train_dataset, shuffle=True, device=device, generator=train_generator)
    validation_loader = create_dataloader(validation_dataset, shuffle=False, device=device)
    logging.info(f'Train samples: {len(train_dataset)}')
    logging.info(f'Validation samples: {len(validation_dataset)}')
    logging.info('Test dataset remains unloaded during training/model selection.')
    model = BigCity4FineTune(device, checkpoint=pretrained_checkpoint).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)
    total_training_steps = len(train_loader) * args.train_epochs
    if total_training_steps <= 0:
        raise RuntimeError('Total training steps must be positive.')
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    logging.info(f'Total maximum optimizer steps: {total_training_steps}')
    logging.info(f'Warmup optimizer steps: {warmup_steps}')
    best_accuracy = float('-inf')
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    checkpoint_path = os.path.join(args.checkpoint_path, 'best_next_hop.pth')
    for epoch in range(1, args.train_epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        validation_metrics = evaluate(model, validation_loader, device, 'validation')
        record = {'epoch': epoch, 'train_loss': train_stats['loss'], 'train_seconds': train_stats['seconds'], 'train_batches_per_second': train_stats['batches_per_second'], 'validation_loss': validation_metrics['loss'], 'validation_accuracy': validation_metrics['accuracy'], 'validation_mrr@5': validation_metrics['mrr@5'], 'validation_ndcg@5': validation_metrics['ndcg@5'], 'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(record)
        logging.info(f'Epoch result: {record}')
        wandb.log(record)
        current_accuracy = validation_metrics[PRIMARY_METRIC]
        if validation_improved(current_accuracy, best_accuracy):
            best_accuracy = current_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_payload = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'validation_metrics': validation_metrics, 'configuration': {'city': args.city, 'seq_len': args.seq_len, 'sample_rate': args.sample_rate, 'dynamic_mode': args.dynamic_mode, 'seed': args.seed, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'weight_decay': args.weight, 'max_epochs': args.train_epochs, 'patience': args.patience, 'selection_metric': PRIMARY_METRIC}}
            save_checkpoint_atomic(checkpoint_path, checkpoint_payload)
            logging.info(f'New best checkpoint: epoch={epoch}, validation_accuracy={best_accuracy:.6f}')
        else:
            epochs_without_improvement += 1
            logging.info(f'No validation Accuracy improvement. Patience: {epochs_without_improvement}/{args.patience}')
        if early_stopping_reached(epochs_without_improvement, args.patience):
            logging.info('Early stopping triggered.')
            break
    history_path = os.path.join(log_dir, 'next_hop_history.csv')
    save_history_csv(history, history_path)
    if best_epoch is None:
        raise RuntimeError('No best checkpoint was created.')
    logging.info(f'Reloading best checkpoint from epoch {best_epoch}.')
    checkpoint_reloaded = False
    best_checkpoint = load_best_checkpoint(checkpoint_path, model, device)
    checkpoint_reloaded = True
    logging.info('Loading test dataset after model selection.')
    test_dataset = load_test_dataset_after_model_selection(test_cache_path, best_epoch=best_epoch, checkpoint_reloaded=checkpoint_reloaded)
    test_loader = create_dataloader(test_dataset, shuffle=False, device=device)
    test_metrics = evaluate(model, test_loader, device, 'test')
    logging.info(f'FINAL TEST RESULTS: {test_metrics}')
    if args.semantic_mode == 'poi_tfidf':
        semantic_cache_path = global_vars.road_semantic_tensor_file
        semantic_representation = 'l2_normalized_poi_tfidf'
        semantic_projection = f'Linear(204,{args.d_model},bias=False)'
        semantic_projection_initialization = 'zeros'
    elif args.semantic_mode == 'poi_qwen_tfidf':
        semantic_cache_path = global_vars.road_language_semantic_tensor_file
        semantic_representation = 'l2_normalized_qwen_tfidf_road_semantics'
        semantic_projection = f'Linear(1024,{args.d_model},bias=False)'
        semantic_projection_initialization = 'zeros'
    else:
        semantic_cache_path = None
        semantic_representation = None
        semantic_projection = None
        semantic_projection_initialization = None
    final_results = {'best_epoch': best_epoch, 'best_validation_metrics': best_checkpoint['validation_metrics'], 'test_metrics': test_metrics, 'dataset_sizes': {'train': len(train_dataset), 'validation': len(validation_dataset), 'test': len(test_dataset)}, 'files': {'pretrained_checkpoint': os.path.abspath(pretrained_checkpoint), 'pretrained_sha256': sha256_file(pretrained_checkpoint), 'metadata': os.path.abspath(metadata_path), 'train_cache': os.path.abspath(train_cache_path), 'validation_cache': os.path.abspath(validation_cache_path), 'test_cache': os.path.abspath(test_cache_path), 'train_cache_sha256': sha256_file(train_cache_path), 'validation_cache_sha256': sha256_file(validation_cache_path), 'test_cache_sha256': sha256_file(test_cache_path)}, 'metadata': metadata, 'semantic': {'mode': args.semantic_mode, 'cache': os.path.abspath(semantic_cache_path) if semantic_cache_path is not None else None, 'cache_sha256': sha256_file(semantic_cache_path) if semantic_cache_path is not None else None, 'representation': semantic_representation, 'projection': semantic_projection, 'projection_initialization': semantic_projection_initialization}, 'raw_trajectory_dataframe_loaded': file_loader.traj_data is not None}
    final_results_path = os.path.join(log_dir, 'next_hop_final_results.json')
    save_json(final_results, final_results_path)
    wandb.log({'best_epoch': best_epoch, 'test_accuracy': test_metrics['accuracy'], 'test_mrr@5': test_metrics['mrr@5'], 'test_ndcg@5': test_metrics['ndcg@5'], 'test_loss': test_metrics['loss']})
    return final_results

def main():
    """
    Entry point for the controlled next-hop experiment.
    """
    if args.patience <= 0:
        raise ValueError('--patience must be a positive integer.')
    if args.train_epochs <= 0:
        raise ValueError('--train_epochs must be a positive integer.')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be a positive integer.')
    log_dir = make_log_dir(args.log_path, args.checkpoint_path)
    init_logger(log_dir)
    device = resolve_device()
    pretrained_checkpoint = resolve_pretrained_checkpoint()
    logging.info(f'Using device: {device}')
    logging.info('Controlled next-hop experiment.')
    logging.info(f'Configuration: city={args.city}, seq_len={args.seq_len}, sample_rate={args.sample_rate}, dynamic_mode={args.dynamic_mode}, seed={args.seed}, batch_size={args.batch_size}, learning_rate={args.learning_rate}, weight_decay={args.weight}, max_epochs={args.train_epochs}, patience={args.patience}')
    project_name = 'bigcity-dev' if args.develop else 'bigcity'
    wandb.init(mode=args.wandb_mode, project=project_name, config=args, name=f'next-hop-{args.city}-seed{args.seed}')
    try:
        results = run_experiment(device, pretrained_checkpoint, log_dir)
        logging.info('Experiment completed.')
        logging.info(f"Final results: {results['test_metrics']}")
    except KeyboardInterrupt:
        logging.info('Experiment interrupted by user.')
        raise
    finally:
        wandb.finish()
if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.error('\n' + traceback.format_exc())
        raise

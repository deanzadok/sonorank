"""
Training script for multi-finger temporal ranking model.

Extends the single-finger temporal ranking approach to 5 parallel comparison heads
(one per finger) sharing a single backbone (encoder + transformer). Each sample
rotates through 6 modes: index, middle, ring, pinky, thumb, rest.

Loss design:
  - Primary active finger: BCE ranking loss (weight 1.0)
  - Secondary active fingers: BCE ranking loss (reduced weight)
  - Inactive fingers: uncertainty penalty (logit^2, pushes toward P=0.5)
  - Rest mode: all 5 fingers get uncertainty penalty

Validation metric: mean AUC across all 5 fingers.

Implements two-phase training:
  Phase 1: Frozen encoder, train transformer + comparison heads
  Phase 2: Full fine-tuning with discriminative learning rates

Reader's guide: this is Level 2 of SonoRank. Each batch contains paired
sequences plus masks describing the primary, secondary, and inactive fingers.
``compute_multi_finger_loss`` applies the corresponding terms, while ``main``
loads the Level 1 encoder and preserves the fold-specific split in checkpoints.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from trainer.model import MultiFingerTemporalRankingModel, count_parameters
from trainer.load import (
    create_dataloaders,
    create_subject_split,
    FINGER_NAMES,
    TEMPORAL_SEQUENCE_LENGTH,
)
from trainer.utils import SummaryLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-finger temporal ranking model')

    # Data
    parser.add_argument('--data-dir', type=str,
                       default='data/processed_h5/h5_files',
                       help='Path to H5 files directory')
    parser.add_argument('--train-subjects', type=int, nargs='+',
                       default=[1, 4, 5, 6, 10, 17, 19, 20, 21, 23],
                       help='Subject IDs for training')
    parser.add_argument('--val-subjects', type=int, nargs='+',
                       default=[24],
                       help='Subject IDs for validation')
    parser.add_argument('--test-subjects', type=int, nargs='+',
                       default=[25],
                       help='Subject IDs for testing')
    parser.add_argument('--use-masks-only', action='store_true',
                       help='Use segmentation masks instead of RGB images')
    parser.add_argument('--exclude-playing', action='store_true',
                       help='Exclude multi_playing sessions from training (default: False, use all sessions)')
    parser.add_argument('--exclude-pinch', action='store_true',
                       help='Exclude pinch sessions from training (default: False, use all sessions)')
    parser.add_argument('--exclude-wrist', action='store_true',
                       help='Exclude wrist sessions from training (default: False, use all sessions)')

    # Model - Encoder
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension (must match pretrained encoder)')
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Base channels for encoder (must match pretrained encoder)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for transformer FFN and comparison heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--pretrained-embedding-checkpoint', type=str,
                       default='',
                       help='Path to pretrained EmbeddingAutoencoder checkpoint')
    parser.add_argument('--encoder-init', type=str, default='embedding',
                       choices=['embedding', 'random'],
                       help='Encoder initialization: embedding (from checkpoint) or random')

    # Model - Transformer
    parser.add_argument('--sequence-length', type=int, default=TEMPORAL_SEQUENCE_LENGTH,
                       help=f'Number of frames per sequence (default: {TEMPORAL_SEQUENCE_LENGTH})')
    parser.add_argument('--sequence-stride', type=int, default=1,
                       help='Stride between frames in a sequence')
    parser.add_argument('--num-transformer-layers', type=int, default=2,
                       help='Number of transformer encoder layers')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')

    # Multi-finger loss weights
    parser.add_argument('--secondary-loss-weight', type=float, default=0.3,
                       help='Loss weight for secondary active fingers (default: 0.3)')
    parser.add_argument('--inactive-loss-weight', type=float, default=0.1,
                       help='Loss weight for inactive finger uncertainty penalty (default: 0.1)')

    # Pair-level uncertainty penalty
    parser.add_argument('--uncertainty-weight', type=float, default=1.0,
                       help='Weight for uncertainty penalty on ambiguous pairs (default: 1.0, set to 0.0 to disable)')
    parser.add_argument('--uncertainty-threshold-deg', type=float, default=5.0,
                       help='Threshold in degrees: pairs with delta difference below this are ambiguous (default: 5.0)')
    parser.add_argument('--rest-threshold-deg', type=float, default=17.5,
                       help='Threshold in degrees: default if both sequences have delta below this, treat as rest (default: 17.5)')
    parser.add_argument('--per-finger-rest-thresholds-deg', type=float, nargs=5, default=None,
                       metavar=('IDX', 'MID', 'RING', 'PNK', 'THB'),
                       help='Per-finger rest thresholds in degrees (index middle ring pinky thumb). '
                            'Overrides --rest-threshold-deg when provided.')
    parser.add_argument('--dynamic-secondary', action='store_true', default=False,
                       help='Use dynamic secondary weight: min(1, secondary_delta/primary_delta) '
                            'instead of fixed --secondary-loss-weight')

    # Training - Phase 1
    parser.add_argument('--phase1-epochs', type=int, default=10,
                       help='Number of epochs for Phase 1 (frozen encoder)')
    parser.add_argument('--phase1-lr', type=float, default=1e-3,
                       help='Learning rate for Phase 1 (transformer + heads)')

    # Training - Phase 2
    parser.add_argument('--phase2-epochs', type=int, default=20,
                       help='Number of epochs for Phase 2 (full fine-tuning)')
    parser.add_argument('--encoder-lr-early', type=float, default=1e-5,
                       help='Learning rate for early encoder layers in Phase 2')
    parser.add_argument('--encoder-lr-late', type=float, default=5e-5,
                       help='Learning rate for late encoder layers in Phase 2')
    parser.add_argument('--transformer-lr', type=float, default=1e-4,
                       help='Learning rate for transformer in Phase 2')
    parser.add_argument('--head-lr', type=float, default=3e-4,
                       help='Learning rate for comparison heads in Phase 2')

    # Training - General
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping max norm')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                       help='Number of warmup epochs')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/multi_finger_ranking',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs')

    # Device
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Dataset size override
    parser.add_argument('--samples-per-epoch', type=int, default=None,
                       help='Override total samples per epoch (for quick testing)')
    parser.add_argument('--samples-per-sequence', type=int, default=None,
                       help='Samples per sequence per epoch (default: 10000)')

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_multi_finger_loss(logits_dict, labels, active_mask, primary_finger_idx,
                               secondary_weight, inactive_weight,
                               deltas_A=None, deltas_B=None,
                               uncertainty_threshold_rad=0.0, rest_threshold_rad=0.0,
                               uncertainty_weight=1.0,
                               per_finger_rest_thresholds_rad=None,
                               dynamic_secondary=False):
    """
    Compute multi-finger ranking loss with optional pair-level uncertainty.

    Two distinct uncertainty mechanisms:
      1. Rest threshold: if BOTH delta_A and delta_B for a finger are below
         rest_threshold_rad (or per-finger threshold), neither sequence has
         meaningful motion -> uncertainty penalty.
      2. Uncertainty threshold: if |delta_A - delta_B| for a finger is below
         uncertainty_threshold_rad, the pair difference is too small to rank -> uncertainty penalty.

    Args:
        logits_dict: dict mapping finger_name -> [batch] logits
        labels: [batch, 5] tensor of binary labels per finger
        active_mask: [batch, 5] tensor (1=active, 0=inactive)
        primary_finger_idx: [batch] tensor (-1 for rest mode)
        secondary_weight: float, base loss weight for secondary active fingers
            (used as fallback when dynamic_secondary=False)
        inactive_weight: float, loss weight for uncertainty penalty on inactive fingers
        deltas_A: [batch, 5] tensor of delta angles for sequence A (optional)
        deltas_B: [batch, 5] tensor of delta angles for sequence B (optional)
        uncertainty_threshold_rad: float, pairs with delta diff below this are ambiguous
        rest_threshold_rad: float, default if both deltas below this, treat as rest
        uncertainty_weight: float, weight for uncertainty penalty on ambiguous pairs
        per_finger_rest_thresholds_rad: [5] tensor of per-finger rest thresholds (optional,
            overrides rest_threshold_rad when provided)
        dynamic_secondary: bool, if True compute secondary weight as
            min(1.0, secondary_delta / primary_delta) instead of fixed weight

    Returns:
        Scalar loss tensor
    """
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss(reduction='none')
    use_pair_uncertainty = (deltas_A is not None and uncertainty_weight > 0)

    # Pre-compute per-sample primary finger delta for dynamic secondary weighting
    if dynamic_secondary and deltas_A is not None:
        # For each sample, get the max delta of the primary finger
        # primary_finger_idx is [batch], values 0-4 or -1 (rest)
        batch_size = deltas_A.shape[0]
        # max(dA, dB) for each finger, shape [batch, 5]
        max_deltas = torch.max(deltas_A, deltas_B)
        # Gather primary finger's max delta for each sample
        # Clamp to avoid indexing with -1 (rest mode)
        safe_idx = primary_finger_idx.clamp(min=0).unsqueeze(1)  # [batch, 1]
        primary_delta = max_deltas.gather(1, safe_idx).squeeze(1)  # [batch]
        # Avoid division by zero
        primary_delta = primary_delta.clamp(min=1e-6)

    for i, finger in enumerate(FINGER_NAMES):
        logit = logits_dict[finger]  # [batch]
        label = labels[:, i]         # [batch]
        active = active_mask[:, i]   # [batch]
        is_primary = (primary_finger_idx == i).float()  # [batch]

        # BCE loss (for active fingers)
        bce_loss = bce(logit, label)  # [batch]

        # Uncertainty penalty: push logit toward 0 -> sigmoid -> 0.5
        uncertainty_loss = logit ** 2  # [batch]

        # Weight assignment per sample
        primary_w = is_primary * active  # 1 if primary AND active
        secondary_w = (1 - is_primary) * active  # 1 if not primary but active
        inactive_w = 1 - active  # 1 if inactive

        # Compute dynamic secondary weight if enabled
        if dynamic_secondary and deltas_A is not None:
            # This finger's max delta
            finger_delta = max_deltas[:, i]  # [batch]
            # Weight = min(1.0, this_finger_delta / primary_finger_delta)
            dyn_sec_w = torch.clamp(finger_delta / primary_delta, max=1.0)  # [batch]
        else:
            dyn_sec_w = secondary_weight  # scalar fallback

        if use_pair_uncertainty:
            dA = deltas_A[:, i]  # [batch]
            dB = deltas_B[:, i]  # [batch]

            # Rest check: per-finger or global threshold
            if per_finger_rest_thresholds_rad is not None:
                finger_rest_thresh = per_finger_rest_thresholds_rad[i]
            else:
                finger_rest_thresh = rest_threshold_rad
            both_rest = ((dA < finger_rest_thresh) & (dB < finger_rest_thresh)).float()

            # Uncertainty check: pair difference too small to rank reliably
            delta_diffs = torch.abs(dA - dB)  # [batch]
            pair_ambiguous = (delta_diffs < uncertainty_threshold_rad).float()

            # Combined: ambiguous if either condition is true
            ambiguous = torch.clamp(both_rest + pair_ambiguous, max=1.0)
            clear = 1.0 - ambiguous

            # For active fingers: clear pairs get BCE, ambiguous pairs get uncertainty
            active_loss = clear * bce_loss + ambiguous * uncertainty_weight * uncertainty_loss

            finger_loss = (
                primary_w * active_loss +
                secondary_w * dyn_sec_w * active_loss +
                inactive_w * inactive_weight * uncertainty_loss
            )
        else:
            finger_loss = (
                primary_w * bce_loss +
                secondary_w * dyn_sec_w * bce_loss +
                inactive_w * inactive_weight * uncertainty_loss
            )

        total_loss += finger_loss.mean()

    return total_loss / 5.0  # Average across fingers


def compute_multi_finger_metrics(logits_dict, labels, active_mask):
    """
    Compute per-finger AUC and mean AUC.

    Args:
        logits_dict: dict mapping finger_name -> [N] logits (concatenated across batches)
        labels: [N, 5] tensor
        active_mask: [N, 5] tensor

    Returns:
        dict with 'auc_{finger}' for each finger and 'mean_auc'
    """
    metrics = {}
    aucs = []

    for i, finger in enumerate(FINGER_NAMES):
        logit = logits_dict[finger].cpu().numpy()
        label = labels[:, i].cpu().numpy()
        active = active_mask[:, i].cpu().numpy().astype(bool)

        if active.sum() >= 10 and len(np.unique(label[active])) > 1:
            auc = roc_auc_score(label[active], logit[active])
        else:
            auc = 0.5  # Default when not enough active samples
        metrics[f'auc_{finger}'] = auc
        aucs.append(auc)

    metrics['mean_auc'] = float(np.mean(aucs))
    return metrics


def train_epoch(model, dataloader, optimizer, device, phase, grad_clip,
                secondary_weight, inactive_weight,
                uncertainty_weight=0.0, uncertainty_threshold_rad=0.0,
                rest_threshold_rad=0.0,
                per_finger_rest_thresholds_rad=None,
                dynamic_secondary=False):
    """Train for one epoch."""
    model.train()
    use_pair_uncertainty = uncertainty_weight > 0

    total_loss = 0.0
    all_logits = {finger: [] for finger in FINGER_NAMES}
    all_labels = []
    all_active_masks = []

    pbar = tqdm(dataloader, desc=f'Phase {phase} Training')
    for batch in pbar:
        sequence_A = batch['sequence_A'].to(device)
        sequence_B = batch['sequence_B'].to(device)
        labels = batch['labels'].to(device)
        active_mask = batch['active_mask'].to(device)
        primary_finger_idx = batch['primary_finger_idx'].to(device)

        # Forward pass
        logits_dict = model(sequence_A, sequence_B)

        # Loss
        loss_kwargs = {'dynamic_secondary': dynamic_secondary}
        if use_pair_uncertainty:
            loss_kwargs['deltas_A'] = batch['deltas_A'].to(device)
            loss_kwargs['deltas_B'] = batch['deltas_B'].to(device)
            loss_kwargs['uncertainty_threshold_rad'] = uncertainty_threshold_rad
            loss_kwargs['rest_threshold_rad'] = rest_threshold_rad
            loss_kwargs['uncertainty_weight'] = uncertainty_weight
            if per_finger_rest_thresholds_rad is not None:
                loss_kwargs['per_finger_rest_thresholds_rad'] = per_finger_rest_thresholds_rad

        loss = compute_multi_finger_loss(
            logits_dict, labels, active_mask, primary_finger_idx,
            secondary_weight, inactive_weight, **loss_kwargs,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        for finger in FINGER_NAMES:
            all_logits[finger].append(logits_dict[finger].detach())
        all_labels.append(labels.detach())
        all_active_masks.append(active_mask.detach())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute metrics
    cat_logits = {finger: torch.cat(all_logits[finger], dim=0) for finger in FINGER_NAMES}
    cat_labels = torch.cat(all_labels, dim=0)
    cat_active = torch.cat(all_active_masks, dim=0)

    metrics = compute_multi_finger_metrics(cat_logits, cat_labels, cat_active)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device, secondary_weight, inactive_weight,
                   uncertainty_weight=0.0, uncertainty_threshold_rad=0.0,
                   rest_threshold_rad=0.0,
                   per_finger_rest_thresholds_rad=None,
                   dynamic_secondary=False):
    """Evaluate model on validation/test set."""
    model.eval()
    use_pair_uncertainty = uncertainty_weight > 0

    total_loss = 0.0
    all_logits = {finger: [] for finger in FINGER_NAMES}
    all_labels = []
    all_active_masks = []

    pbar = tqdm(dataloader, desc='Evaluating')
    for batch in pbar:
        sequence_A = batch['sequence_A'].to(device)
        sequence_B = batch['sequence_B'].to(device)
        labels = batch['labels'].to(device)
        active_mask = batch['active_mask'].to(device)
        primary_finger_idx = batch['primary_finger_idx'].to(device)

        logits_dict = model(sequence_A, sequence_B)

        loss_kwargs = {'dynamic_secondary': dynamic_secondary}
        if use_pair_uncertainty:
            loss_kwargs['deltas_A'] = batch['deltas_A'].to(device)
            loss_kwargs['deltas_B'] = batch['deltas_B'].to(device)
            loss_kwargs['uncertainty_threshold_rad'] = uncertainty_threshold_rad
            loss_kwargs['rest_threshold_rad'] = rest_threshold_rad
            loss_kwargs['uncertainty_weight'] = uncertainty_weight
            if per_finger_rest_thresholds_rad is not None:
                loss_kwargs['per_finger_rest_thresholds_rad'] = per_finger_rest_thresholds_rad

        loss = compute_multi_finger_loss(
            logits_dict, labels, active_mask, primary_finger_idx,
            secondary_weight, inactive_weight, **loss_kwargs,
        )

        total_loss += loss.item()
        for finger in FINGER_NAMES:
            all_logits[finger].append(logits_dict[finger])
        all_labels.append(labels)
        all_active_masks.append(active_mask)

    cat_logits = {finger: torch.cat(all_logits[finger], dim=0) for finger in FINGER_NAMES}
    cat_labels = torch.cat(all_labels, dim=0)
    cat_active = torch.cat(all_active_masks, dim=0)

    metrics = compute_multi_finger_metrics(cat_logits, cat_labels, cat_active)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, phase, metrics, output_dir,
                    is_best=False, save_regular=True):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
    }

    checkpoint_path = None

    if save_regular:
        checkpoint_path = output_dir / f'checkpoint_phase{phase}_epoch{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = output_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
        if checkpoint_path is None:
            checkpoint_path = best_path

    return checkpoint_path


def create_optimizer_phase1(model, lr, weight_decay):
    """Create optimizer for Phase 1 (transformer + heads only, encoder frozen)."""
    params = list(model.transformer_encoder.parameters()) + \
             [model.pos_embedding]
    # Add all 5 comparison heads
    for finger in FINGER_NAMES:
        params.extend(list(model.comparison_heads[finger].parameters()))
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def create_optimizer_phase2(model, encoder_lr_early, encoder_lr_late, transformer_lr,
                            head_lr, weight_decay):
    """Create optimizer for Phase 2 with discriminative learning rates."""
    param_groups = model.get_discriminative_lr_params(
        encoder_lr_early=encoder_lr_early,
        encoder_lr_late=encoder_lr_late,
        transformer_lr=transformer_lr,
        heads_lr=head_lr,
    )
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def create_scheduler(optimizer, scheduler_type, num_epochs, warmup_epochs=3):
    """Create learning rate scheduler."""
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        effective_warmup = min(warmup_epochs, max(1, num_epochs - 1))

        if num_epochs <= 2:
            return None

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=effective_warmup,
        )

        cosine_epochs = max(1, num_epochs - effective_warmup)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[effective_warmup],
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def format_per_finger_metrics(metrics, prefix=''):
    """Format per-finger AUC values for display."""
    parts = []
    for finger in FINGER_NAMES:
        key = f'auc_{finger}'
        if key in metrics:
            parts.append(f"{finger[:3]}={metrics[key]:.4f}")
    return f"{prefix}AUC: mean={metrics.get('mean_auc', 0):.4f} [{', '.join(parts)}]"


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Output directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_subjects_str = '_'.join(map(str, args.test_subjects))
        val_subjects_str = '_'.join(map(str, args.val_subjects))
        args.exp_name = f"multi_finger_ranking_{timestamp}_test{test_subjects_str}_val{val_subjects_str}"

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Summary logger
    summary = SummaryLogger(output_dir)
    summary.log_config({
        'train_subjects': args.train_subjects,
        'val_subjects': args.val_subjects,
        'test_subjects': args.test_subjects,
        'embedding_dim': args.embedding_dim,
        'sequence_length': args.sequence_length,
        'sequence_stride': args.sequence_stride,
        'batch_size': args.batch_size,
        'secondary_loss_weight': args.secondary_loss_weight,
        'inactive_loss_weight': args.inactive_loss_weight,
        'uncertainty_weight': args.uncertainty_weight,
        'uncertainty_threshold_deg': args.uncertainty_threshold_deg,
        'rest_threshold_deg': args.rest_threshold_deg,
        'per_finger_rest_thresholds_deg': args.per_finger_rest_thresholds_deg,
        'dynamic_secondary': args.dynamic_secondary,
        'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs,
    })

    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved configuration to: {config_path}")

    # Determine which session types to include
    all_types = ['finger_flex', 'pinch', 'wrist', 'multi_playing']
    if args.exclude_playing:
        all_types.remove('multi_playing')
    if args.exclude_pinch:
        all_types.remove('pinch')
    if args.exclude_wrist:
        all_types.remove('wrist')
    if args.exclude_playing or args.exclude_pinch or args.exclude_wrist:
        session_types = all_types
        print(f"Using session types: {session_types}")
    else:
        session_types = None
        print("Using all available session types (finger_flex, pinch, wrist, playing)")

    # Data splits
    splits = create_subject_split(
        data_dir=args.data_dir,
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
        session_types=session_types,
    )

    # Dataloaders
    print(f"\nCreating dataloaders with mode='multi_finger_temporal_ranking'")
    print(f"  Sequence length: {args.sequence_length} frames, stride: {args.sequence_stride}")
    print(f"  Loss weights: secondary={args.secondary_loss_weight}, inactive={args.inactive_loss_weight}")
    dataloaders = create_dataloaders(
        splits=splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        mode='multi_finger_temporal_ranking',
        use_masks_only=args.use_masks_only,
        sequence_length=args.sequence_length,
        stride=args.sequence_stride,
        samples_per_epoch=args.samples_per_epoch,
        samples_per_sequence=args.samples_per_sequence,
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders.get('test', None)

    # Model
    print(f"\nCreating MultiFingerTemporalRankingModel")
    print(f"  Embedding dim: {args.embedding_dim}, Base channels: {args.base_channels}")
    print(f"  Transformer: {args.num_transformer_layers} layers, {args.num_heads} heads")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Comparison heads: 5 (one per finger)")

    model = MultiFingerTemporalRankingModel(
        embedding_dim=args.embedding_dim,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
    )

    # Load pretrained encoder
    if args.encoder_init == 'embedding' and args.pretrained_embedding_checkpoint:
        model.load_pretrained_encoder(args.pretrained_embedding_checkpoint)
    elif args.encoder_init == 'embedding' and not args.pretrained_embedding_checkpoint:
        print("  WARNING: encoder-init='embedding' but no checkpoint provided, using random init")

    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Pair-level uncertainty settings
    uncertainty_weight = args.uncertainty_weight
    uncertainty_threshold_rad = np.deg2rad(args.uncertainty_threshold_deg)
    rest_threshold_rad = np.deg2rad(args.rest_threshold_deg)

    # Per-finger rest thresholds (tensor on device, or None)
    per_finger_rest_thresholds_rad = None
    if args.per_finger_rest_thresholds_deg is not None:
        per_finger_rest_thresholds_rad = torch.tensor(
            [np.deg2rad(d) for d in args.per_finger_rest_thresholds_deg],
            dtype=torch.float32, device=device,
        )

    dynamic_secondary = args.dynamic_secondary

    if uncertainty_weight > 0:
        print(f"\n  Pair-level uncertainty: weight={uncertainty_weight}")
        print(f"    Uncertainty threshold: {args.uncertainty_threshold_deg} deg (pair delta diff)")
        if per_finger_rest_thresholds_rad is not None:
            print(f"    Per-finger rest thresholds (deg): "
                  f"idx={args.per_finger_rest_thresholds_deg[0]}, "
                  f"mid={args.per_finger_rest_thresholds_deg[1]}, "
                  f"ring={args.per_finger_rest_thresholds_deg[2]}, "
                  f"pnk={args.per_finger_rest_thresholds_deg[3]}, "
                  f"thb={args.per_finger_rest_thresholds_deg[4]}")
        else:
            print(f"    Rest threshold: {args.rest_threshold_deg} deg (both deltas below -> rest)")
    if dynamic_secondary:
        print(f"  Dynamic secondary weight: enabled (ratio-based)")

    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')

    history = {'train': [], 'val': []}
    best_val_mean_auc = 0.0
    total_epochs = 0

    # ============================
    # Phase 1: Frozen encoder
    # ============================
    if args.phase1_epochs > 0:
        print("\n" + "=" * 80)
        print("PHASE 1: Training transformer + comparison heads (frozen encoder)")
        print("=" * 80)
        summary.log_phase_start(1, "Training transformer + comparison heads (frozen encoder)")

        model.freeze_encoder()

        optimizer = create_optimizer_phase1(
            model, lr=args.phase1_lr, weight_decay=args.weight_decay,
        )
        scheduler = create_scheduler(
            optimizer, scheduler_type=args.scheduler,
            num_epochs=args.phase1_epochs,
            warmup_epochs=min(args.warmup_epochs, args.phase1_epochs),
        )

        for epoch in range(1, args.phase1_epochs + 1):
            total_epochs += 1
            print(f"\nPhase 1 - Epoch {epoch}/{args.phase1_epochs}")

            train_metrics = train_epoch(
                model, train_loader, optimizer, device, phase=1,
                grad_clip=args.grad_clip,
                secondary_weight=args.secondary_loss_weight,
                inactive_weight=args.inactive_loss_weight,
                uncertainty_weight=uncertainty_weight,
                uncertainty_threshold_rad=uncertainty_threshold_rad,
                rest_threshold_rad=rest_threshold_rad,
                per_finger_rest_thresholds_rad=per_finger_rest_thresholds_rad,
                dynamic_secondary=dynamic_secondary,
            )

            val_metrics = evaluate_model(
                model, val_loader, device,
                secondary_weight=args.secondary_loss_weight,
                inactive_weight=args.inactive_loss_weight,
                uncertainty_weight=uncertainty_weight,
                uncertainty_threshold_rad=uncertainty_threshold_rad,
                rest_threshold_rad=rest_threshold_rad,
                per_finger_rest_thresholds_rad=per_finger_rest_thresholds_rad,
                dynamic_secondary=dynamic_secondary,
            )

            print(f"Train - Loss: {train_metrics['loss']:.4f}, {format_per_finger_metrics(train_metrics)}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, {format_per_finger_metrics(val_metrics)}")

            # TensorBoard
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'Phase1/Train/{key}', value, total_epochs)
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'Phase1/Val/{key}', value, total_epochs)

            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            is_best = val_metrics['mean_auc'] > best_val_mean_auc
            if is_best:
                best_val_mean_auc = val_metrics['mean_auc']
                print(f"New best mean AUC: {best_val_mean_auc:.4f}")

            summary.log_epoch(
                epoch, args.phase1_epochs, train_metrics, val_metrics,
                is_best=is_best, best_value=best_val_mean_auc if is_best else None,
            )

            if epoch % args.save_freq == 0 or is_best:
                save_regular = (epoch % args.save_freq == 0)
                save_checkpoint(
                    model, optimizer, scheduler, epoch, phase=1,
                    metrics=val_metrics, output_dir=output_dir,
                    is_best=is_best, save_regular=save_regular,
                )

            if scheduler is not None:
                scheduler.step()
    else:
        print("\n" + "=" * 80)
        print("PHASE 1: Skipped (phase1_epochs=0)")
        print("=" * 80)
        summary.log_message("Phase 1: Skipped (phase1_epochs=0)")

    # ============================
    # Phase 2: Full fine-tuning
    # ============================
    print("\n" + "=" * 80)
    print("PHASE 2: Full fine-tuning with discriminative learning rates")
    print("=" * 80)
    summary.log_phase_start(2, "Full fine-tuning with discriminative learning rates")

    model.unfreeze_encoder()

    optimizer = create_optimizer_phase2(
        model,
        encoder_lr_early=args.encoder_lr_early,
        encoder_lr_late=args.encoder_lr_late,
        transformer_lr=args.transformer_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = create_scheduler(
        optimizer, scheduler_type=args.scheduler,
        num_epochs=args.phase2_epochs,
        warmup_epochs=min(args.warmup_epochs, args.phase2_epochs),
    )

    for epoch in range(1, args.phase2_epochs + 1):
        total_epochs += 1
        print(f"\nPhase 2 - Epoch {epoch}/{args.phase2_epochs}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, phase=2,
            grad_clip=args.grad_clip,
            secondary_weight=args.secondary_loss_weight,
            inactive_weight=args.inactive_loss_weight,
            uncertainty_weight=uncertainty_weight,
            uncertainty_threshold_rad=uncertainty_threshold_rad,
            rest_threshold_rad=rest_threshold_rad,
            per_finger_rest_thresholds_rad=per_finger_rest_thresholds_rad,
            dynamic_secondary=dynamic_secondary,
        )

        val_metrics = evaluate_model(
            model, val_loader, device,
            secondary_weight=args.secondary_loss_weight,
            inactive_weight=args.inactive_loss_weight,
            uncertainty_weight=uncertainty_weight,
            uncertainty_threshold_rad=uncertainty_threshold_rad,
            rest_threshold_rad=rest_threshold_rad,
            per_finger_rest_thresholds_rad=per_finger_rest_thresholds_rad,
            dynamic_secondary=dynamic_secondary,
        )

        print(f"Train - Loss: {train_metrics['loss']:.4f}, {format_per_finger_metrics(train_metrics)}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, {format_per_finger_metrics(val_metrics)}")

        # TensorBoard
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'Phase2/Train/{key}', value, total_epochs)
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'Phase2/Val/{key}', value, total_epochs)

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        is_best = val_metrics['mean_auc'] > best_val_mean_auc
        if is_best:
            best_val_mean_auc = val_metrics['mean_auc']
            print(f"New best mean AUC: {best_val_mean_auc:.4f}")

        summary.log_epoch(
            epoch, args.phase2_epochs, train_metrics, val_metrics,
            is_best=is_best, best_value=best_val_mean_auc if is_best else None,
        )

        if epoch % args.save_freq == 0 or is_best:
            save_regular = (epoch % args.save_freq == 0)
            save_checkpoint(
                model, optimizer, scheduler, epoch, phase=2,
                metrics=val_metrics, output_dir=output_dir,
                is_best=is_best, save_regular=save_regular,
            )

        if scheduler is not None:
            scheduler.step()

    # ============================
    # Final evaluation on test set
    # ============================
    if test_loader is not None:
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 80)

        best_checkpoint = output_dir / 'checkpoint_best.pth'
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}, phase {checkpoint['phase']}")

        test_metrics = evaluate_model(
            model, test_loader, device,
            secondary_weight=args.secondary_loss_weight,
            inactive_weight=args.inactive_loss_weight,
            uncertainty_weight=uncertainty_weight,
            uncertainty_threshold_rad=uncertainty_threshold_rad,
            rest_threshold_rad=rest_threshold_rad,
            per_finger_rest_thresholds_rad=per_finger_rest_thresholds_rad,
            dynamic_secondary=dynamic_secondary,
        )

        print(f"\nTest Metrics:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  {format_per_finger_metrics(test_metrics)}")

        test_metrics_path = output_dir / 'test_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nSaved test metrics to: {test_metrics_path}")

        summary.log_test_results(test_metrics)

    # Save history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    writer.close()
    summary.log_completion(best_metric_name="Mean AUC", best_metric_value=best_val_mean_auc)
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

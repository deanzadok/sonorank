"""
Training script for CBMF (Configuration-Based Multi-Frame) baseline.

Reimplements the two-stage CBMF architecture from:
  Zadok et al., "Inferring Fine Finger Motions for Prosthetic Control",
  ASME J. Biomech. Eng., 2025.

Phase 1: Train UNet encoder + GRU to predict per-frame joint angles (MSE loss).
Phase 2: Train full model (encoder + encoder GRU + decoder GRU) to predict
         flex labels from predicted configs + residual encoder features
         (lambda*MSE + BCE loss).

No rest reference is used (unlike SonoRank). Trains from scratch (no pretrained backbone).

Reader's guide: phase 1 learns the kinematic configuration representation and
phase 2 adds flex classification. Separate train and validation functions make
the phase boundary explicit; ``train`` owns checkpoint loading, scheduling,
and subject-disjoint data construction.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from trainer.model import CBMFModel, count_parameters
from trainer.load import create_subject_split, create_dataloaders
from trainer.utils import SummaryLogger

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']


def train_epoch_phase1(model, loader, optimizer, device, epoch, grad_clip=1.0):
    """Phase 1: train encoder + encoder GRU on joint angle prediction (MSE)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc=f"Phase1 Epoch {epoch}", leave=False):
        images = batch['image_sequence'].to(device)           # [B, T, 3, H, W]
        joint_targets = batch['joint_angle_targets'].to(device)  # [B, T, 10]

        config_pred = model.forward_phase1(images)  # [B, T, 10]
        loss = nn.functional.mse_loss(config_pred, joint_targets)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_epoch_phase1(model, loader, device, epoch):
    """Phase 1 validation: MSE on joint angles."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc=f"Phase1 Val {epoch}", leave=False):
        images = batch['image_sequence'].to(device)
        joint_targets = batch['joint_angle_targets'].to(device)

        config_pred = model.forward_phase1(images)
        loss = nn.functional.mse_loss(config_pred, joint_targets)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_epoch_phase2(model, loader, criterion_flex, optimizer, device, epoch,
                       lambda_config=4.0, grad_clip=1.0, threshold=0.5):
    """Phase 2: train full model with lambda*MSE(config) + BCE(flex)."""
    model.train()
    total_loss = 0.0
    total_config_loss = 0.0
    total_flex_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Phase2 Epoch {epoch}", leave=False):
        images = batch['image_sequence'].to(device)
        joint_targets = batch['joint_angle_targets'].to(device)
        flex_targets_pf = batch['flex_labels_per_frame'].to(device)  # [B, T, 5]

        config_pred, flex_logits, flex_logits_pf = model(images)

        # Config loss: MSE on per-frame joint angles
        config_loss = nn.functional.mse_loss(config_pred, joint_targets)

        # Flex loss: BCE on per-frame flex predictions
        flex_loss = criterion_flex(flex_logits_pf, flex_targets_pf)

        loss = lambda_config * config_loss + flex_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_config_loss += config_loss.item()
        total_flex_loss += flex_loss.item()
        num_batches += 1

        # Collect predictions for metrics (use window-level majority vote)
        flex_labels_window = batch['flex_labels']  # [B, 5]
        preds = (torch.sigmoid(flex_logits) >= threshold).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(flex_labels_window.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute per-finger and aggregate metrics
    per_finger_f1 = []
    per_finger_precision = []
    per_finger_recall = []
    for i in range(5):
        per_finger_f1.append(f1_score(all_labels[:, i], all_preds[:, i], zero_division=0))
        per_finger_precision.append(precision_score(all_labels[:, i], all_preds[:, i], zero_division=0))
        per_finger_recall.append(recall_score(all_labels[:, i], all_preds[:, i], zero_division=0))

    mean_f1 = np.mean(per_finger_f1)
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())

    avg_loss = total_loss / max(num_batches, 1)
    avg_config = total_config_loss / max(num_batches, 1)
    avg_flex = total_flex_loss / max(num_batches, 1)

    metrics = {
        'mean_f1': mean_f1,
        'accuracy': accuracy,
        'per_finger_f1': per_finger_f1,
        'per_finger_precision': per_finger_precision,
        'per_finger_recall': per_finger_recall,
        'config_loss': avg_config,
        'flex_loss': avg_flex,
    }
    return avg_loss, metrics


@torch.no_grad()
def validate_epoch_phase2(model, loader, criterion_flex, device, epoch,
                          lambda_config=4.0, threshold=0.5):
    """Phase 2 validation."""
    model.eval()
    total_loss = 0.0
    total_config_loss = 0.0
    total_flex_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Phase2 Val {epoch}", leave=False):
        images = batch['image_sequence'].to(device)
        joint_targets = batch['joint_angle_targets'].to(device)
        flex_targets_pf = batch['flex_labels_per_frame'].to(device)

        config_pred, flex_logits, flex_logits_pf = model(images)

        config_loss = nn.functional.mse_loss(config_pred, joint_targets)
        flex_loss = criterion_flex(flex_logits_pf, flex_targets_pf)
        loss = lambda_config * config_loss + flex_loss

        total_loss += loss.item()
        total_config_loss += config_loss.item()
        total_flex_loss += flex_loss.item()
        num_batches += 1

        flex_labels_window = batch['flex_labels']
        preds = (torch.sigmoid(flex_logits) >= threshold).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(flex_labels_window.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    per_finger_f1 = []
    per_finger_precision = []
    per_finger_recall = []
    for i in range(5):
        per_finger_f1.append(f1_score(all_labels[:, i], all_preds[:, i], zero_division=0))
        per_finger_precision.append(precision_score(all_labels[:, i], all_preds[:, i], zero_division=0))
        per_finger_recall.append(recall_score(all_labels[:, i], all_preds[:, i], zero_division=0))

    mean_f1 = np.mean(per_finger_f1)
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())

    avg_loss = total_loss / max(num_batches, 1)
    avg_config = total_config_loss / max(num_batches, 1)
    avg_flex = total_flex_loss / max(num_batches, 1)

    metrics = {
        'mean_f1': mean_f1,
        'accuracy': accuracy,
        'per_finger_f1': per_finger_f1,
        'per_finger_precision': per_finger_precision,
        'per_finger_recall': per_finger_recall,
        'config_loss': avg_config,
        'flex_loss': avg_flex,
    }
    return avg_loss, metrics


def train(args):
    """Main training function."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using device: {device} (GPU {args.gpu_id})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CUDA not available)")

    # Sequence parameters
    ranking_seq_len = args.sequence_length
    ranking_stride = args.sequence_stride
    ranking_span = (ranking_seq_len - 1) * ranking_stride + 1

    print(f"\nSequence params: seq_len={ranking_seq_len}, stride={ranking_stride}, span={ranking_span}")

    # Create model
    print("\n" + "=" * 80)
    print("CREATING CBMF MODEL")
    print("=" * 80)

    model = CBMFModel(
        in_channels=3,
        n_filters=args.n_filters,
        ir_dim=args.ir_dim,
        encoder_dropout=args.encoder_dropout,
        num_joints=CBMFModel.NUM_JOINTS,
        num_fingers=CBMFModel.NUM_FINGERS,
        enc_gru1_hidden=args.enc_gru1_hidden,
        enc_gru2_hidden=args.enc_gru2_hidden,
        dec_gru1_hidden=args.dec_gru1_hidden,
        dec_gru2_hidden=args.dec_gru2_hidden,
    )
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create data splits
    print("\n" + "=" * 80)
    print("CREATING DATA SPLITS")
    print("=" * 80)

    train_subjects = args.train_subjects
    val_subjects = args.val_subjects
    test_subjects = args.test_subjects

    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects: {val_subjects}")
    print(f"Test subjects: {test_subjects}")

    # Session type filtering
    session_types = None
    if args.exclude_playing or args.exclude_pinch or args.exclude_wrist:
        all_types = ['finger_flex', 'pinch', 'wrist', 'multi_playing']
        if args.exclude_playing:
            all_types.remove('multi_playing')
        if args.exclude_pinch:
            all_types.remove('pinch')
        if args.exclude_wrist:
            all_types.remove('wrist')
        session_types = all_types
        print(f"Session type filtering (using only: {session_types})")

    # Output directory
    base_output_dir = Path(args.output_dir)
    if args.exp_name:
        run_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_str = '_'.join(str(s) for s in test_subjects)
        run_name = f"cbmf_{timestamp}_test{test_str}"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Summary logger
    summary = SummaryLogger(output_dir)
    summary.log_config({
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'sequence_length': ranking_seq_len,
        'sequence_stride': ranking_stride,
        'ranking_span': ranking_span,
        'n_filters': args.n_filters,
        'ir_dim': args.ir_dim,
        'enc_gru1_hidden': args.enc_gru1_hidden,
        'enc_gru2_hidden': args.enc_gru2_hidden,
        'dec_gru1_hidden': args.dec_gru1_hidden,
        'dec_gru2_hidden': args.dec_gru2_hidden,
        'lambda_config': args.lambda_config,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs,
        'use_fixed_labels': args.use_fixed_labels,
    })

    splits = create_subject_split(
        data_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        session_types=session_types,
    )

    # Create dataloaders
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)

    dataloaders = create_dataloaders(
        splits=splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        mode='cbmf',
        stride=args.window_stride,
        ranking_seq_len=ranking_seq_len,
        ranking_stride=ranking_stride,
        vel_threshold_deg_s=args.vel_threshold_deg_s,
        vel_smooth_window=args.vel_smooth_window,
        angle_pct_threshold=args.angle_pct_threshold,
        thumb_vel_threshold_deg_s=args.thumb_vel_threshold_deg_s,
        thumb_angle_pct_threshold=args.thumb_angle_pct_threshold,
        use_fixed_labels=args.use_fixed_labels,
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Optional: subsample training set
    if args.samples_per_epoch is not None:
        train_dataset = train_loader.dataset
        sampler = RandomSampler(train_dataset, num_samples=args.samples_per_epoch)
        print(f"  Using RandomSampler: {args.samples_per_epoch} samples/epoch "
              f"from {len(train_dataset)} total windows")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    # Flex loss: BCEWithLogitsLoss with pos_weight
    train_dataset = train_loader.dataset
    all_flex = np.concatenate(
        [rec['flex_labels'] for rec in train_dataset.recordings], axis=0
    )
    pos_counts = all_flex.sum(axis=0)
    neg_counts = all_flex.shape[0] - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32)
    print(f"\nClass balance (flex rate per finger):")
    for i, name in enumerate(FINGER_NAMES):
        print(f"  {name}: {pos_counts[i]:.0f} pos / {neg_counts[i]:.0f} neg "
              f"(flex rate {pos_counts[i]/all_flex.shape[0]:.1%}, pos_weight={pos_weight[i]:.2f})")
    criterion_flex = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Save config
    config = vars(args).copy()
    config['ranking_seq_len'] = ranking_seq_len
    config['ranking_stride'] = ranking_stride
    config['ranking_span'] = ranking_span
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    config['num_joints'] = CBMFModel.NUM_JOINTS
    config['joint_columns'] = CBMFModel.JOINT_COLUMNS
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ====================================================================
    # PHASE 1: Joint angle prediction (MSE)
    # ====================================================================
    best_val_mean_f1 = 0.0
    best_epoch = 0
    total_epochs = args.phase1_epochs + args.phase2_epochs

    if args.phase1_epochs > 0:
        print("\n" + "=" * 80)
        print(f"PHASE 1: JOINT ANGLE PREDICTION ({args.phase1_epochs} epochs)")
        print("=" * 80)
        summary.log_phase_start(1, "Joint angle prediction (MSE)")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_mse = float('inf')
        for epoch in range(1, args.phase1_epochs + 1):
            epoch_start_time = time.time()

            train_loss = train_epoch_phase1(
                model, train_loader, optimizer, device, epoch,
                grad_clip=args.grad_clip,
            )
            val_loss = validate_epoch_phase1(model, val_loader, device, epoch)

            epoch_time = time.time() - epoch_start_time
            is_best_mse = val_loss < best_val_mse
            if is_best_mse:
                best_val_mse = val_loss
                torch.save({
                    'epoch': epoch, 'phase': 1,
                    'model_state_dict': model.state_dict(),
                    'val_mse': val_loss,
                    'args': vars(args),
                }, output_dir / 'checkpoint_phase1_best.pth')

            print(f"Phase 1 Epoch {epoch}/{args.phase1_epochs} ({epoch_time:.1f}s): "
                  f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}"
                  f"{' *best*' if is_best_mse else ''}")

            train_metrics_dict = {'loss': train_loss, 'accuracy': 0.0, 'mean_f1': 0.0}
            val_metrics_dict = {'loss': val_loss, 'accuracy': 0.0, 'mean_f1': 0.0}
            summary.log_epoch(
                epoch, total_epochs, train_metrics_dict, val_metrics_dict,
                is_best=is_best_mse, best_value=best_val_mse if is_best_mse else None,
                metric_name="Val MSE",
            )

    # ====================================================================
    # PHASE 2: Flex classification (lambda*MSE + BCE)
    # ====================================================================
    if args.phase2_epochs > 0:
        print("\n" + "=" * 80)
        print(f"PHASE 2: FLEX CLASSIFICATION ({args.phase2_epochs} epochs)")
        print(f"  Loss: {args.lambda_config}*MSE(config) + BCE(flex)")
        print("=" * 80)
        summary.log_phase_start(2, f"Flex classification (lambda={args.lambda_config})")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.phase2_epochs + 1):
            global_epoch = args.phase1_epochs + epoch
            epoch_start_time = time.time()

            train_loss, train_metrics = train_epoch_phase2(
                model, train_loader, criterion_flex, optimizer, device, global_epoch,
                lambda_config=args.lambda_config,
                grad_clip=args.grad_clip,
                threshold=args.threshold,
            )
            val_loss, val_metrics = validate_epoch_phase2(
                model, val_loader, criterion_flex, device, global_epoch,
                lambda_config=args.lambda_config,
                threshold=args.threshold,
            )

            epoch_time = time.time() - epoch_start_time
            print(f"\nPhase 2 Epoch {epoch}/{args.phase2_epochs} (global {global_epoch}, {epoch_time:.1f}s):")
            print(f"  Total Loss: {train_loss:.4f} | Val Total: {val_loss:.4f}")
            print(f"  Config MSE: {train_metrics['config_loss']:.6f} | Val: {val_metrics['config_loss']:.6f}")
            print(f"  Flex BCE: {train_metrics['flex_loss']:.4f} | Val: {val_metrics['flex_loss']:.4f}")
            print(f"  Train Mean F1: {train_metrics['mean_f1']:.4f} | Val Mean F1: {val_metrics['mean_f1']:.4f}")

            print("  Per-finger Val F1:", end="")
            for i, name in enumerate(FINGER_NAMES):
                print(f" {name}={val_metrics['per_finger_f1'][i]:.3f}", end="")
            print()

            # Save checkpoint
            checkpoint = {
                'epoch': global_epoch,
                'phase': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mean_f1': val_metrics['mean_f1'],
                'args': vars(args),
            }

            if args.save_freq > 0 and epoch % args.save_freq == 0:
                torch.save(checkpoint, output_dir / f'checkpoint_phase2_epoch{epoch:03d}.pth')

            is_best = val_metrics['mean_f1'] > best_val_mean_f1
            if is_best:
                best_val_mean_f1 = val_metrics['mean_f1']
                best_epoch = global_epoch
                torch.save(checkpoint, output_dir / 'checkpoint_best.pth')
                print(f"  * New best model! Val Mean F1: {best_val_mean_f1:.4f}")

            train_metrics_dict = {
                'loss': train_loss,
                'accuracy': train_metrics['accuracy'],
                'mean_f1': train_metrics['mean_f1'],
            }
            val_metrics_dict = {
                'loss': val_loss,
                'accuracy': val_metrics['accuracy'],
                'mean_f1': val_metrics['mean_f1'],
            }
            summary.log_epoch(
                global_epoch, total_epochs, train_metrics_dict, val_metrics_dict,
                is_best=is_best, best_value=best_val_mean_f1 if is_best else None,
                metric_name="Mean F1",
            )
            print()

    # Save final checkpoint
    final_checkpoint = {
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'best_val_mean_f1': best_val_mean_f1,
        'best_epoch': best_epoch,
        'args': vars(args),
    }
    torch.save(final_checkpoint, output_dir / 'checkpoint_final.pth')

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val Mean F1: {best_val_mean_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")
    summary.log_completion(best_metric_name="Mean F1", best_metric_value=best_val_mean_f1)


def main():
    parser = argparse.ArgumentParser(
        description='Train CBMF baseline (Configuration-Based Multi-Frame)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data parameters
    parser.add_argument('--data-dir', type=str,
                        default='/home/deanz/neural-hs/muscle_attention/data/processed_h5/h5_files',
                        help='Directory containing processed H5 files')
    parser.add_argument('--output-dir', type=str, default='./outputs/cbmf',
                        help='Output directory for checkpoints and logs')

    # Architecture parameters (matching original CBMF)
    parser.add_argument('--n-filters', type=int, default=32,
                        help='Base number of filters in UNet encoder')
    parser.add_argument('--ir-dim', type=int, default=128,
                        help='Intermediate representation dimension')
    parser.add_argument('--encoder-dropout', type=float, default=0.3,
                        help='Dropout rate in encoder blocks 3-4')
    parser.add_argument('--enc-gru1-hidden', type=int, default=1024,
                        help='Hidden dim for encoder GRU layer 1')
    parser.add_argument('--enc-gru2-hidden', type=int, default=256,
                        help='Hidden dim for encoder GRU layer 2')
    parser.add_argument('--dec-gru1-hidden', type=int, default=256,
                        help='Hidden dim for decoder GRU layer 1')
    parser.add_argument('--dec-gru2-hidden', type=int, default=128,
                        help='Hidden dim for decoder GRU layer 2')

    # Sequence parameters
    parser.add_argument('--sequence-length', type=int, default=16,
                        help='Number of frames per sequence')
    parser.add_argument('--sequence-stride', type=int, default=2,
                        help='Stride between sampled frames')
    parser.add_argument('--window-stride', type=int, default=1,
                        help='Stride between sliding windows in raw frames')
    parser.add_argument('--samples-per-epoch', type=int, default=None,
                        help='Number of training samples per epoch (random subset)')

    # Flex label parameters
    parser.add_argument('--vel-threshold-deg-s', type=float, default=80.0,
                        help='Velocity threshold in deg/s for flex onset detection')
    parser.add_argument('--vel-smooth-window', type=int, default=5,
                        help='Smoothing window in frames for velocity computation')
    parser.add_argument('--angle-pct-threshold', type=float, default=0.1,
                        help='Fraction of finger range for angle-based offset detection')
    parser.add_argument('--thumb-vel-threshold-deg-s', type=float, default=50.0,
                        help='Velocity threshold in deg/s for the thumb')
    parser.add_argument('--thumb-angle-pct-threshold', type=float, default=0.1,
                        help='Angle pct threshold for the thumb')
    parser.add_argument('--use-fixed-labels', action='store_true',
                        help='Use manually corrected labels from labeling/fixed_labels/')

    # Training parameters
    parser.add_argument('--phase1-epochs', type=int, default=20,
                        help='Phase 1 epochs (joint angle prediction)')
    parser.add_argument('--phase2-epochs', type=int, default=20,
                        help='Phase 2 epochs (flex classification)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (Adam)')
    parser.add_argument('--lambda-config', type=float, default=4.0,
                        help='Weight for config MSE loss in Phase 2')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (0 to match original Adam)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for flex predictions')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (0 to disable)')

    # Data loading
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size')

    # Hardware
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID to use')

    # Data split
    parser.add_argument('--train-subjects', type=int, nargs='+',
                        default=[1, 4, 5, 6, 10, 17, 19, 20],
                        help='Subject IDs for training')
    parser.add_argument('--val-subjects', type=int, nargs='+',
                        default=[21, 23],
                        help='Subject IDs for validation')
    parser.add_argument('--test-subjects', type=int, nargs='+',
                        default=[24, 25],
                        help='Subject IDs for testing')

    # Data filtering
    parser.add_argument('--exclude-playing', action='store_true', default=False,
                        help='Exclude multi_playing sessions')
    parser.add_argument('--exclude-pinch', action='store_true', default=False,
                        help='Exclude pinch sessions')
    parser.add_argument('--exclude-wrist', action='store_true', default=False,
                        help='Exclude wrist sessions')

    # Experiment name
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name for output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("CBMF BASELINE CONFIGURATION")
    print("=" * 80)
    for arg in vars(args):
        print(f"  {arg:30s}: {getattr(args, arg)}")
    print("=" * 80)

    train(args)


if __name__ == '__main__':
    main()

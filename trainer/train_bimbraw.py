"""
Training script for Bimbraw 2023 (Modified VGG16) baseline.

Reimplements the single-frame CNN from:
  Bimbraw et al., "Simultaneous Estimation of Hand Configurations and Finger
  Joint Angles using Forearm Ultrasound", IEEE T-MRB, 2023.

Single-frame flex classification using a modified VGG16 architecture.
No temporal processing: each frame is classified independently.

Reader's guide: ``SingleFrameFlexDataset`` converts each labeled HDF5 frame
into an independent example. ``train`` builds the subject-disjoint datasets,
runs optimization, and finally saves frame probabilities in the same format
used by the common event-evaluation pipeline.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from trainer.model import BimbrawVGG, count_parameters
from trainer.load import (
    compute_flex_intervals, load_fixed_labels,
    create_subject_split,
    FINGER_NAMES, FINGER_JOINTS,
)
from trainer.utils import SummaryLogger


class SingleFrameFlexDataset(Dataset):
    """Single-frame dataset for per-frame flex classification.

    Loads all frames from H5 files and provides (image, flex_label) pairs.
    """

    def __init__(self, h5_files, transform=None, use_fixed_labels=True,
                 vel_threshold_deg_s=80.0, vel_smooth_window=5,
                 angle_pct_threshold=0.1, thumb_vel_threshold_deg_s=50.0,
                 thumb_angle_pct_threshold=0.1):
        self.transform = transform
        self.images = []
        self.labels = []
        self.recording_boundaries = []  # (start_idx, end_idx, h5_path, h5_stem)

        idx = 0
        for h5_file in h5_files:
            h5_file = Path(h5_file)
            df = pd.read_hdf(h5_file, 'df')
            if len(df) == 0:
                continue

            # Summed joint angles for flex label computation
            joint_angles_summed = np.zeros((len(df), 5), dtype=np.float32)
            for j, finger in enumerate(FINGER_NAMES):
                for jc in FINGER_JOINTS[finger]:
                    if jc in df.columns:
                        joint_angles_summed[:, j] += df[jc].values.astype(np.float32)

            if use_fixed_labels:
                flex_labels = load_fixed_labels(h5_file.stem, len(df))
                if flex_labels is None:
                    continue
            else:
                flex_labels = compute_flex_intervals(
                    joint_angles_summed, fps=20.0,
                    vel_threshold_deg_s=vel_threshold_deg_s,
                    smooth_window=vel_smooth_window,
                    angle_pct_threshold=angle_pct_threshold,
                    thumb_vel_threshold_deg_s=thumb_vel_threshold_deg_s,
                    thumb_angle_pct_threshold=thumb_angle_pct_threshold,
                )

            for i in range(len(df)):
                self.images.append(np.array(df['img'].iloc[i]))
                self.labels.append(flex_labels[i])

            self.recording_boundaries.append(
                (idx, idx + len(df), str(h5_file), h5_file.stem))
            idx += len(df)

        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"  Loaded {len(self.images)} frames from {len(self.recording_boundaries)} recordings")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        return {
            'image': image,
            'flex_labels': torch.from_numpy(self.labels[idx]).float(),
        }


def get_transforms(image_size=224, is_train=True):
    """Get albumentations transforms."""
    if is_train:
        return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0),
            ToTensorV2(),
        ])


def train_epoch(model, loader, optimizer, criterion, device, epoch,
                grad_clip=1.0, threshold=0.5):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        images = batch['image'].to(device)
        labels = batch['flex_labels'].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        preds = (torch.sigmoid(logits) >= threshold).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    per_finger_f1 = []
    for i in range(5):
        per_finger_f1.append(
            f1_score(all_labels[:, i], all_preds[:, i], zero_division=0))

    return total_loss / max(num_batches, 1), np.mean(per_finger_f1), per_finger_f1


def validate_epoch(model, loader, criterion, device, epoch, threshold=0.5):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Val {epoch}", leave=False):
            images = batch['image'].to(device)
            labels = batch['flex_labels'].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            preds = (torch.sigmoid(logits) >= threshold).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    per_finger_f1 = []
    per_finger_pre = []
    per_finger_rec = []
    for i in range(5):
        per_finger_f1.append(
            f1_score(all_labels[:, i], all_preds[:, i], zero_division=0))
        per_finger_pre.append(
            precision_score(all_labels[:, i], all_preds[:, i], zero_division=0))
        per_finger_rec.append(
            recall_score(all_labels[:, i], all_preds[:, i], zero_division=0))

    metrics = {
        'mean_f1': np.mean(per_finger_f1),
        'per_finger_f1': per_finger_f1,
        'per_finger_precision': per_finger_pre,
        'per_finger_recall': per_finger_rec,
    }
    return total_loss / max(num_batches, 1), metrics


def save_raw_predictions(model, dataset, device, output_dir):
    """Save per-frame probability signals for evaluation pipeline."""
    event_dir = output_dir / 'event_detection'
    event_dir.mkdir(exist_ok=True)

    model.eval()
    val_transform = get_transforms(is_train=False)

    for start_idx, end_idx, h5_path, h5_stem in dataset.recording_boundaries:
        num_frames = end_idx - start_idx
        probs = np.zeros((num_frames, 5), dtype=np.float64)

        with torch.no_grad():
            for i in range(num_frames):
                image = dataset.images[start_idx + i].copy()
                transformed = val_transform(image=image)
                img_tensor = transformed['image'].unsqueeze(0).to(device)
                logits = model(img_tensor)
                probs[i] = torch.sigmoid(logits).cpu().numpy()[0]

        prob_signals = {}
        gt_labels = {}
        for finger_idx, name in enumerate(FINGER_NAMES):
            prob_signals[name] = probs[:, finger_idx].tolist()
            gt_labels[name] = dataset.labels[start_idx:end_idx, finger_idx].tolist()

        raw_data = {
            'h5_file': h5_path,
            'num_frames': num_frames,
            'method': 'bimbraw',
            'prob_signals': prob_signals,
            'gt_flex_labels': gt_labels,
        }
        out_path = event_dir / f'raw_predictions_{h5_stem}.json'
        with open(out_path, 'w') as f:
            json.dump(raw_data, f)

    print(f"  Raw predictions saved to: {event_dir}")


def train(args):
    """Main training function."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using device: {device} (GPU {args.gpu_id})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CUDA not available)")

    # Print config
    print("=" * 80)
    print("BIMBRAW VGG BASELINE CONFIGURATION")
    print("=" * 80)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:35s}: {v}")
    print("=" * 80)

    # Create model
    print("\nCreating BimbrawVGG model...")
    model = BimbrawVGG(num_fingers=5, dropout=args.dropout)
    model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

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

    # Create data splits
    train_subjects = args.train_subjects
    val_subjects = args.val_subjects
    test_subjects = args.test_subjects

    print(f"\nTrain subjects: {train_subjects}")
    print(f"Val subjects: {val_subjects}")
    print(f"Test subjects: {test_subjects}")

    splits = create_subject_split(
        data_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        session_types=session_types,
    )

    # Output directory
    base_output_dir = Path(args.output_dir)
    if args.exp_name:
        run_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_str = '_'.join(str(s) for s in test_subjects)
        run_name = f"bimbraw_{timestamp}_test{test_str}"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = SummaryLogger(output_dir)
    summary.log_config({
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'epochs': args.epochs,
        'lr': args.lr,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'use_fixed_labels': args.use_fixed_labels,
    })

    # Label kwargs
    label_kwargs = dict(
        use_fixed_labels=args.use_fixed_labels,
        vel_threshold_deg_s=args.vel_threshold_deg_s,
        vel_smooth_window=args.vel_smooth_window,
        angle_pct_threshold=args.angle_pct_threshold,
        thumb_vel_threshold_deg_s=args.thumb_vel_threshold_deg_s,
        thumb_angle_pct_threshold=args.thumb_angle_pct_threshold,
    )

    # Create datasets
    print("\nLoading train data...")
    train_dataset = SingleFrameFlexDataset(
        splits['train'], transform=get_transforms(is_train=True), **label_kwargs)
    print("Loading val data...")
    val_dataset = SingleFrameFlexDataset(
        splits['val'], transform=get_transforms(is_train=False), **label_kwargs)
    print("Loading test data...")
    test_dataset = SingleFrameFlexDataset(
        splits['test'], transform=get_transforms(is_train=False), **label_kwargs)

    # Dataloaders
    if args.samples_per_epoch is not None:
        sampler = RandomSampler(train_dataset, num_samples=args.samples_per_epoch)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print(f"  Using RandomSampler: {args.samples_per_epoch} samples/epoch")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Class balance and loss
    pos_counts = train_dataset.labels.sum(axis=0)
    neg_counts = len(train_dataset) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32)
    print(f"\nClass balance:")
    for i, name in enumerate(FINGER_NAMES):
        print(f"  {name}: {pos_counts[i]:.0f} pos / {neg_counts[i]:.0f} neg "
              f"(flex rate {pos_counts[i]/len(train_dataset):.1%}, pos_weight={pos_weight[i]:.2f})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Save config
    config = vars(args).copy()
    config['total_params'] = total_params
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_f1, train_pf_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            grad_clip=args.grad_clip, threshold=args.threshold)

        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch,
            threshold=args.threshold)

        epoch_time = time.time() - t0
        val_f1 = val_metrics['mean_f1']

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mean_f1': val_f1,
                'val_per_finger_f1': val_metrics['per_finger_f1'],
                'args': vars(args),
            }, output_dir / 'checkpoint_best.pth')

        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s): "
              f"Train Loss={train_loss:.4f} F1={train_f1:.4f} | "
              f"Val Loss={val_loss:.4f} F1={val_f1:.4f}"
              f"{'  *best*' if is_best else ''}")

        print("  Per-finger Val F1:", end="")
        for i, name in enumerate(FINGER_NAMES):
            print(f" {name}={val_metrics['per_finger_f1'][i]:.3f}", end="")
        print()

        train_metrics_dict = {'loss': train_loss, 'accuracy': 0.0, 'mean_f1': train_f1}
        val_metrics_dict = {'loss': val_loss, 'accuracy': 0.0, 'mean_f1': val_f1}
        summary.log_epoch(
            epoch, args.epochs, train_metrics_dict, val_metrics_dict,
            is_best=is_best, best_value=best_val_f1 if is_best else None,
            metric_name="Val Mean F1",
        )

        # Save periodic checkpoint
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mean_f1': val_f1,
                'args': vars(args),
            }, output_dir / f'checkpoint_epoch{epoch:03d}.pth')

    # Save final checkpoint
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_mean_f1': val_f1,
        'args': vars(args),
    }, output_dir / 'checkpoint_final.pth')

    print(f"\nBest Val Mean F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Load best model and save raw predictions on test set
    print("\nGenerating test predictions from best model...")
    best_ckpt = torch.load(output_dir / 'checkpoint_best.pth',
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    save_raw_predictions(model, test_dataset, device, output_dir)

    summary.log_message(f"\nBest Val Mean F1: {best_val_f1:.4f} at epoch {best_epoch}")
    summary.log_message("TRAINING COMPLETE")

    print(f"\nOutputs saved to: {output_dir}")
    print("DONE")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bimbraw VGG baseline training')

    # Data paths
    parser.add_argument('--data-dir', type=str,
                        default='./data/processed_h5/h5_files')
    parser.add_argument('--output-dir', type=str, default='./outputs/bimbraw')

    # Subject splits
    parser.add_argument('--train-subjects', type=int, nargs='+',
                        default=[1, 4, 5, 6, 10, 17, 19, 20])
    parser.add_argument('--val-subjects', type=int, nargs='+',
                        default=[21, 23])
    parser.add_argument('--test-subjects', type=int, nargs='+',
                        default=[24, 25])

    # Flex label parameters
    parser.add_argument('--vel-threshold-deg-s', type=float, default=80.0)
    parser.add_argument('--vel-smooth-window', type=int, default=5)
    parser.add_argument('--angle-pct-threshold', type=float, default=0.1)
    parser.add_argument('--thumb-vel-threshold-deg-s', type=float, default=50.0)
    parser.add_argument('--thumb-angle-pct-threshold', type=float, default=0.1)
    parser.add_argument('--use-fixed-labels', action='store_true')

    # Session filtering
    parser.add_argument('--exclude-playing', action='store_true')
    parser.add_argument('--exclude-pinch', action='store_true')
    parser.add_argument('--exclude-wrist', action='store_true')

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--samples-per-epoch', type=int, default=None)

    # System
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--exp-name', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)

"""
Training script for flex embedding classifier.

Fine-tunes a pretrained MultiFingerTemporalRankingModel backbone
(encoder + transformer) with a new MLP head for per-sequence
flex/no-flex classification.

Both query and rest sequences are encoded through the shared backbone,
their embeddings are concatenated, and a small MLP predicts 5 binary
flex probabilities (one per finger). Labels are majority-vote over
the raw frames in each query window.

Reader's guide: this is Level 3 of SonoRank. ``train`` loads the matching
ranking checkpoint for one fold, replaces its comparison output with a flexion
head, and fine-tunes on fixed labels. Variant flags control whether ranking
weights and the rest reference are used for the ablation runs.
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
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm import tqdm

from trainer.model import FlexEmbeddingClassifier, count_parameters
from trainer.load import create_subject_split, create_dataloaders
from trainer.utils import SummaryLogger

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
CLASS_NAMES = ['rest', 'index', 'middle', 'ring', 'pinky', 'thumb']


class FocalLoss(nn.Module):
    """Focal loss for binary classification with class imbalance."""

    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_metrics(logits, labels, threshold=0.5):
    """Compute classification metrics for per-sequence flex prediction.

    Args:
        logits: [N, 5] raw logits
        labels: [N, 5] binary labels
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_np = preds.cpu().numpy().reshape(-1)
    labels_np = labels.cpu().numpy().reshape(-1)

    accuracy = accuracy_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)

    per_finger_f1 = []
    per_finger_precision = []
    per_finger_recall = []
    for i in range(5):
        p = preds[:, i].cpu().numpy()
        l = labels[:, i].cpu().numpy()
        per_finger_f1.append(f1_score(l, p, zero_division=0))
        per_finger_precision.append(precision_score(l, p, zero_division=0))
        per_finger_recall.append(recall_score(l, p, zero_division=0))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_f1': np.mean(per_finger_f1),
        'per_finger_f1': per_finger_f1,
        'per_finger_precision': per_finger_precision,
        'per_finger_recall': per_finger_recall,
    }


def compute_six_class_metrics(logits, labels):
    """Compute classification metrics for 6-class dominant finger prediction.

    Args:
        logits: [N, 6] raw logits
        labels: [N] integer class labels (0=rest, 1-5=fingers)
    """
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    accuracy = accuracy_score(labels_np, preds)
    macro_f1 = f1_score(labels_np, preds, average='macro', labels=range(6), zero_division=0)
    macro_precision = precision_score(labels_np, preds, average='macro', labels=range(6), zero_division=0)
    macro_recall = recall_score(labels_np, preds, average='macro', labels=range(6), zero_division=0)
    per_class_f1 = f1_score(labels_np, preds, average=None, labels=range(6), zero_division=0)
    per_class_precision = precision_score(labels_np, preds, average=None, labels=range(6), zero_division=0)
    per_class_recall = recall_score(labels_np, preds, average=None, labels=range(6), zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'mean_f1': macro_f1,
        'per_class_f1': list(per_class_f1),
        'per_class_precision': list(per_class_precision),
        'per_class_recall': list(per_class_recall),
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch,
                grad_clip=1.0, threshold=0.5, six_class=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        image_sequence = batch['image_sequence'].to(device)  # [B, T, 3, H, W]
        rest_sequence = batch['rest_sequence'].to(device)    # [B, T, 3, H, W]

        if six_class:
            labels = batch['dominant_label'].to(device)      # [B] long
        else:
            labels = batch['flex_labels'].to(device)         # [B, 5]

        optimizer.zero_grad()
        logits = model(image_sequence, rest_sequence)
        loss = criterion(logits, labels)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if six_class:
        metrics = compute_six_class_metrics(all_logits, all_labels)
    else:
        metrics = compute_metrics(all_logits, all_labels, threshold=threshold)

    avg_loss = running_loss / len(dataloader)
    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device, epoch, threshold=0.5,
                   six_class=False):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")

    with torch.no_grad():
        for batch in pbar:
            image_sequence = batch['image_sequence'].to(device)
            rest_sequence = batch['rest_sequence'].to(device)

            if six_class:
                labels = batch['dominant_label'].to(device)
            else:
                labels = batch['flex_labels'].to(device)

            logits = model(image_sequence, rest_sequence)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            all_logits.append(logits)
            all_labels.append(labels)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    if not all_logits:
        print("  Warning: no validation batches processed!")
        return 0.0, {'mean_f1': 0.0, 'mean_accuracy': 0.0}

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if six_class:
        metrics = compute_six_class_metrics(all_logits, all_labels)
    else:
        metrics = compute_metrics(all_logits, all_labels, threshold=threshold)

    avg_loss = running_loss / len(dataloader)
    return avg_loss, metrics


def train(args):
    """Main training function."""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using device: {device} (GPU {args.gpu_id})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CUDA not available)")

    # Load ranking model config
    print("\n" + "=" * 80)
    print("LOADING RANKING MODEL CONFIG")
    print("=" * 80)

    ranking_checkpoint = Path(args.ranking_checkpoint)
    if args.ranking_config:
        ranking_config_path = Path(args.ranking_config)
    else:
        ranking_config_path = ranking_checkpoint.parent / 'config.json'

    with open(ranking_config_path) as f:
        ranking_config = json.load(f)

    ranking_seq_len = ranking_config['sequence_length']
    ranking_stride = ranking_config.get('sequence_stride', 3)
    ranking_span = (ranking_seq_len - 1) * ranking_stride + 1

    print(f"  Ranking model: seq_len={ranking_seq_len}, stride={ranking_stride}, span={ranking_span}")

    # Create model
    print("\n" + "=" * 80)
    print("CREATING FLEX EMBEDDING CLASSIFIER")
    print("=" * 80)

    num_fingers = 6 if args.six_class else 5
    model = FlexEmbeddingClassifier(
        embedding_dim=ranking_config['embedding_dim'],
        base_channels=ranking_config['base_channels'],
        hidden_dim=ranking_config['hidden_dim'],
        dropout=ranking_config.get('dropout', 0.3),
        sequence_length=ranking_seq_len,
        num_transformer_layers=ranking_config['num_transformer_layers'],
        num_heads=ranking_config['num_heads'],
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        no_rest=args.no_rest,
        num_fingers=num_fingers,
    )

    # Load pretrained backbone weights
    model.load_ranking_weights(str(ranking_checkpoint), str(ranking_config_path),
                               device, encoder_only=args.encoder_only)
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"MLP hidden dim: {args.mlp_hidden_dim}")
    print(f"MLP dropout: {args.mlp_dropout}")
    print(f"No rest reference: {args.no_rest}")
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

    # Create output directory
    base_output_dir = Path(args.output_dir)
    if args.exp_name:
        run_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_str = '_'.join(str(s) for s in test_subjects)
        run_name = f"flex_embedding_{timestamp}_test{test_str}"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Initialize summary logger
    summary = SummaryLogger(output_dir)
    summary.log_config({
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'ranking_checkpoint': str(ranking_checkpoint),
        'ranking_seq_len': ranking_seq_len,
        'ranking_stride': ranking_stride,
        'ranking_span': ranking_span,
        'mlp_hidden_dim': args.mlp_hidden_dim,
        'mlp_dropout': args.mlp_dropout,
        'window_stride': args.window_stride,
        'samples_per_epoch': args.samples_per_epoch,
        'vel_threshold_deg_s': args.vel_threshold_deg_s,
        'vel_smooth_window': args.vel_smooth_window,
        'angle_pct_threshold': args.angle_pct_threshold,
        'thumb_vel_threshold_deg_s': args.thumb_vel_threshold_deg_s,
        'thumb_angle_pct_threshold': args.thumb_angle_pct_threshold,
        'exclude_playing': args.exclude_playing,
        'exclude_pinch': args.exclude_pinch,
        'exclude_wrist': args.exclude_wrist,
        'flex_weighted_sampling': args.flex_weighted_sampling,
        'batch_size': args.batch_size,
        'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs,
        'freeze_backbone': args.freeze_backbone,
        'no_rest': args.no_rest,
        'loss': args.loss,
        'six_class': args.six_class,
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
        mode='flex_embedding',
        sequence_length=ranking_seq_len,
        stride=args.window_stride,
        ranking_seq_len=ranking_seq_len,
        ranking_stride=ranking_stride,
        flex_threshold_deg=args.flex_threshold_deg,
        rest_threshold_deg=args.rest_threshold_deg,
        vel_threshold_deg_s=args.vel_threshold_deg_s,
        vel_smooth_window=args.vel_smooth_window,
        angle_pct_threshold=args.angle_pct_threshold,
        thumb_vel_threshold_deg_s=args.thumb_vel_threshold_deg_s,
        thumb_angle_pct_threshold=args.thumb_angle_pct_threshold,
        use_fixed_labels=args.use_fixed_labels,
        six_class=args.six_class,
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Replace train loader with sampler (weighted or uniform)
    if args.samples_per_epoch is not None or args.flex_weighted_sampling:
        train_dataset = train_loader.dataset
        num_samples = args.samples_per_epoch or len(train_dataset)

        if args.flex_weighted_sampling:
            weights = []
            for item in train_dataset.data:
                rec = train_dataset.recordings[item['recording_idx']]
                if args.six_class:
                    raw_dominant = rec['dominant_labels'][item['start']:item['start'] + ranking_span]
                    active_frac_w = (raw_dominant > 0).mean()
                    weights.append(max(active_frac_w, 0.01))
                else:
                    raw_labels = rec['flex_labels'][item['start']:item['start'] + ranking_span]
                    flex_density = raw_labels.mean()
                    weights.append(max(flex_density, 0.01))
            weights = torch.tensor(weights, dtype=torch.float64)
            sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
            active_frac = (weights > 0.01).float().mean()
            mean_weight = weights.mean()
            print(f"  Using WeightedRandomSampler: {num_samples} samples/epoch from {len(train_dataset)} total windows")
            print(f"  Windows with activity: {active_frac:.1%}, mean weight: {mean_weight:.4f}")
        else:
            sampler = RandomSampler(train_dataset, num_samples=num_samples)
            print(f"  Using RandomSampler: {num_samples} samples/epoch from {len(train_dataset)} total windows")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    # Loss function
    train_dataset = train_loader.dataset
    if args.six_class:
        # Compute class weights from dominant label distribution
        all_dominant = np.concatenate(
            [rec['dominant_labels'] for rec in train_dataset.recordings], axis=0
        )
        class_counts = np.bincount(all_dominant, minlength=6).astype(np.float64)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        class_weights /= class_weights.sum()
        print(f"\nClass distribution (6-class dominant finger):")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {class_counts[i]:.0f} frames ({class_counts[i]/len(all_dominant):.1%}), weight={class_weights[i]:.4f}")
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
        )
        print(f"\nUsing CrossEntropyLoss with class weights")
    else:
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

        if args.loss == 'focal':
            print(f"\nUsing FocalLoss with gamma={args.focal_gamma}")
            criterion = FocalLoss(gamma=args.focal_gamma, reduction='mean')
        elif args.loss == 'bce':
            print(f"\nUsing BCEWithLogitsLoss with pos_weight")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        else:
            raise ValueError(f"Unknown loss function: {args.loss}")

        print(f"Decision threshold: {args.threshold}")

    # Save config
    config = vars(args).copy()
    config['ranking_seq_len'] = ranking_seq_len
    config['ranking_stride'] = ranking_stride
    config['ranking_span'] = ranking_span
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ====================================================================
    # PHASE 1: Frozen encoder (optional)
    # ====================================================================
    best_val_mean_f1 = 0.0
    best_epoch = 0
    total_epochs = args.phase1_epochs + args.phase2_epochs

    if args.phase1_epochs > 0:
        if args.freeze_backbone:
            freeze_label = "FROZEN BACKBONE (encoder + transformer)"
            print("\n" + "=" * 80)
            print(f"PHASE 1: {freeze_label} ({args.phase1_epochs} epochs)")
            print("=" * 80)
            summary.log_phase_start(1, freeze_label)
            model.freeze_backbone()
        else:
            freeze_label = "FROZEN ENCODER"
            print("\n" + "=" * 80)
            print(f"PHASE 1: {freeze_label} ({args.phase1_epochs} epochs)")
            print("=" * 80)
            summary.log_phase_start(1, freeze_label)
            model.freeze_encoder()
        trainable_params_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (phase 1): {trainable_params_p1:,}")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.phase1_lr,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.phase1_epochs, eta_min=args.phase1_lr * 0.01,
        )

        for epoch in range(1, args.phase1_epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                grad_clip=args.grad_clip, threshold=args.threshold,
                six_class=args.six_class,
            )
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch, threshold=args.threshold,
                six_class=args.six_class,
            )
            scheduler.step()

            epoch_time = time.time() - epoch_start_time
            print(f"\nPhase 1 Epoch {epoch}/{args.phase1_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Mean F1: {train_metrics['mean_f1']:.4f} | Val Mean F1: {val_metrics['mean_f1']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

            if args.six_class:
                print("  Per-class Val F1:", end="")
                for i, name in enumerate(CLASS_NAMES):
                    print(f" {name}={val_metrics['per_class_f1'][i]:.3f}", end="")
                print()
            else:
                print("  Per-finger Val F1:", end="")
                for i, name in enumerate(FINGER_NAMES):
                    print(f" {name}={val_metrics['per_finger_f1'][i]:.3f}", end="")
                print()

            is_best = val_metrics['mean_f1'] > best_val_mean_f1
            if is_best:
                best_val_mean_f1 = val_metrics['mean_f1']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch, 'phase': 1,
                    'model_state_dict': model.state_dict(),
                    'val_mean_f1': val_metrics['mean_f1'],
                    'args': vars(args),
                }, output_dir / 'checkpoint_best.pth')
                print(f"  * New best model! Val Mean F1: {best_val_mean_f1:.4f}")

            train_metrics_dict = {'loss': train_loss, 'accuracy': train_metrics['accuracy'], 'mean_f1': train_metrics['mean_f1']}
            val_metrics_dict = {'loss': val_loss, 'accuracy': val_metrics['accuracy'], 'mean_f1': val_metrics['mean_f1']}
            summary.log_epoch(
                epoch, total_epochs, train_metrics_dict, val_metrics_dict,
                is_best=is_best, best_value=best_val_mean_f1 if is_best else None,
                metric_name="Mean F1",
            )
            print()

    # ====================================================================
    # PHASE 2: Full fine-tuning with discriminative LRs
    # ====================================================================
    if args.phase2_epochs > 0:
        print("\n" + "=" * 80)
        print(f"PHASE 2: FULL FINE-TUNING ({args.phase2_epochs} epochs)")
        print("=" * 80)
        summary.log_phase_start(2, "Full fine-tuning")

        model.unfreeze_encoder()
        param_groups = model.get_discriminative_lr_params(
            encoder_lr_early=args.encoder_lr_early,
            encoder_lr_late=args.encoder_lr_late,
            transformer_lr=args.transformer_lr,
            flex_head_lr=args.flex_head_lr,
        )

        trainable_params_p2 = sum(p.numel() for g in param_groups for p in g['params'])
        print(f"Trainable parameters (phase 2): {trainable_params_p2:,}")
        print(f"Learning rates:")
        print(f"  Encoder (early): {args.encoder_lr_early}")
        print(f"  Encoder (late):  {args.encoder_lr_late}")
        print(f"  Transformer:     {args.transformer_lr}")
        print(f"  Flex head:       {args.flex_head_lr}")

        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.phase2_epochs, eta_min=1e-6,
        )

    for epoch in range(1, args.phase2_epochs + 1):
        global_epoch = args.phase1_epochs + epoch
        epoch_start_time = time.time()

        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, global_epoch,
            grad_clip=args.grad_clip, threshold=args.threshold,
            six_class=args.six_class,
        )
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, global_epoch, threshold=args.threshold,
            six_class=args.six_class,
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        print(f"\nPhase 2 Epoch {epoch}/{args.phase2_epochs} (global {global_epoch}, {epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Train Mean F1: {train_metrics['mean_f1']:.4f} | Val Mean F1: {val_metrics['mean_f1']:.4f}")
        print(f"  LR: enc_early={optimizer.param_groups[0]['lr']:.2e} "
              f"enc_late={optimizer.param_groups[1]['lr']:.2e} "
              f"transformer={optimizer.param_groups[2]['lr']:.2e} "
              f"head={optimizer.param_groups[3]['lr']:.2e}")

        if args.six_class:
            print("  Per-class Val F1:", end="")
            for i, name in enumerate(CLASS_NAMES):
                print(f" {name}={val_metrics['per_class_f1'][i]:.3f}", end="")
            print()

            print("  Per-class Val Prec:", end="")
            for i, name in enumerate(CLASS_NAMES):
                print(f" {name}={val_metrics['per_class_precision'][i]:.3f}", end="")
            print()

            print("  Per-class Val Rec:", end="")
            for i, name in enumerate(CLASS_NAMES):
                print(f" {name}={val_metrics['per_class_recall'][i]:.3f}", end="")
            print()
        else:
            print("  Per-finger Val F1:", end="")
            for i, name in enumerate(FINGER_NAMES):
                print(f" {name}={val_metrics['per_finger_f1'][i]:.3f}", end="")
            print()

            print("  Per-finger Val Prec:", end="")
            for i, name in enumerate(FINGER_NAMES):
                print(f" {name}={val_metrics['per_finger_precision'][i]:.3f}", end="")
            print()

            print("  Per-finger Val Rec:", end="")
            for i, name in enumerate(FINGER_NAMES):
                print(f" {name}={val_metrics['per_finger_recall'][i]:.3f}", end="")
            print()

        # Save checkpoint
        checkpoint = {
            'epoch': global_epoch,
            'phase': 2,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mean_f1': val_metrics['mean_f1'],
            'args': vars(args),
            'ranking_checkpoint': str(ranking_checkpoint),
        }

        torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')

        if args.save_freq > 0 and epoch % args.save_freq == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_phase2_epoch{epoch:03d}.pth')

        is_best = val_metrics['mean_f1'] > best_val_mean_f1
        if is_best:
            best_val_mean_f1 = val_metrics['mean_f1']
            best_epoch = global_epoch
            torch.save(checkpoint, output_dir / 'checkpoint_best.pth')
            print(f"  * New best model! Val Mean F1: {best_val_mean_f1:.4f}")

        train_metrics_dict = {'loss': train_loss, 'accuracy': train_metrics['accuracy'], 'mean_f1': train_metrics['mean_f1']}
        val_metrics_dict = {'loss': val_loss, 'accuracy': val_metrics['accuracy'], 'mean_f1': val_metrics['mean_f1']}
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

    # Training complete
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val Mean F1: {best_val_mean_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")

    summary.log_completion(best_metric_name="Mean F1", best_metric_value=best_val_mean_f1)


def main():
    parser = argparse.ArgumentParser(
        description='Train flex embedding classifier (MLP on ranking model embeddings)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data parameters
    parser.add_argument('--data-dir', type=str,
                        default='/home/deanz/neural-hs/muscle_attention/data/processed_h5/h5_files',
                        help='Directory containing processed H5 files')
    parser.add_argument('--output-dir', type=str,
                        default='./outputs/flex_embedding',
                        help='Output directory for checkpoints and logs')

    # Ranking model (pretrained backbone)
    parser.add_argument('--ranking-checkpoint', type=str, required=True,
                        help='Path to trained multi-finger ranking model checkpoint')
    parser.add_argument('--ranking-config', type=str, default=None,
                        help='Path to ranking model config.json (default: auto-detect from checkpoint dir)')
    parser.add_argument('--encoder-only', action='store_true',
                        help='Load only CNN encoder weights from ranking checkpoint, '
                             'leaving transformer randomly initialized')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='In phase 1, freeze entire backbone (encoder + transformer), '
                             'training only the MLP flex head')
    parser.add_argument('--no-rest', action='store_true',
                        help='Query-only mode: classify from query embedding alone, '
                             'without concatenating a rest reference embedding')

    # MLP head parameters
    parser.add_argument('--mlp-hidden-dim', type=int, default=128,
                        help='Hidden dimension for MLP flex head')
    parser.add_argument('--mlp-dropout', type=float, default=0.3,
                        help='Dropout rate for MLP flex head')

    # Window parameters
    parser.add_argument('--window-stride', type=int, default=8,
                        help='Stride between windows in raw frames')
    parser.add_argument('--samples-per-epoch', type=int, default=None,
                        help='Number of training samples per epoch (random subset). If None, use all.')

    # Flex label parameters
    parser.add_argument('--flex-threshold-deg', type=float, default=10.0,
                        help='(deprecated) Threshold in degrees below median to classify as flexing')
    parser.add_argument('--rest-threshold-deg', type=float, default=5.0,
                        help='Max summed delta in degrees for a frame to be considered rest')
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
                        help='Use manually corrected labels from labeling/fixed_labels/ '
                             'instead of auto-computed flex intervals')
    parser.add_argument('--six-class', action='store_true',
                        help='Use 6-class dominant finger classification (rest + 5 fingers) '
                             'instead of per-finger binary classification. '
                             'Requires --use-fixed-labels (dominant_sequence from JSONs).')

    # Training parameters — Phase 1 (frozen encoder)
    parser.add_argument('--phase1-epochs', type=int, default=5,
                        help='Number of Phase 1 epochs (frozen encoder)')
    parser.add_argument('--phase1-lr', type=float, default=1e-3,
                        help='Learning rate for Phase 1')

    # Training parameters — Phase 2 (full fine-tuning)
    parser.add_argument('--phase2-epochs', type=int, default=30,
                        help='Number of Phase 2 epochs (full fine-tuning)')
    parser.add_argument('--encoder-lr-early', type=float, default=1e-5,
                        help='LR for early encoder layers (conv1-3)')
    parser.add_argument('--encoder-lr-late', type=float, default=5e-5,
                        help='LR for late encoder layers (conv4-5, fc)')
    parser.add_argument('--transformer-lr', type=float, default=5e-5,
                        help='LR for transformer + positional embeddings')
    parser.add_argument('--flex-head-lr', type=float, default=3e-4,
                        help='LR for MLP flex head')

    # General training parameters
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (0 to disable)')

    # Loss function parameters
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss function')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for predictions')

    # Data loading parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size')

    # Hardware parameters
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID to use')

    # Data split parameters
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
    parser.add_argument('--flex-weighted-sampling', action='store_true', default=False,
                        help='Use weighted sampling proportional to flex activity per window')

    # Experiment name
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name for output directory')

    args = parser.parse_args()

    if args.six_class and not args.use_fixed_labels:
        parser.error("--six-class requires --use-fixed-labels (dominant_sequence only exists in fixed label JSONs)")

    print("=" * 80)
    print("FLEX EMBEDDING CLASSIFIER CONFIGURATION")
    print("=" * 80)
    for arg in vars(args):
        print(f"  {arg:30s}: {getattr(args, arg)}")
    print("=" * 80)

    train(args)


if __name__ == '__main__':
    main()

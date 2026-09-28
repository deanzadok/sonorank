"""
Training script for embedding autoencoder on muscle ultrasound images.

Trains an autoencoder to compress 224x224 images to a 64-dim embedding
and reconstruct them back. The learned embeddings can be used as tokens
for a transformer model in a later stage.

Reader's guide: this is Level 1 of SonoRank. ``train`` creates one
subject-disjoint autoencoder per fold and saves the encoder weights consumed by
Level 2; PSNR is a reconstruction diagnostic rather than a flexion metric.
"""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from trainer.model import EmbeddingAutoencoder, count_parameters
from trainer.load import create_subject_split, create_dataloaders
from trainer.utils import SummaryLogger


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio from MSE.

    Args:
        mse: Mean Squared Error
        max_val: Maximum pixel value (1.0 for normalized images)

    Returns:
        PSNR in dB
    """
    if mse == 0:
        return float('inf')
    return 10 * math.log10(max_val ** 2 / mse)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()

    running_loss = 0.0
    running_psnr = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        images = batch['image'].to(device)

        # Forward pass
        optimizer.zero_grad()
        reconstructed, embedding = model(images)
        loss = criterion(reconstructed, images)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        psnr = compute_psnr(loss.item())
        running_psnr += psnr

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr:.1f}dB',
        })

    # Epoch averages
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches

    return avg_loss, avg_psnr


def validate_epoch(model, dataloader, criterion, device, epoch):
    """
    Validate for one epoch.
    """
    model.eval()

    running_loss = 0.0
    running_psnr = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)

            # Forward pass
            reconstructed, embedding = model(images)
            loss = criterion(reconstructed, images)

            # Metrics
            running_loss += loss.item()
            psnr = compute_psnr(loss.item())
            running_psnr += psnr

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.1f}dB',
            })

    # Epoch averages
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches

    return avg_loss, avg_psnr


def train(args):
    """
    Main training function.
    """
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using device: {device} (GPU {args.gpu_id})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CUDA not available)")

    # Create data splits
    print("\n" + "=" * 80)
    print("CREATING DATA SPLITS")
    print("=" * 80)

    train_subjects = args.train_subjects
    val_subjects = args.val_subjects
    test_subjects = args.test_subjects
    session_types = args.session_types

    print(f"Session types: {session_types}")
    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects: {val_subjects}")
    print(f"Test subjects: {test_subjects}")

    # Create output directory with experiment name
    base_output_dir = Path(args.output_dir)
    if args.exp_name:
        run_name = args.exp_name
    else:
        # Auto-generate name with timestamp and test subject info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_str = '_'.join(str(s) for s in test_subjects)
        run_name = f"embedding_{timestamp}_test{test_str}"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Initialize summary logger
    summary = SummaryLogger(output_dir)
    summary.log_config({
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'embedding_dim': args.embedding_dim,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
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
        mode='embedding',
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)

    model = EmbeddingAutoencoder(
        embedding_dim=args.embedding_dim,
        base_channels=args.base_channels,
        in_channels=3,
    )

    total_params, trainable_params = count_parameters(model)
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Base channels: {args.base_channels}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(device)

    # Loss function (MSE for reconstruction)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize loss
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    summary.log_phase_start(1, "Autoencoder training")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss, train_psnr = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_psnr = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"  Train PSNR: {train_psnr:.2f}dB | Val PSNR: {val_psnr:.2f}dB")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'args': vars(args),
        }

        # Save latest checkpoint
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')

        # Save periodic checkpoint
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch{epoch}.pth')

        # Save best checkpoint (lowest validation loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(checkpoint, output_dir / 'checkpoint_best.pth')
            print(f"  * New best model! Val Loss: {best_val_loss:.6f}")

        # Log to summary
        train_metrics = {'loss': train_loss, 'psnr': train_psnr}
        val_metrics = {'loss': val_loss, 'psnr': val_psnr}
        summary.log_epoch(
            epoch, args.epochs, train_metrics, val_metrics,
            is_best=is_best, best_value=best_val_loss if is_best else None,
            metric_name="Val Loss",
        )

        print()

    # Training complete
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val Loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")

    summary.log_completion(best_metric_name="Val Loss", best_metric_value=best_val_loss)


def main():
    parser = argparse.ArgumentParser(
        description='Train embedding autoencoder for muscle ultrasound images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data parameters
    parser.add_argument('--data-dir', type=str,
                        default='/home/deanz/neural-hs/muscle_attention/data/processed_h5/h5_files',
                        help='Directory containing processed H5 files')
    parser.add_argument('--output-dir', type=str,
                        default='./outputs/embedding',
                        help='Output directory for checkpoints and logs')

    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help='Dimension of the embedding vector')
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Number of channels in first conv layer')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (0 to disable)')

    # Data loading parameters
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size (assumes square images)')

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

    # Session type parameters
    parser.add_argument('--session-types', type=str, nargs='+',
                        default=['finger_flex', 'wrist', 'pinch'],
                        choices=['playing', 'finger_flex', 'wrist', 'pinch'],
                        help='Session types to include (default: finger_flex wrist pinch)')

    # Experiment name
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name for output directory (auto-generated if not specified)')

    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    for arg in vars(args):
        print(f"{arg:20s}: {getattr(args, arg)}")
    print("=" * 80)

    train(args)


if __name__ == "__main__":
    main()

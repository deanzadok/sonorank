"""
Visualize flex embedding classifier predictions on finger flex sessions.

Loads a trained FlexEmbeddingClassifier checkpoint, slides windows across
a finger_flex H5 recording, and plots per-finger P(flex) predictions
alongside ground-truth flex labels and raw joint angles.

Usage:
    python -m evaluation.visualize_flex_embedding \
        --checkpoint outputs/flex_embedding/.../checkpoint_best.pth \
        --h5-files data/.../recordings_04_gestures_enrollment01_finger_flex01.h5 \
        --gpu-id 0

    # Multiple files
    python -m evaluation.visualize_flex_embedding \
        --checkpoint outputs/flex_embedding/.../checkpoint_best.pth \
        --h5-files data/.../recordings_04_gestures_*_finger_flex*.h5 \
        --gpu-id 0 --eval-stride 4 --output-dir evaluation/flex_embedding_viz

Reader's guide: the script constructs the same query and rest inputs used in
evaluation, runs inference over time, and aligns the resulting probabilities
with kinematics and fixed labels for inspection.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from trainer.model import FlexEmbeddingClassifier
from trainer.load import compute_flex_intervals

import albumentations as A
from albumentations.pytorch import ToTensorV2


FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
FINGER_JOINTS = {
    'index': ['joint11', 'joint12'],
    'middle': ['joint21', 'joint22'],
    'ring': ['joint31', 'joint32'],
    'pinky': ['joint41', 'joint42'],
    'thumb': ['jointt5', 'jointt6'],
}
FPS = 20
FINGER_COLORS = {
    'index': '#1f77b4',
    'middle': '#ff7f0e',
    'ring': '#2ca02c',
    'pinky': '#d62728',
    'thumb': '#9467bd',
}


# ============================================================
# Data loading
# ============================================================

def load_h5_data(h5_path):
    """Load images and summed joint angles from H5 file."""
    print(f"Loading {h5_path}...")
    df = pd.read_hdf(h5_path, 'df')
    print(f"  {len(df)} frames")

    images = [np.array(img) for img in df['img']]

    # Summed joint angles per finger (matching training)
    joint_angles = np.zeros((len(df), 5), dtype=np.float32)
    for j, finger in enumerate(FINGER_NAMES):
        for col in FINGER_JOINTS[finger]:
            if col in df.columns:
                joint_angles[:, j] += df[col].values.astype(np.float32)
            else:
                print(f"  Warning: {col} ({finger}) not found")

    return images, joint_angles, len(df)


def create_transform(image_size=224):
    """Create val/test transform."""
    return A.ReplayCompose([
        A.Resize(height=image_size, width=image_size, interpolation=1),
        ToTensorV2(),
    ])


def prepare_sequence(images, start_idx, seq_len, stride, transform):
    """Build a model-ready sequence tensor from raw images.

    Returns: [1, seq_len, 3, H, W] tensor
    """
    frames = []
    replay = None
    for i in range(seq_len):
        img = images[start_idx + i * stride].copy()
        img = img.astype(np.float32) / 255.0
        if replay is None:
            result = transform(image=img)
            replay = result.get('replay', None)
        else:
            result = A.ReplayCompose.replay(replay, image=img)
        frames.append(result['image'])
    return torch.stack(frames, dim=0).unsqueeze(0)


# ============================================================
# Model loading
# ============================================================

def load_flex_embedding_model(checkpoint_path, config_path, device):
    """Load FlexEmbeddingClassifier from checkpoint."""
    with open(config_path) as f:
        config = json.load(f)

    # Load ranking config to get backbone architecture params
    ranking_checkpoint = config['ranking_checkpoint']
    ranking_config_path = config.get('ranking_config')
    if ranking_config_path is None:
        ranking_config_path = str(Path(ranking_checkpoint).parent / 'config.json')
    with open(ranking_config_path) as f:
        ranking_config = json.load(f)

    model = FlexEmbeddingClassifier(
        embedding_dim=ranking_config['embedding_dim'],
        base_channels=ranking_config['base_channels'],
        hidden_dim=ranking_config['hidden_dim'],
        dropout=ranking_config.get('dropout', 0.3),
        sequence_length=ranking_config['sequence_length'],
        num_transformer_layers=ranking_config['num_transformer_layers'],
        num_heads=ranking_config['num_heads'],
        mlp_hidden_dim=config['mlp_hidden_dim'],
        mlp_dropout=config['mlp_dropout'],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    metrics = checkpoint.get('metrics', {})
    mean_f1 = metrics.get('mean_f1', 'N/A')
    print(f"  Loaded from epoch {epoch}, mean F1: {mean_f1}")

    return model, config, ranking_config


# ============================================================
# Rest reference detection
# ============================================================

def find_rest_window(joint_angles, span, rest_threshold_deg=5.0):
    """Find the best rest window: minimum total angular change across all fingers.

    Args:
        joint_angles: [num_frames, 5] summed joint angles in radians
        span: window size in raw frames
        rest_threshold_deg: max total angular change in degrees for a valid rest

    Returns:
        rest_start: start frame of best rest window, or None if none found
    """
    num_frames = joint_angles.shape[0]
    rest_threshold_rad = np.deg2rad(rest_threshold_deg)

    best_start = None
    best_delta = float('inf')

    for start in range(num_frames - span + 1):
        end = start + span - 1
        total_delta = np.sum(np.abs(joint_angles[end] - joint_angles[start]))
        if total_delta < best_delta:
            best_delta = total_delta
            best_start = start

    if best_delta < rest_threshold_rad:
        return best_start
    else:
        print(f"  Warning: best rest window has {np.rad2deg(best_delta):.1f} deg change "
              f"(threshold: {rest_threshold_deg} deg), using it anyway")
        return best_start


# ============================================================
# Inference
# ============================================================

def run_sliding_window_inference(model, images, rest_seq, seq_len, stride_model,
                                  eval_stride, device, transform):
    """Run flex embedding model across the recording with sliding windows.

    Args:
        model: FlexEmbeddingClassifier
        images: list of numpy images
        rest_seq: [1, seq_len, 3, H, W] rest reference tensor
        seq_len: number of frames per sequence (after striding)
        stride_model: ranking stride (frame gap within a sequence)
        eval_stride: stride between evaluation windows (in raw frames)
        device: torch device
        transform: albumentations transform

    Returns:
        results: list of dicts with 'frame', 'probs' keys
    """
    span = (seq_len - 1) * stride_model + 1
    num_frames = len(images)
    results = []

    eval_positions = list(range(0, num_frames - span + 1, eval_stride))

    for window_start in tqdm(eval_positions, desc="Inference"):
        query_seq = prepare_sequence(images, window_start, seq_len, stride_model, transform)
        query_seq = query_seq.to(device)

        with torch.no_grad():
            logits = model(query_seq, rest_seq)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        center_frame = window_start + span // 2
        results.append({
            'frame': center_frame,
            'window_start': window_start,
            'probs': probs,
        })

    return results


# ============================================================
# Plotting
# ============================================================

def plot_session(results, joint_angles, flex_labels, num_frames, h5_stem,
                 threshold=0.5, output_path=None):
    """Create multi-panel diagnostic plot matching classify_refs style.

    Panels:
      1. Joint angles (degrees) per finger — raw kinematics context
      2. GT flex labels — colored bands per finger
      3. Model P(flex) per finger — line plot with threshold
      4. Per-finger: GT band + P(flex) overlay (one panel per finger)
    """
    time = np.arange(num_frames) / FPS
    frames_arr = np.array([r['frame'] for r in results])
    time_pred = frames_arr / FPS

    probs = np.array([r['probs'] for r in results])  # [N, 5]

    fig, axes = plt.subplots(3 + 5, 1, figsize=(18, 24), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1.2] + [0.8] * 5})

    # --- Panel 1: Joint angles ---
    ax = axes[0]
    for j, finger in enumerate(FINGER_NAMES):
        angles_deg = np.rad2deg(joint_angles[:, j])
        ax.plot(time, angles_deg, color=FINGER_COLORS[finger], alpha=0.8,
                linewidth=1, label=finger)
    ax.set_ylabel('Joint Angle (deg)')
    ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.set_title(f'{h5_stem} — Flex Embedding Predictions', fontsize=12, fontweight='bold')

    # --- Panel 2: GT flex labels as colored bands ---
    ax = axes[1]
    for j, finger in enumerate(FINGER_NAMES):
        gt = flex_labels[:, j]
        ax.fill_between(time, j - 0.4, j + 0.4, where=gt > 0.5,
                        color=FINGER_COLORS[finger], alpha=0.6)
    ax.set_yticks(range(5))
    ax.set_yticklabels(FINGER_NAMES, fontsize=9)
    ax.set_ylabel('GT Flex')
    ax.set_ylim(-0.5, 4.5)
    ax.set_title('Ground Truth Flex Intervals')

    # --- Panel 3: All P(flex) curves ---
    ax = axes[2]
    for j, finger in enumerate(FINGER_NAMES):
        ax.plot(time_pred, probs[:, j], color=FINGER_COLORS[finger],
                alpha=0.8, linewidth=1.2, label=finger)
    ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5,
               label=f'threshold={threshold}')
    ax.set_ylabel('P(flex)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8, ncol=6)
    ax.set_title('Model Predictions — P(flex) per Finger')

    # --- Panels 4-8: Per-finger detail ---
    for j, finger in enumerate(FINGER_NAMES):
        ax = axes[3 + j]

        # GT band
        gt = flex_labels[:, j]
        ax.fill_between(time, 0, 1, where=gt > 0.5,
                        color=FINGER_COLORS[finger], alpha=0.15, label='GT flex')

        # P(flex) curve
        ax.plot(time_pred, probs[:, j], color=FINGER_COLORS[finger],
                linewidth=1.5, alpha=0.9, label='P(flex)')

        # Threshold
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.4)

        ax.set_ylabel(f'{finger}', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        if j == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    if output_path is None:
        output_path = f'{h5_stem}_flex_embedding.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


# ============================================================
# Metrics
# ============================================================

def compute_metrics(results, flex_labels, span, threshold=0.5):
    """Compute per-finger and mean F1 from sliding window predictions."""
    all_preds = []
    all_gt = []

    for r in results:
        start = r['window_start']
        raw_gt = flex_labels[start:start + span]
        gt_majority = (raw_gt.mean(axis=0) >= 0.5).astype(np.float32)
        pred = (r['probs'] >= threshold).astype(np.float32)
        all_preds.append(pred)
        all_gt.append(gt_majority)

    all_preds = np.array(all_preds)
    all_gt = np.array(all_gt)

    print(f"\n  Per-finger metrics (threshold={threshold}):")
    print(f"  {'Finger':>8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Flex%':>6}  {'Pred%':>6}")
    print(f"  {'-' * 44}")

    per_finger = {}
    f1s = []
    for j, finger in enumerate(FINGER_NAMES):
        tp = ((all_preds[:, j] == 1) & (all_gt[:, j] == 1)).sum()
        fp = ((all_preds[:, j] == 1) & (all_gt[:, j] == 0)).sum()
        fn = ((all_preds[:, j] == 0) & (all_gt[:, j] == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        flex_rate = float(all_gt[:, j].mean())
        pred_rate = float(all_preds[:, j].mean())
        f1s.append(f1)

        per_finger[finger] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'flex_rate': flex_rate,
            'pred_rate': pred_rate,
        }

        print(f"  {finger:>8}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {flex_rate:>5.1%}  {pred_rate:>5.1%}")

    mean_f1 = float(np.mean(f1s))
    print(f"  {'MEAN':>8}  {'':>6}  {'':>6}  {mean_f1:>6.3f}")
    return mean_f1, per_finger


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize flex embedding classifier predictions on finger flex sessions')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to FlexEmbeddingClassifier checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (default: auto-detect from checkpoint dir)')
    parser.add_argument('--h5-files', type=str, nargs='+', required=True,
                        help='H5 files to evaluate')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--eval-stride', type=int, default=4,
                        help='Stride between evaluation windows in raw frames (default: 4)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='P(flex) threshold for binary prediction (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same dir as checkpoint)')
    parser.add_argument('--rest-threshold-deg', type=float, default=5.0,
                        help='Rest detection threshold in degrees (default: 5.0)')

    # Flex label params (should match training config)
    parser.add_argument('--vel-threshold-deg-s', type=float, default=None,
                        help='Override velocity threshold (default: from config)')
    parser.add_argument('--vel-smooth-window', type=int, default=None,
                        help='Override velocity smooth window (default: from config)')
    parser.add_argument('--angle-pct-threshold', type=float, default=None,
                        help='Override angle pct threshold (default: from config)')
    parser.add_argument('--thumb-vel-threshold-deg-s', type=float, default=None,
                        help='Override thumb velocity threshold (default: from config)')
    parser.add_argument('--thumb-angle-pct-threshold', type=float, default=None,
                        help='Override thumb angle pct threshold (default: from config)')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    config_path = args.config or str(Path(args.checkpoint).parent / 'config.json')
    model, config, ranking_config = load_flex_embedding_model(
        args.checkpoint, config_path, device)

    seq_len = ranking_config['sequence_length']
    stride_model = ranking_config.get('sequence_stride', ranking_config.get('stride', 3))
    span = (seq_len - 1) * stride_model + 1
    image_size = config.get('image_size', 224)

    # Flex label params: use config values unless overridden
    vel_threshold = args.vel_threshold_deg_s or config.get('vel_threshold_deg_s', 80.0)
    vel_smooth = args.vel_smooth_window or config.get('vel_smooth_window', 5)
    angle_pct = args.angle_pct_threshold or config.get('angle_pct_threshold', 0.1)
    thumb_vel = args.thumb_vel_threshold_deg_s or config.get('thumb_vel_threshold_deg_s', 50.0)
    thumb_pct = args.thumb_angle_pct_threshold or config.get('thumb_angle_pct_threshold', 0.1)

    print(f"\nModel config:")
    print(f"  seq_len={seq_len}, stride={stride_model}, span={span}")
    print(f"  MLP: hidden={config['mlp_hidden_dim']}, dropout={config['mlp_dropout']}")
    print(f"\nFlex label params:")
    print(f"  vel_threshold={vel_threshold} deg/s, smooth={vel_smooth}")
    print(f"  angle_pct={angle_pct}, thumb_vel={thumb_vel}, thumb_pct={thumb_pct}")

    transform = create_transform(image_size)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each H5 file
    all_metrics = {}
    for h5_path in args.h5_files:
        h5_path = Path(h5_path)
        print(f"\n{'=' * 60}")
        print(f"Processing: {h5_path.name}")
        print(f"{'=' * 60}")

        images, joint_angles, num_frames = load_h5_data(str(h5_path))

        if num_frames < span:
            print(f"  Skipping: only {num_frames} frames, need {span}")
            continue

        # Compute GT flex labels
        flex_labels = compute_flex_intervals(
            joint_angles, fps=FPS,
            vel_threshold_deg_s=vel_threshold,
            smooth_window=vel_smooth,
            angle_pct_threshold=angle_pct,
            thumb_vel_threshold_deg_s=thumb_vel,
            thumb_angle_pct_threshold=thumb_pct,
        )
        print(f"  GT flex rate per finger: " + ", ".join(
            f"{f}={flex_labels[:, j].mean():.1%}" for j, f in enumerate(FINGER_NAMES)))

        # Find rest reference window
        rest_start = find_rest_window(joint_angles, span,
                                       rest_threshold_deg=args.rest_threshold_deg)
        print(f"  Rest reference: frames {rest_start}-{rest_start + span - 1}")

        rest_seq = prepare_sequence(images, rest_start, seq_len, stride_model, transform)
        rest_seq = rest_seq.to(device)

        # Run inference
        results = run_sliding_window_inference(
            model, images, rest_seq, seq_len, stride_model,
            args.eval_stride, device, transform)

        # Metrics
        mean_f1, per_finger = compute_metrics(results, flex_labels, span,
                                              threshold=args.threshold)
        all_metrics[h5_path.stem] = {
            'mean_f1': mean_f1,
            'per_finger': per_finger,
            'num_frames': num_frames,
        }

        # Plot
        plot_path = str(output_dir / f'{h5_path.stem}_flex_embedding.png')
        plot_session(results, joint_angles, flex_labels, num_frames,
                     h5_path.stem, threshold=args.threshold, output_path=plot_path)

    # Save metrics JSON (per-subject filename to avoid overwrites in parallel runs)
    if all_metrics:
        # Compute aggregate metrics across all sessions
        all_f1s = [m['mean_f1'] for m in all_metrics.values()]
        aggregate = {
            'overall_mean_f1': float(np.mean(all_f1s)),
            'per_finger_mean': {},
        }
        for finger in FINGER_NAMES:
            finger_f1s = [m['per_finger'][finger]['f1'] for m in all_metrics.values()]
            finger_precs = [m['per_finger'][finger]['precision'] for m in all_metrics.values()]
            finger_recs = [m['per_finger'][finger]['recall'] for m in all_metrics.values()]
            aggregate['per_finger_mean'][finger] = {
                'f1': float(np.mean(finger_f1s)),
                'precision': float(np.mean(finger_precs)),
                'recall': float(np.mean(finger_recs)),
            }

        metrics_out = {
            'checkpoint': str(args.checkpoint),
            'threshold': args.threshold,
            'eval_stride': args.eval_stride,
            'aggregate': aggregate,
            'per_session': all_metrics,
        }

        # Derive subject ID from H5 filenames (e.g. recordings_04_gestures_... → 04)
        subject_ids = set()
        for stem in all_metrics.keys():
            m = re.match(r'recordings_(\d+)_', stem)
            if m:
                subject_ids.add(m.group(1))
        subject_suffix = '_'.join(sorted(subject_ids)) if subject_ids else 'unknown'
        metrics_path = output_dir / f'metrics_subject{subject_suffix}.json'

        with open(metrics_path, 'w') as f:
            json.dump(metrics_out, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()

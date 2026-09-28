"""
Evaluate a single CBMFModel checkpoint for event-level flex detection.

Analogous to evaluate_flex_embedding_events.py but for CBMF models (direct
classifier, no rest reference). For the checkpoint's test subject, processes all
finger_flex H5 files:
  1. Runs sliding-window inference -> per-frame P(flex) per finger
  2. Saves raw predictions + full-frame probability signals as JSON
  3. Runs grid-search event detection and saves metrics

All output is stored in {checkpoint_dir}/event_detection/.

Usage:
    python -m evaluation.evaluate_cbmf_events \
        --checkpoint outputs/experiments/methods_fixed_labels/cbmf/.../checkpoint_best.pth \
        --gpu-id 0

    # Skip if results already exist:
    python -m evaluation.evaluate_cbmf_events \
        --checkpoint ... --gpu-id 0 --skip-existing

Reader's guide: inference produces one probability trace for every finger and
session. Ground-truth labels and event matching are computed afterward so raw
predictions remain reusable when post-processing parameters change.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from trainer.model import CBMFModel
from trainer.load import compute_flex_intervals, load_fixed_labels
from evaluation.event_detection_metrics import (
    evaluate_finger, aggregate_results, load_joint_angles,
    extract_per_finger_signal,
)

DATA_DIR = "data/processed_h5/h5_files"
FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
FPS = 20


def find_h5_files(subject_id, data_dir=DATA_DIR):
    """Find all finger_flex H5 files for a given subject."""
    data_dir = Path(data_dir)
    pattern = f"recordings_{subject_id:02d}_*finger_flex*.h5"
    files = sorted(data_dir.glob(pattern))
    if not files:
        pattern = f"recordings_{subject_id}_*finger_flex*.h5"
        files = sorted(data_dir.glob(pattern))
    return [str(f) for f in files]


def load_h5_data(h5_path):
    """Load images from H5 file."""
    import pandas as pd
    print(f"Loading {h5_path}...")
    df = pd.read_hdf(h5_path, 'df')
    print(f"  {len(df)} frames")
    images = [np.array(img) for img in df['img']]
    return images, len(df)


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


def load_cbmf_model(checkpoint_path, config_path, device):
    """Load CBMFModel from checkpoint."""
    with open(config_path) as f:
        config = json.load(f)

    model = CBMFModel(
        in_channels=3,
        n_filters=config.get('n_filters', 32),
        ir_dim=config.get('ir_dim', 128),
        encoder_dropout=config.get('encoder_dropout', 0.3),
        num_joints=config.get('num_joints', CBMFModel.NUM_JOINTS),
        num_fingers=CBMFModel.NUM_FINGERS,
        enc_gru1_hidden=config.get('enc_gru1_hidden', 1024),
        enc_gru2_hidden=config.get('enc_gru2_hidden', 256),
        dec_gru1_hidden=config.get('dec_gru1_hidden', 256),
        dec_gru2_hidden=config.get('dec_gru2_hidden', 128),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    phase = checkpoint.get('phase', '?')
    print(f"Loaded CBMFModel from phase {phase}, epoch {epoch}")

    return model, config


def run_inference(model, images, seq_len, stride_model, eval_stride,
                  device, transform):
    """Run sliding-window CBMF inference on one session.

    Args:
        model: CBMFModel
        images: list of numpy images
        seq_len: model sequence length
        stride_model: frame stride within a sequence
        eval_stride: stride between evaluation windows
        device: torch device
        transform: albumentations transform

    Returns:
        list of dicts with 'frame', 'p_vs_rest' (dict of floats per finger)
    """
    span = (seq_len - 1) * stride_model + 1
    num_frames = len(images)
    results = []

    eval_positions = list(range(span - 1, num_frames, eval_stride))

    for frame_idx in tqdm(eval_positions, desc="  inference", leave=False):
        window_start = frame_idx - span + 1
        query_seq = prepare_sequence(
            images, window_start, seq_len, stride_model, transform).to(device)

        with torch.no_grad():
            _, flex_logits, _ = model(query_seq)
            probs = torch.sigmoid(flex_logits).cpu().numpy()[0]  # [5]

        # Use 'p_vs_rest' key for compatibility with evaluate_methods_fair
        p_vs_rest = {f: round(float(probs[i]), 6) for i, f in enumerate(FINGER_NAMES)}
        results.append({
            'frame': frame_idx,
            'p_vs_rest': p_vs_rest,
        })

    return results


def compute_gt_flex_labels(h5_paths, vel_threshold_deg_s=80.0,
                           angle_pct_threshold=0.1,
                           thumb_vel_threshold_deg_s=50.0,
                           thumb_angle_pct_threshold=0.1,
                           use_fixed_labels=True):
    """Load GT flex labels for all H5 files."""
    h5_cache = {}
    for h5_path in h5_paths:
        joint_angles_rad, num_frames = load_joint_angles(h5_path)
        gt_flex_labels = None
        if use_fixed_labels:
            h5_stem = Path(h5_path).stem
            gt_flex_labels = load_fixed_labels(h5_stem, num_frames)
        if gt_flex_labels is None:
            gt_flex_labels = compute_flex_intervals(
                joint_angles_rad, fps=FPS,
                vel_threshold_deg_s=vel_threshold_deg_s,
                angle_pct_threshold=angle_pct_threshold,
                thumb_vel_threshold_deg_s=thumb_vel_threshold_deg_s,
                thumb_angle_pct_threshold=thumb_angle_pct_threshold,
            )
        h5_cache[h5_path] = (joint_angles_rad, num_frames, gt_flex_labels)
    return h5_cache


def compute_event_metrics(prediction_logs, h5_cache, eval_stride):
    """Compute event detection metrics with grid search optimization."""
    threshold_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    min_duration_range = [4, 6, 8, 10, 14, 20]
    max_gap_range = [0, 4, 8, 12, 16, 24, 32]

    from itertools import product

    best_score = -1
    best_params = {}

    for thresh, min_dur, max_gap in product(
            threshold_range, min_duration_range, max_gap_range):
        all_finger_results = []

        for h5_path, comparison_log in prediction_logs.items():
            _, num_frames, gt_flex_labels = h5_cache[h5_path]

            for finger_idx, finger in enumerate(FINGER_NAMES):
                result = evaluate_finger(
                    comparison_log, finger, finger_idx,
                    gt_flex_labels, num_frames, eval_stride,
                    'p_vs_rest', thresh, min_dur, max_gap,
                    normalize=False)
                all_finger_results.append(result)

        overall = aggregate_results(all_finger_results)
        score = overall['micro']['f1']

        if score > best_score:
            best_score = score
            best_params = {
                'threshold': thresh,
                'min_duration': min_dur,
                'max_gap': max_gap,
            }

    # Re-run with best params for per-file breakdown
    per_file_metrics = []
    overall_finger_results = []

    for h5_path, comparison_log in prediction_logs.items():
        _, num_frames, gt_flex_labels = h5_cache[h5_path]
        h5_stem = Path(h5_path).stem

        file_finger_results = []
        for finger_idx, finger in enumerate(FINGER_NAMES):
            result = evaluate_finger(
                comparison_log, finger, finger_idx,
                gt_flex_labels, num_frames, eval_stride,
                'p_vs_rest', best_params['threshold'],
                best_params['min_duration'], best_params['max_gap'],
                normalize=False)
            file_finger_results.append(result)
            overall_finger_results.append(result)

        file_agg = aggregate_results(file_finger_results)

        finger_events = {}
        for r in file_finger_results:
            finger_events[r['finger']] = {
                'gt_events': [{'start': int(s), 'end': int(e)}
                              for s, e in r['gt_events']],
                'pred_events': [{'start': int(s), 'end': int(e)}
                                for s, e in r['pred_events']],
            }

        per_file_metrics.append({
            'h5_file': str(h5_path),
            'h5_stem': h5_stem,
            'micro': file_agg['micro'],
            'macro': file_agg['macro'],
            'per_finger': file_agg['per_finger'],
            'events': finger_events,
        })

    overall = aggregate_results(overall_finger_results)
    return best_params, per_file_metrics, overall


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a CBMF checkpoint for event-level flex detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint_best.pth')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--eval-stride', type=int, default=8,
                        help='Stride between evaluation windows (default: 8)')
    parser.add_argument('--thumb-vel-threshold', type=float, default=50.0,
                        help='Thumb velocity threshold for GT flex labels')
    parser.add_argument('--no-fixed-labels', dest='use_fixed_labels',
                        action='store_false', default=True,
                        help='Disable fixed (manually corrected) GT labels')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if event_detection/metrics.json already exists')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent
    config_path = checkpoint_dir / 'config.json'
    output_dir = checkpoint_dir / 'event_detection'

    if args.skip_existing and (output_dir / 'metrics.json').exists():
        print(f"SKIP: {checkpoint_dir.name} (metrics.json exists)")
        return

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    test_subject = config['test_subjects'][0]
    seq_len = config.get('ranking_seq_len', config.get('sequence_length', 16))
    stride = config.get('ranking_stride', config.get('sequence_stride', 2))
    span = config.get('ranking_span', (seq_len - 1) * stride + 1)
    image_size = config.get('image_size', 224)

    print(f"Checkpoint: {checkpoint_dir.name}")
    print(f"Test subject: {test_subject}")
    print(f"Sequence: T={seq_len}, stride={stride}, span={span}")

    # Find H5 files
    data_dir = config.get('data_dir', DATA_DIR)
    if not Path(data_dir).exists():
        data_dir = DATA_DIR
    h5_files = find_h5_files(test_subject, data_dir)
    if not h5_files:
        print(f"ERROR: no finger_flex H5 files for subject {test_subject}")
        return
    print(f"H5 files: {len(h5_files)}")

    # Load model
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model, config = load_cbmf_model(
        str(checkpoint_path), str(config_path), device)
    transform = create_transform(image_size)

    # Load all H5 data
    all_data = {}
    for h5_path in h5_files:
        images, num_frames = load_h5_data(h5_path)
        all_data[h5_path] = {'images': images, 'num_frames': num_frames}

    # Run inference per session
    print(f"\nRunning inference (eval_stride={args.eval_stride})...")
    prediction_logs = {}

    for h5_path, data in all_data.items():
        images = data['images']
        num_frames = data['num_frames']
        h5_stem = Path(h5_path).stem

        eval_positions = list(range(span - 1, num_frames, args.eval_stride))
        print(f"  {h5_stem}: {num_frames} frames, {len(eval_positions)} windows")

        log = run_inference(
            model, images, seq_len, stride, args.eval_stride,
            device, transform)
        prediction_logs[h5_path] = log

    # Compute GT flex labels
    gt_label_params = {
        'vel_threshold_deg_s': 80.0,
        'angle_pct_threshold': 0.1,
        'thumb_vel_threshold_deg_s': args.thumb_vel_threshold,
        'thumb_angle_pct_threshold': 0.1,
    }
    h5_cache = compute_gt_flex_labels(
        prediction_logs,
        vel_threshold_deg_s=gt_label_params['vel_threshold_deg_s'],
        angle_pct_threshold=gt_label_params['angle_pct_threshold'],
        thumb_vel_threshold_deg_s=gt_label_params['thumb_vel_threshold_deg_s'],
        thumb_angle_pct_threshold=gt_label_params['thumb_angle_pct_threshold'],
        use_fixed_labels=args.use_fixed_labels,
    )

    # Save raw predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    params_data = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_name': checkpoint_dir.name,
        'test_subject': test_subject,
        'model_type': 'cbmf',
        'model': {
            'seq_len': seq_len,
            'stride': stride,
            'span': span,
            'image_size': image_size,
        },
        'inference': {
            'eval_stride': args.eval_stride,
            'signal': 'p_vs_rest',
        },
        'use_fixed_labels': args.use_fixed_labels,
        'gt_flex_labels': gt_label_params,
        'grid_search': {
            'threshold_range': [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            'min_duration_range': [4, 6, 8, 10, 14, 20],
            'max_gap_range': [0, 4, 8, 12, 16, 24, 32],
            'target_metric': 'f1',
        },
        'h5_files': [str(f) for f in h5_files],
    }
    params_path = output_dir / 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params_data, f, indent=2)
    print(f"  Saved: {params_path.name}")

    for h5_path, log in prediction_logs.items():
        h5_stem = Path(h5_path).stem
        _, _, gt_flex_labels = h5_cache[h5_path]
        num_frames = all_data[h5_path]['num_frames']

        gt_per_finger = {}
        for finger_idx, finger in enumerate(FINGER_NAMES):
            col = gt_flex_labels[:, finger_idx]
            gt_per_finger[finger] = col.tolist()

        # Full-frame probability signals
        prob_signals = {}
        for finger in FINGER_NAMES:
            signal = extract_per_finger_signal(
                log, finger, 'p_vs_rest', num_frames, args.eval_stride)
            prob_signals[finger] = [round(float(v), 4) if not np.isnan(v) else None
                                    for v in signal]

        raw_path = output_dir / f'raw_predictions_{h5_stem}.json'
        raw_data = {
            'h5_file': str(h5_path),
            'checkpoint': str(checkpoint_path),
            'test_subject': test_subject,
            'model_type': 'cbmf',
            'seq_len': seq_len,
            'stride': stride,
            'span': span,
            'eval_stride': args.eval_stride,
            'fingers_with_refs': [],
            'num_frames': num_frames,
            'gt_flex_labels': gt_per_finger,
            'prob_signals': prob_signals,
            'predictions': log,
        }
        with open(raw_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
        print(f"  Saved: {raw_path.name}")

    # Compute event detection metrics
    print(f"\nComputing event detection metrics (grid search)...")
    best_params, per_file_metrics, overall = compute_event_metrics(
        prediction_logs, h5_cache, args.eval_stride)

    # Save metrics
    metrics_data = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_name': checkpoint_dir.name,
        'test_subject': test_subject,
        'model_type': 'cbmf',
        'seq_len': seq_len,
        'stride': stride,
        'best_params': best_params,
        'overall': {
            'micro': overall['micro'],
            'macro': overall['macro'],
            'per_finger': overall['per_finger'],
        },
        'per_session': per_file_metrics,
    }
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    # Print summary
    m = overall['micro']
    print(f"\n{'='*60}")
    print(f"Results: {checkpoint_dir.name}")
    print(f"  Best params: thresh={best_params['threshold']}, "
          f"min_dur={best_params['min_duration']}, "
          f"max_gap={best_params['max_gap']}")
    print(f"  Micro: P={m['precision']:.3f} R={m['recall']:.3f} "
          f"F1={m['f1']:.3f} Acc={m['accuracy']:.3f}")
    print(f"\n  Per session:")
    for pf in per_file_metrics:
        pm = pf['micro']
        print(f"    {pf['h5_stem']}: "
              f"P={pm['precision']:.3f} R={pm['recall']:.3f} "
              f"F1={pm['f1']:.3f} Acc={pm['accuracy']:.3f}")
    print(f"\n  Saved: {metrics_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

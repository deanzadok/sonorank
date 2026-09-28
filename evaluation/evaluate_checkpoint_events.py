"""
Evaluate a single ranking checkpoint for event-level flex detection.

For the checkpoint's test subject, processes all finger_flex H5 files:
  1. Auto-detects per-finger flex reference windows from kinematics
  2. Runs sliding-window comparisons (P(query > rest), P(query > flex_i))
  3. Saves raw per-frame predictions as JSON (for future metric recomputation)
  4. Runs grid-search event detection and saves metrics

All output is stored in {checkpoint_dir}/event_detection/.

Usage:
    python -m evaluation.evaluate_checkpoint_events \
        --checkpoint outputs/experiments/seqlen_sweep/stride2/.../checkpoint_best.pth \
        --gpu-id 0

    # Skip if results already exist:
    python -m evaluation.evaluate_checkpoint_events \
        --checkpoint ... --gpu-id 0 --skip-existing

Reader's guide: this file evaluates the ranking model itself by constructing
reference sequences and logging pairwise comparisons. It differs from
``evaluate_flex_embedding_events`` which evaluates the downstream classifier.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from evaluation.eval_utils import (
    load_h5_data, load_multi_finger_model,
    create_transform, prepare_sequence, FPS,
)
from evaluation.classify_with_ranking_references import (
    find_reference_windows_multi, FINGER_NAMES,
)
from evaluation.event_detection_metrics import (
    evaluate_finger, aggregate_results, load_joint_angles,
    extract_per_finger_signal,
)
from trainer.load import compute_flex_intervals, load_fixed_labels

DATA_DIR = "data/processed_h5/h5_files"


def find_h5_files(subject_id, data_dir=DATA_DIR):
    """Find all finger_flex H5 files for a given subject."""
    data_dir = Path(data_dir)
    pattern = f"recordings_{subject_id:02d}_*finger_flex*.h5"
    files = sorted(data_dir.glob(pattern))
    if not files:
        pattern = f"recordings_{subject_id}_*finger_flex*.h5"
        files = sorted(data_dir.glob(pattern))
    return [str(f) for f in files]


def build_static_rest(images, frame_idx, seq_len, transform, device):
    """Build rest reference by duplicating a single frame seq_len times."""
    img = images[frame_idx].copy().astype(np.float32) / 255.0
    result = transform(image=img)
    frame_tensor = result['image']  # [3, H, W]
    return frame_tensor.unsqueeze(0).repeat(
        seq_len, 1, 1, 1).unsqueeze(0).to(device)  # [1, T, 3, H, W]


def run_comparisons(model, all_data, flex_info, seq_len, stride, span,
                    eval_stride, transform, device, static_rest_frame=3):
    """Run sliding-window comparisons for all H5 files.

    Returns:
        dict: h5_path -> list of comparison entries, each with:
            frame, p_vs_rest (dict), p_vs_flex (dict of dicts)
    """
    fingers_with_refs = sorted(flex_info.keys(), key=FINGER_NAMES.index)

    # Prepare flex reference tensors
    ref_flex = {}
    for finger, (h5, start) in flex_info.items():
        ref_flex[finger] = prepare_sequence(
            all_data[h5]['images'], start, seq_len, stride, transform
        ).to(device)

    results = {}

    for h5_path, data in all_data.items():
        images = data['images']
        num_frames = data['num_frames']
        h5_stem = Path(h5_path).stem

        # Build static rest reference from this file's early frame
        rest_frame_idx = min(static_rest_frame, num_frames - 1)
        ref_rest = build_static_rest(images, rest_frame_idx, seq_len,
                                     transform, device)

        print(f"  {h5_stem}: {num_frames} frames, "
              f"{len(range(span - 1, num_frames, eval_stride))} windows")

        comparison_log = []
        eval_positions = list(range(span - 1, num_frames, eval_stride))

        for frame_idx in tqdm(eval_positions, desc=f"  {h5_stem}", leave=False):
            window_start = frame_idx - span + 1

            # Prepare query
            query_seq = prepare_sequence(
                images, window_start, seq_len, stride, transform).to(device)

            with torch.no_grad():
                # Compare against rest
                logits_rest = model(query_seq, ref_rest)
                p_vs_rest = {
                    f: float(torch.sigmoid(logits_rest[f]).item())
                    for f in FINGER_NAMES
                }

                # Compare against each flex reference
                p_vs_flex = {}
                for ref_finger, ref_seq in ref_flex.items():
                    logits_flex = model(query_seq, ref_seq)
                    p_vs_flex[ref_finger] = {
                        f: float(torch.sigmoid(logits_flex[f]).item())
                        for f in FINGER_NAMES
                    }

            comparison_log.append({
                'frame': frame_idx,
                'p_vs_rest': p_vs_rest,
                'p_vs_flex': p_vs_flex,
            })

        results[h5_path] = comparison_log

    return results, fingers_with_refs


def compute_gt_flex_labels(comparison_logs, vel_threshold_deg_s=80.0,
                           angle_pct_threshold=0.1,
                           thumb_vel_threshold_deg_s=50.0,
                           thumb_angle_pct_threshold=0.1,
                           use_fixed_labels=True):
    """Load GT flex labels for all H5 files in comparison_logs.

    When use_fixed_labels is True, tries manually corrected labels first
    and falls back to auto-computed labels if no JSON exists.

    Returns:
        dict: h5_path -> (joint_angles_rad, num_frames, gt_flex_labels)
    """
    h5_cache = {}
    for h5_path in comparison_logs:
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


def compute_event_metrics(comparison_logs, fingers_with_refs, eval_stride,
                          h5_cache):
    """Compute event detection metrics with grid search optimization.

    Args:
        comparison_logs: dict h5_path -> list of comparison entries
        fingers_with_refs: list of finger names with flex references
        eval_stride: stride between evaluation windows
        h5_cache: dict h5_path -> (joint_angles_rad, num_frames, gt_flex_labels)

    Returns:
        best_params: dict with threshold, min_duration, max_gap
        per_file_metrics: list of per-file results (with macro, per_finger, events)
        overall_metrics: aggregated micro/macro metrics
    """
    # Grid search ranges
    threshold_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    min_duration_range = [4, 6, 8, 10, 14, 20]
    max_gap_range = [0, 4, 8, 12, 16, 24, 32]

    # Grid search
    best_score = -1
    best_params = {}
    best_all_results = []

    from itertools import product
    total = len(threshold_range) * len(min_duration_range) * len(max_gap_range)

    for thresh, min_dur, max_gap in product(
            threshold_range, min_duration_range, max_gap_range):
        all_finger_results = []

        for h5_path, comparison_log in comparison_logs.items():
            _, num_frames, gt_flex_labels = h5_cache[h5_path]

            for finger_idx, finger in enumerate(FINGER_NAMES):
                if finger not in fingers_with_refs:
                    continue
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
            best_all_results = all_finger_results

    # Re-run with best params to get per-file breakdown
    per_file_metrics = []
    overall_finger_results = []

    for h5_path, comparison_log in comparison_logs.items():
        _, num_frames, gt_flex_labels = h5_cache[h5_path]
        h5_stem = Path(h5_path).stem

        file_finger_results = []
        for finger_idx, finger in enumerate(FINGER_NAMES):
            if finger not in fingers_with_refs:
                continue
            result = evaluate_finger(
                comparison_log, finger, finger_idx,
                gt_flex_labels, num_frames, eval_stride,
                'p_vs_rest', best_params['threshold'],
                best_params['min_duration'], best_params['max_gap'],
                normalize=False)
            file_finger_results.append(result)
            overall_finger_results.append(result)

        file_agg = aggregate_results(file_finger_results)

        # Collect per-finger event lists for this session
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
        description='Evaluate a ranking checkpoint for event-level flex detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint_best.pth')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--eval-stride', type=int, default=8,
                        help='Stride between evaluation windows (default: 8)')
    parser.add_argument('--static-rest-frame', type=int, default=3,
                        help='Frame index for static rest reference (default: 3)')
    parser.add_argument('--thumb-vel-threshold', type=float, default=50.0,
                        help='Thumb velocity threshold for GT flex labels')
    parser.add_argument('--no-fixed-labels', dest='use_fixed_labels',
                        action='store_false', default=True,
                        help='Disable fixed (manually corrected) GT labels; use auto-computed instead')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if event_detection/metrics.json already exists')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent
    config_path = checkpoint_dir / 'config.json'
    output_dir = checkpoint_dir / 'event_detection'

    # Skip if already done
    if args.skip_existing and (output_dir / 'metrics.json').exists():
        print(f"SKIP: {checkpoint_dir.name} (metrics.json exists)")
        return

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    test_subject = config['test_subjects'][0]
    seq_len = config['sequence_length']
    stride = config.get('sequence_stride', config.get('stride', 1))
    span = (seq_len - 1) * stride + 1
    image_size = config.get('image_size', 224)

    print(f"Checkpoint: {checkpoint_dir.name}")
    print(f"Test subject: {test_subject}")
    print(f"Sequence: T={seq_len}, stride={stride}, span={span}")

    # Find H5 files
    data_dir = config.get('data_dir', DATA_DIR)
    h5_files = find_h5_files(test_subject, data_dir)
    if not h5_files:
        print(f"ERROR: no finger_flex H5 files for subject {test_subject}")
        return
    print(f"H5 files: {len(h5_files)}")

    # Load model
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model, _ = load_multi_finger_model(str(checkpoint_path), str(config_path), device)
    transform = create_transform(image_size)

    # Load all H5 data
    all_data = {}
    for h5_path in h5_files:
        images, joint_angles, num_frames = load_h5_data(h5_path)
        all_data[h5_path] = {
            'images': images, 'joint_angles': joint_angles, 'num_frames': num_frames,
        }

    # Find flex reference windows
    _, flex_info = find_reference_windows_multi(all_data, span)
    fingers_with_refs = sorted(flex_info.keys(), key=FINGER_NAMES.index)
    print(f"\nFlex references found for: {', '.join(fingers_with_refs)}")
    for finger, (h5, start) in flex_info.items():
        print(f"  {finger:>6}: {Path(h5).name} frames {start}-{start + span - 1}")

    # Run comparisons
    print(f"\nRunning comparisons (eval_stride={args.eval_stride})...")
    comparison_logs, fingers_with_refs = run_comparisons(
        model, all_data, flex_info, seq_len, stride, span,
        args.eval_stride, transform, device,
        static_rest_frame=args.static_rest_frame,
    )

    # Compute GT flex labels (needed for raw predictions and metrics)
    gt_label_params = {
        'vel_threshold_deg_s': 80.0,
        'angle_pct_threshold': 0.1,
        'thumb_vel_threshold_deg_s': args.thumb_vel_threshold,
        'thumb_angle_pct_threshold': 0.1,
    }
    h5_cache = compute_gt_flex_labels(
        comparison_logs,
        vel_threshold_deg_s=gt_label_params['vel_threshold_deg_s'],
        angle_pct_threshold=gt_label_params['angle_pct_threshold'],
        thumb_vel_threshold_deg_s=gt_label_params['thumb_vel_threshold_deg_s'],
        thumb_angle_pct_threshold=gt_label_params['thumb_angle_pct_threshold'],
        use_fixed_labels=args.use_fixed_labels,
    )

    # Save raw predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reproducibility params
    params_data = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_name': checkpoint_dir.name,
        'test_subject': test_subject,
        'model': {
            'seq_len': seq_len,
            'stride': stride,
            'span': span,
            'image_size': image_size,
        },
        'inference': {
            'eval_stride': args.eval_stride,
            'static_rest_frame': args.static_rest_frame,
            'signal': 'p_vs_rest',
        },
        'flex_references': {f: {'h5': str(h5), 'start': s}
                            for f, (h5, s) in flex_info.items()},
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

    for h5_path, log in comparison_logs.items():
        h5_stem = Path(h5_path).stem
        _, _, gt_flex_labels = h5_cache[h5_path]

        num_frames = all_data[h5_path]['num_frames']

        # Per-finger GT binary labels for this session
        gt_per_finger = {}
        for finger_idx, finger in enumerate(FINGER_NAMES):
            col = gt_flex_labels[:, finger_idx]
            gt_per_finger[finger] = col.tolist()

        # Per-finger full-frame probability signals (interpolated from stride)
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
            'seq_len': seq_len,
            'stride': stride,
            'span': span,
            'eval_stride': args.eval_stride,
            'static_rest_frame': args.static_rest_frame,
            'fingers_with_refs': fingers_with_refs,
            'flex_refs': {f: {'h5': str(h5), 'start': s}
                          for f, (h5, s) in flex_info.items()},
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
        comparison_logs, fingers_with_refs, args.eval_stride,
        h5_cache=h5_cache,
    )

    # Save metrics
    metrics_data = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_name': checkpoint_dir.name,
        'test_subject': test_subject,
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

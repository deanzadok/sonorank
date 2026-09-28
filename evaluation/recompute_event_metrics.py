"""
Recompute event detection metrics from saved raw predictions.

Two improvements over the original evaluate_checkpoint_events.py:
1. Per-finger thresholds optimized on seq16 data, then applied to all seq lengths.
2. IoU-based matching (IoU >= 0.5) in addition to any-overlap matching.

Saves results as metrics_fixed_thresh.json and metrics_iou50.json inside
each checkpoint's event_detection/ folder.

Usage:
    python -m evaluation.recompute_event_metrics
    python -m evaluation.recompute_event_metrics --seq-lengths 2 4 8 16
    python -m evaluation.recompute_event_metrics --reference-seq 16

Reader's guide: this script never performs model inference. It reloads stored
probability traces, selects post-processing parameters, rematches intervals,
and writes a new metric summary for comparison.
"""

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from evaluation.event_detection_metrics import (
    FINGER_NAMES, FPS, load_joint_angles,
    extract_per_finger_signal, threshold_and_postprocess,
    binary_to_events, compute_rest_intervals, intervals_overlap,
    aggregate_results,
)
from trainer.load import compute_flex_intervals, load_fixed_labels


# ---------------------------------------------------------------------------
# IoU matching
# ---------------------------------------------------------------------------

def compute_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Compute IoU between two half-open intervals [a0, a1) and [b0, b1)."""
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    intersection = max(0, inter_end - inter_start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def match_events_iou(
    pred_events: List[Tuple[int, int]],
    gt_events: List[Tuple[int, int]],
    rest_intervals: List[Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> Dict:
    """Match predicted events to GT events using IoU threshold.

    TP: GT event that has at least one predicted event with IoU >= threshold.
    FN: GT event with no prediction reaching IoU threshold.
    FP: Predicted event whose best IoU with any GT event < threshold.
    TN: Rest interval with no false alarm predictions.
    """
    gt_matched = set()
    pred_matched = set()

    for gi, gt in enumerate(gt_events):
        for pi, pred in enumerate(pred_events):
            if compute_iou(pred, gt) >= iou_threshold:
                gt_matched.add(gi)
                pred_matched.add(pi)
                break

    tp = len(gt_matched)
    fn = len(gt_events) - tp

    # FP: predicted events not matched to any GT
    false_alarm_preds = [pred_events[pi] for pi in range(len(pred_events))
                         if pi not in pred_matched]
    fp = len(false_alarm_preds)

    # TN: rest intervals with no false alarm predictions
    rest_clean = 0
    for rest in rest_intervals:
        has_fa = any(intervals_overlap(fa, rest) for fa in false_alarm_preds)
        if not has_fa:
            rest_clean += 1
    tn = rest_clean

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
        'num_gt_events': len(gt_events), 'num_pred_events': len(pred_events),
        'num_rest_intervals': len(rest_intervals),
    }


# ---------------------------------------------------------------------------
# Per-finger evaluation with configurable matching
# ---------------------------------------------------------------------------

def evaluate_finger_from_raw(
    predictions: List[dict],
    finger: str,
    finger_idx: int,
    gt_flex_labels: np.ndarray,
    num_frames: int,
    eval_stride: int,
    threshold: float,
    min_duration: int,
    max_gap: int,
    match_fn='overlap',
    iou_threshold: float = 0.5,
) -> Dict:
    """Evaluate one finger from raw predictions with configurable matching."""
    signal = extract_per_finger_signal(
        predictions, finger, 'p_vs_rest', num_frames, eval_stride)

    pred_binary = threshold_and_postprocess(
        signal, threshold, min_duration, max_gap)

    pred_events = binary_to_events(pred_binary)
    gt_binary = gt_flex_labels[:, finger_idx]
    gt_events = binary_to_events(gt_binary)
    rest_intervals = compute_rest_intervals(gt_events, num_frames)

    if match_fn == 'iou':
        metrics = match_events_iou(
            pred_events, gt_events, rest_intervals, iou_threshold)
    else:
        from evaluation.event_detection_metrics import match_events_overlap
        metrics = match_events_overlap(pred_events, gt_events, rest_intervals)

    metrics['finger'] = finger
    return metrics


# ---------------------------------------------------------------------------
# Load raw predictions and GT for a checkpoint folder
# ---------------------------------------------------------------------------

def load_checkpoint_data(event_dir: Path, vel_threshold_deg_s=80.0,
                         angle_pct_threshold=0.1,
                         thumb_vel_threshold_deg_s=50.0,
                         thumb_angle_pct_threshold=0.1,
                         use_fixed_labels=True):
    """Load all raw prediction files and GT flex labels for a checkpoint.

    When use_fixed_labels is True, tries manually corrected labels first
    and falls back to auto-computed labels if no JSON exists.
    """
    raw_files = sorted(event_dir.glob('raw_predictions_*.json'))
    if not raw_files:
        return None

    data = []
    for raw_path in raw_files:
        with open(raw_path) as f:
            raw = json.load(f)

        h5_path = raw['h5_file']
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
        data.append({
            'h5_path': h5_path,
            'h5_stem': Path(h5_path).stem,
            'predictions': raw['predictions'],
            'num_frames': raw['num_frames'],
            'eval_stride': raw['eval_stride'],
            'fingers_with_refs': raw['fingers_with_refs'],
            'gt_flex_labels': gt_flex_labels,
        })

    return data


# ---------------------------------------------------------------------------
# Grid search for per-finger thresholds
# ---------------------------------------------------------------------------

def find_per_finger_thresholds(
    all_checkpoint_data,
    threshold_range,
    min_duration_range,
    max_gap_range,
    target_metric='f1',
    match_fn='overlap',
    iou_threshold=0.5,
):
    """Find best (threshold, min_duration, max_gap) per finger across all
    seq16 checkpoints, optimizing the given metric.

    Returns dict: finger -> {threshold, min_duration, max_gap, metric_value}
    """
    # Collect all file data across all seq16 checkpoints
    all_files = []
    for ckpt_data in all_checkpoint_data:
        all_files.extend(ckpt_data)

    combos = list(product(threshold_range, min_duration_range, max_gap_range))

    best_per_finger = {}

    for finger_idx, finger in enumerate(FINGER_NAMES):
        best_score = -1
        best_params = {}

        for thresh, min_dur, max_gap in combos:
            finger_results = []
            for file_data in all_files:
                if finger not in file_data['fingers_with_refs']:
                    continue
                result = evaluate_finger_from_raw(
                    file_data['predictions'], finger, finger_idx,
                    file_data['gt_flex_labels'], file_data['num_frames'],
                    file_data['eval_stride'],
                    thresh, min_dur, max_gap,
                    match_fn=match_fn, iou_threshold=iou_threshold)
                finger_results.append(result)

            if not finger_results:
                continue

            # Aggregate this finger across all files
            total_tp = sum(r['tp'] for r in finger_results)
            total_fp = sum(r['fp'] for r in finger_results)
            total_fn = sum(r['fn'] for r in finger_results)
            total_tn = sum(r['tn'] for r in finger_results)

            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            acc = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) \
                if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0

            scores = {'f1': f1, 'precision': p, 'recall': r, 'accuracy': acc}
            score = scores[target_metric]

            if score > best_score:
                best_score = score
                best_params = {
                    'threshold': thresh,
                    'min_duration': min_dur,
                    'max_gap': max_gap,
                    target_metric: score,
                    'precision': p, 'recall': r, 'f1': f1, 'accuracy': acc,
                    'tp': total_tp, 'fp': total_fp,
                    'fn': total_fn, 'tn': total_tn,
                }

        best_per_finger[finger] = best_params
        print(f"  {finger:>6}: thresh={best_params.get('threshold', 'N/A')}, "
              f"min_dur={best_params.get('min_duration', 'N/A')}, "
              f"max_gap={best_params.get('max_gap', 'N/A')}, "
              f"{target_metric}={best_params.get(target_metric, 0):.3f}")

    return best_per_finger


# ---------------------------------------------------------------------------
# Apply fixed thresholds to a checkpoint
# ---------------------------------------------------------------------------

def evaluate_checkpoint_fixed(checkpoint_data, per_finger_params, match_fn='overlap',
                              iou_threshold=0.5):
    """Evaluate a checkpoint using fixed per-finger thresholds."""
    all_finger_results = []
    per_session = []

    for file_data in checkpoint_data:
        session_results = []
        for finger_idx, finger in enumerate(FINGER_NAMES):
            if finger not in file_data['fingers_with_refs']:
                continue
            if finger not in per_finger_params:
                continue

            params = per_finger_params[finger]
            result = evaluate_finger_from_raw(
                file_data['predictions'], finger, finger_idx,
                file_data['gt_flex_labels'], file_data['num_frames'],
                file_data['eval_stride'],
                params['threshold'], params['min_duration'], params['max_gap'],
                match_fn=match_fn, iou_threshold=iou_threshold)
            session_results.append(result)
            all_finger_results.append(result)

        session_agg = aggregate_results(session_results)
        per_session.append({
            'h5_stem': file_data['h5_stem'],
            'micro': session_agg['micro'],
            'per_finger': session_agg['per_finger'],
        })

    overall = aggregate_results(all_finger_results)
    return overall, per_session


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Recompute event metrics with per-finger thresholds and IoU matching')
    parser.add_argument('--sweep-dir', type=str,
                        default='outputs/experiments/seqlen_sweep/stride2',
                        help='Directory containing seqlen sweep checkpoints')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[2, 4, 8, 16],
                        help='Sequence lengths to evaluate')
    parser.add_argument('--reference-seq', type=int, default=16,
                        help='Sequence length to optimize thresholds on (default: 16)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--target-metric', type=str, default='f1',
                        choices=['f1', 'accuracy', 'recall', 'precision'])
    parser.add_argument('--thumb-vel-threshold', type=float, default=50.0)
    parser.add_argument('--no-fixed-labels', dest='use_fixed_labels',
                        action='store_false', default=True,
                        help='Disable fixed (manually corrected) GT labels; use auto-computed instead')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    # Grid search ranges
    threshold_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    min_duration_range = [4, 6, 8, 10, 14, 20]
    max_gap_range = [0, 4, 8, 12, 16, 24, 32]

    # -----------------------------------------------------------------------
    # Step 1: Load seq16 data and find per-finger thresholds
    # -----------------------------------------------------------------------
    ref_seq = args.reference_seq
    print(f"Step 1: Finding per-finger thresholds on seq{ref_seq} data...")

    ref_pattern = f"seqlen_sweep_seq{ref_seq}_stride*"
    ref_folders = sorted(sweep_dir.glob(ref_pattern))
    if not ref_folders:
        print(f"ERROR: no folders matching {ref_pattern} in {sweep_dir}")
        return

    ref_data_all = []
    for folder in ref_folders:
        event_dir = folder / 'event_detection'
        data = load_checkpoint_data(
            event_dir, thumb_vel_threshold_deg_s=args.thumb_vel_threshold,
            use_fixed_labels=args.use_fixed_labels)
        if data:
            ref_data_all.append(data)
            print(f"  Loaded {folder.name}: {len(data)} files")

    # --- Optimize for overlap matching ---
    print(f"\nOptimizing per-finger thresholds for OVERLAP matching "
          f"(target: {args.target_metric})...")
    per_finger_params_overlap = find_per_finger_thresholds(
        ref_data_all, threshold_range, min_duration_range, max_gap_range,
        target_metric=args.target_metric, match_fn='overlap')

    # --- Optimize for IoU matching ---
    print(f"\nOptimizing per-finger thresholds for IoU>={args.iou_threshold} matching "
          f"(target: {args.target_metric})...")
    per_finger_params_iou = find_per_finger_thresholds(
        ref_data_all, threshold_range, min_duration_range, max_gap_range,
        target_metric=args.target_metric, match_fn='iou',
        iou_threshold=args.iou_threshold)

    # Save the reference thresholds
    thresh_output = sweep_dir / 'per_finger_thresholds.json'
    thresh_data = {
        'reference_seq_len': ref_seq,
        'target_metric': args.target_metric,
        'overlap_params': per_finger_params_overlap,
        'iou_params': {
            'iou_threshold': args.iou_threshold,
            'per_finger': per_finger_params_iou,
        },
    }
    with open(thresh_output, 'w') as f:
        json.dump(thresh_data, f, indent=2)
    print(f"\nSaved thresholds: {thresh_output}")

    # -----------------------------------------------------------------------
    # Step 2: Apply fixed thresholds to all seq lengths
    # -----------------------------------------------------------------------
    print(f"\nStep 2: Applying fixed thresholds to seq lengths {args.seq_lengths}...")

    for seq_len in args.seq_lengths:
        pattern = f"seqlen_sweep_seq{seq_len}_stride*"
        folders = sorted(sweep_dir.glob(pattern))
        if not folders:
            print(f"\n  WARNING: no folders for seq_len={seq_len}")
            continue

        print(f"\n  seq_len={seq_len}: {len(folders)} checkpoints")

        for folder in folders:
            event_dir = folder / 'event_detection'
            ckpt_data = load_checkpoint_data(
                event_dir, thumb_vel_threshold_deg_s=args.thumb_vel_threshold,
                use_fixed_labels=args.use_fixed_labels)
            if not ckpt_data:
                continue

            config_path = folder / 'config.json'
            with open(config_path) as f:
                config = json.load(f)

            # --- Overlap matching with overlap-optimized thresholds ---
            overall_overlap, per_session_overlap = evaluate_checkpoint_fixed(
                ckpt_data, per_finger_params_overlap, match_fn='overlap')

            metrics_overlap = {
                'checkpoint_name': folder.name,
                'test_subject': config['test_subjects'][0],
                'seq_len': seq_len,
                'matching': 'overlap',
                'per_finger_params': per_finger_params_overlap,
                'overall': {
                    'micro': overall_overlap['micro'],
                    'macro': overall_overlap['macro'],
                    'per_finger': overall_overlap['per_finger'],
                },
                'per_session': per_session_overlap,
            }
            out_path = event_dir / 'metrics_fixed_thresh.json'
            with open(out_path, 'w') as f:
                json.dump(metrics_overlap, f, indent=2)

            # --- IoU matching with IoU-optimized thresholds ---
            overall_iou, per_session_iou = evaluate_checkpoint_fixed(
                ckpt_data, per_finger_params_iou, match_fn='iou',
                iou_threshold=args.iou_threshold)

            metrics_iou = {
                'checkpoint_name': folder.name,
                'test_subject': config['test_subjects'][0],
                'seq_len': seq_len,
                'matching': f'iou>={args.iou_threshold}',
                'per_finger_params': per_finger_params_iou,
                'overall': {
                    'micro': overall_iou['micro'],
                    'macro': overall_iou['macro'],
                    'per_finger': overall_iou['per_finger'],
                },
                'per_session': per_session_iou,
            }
            out_path = event_dir / 'metrics_iou50.json'
            with open(out_path, 'w') as f:
                json.dump(metrics_iou, f, indent=2)

            m_ov = overall_overlap['micro']
            m_iou = overall_iou['micro']
            print(f"    {folder.name}: "
                  f"overlap F1={m_ov['f1']:.3f} Acc={m_ov['accuracy']:.3f}  "
                  f"IoU50 F1={m_iou['f1']:.3f} Acc={m_iou['accuracy']:.3f}")

    # -----------------------------------------------------------------------
    # Step 3: Print summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    for matching, filename in [('Fixed thresh (overlap)', 'metrics_fixed_thresh.json'),
                               ('Fixed thresh (IoU>=0.5)', 'metrics_iou50.json')]:
        print(f"\n{matching}:")
        print(f"{'SeqLen':>8} {'N':>4} {'F1':>12} {'Precision':>12} "
              f"{'Recall':>12} {'Accuracy':>12}")
        print('-' * 64)

        for seq_len in args.seq_lengths:
            pattern = f"seqlen_sweep_seq{seq_len}_stride*"
            folders = sorted(sweep_dir.glob(pattern))
            f1s, precs, recs, accs = [], [], [], []
            for folder in folders:
                metrics_path = folder / 'event_detection' / filename
                if not metrics_path.exists():
                    continue
                with open(metrics_path) as f:
                    m = json.load(f)
                micro = m['overall']['micro']
                f1s.append(micro['f1'])
                precs.append(micro['precision'])
                recs.append(micro['recall'])
                accs.append(micro['accuracy'])

            if f1s:
                print(f"{seq_len:>8} {len(f1s):>4} "
                      f"{np.mean(f1s):>5.3f}\u00b1{np.std(f1s):.3f} "
                      f"{np.mean(precs):>5.3f}\u00b1{np.std(precs):.3f} "
                      f"{np.mean(recs):>5.3f}\u00b1{np.std(recs):.3f} "
                      f"{np.mean(accs):>5.3f}\u00b1{np.std(accs):.3f}")

    print(f"\nFiles saved in each event_detection/ folder:")
    print(f"  metrics_fixed_thresh.json  (overlap matching, per-finger thresholds from seq{ref_seq})")
    print(f"  metrics_iou50.json         (IoU>={args.iou_threshold} matching, same thresholds)")


if __name__ == '__main__':
    main()

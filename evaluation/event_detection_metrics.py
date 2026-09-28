"""
Event-level flex detection metrics from ranking reference classification output.

Takes the PKL output of classify_with_ranking_references.py, extracts per-finger
P(query > rest) signals, post-processes them into predicted flex intervals, and
computes event-level TP/FP/FN/TN by matching against GT flex labels.

Includes a grid search mode to optimize post-processing hyperparameters.

Usage:
    # Single evaluation with default params
    python -m evaluation.event_detection_metrics \
        --pkl-file evaluation/classify_refs_173051/subject23_log.pkl --plot

    # Grid search to find optimal params
    python -m evaluation.event_detection_metrics \
        --pkl-file evaluation/classify_refs_173051/subject23_log.pkl \
        --optimize --target-metric f1

Reader's guide: probability extraction, optional normalization, temporal
filtering, interval construction, and event matching are separate steps. This
allows operating points to change without rerunning the neural network.
"""

import argparse
import json
import pickle
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trainer.load import FINGER_JOINTS, compute_flex_intervals, load_fixed_labels

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
FINGER_COLORS = {'index': 'C0', 'middle': 'C1', 'ring': 'C2',
                 'pinky': 'C3', 'thumb': 'C4'}
FPS = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True regions. Returns list of (start, end_exclusive)."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def load_joint_angles(h5_path: str) -> Tuple[np.ndarray, int]:
    """Load and sum joint angles per finger from an H5 file.

    Returns:
        joint_angles_rad: [num_frames, 5] summed joint angles in radians.
        num_frames: Total number of frames.
    """
    df = pd.read_hdf(h5_path)
    num_frames = len(df)
    joint_angles = np.zeros((num_frames, 5), dtype=np.float32)
    for j, finger in enumerate(FINGER_NAMES):
        for jc in FINGER_JOINTS[finger]:
            if jc in df.columns:
                joint_angles[:, j] += df[jc].values.astype(np.float32)
    return joint_angles, num_frames


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def extract_per_finger_signal(
    comparison_log: List[dict],
    finger: str,
    signal_key: str,
    num_frames: int,
    eval_stride: int,
) -> np.ndarray:
    """Extract a full-frame-resolution signal for one finger.

    The comparison_log has one entry every eval_stride frames. This expands
    to [num_frames] using sample-and-hold: each value is assigned to the
    stride interval centered on its frame position.
    """
    signal = np.full(num_frames, np.nan, dtype=np.float64)

    frames = []
    values = []
    for entry in comparison_log:
        frame = entry['frame']
        if signal_key in entry and finger in entry[signal_key]:
            frames.append(frame)
            values.append(entry[signal_key][finger])

    if not frames:
        return signal

    # Assign each value to its stride interval
    for i, (f, v) in enumerate(zip(frames, values)):
        # Determine interval boundaries
        if i == 0:
            start = 0
        else:
            start = (frames[i - 1] + f) // 2 + 1
        if i == len(frames) - 1:
            end = num_frames
        else:
            end = (f + frames[i + 1]) // 2 + 1
        start = max(0, start)
        end = min(num_frames, end)
        signal[start:end] = v

    return signal


def znorm_signal(signal: np.ndarray) -> np.ndarray:
    """Z-normalize a signal (ignore NaN). Returns normalized copy."""
    valid = ~np.isnan(signal)
    if valid.sum() < 2:
        return signal.copy()
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    if std < 1e-8:
        return signal.copy()
    out = signal.copy()
    out[valid] = (signal[valid] - mean) / std
    return out


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def threshold_and_postprocess(
    signal: np.ndarray,
    threshold: float,
    min_duration_frames: int,
    max_merge_gap_frames: int,
) -> np.ndarray:
    """Convert continuous signal to post-processed binary prediction.

    Steps:
    1. Threshold: signal >= threshold -> 1, NaN -> 0
    2. Merge nearby intervals: fill gaps <= max_merge_gap_frames
    3. Remove short intervals: delete regions < min_duration_frames
    """
    binary = np.zeros(len(signal), dtype=bool)
    valid = ~np.isnan(signal)
    binary[valid] = signal[valid] >= threshold

    # Step 2: merge nearby intervals
    if max_merge_gap_frames > 0:
        runs = _find_contiguous_runs(binary)
        for i in range(len(runs) - 1):
            gap = runs[i + 1][0] - runs[i][1]
            if gap <= max_merge_gap_frames:
                binary[runs[i][1]:runs[i + 1][0]] = True

    # Step 3: remove short intervals
    if min_duration_frames > 1:
        runs = _find_contiguous_runs(binary)
        for start, end in runs:
            if (end - start) < min_duration_frames:
                binary[start:end] = False

    return binary.astype(np.float32)


def binary_to_events(binary_signal: np.ndarray) -> List[Tuple[int, int]]:
    """Convert per-frame binary signal to list of (start, end_exclusive) events."""
    return _find_contiguous_runs(binary_signal.astype(bool))


# ---------------------------------------------------------------------------
# Event matching (overlap-based)
# ---------------------------------------------------------------------------

def compute_rest_intervals(
    gt_events: List[Tuple[int, int]],
    num_frames: int,
) -> List[Tuple[int, int]]:
    """Derive rest intervals (gaps between GT events).

    Includes the gap before the first GT event and after the last.
    Only returns non-empty intervals (length > 0).
    """
    rest = []
    prev_end = 0
    for start, end in gt_events:
        if start > prev_end:
            rest.append((prev_end, start))
        prev_end = end
    if prev_end < num_frames:
        rest.append((prev_end, num_frames))
    return rest


def intervals_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Check if two half-open intervals [a0, a1) and [b0, b1) overlap."""
    return a[0] < b[1] and b[0] < a[1]


def match_events_overlap(
    pred_events: List[Tuple[int, int]],
    gt_events: List[Tuple[int, int]],
    rest_intervals: List[Tuple[int, int]],
) -> Dict:
    """Match predicted events to GT events using any-overlap matching.

    Definitions:
    - TP: GT event that has at least one overlapping predicted interval.
    - FN: GT event with no overlapping prediction.
    - FP: Predicted event that does NOT overlap any GT event (false alarm).
    - TN: Rest interval that contains no false alarm predictions.

    TP + FN = total GT events.
    Precision = TP / (TP + FP), Recall = TP / (TP + FN).
    Accuracy = (TP + TN) / (TP + FP + FN + TN).
    """
    # TP/FN: which GT events are detected?
    gt_matched = set()
    for gi, gt in enumerate(gt_events):
        for pred in pred_events:
            if intervals_overlap(pred, gt):
                gt_matched.add(gi)
                break

    tp = len(gt_matched)
    fn = len(gt_events) - tp

    # FP: predicted events that don't overlap ANY GT event
    false_alarm_preds = []
    for pred in pred_events:
        overlaps_gt = any(intervals_overlap(pred, gt) for gt in gt_events)
        if not overlaps_gt:
            false_alarm_preds.append(pred)
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
        'num_false_alarm_preds': fp,
    }


# ---------------------------------------------------------------------------
# Per-finger evaluation
# ---------------------------------------------------------------------------

def evaluate_finger(
    comparison_log: List[dict],
    finger: str,
    finger_idx: int,
    gt_flex_labels: np.ndarray,
    num_frames: int,
    eval_stride: int,
    signal_key: str,
    threshold: float,
    min_duration_frames: int,
    max_merge_gap_frames: int,
    normalize: bool = False,
) -> Dict:
    """Full event detection pipeline for one finger."""
    # 1. Extract signal
    signal = extract_per_finger_signal(
        comparison_log, finger, signal_key, num_frames, eval_stride)

    # 1b. Optional z-normalization
    if normalize:
        signal = znorm_signal(signal)

    # 2. Threshold and post-process
    pred_binary = threshold_and_postprocess(
        signal, threshold, min_duration_frames, max_merge_gap_frames)

    # 3. Convert to events
    pred_events = binary_to_events(pred_binary)

    # 4. GT events from flex labels
    gt_binary = gt_flex_labels[:, finger_idx]
    gt_events = binary_to_events(gt_binary)

    # 5. Rest intervals from GT
    rest_intervals = compute_rest_intervals(gt_events, num_frames)

    # 6. Match and compute metrics
    metrics = match_events_overlap(pred_events, gt_events, rest_intervals)
    metrics['finger'] = finger
    metrics['gt_events'] = gt_events
    metrics['pred_events'] = pred_events
    metrics['rest_intervals'] = rest_intervals
    metrics['signal'] = signal
    metrics['pred_binary'] = pred_binary

    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(per_finger_results: List[Dict]) -> Dict:
    """Aggregate per-finger results into micro and macro metrics."""
    # Group by finger
    by_finger = {}
    for r in per_finger_results:
        fname = r['finger']
        if fname not in by_finger:
            by_finger[fname] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                                'num_gt_events': 0, 'num_pred_events': 0}
        for k in ['tp', 'fp', 'fn', 'tn', 'num_gt_events', 'num_pred_events']:
            by_finger[fname][k] += r[k]

    # Per-finger metrics
    finger_metrics = {}
    for fname, counts in by_finger.items():
        tp, fp, fn, tn = counts['tp'], counts['fp'], counts['fn'], counts['tn']
        total = tp + fp + fn + tn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        acc = (tp + tn) / total if total > 0 else 0.0
        finger_metrics[fname] = {
            **counts, 'precision': p, 'recall': rec, 'f1': f1, 'accuracy': acc}

    # Micro-averaged (sum all counts)
    total_tp = sum(c['tp'] for c in by_finger.values())
    total_fp = sum(c['fp'] for c in by_finger.values())
    total_fn = sum(c['fn'] for c in by_finger.values())
    total_tn = sum(c['tn'] for c in by_finger.values())
    total = total_tp + total_fp + total_fn + total_tn
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    micro_acc = (total_tp + total_tn) / total if total > 0 else 0.0

    # Macro-averaged (mean of per-finger)
    macro_p = np.mean([m['precision'] for m in finger_metrics.values()])
    macro_r = np.mean([m['recall'] for m in finger_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in finger_metrics.values()])
    macro_acc = np.mean([m['accuracy'] for m in finger_metrics.values()])

    return {
        'per_finger': finger_metrics,
        'micro': {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn,
            'precision': micro_p, 'recall': micro_r, 'f1': micro_f1,
            'accuracy': micro_acc,
        },
        'macro': {
            'precision': float(macro_p), 'recall': float(macro_r),
            'f1': float(macro_f1), 'accuracy': float(macro_acc),
        },
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    pkl_data: dict,
    h5_cache: dict,
    eval_stride: int,
    signal_key: str,
    target_metric: str,
    threshold_range: List[float],
    min_duration_range: List[int],
    max_gap_range: List[int],
    fingers: List[str],
    normalize: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """Grid search over (threshold, min_duration, max_gap)."""
    combos = list(product(threshold_range, min_duration_range, max_gap_range))
    print(f"Grid search: {len(combos)} combinations "
          f"({len(threshold_range)} thresholds x {len(min_duration_range)} durations "
          f"x {len(max_gap_range)} gaps)")

    all_results = []
    for ci, (thresh, min_dur, max_gap) in enumerate(combos):
        all_finger_results = []
        for file_data in pkl_data['per_file']:
            h5_path = file_data['h5_file']
            comparison_log = file_data['comparison_log']
            _, num_frames, gt_flex_labels = h5_cache[h5_path]

            for finger_idx, finger in enumerate(FINGER_NAMES):
                if finger not in fingers:
                    continue
                result = evaluate_finger(
                    comparison_log, finger, finger_idx,
                    gt_flex_labels, num_frames, eval_stride,
                    signal_key, thresh, min_dur, max_gap,
                    normalize=normalize)
                all_finger_results.append(result)

        agg = aggregate_results(all_finger_results)
        entry = {
            'threshold': thresh, 'min_duration': min_dur, 'max_gap': max_gap,
            **{f'micro_{k}': v for k, v in agg['micro'].items()},
            **{f'macro_{k}': v for k, v in agg['macro'].items()},
        }
        all_results.append(entry)

        if (ci + 1) % 50 == 0:
            print(f"  {ci + 1}/{len(combos)} done...")

    # Sort by target metric (micro-averaged)
    metric_key = f'micro_{target_metric}'
    all_results.sort(key=lambda x: x[metric_key], reverse=True)

    best = all_results[0]
    return best, all_results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_event_detection(
    comparison_log: List[dict],
    finger_results: List[Dict],
    gt_flex_labels: np.ndarray,
    num_frames: int,
    eval_stride: int,
    signal_key: str,
    threshold: float,
    h5_stem: str,
    output_path: str,
    joint_angles_rad: Optional[np.ndarray] = None,
):
    """Generate multi-panel diagnostic plot in classify_refs style.

    Panels:
    1. Raw joint angles per finger (degrees)
    2. P(query > rest) z-normalized per finger with threshold
    3. GT flex events (colored bars on class axis)
    4. Predicted events with TP/FP/FN coloring (colored bars on class axis)
    5-9. Per-finger signal detail with GT/pred shading
    """
    colors = {'index': 'C0', 'middle': 'C1', 'ring': 'C2',
              'pinky': 'C3', 'thumb': 'C4'}
    finger_names_in_results = [r['finger'] for r in finger_results]
    n_fingers = len(finger_results)

    # Build finger -> y-index mapping for class-axis panels
    class_names = ['rest'] + finger_names_in_results
    finger_y = {f: i + 1 for i, f in enumerate(finger_names_in_results)}

    n_panels = 4 + n_fingers  # angles + signal + GT events + pred events + per-finger
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 3 * n_panels), sharex=True)
    all_frames = np.arange(num_frames)
    panel = 0

    # --- Panel 1: Raw joint angles ---
    ax = axes[panel]; panel += 1
    if joint_angles_rad is not None:
        for j, finger in enumerate(FINGER_NAMES):
            if finger in finger_names_in_results:
                angles_deg = np.rad2deg(joint_angles_rad[:, j])
                ax.plot(all_frames, angles_deg, label=finger,
                        color=colors[finger], alpha=0.8, linewidth=1)
    ax.set_ylabel('Joint angle (deg)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{h5_stem} — Raw Joint Angles')

    # --- Panel 2: Z-normalized signal per finger ---
    ax = axes[panel]; panel += 1
    for result in finger_results:
        finger = result['finger']
        signal = result['signal']
        ax.plot(all_frames, signal, label=finger,
                color=colors[finger], alpha=0.8, linewidth=1)
    ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5,
               label=f'threshold={threshold}')
    ax.set_ylabel(f'{signal_key}\n(z-norm)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'P(query > rest) — Z-Normalized per Finger')

    # --- Panel 3: GT flex events (class axis) ---
    ax = axes[panel]; panel += 1
    for result in finger_results:
        finger = result['finger']
        y = finger_y[finger]
        for start, end in result['gt_events']:
            ax.barh(y, end - start, left=start, height=0.6,
                    color=colors[finger], alpha=0.7, edgecolor='none')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_ylabel('Finger')
    ax.set_title('Ground Truth Flex Events')
    ax.set_ylim(-0.5, len(class_names) - 0.5)

    # --- Panel 4: Predicted events with TP/FP/FN (class axis) ---
    ax = axes[panel]; panel += 1
    # Collect overall metrics for title
    total_tp = sum(r['tp'] for r in finger_results)
    total_fp = sum(r['fp'] for r in finger_results)
    total_fn = sum(r['fn'] for r in finger_results)
    total_tn = sum(r['tn'] for r in finger_results)
    total = total_tp + total_fp + total_fn + total_tn
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    micro_acc = (total_tp + total_tn) / total if total > 0 else 0.0

    for result in finger_results:
        finger = result['finger']
        y = finger_y[finger]
        gt_events = result['gt_events']
        for pred_s, pred_e in result['pred_events']:
            is_tp = any(intervals_overlap((pred_s, pred_e), gt) for gt in gt_events)
            c = 'green' if is_tp else 'red'
            ax.barh(y, pred_e - pred_s, left=pred_s, height=0.6,
                    color=c, alpha=0.7, edgecolor='none')
        # Show FN as gray outlines on GT events that were missed
        gt_matched = set()
        for gi, gt in enumerate(gt_events):
            for pred_s, pred_e in result['pred_events']:
                if intervals_overlap((pred_s, pred_e), gt):
                    gt_matched.add(gi)
                    break
        for gi, (gs, ge) in enumerate(gt_events):
            if gi not in gt_matched:
                ax.barh(y, ge - gs, left=gs, height=0.6,
                        facecolor='none', edgecolor='gray', linewidth=1.5,
                        linestyle='--')

    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_ylabel('Finger')
    ax.set_title(f'Predicted Events — '
                 f'TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}  '
                 f'P={micro_p:.2f} R={micro_r:.2f} F1={micro_f1:.2f} Acc={micro_acc:.2f}')
    ax.set_ylim(-0.5, len(class_names) - 0.5)

    # Add legend for TP/FP/FN
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='TP'),
                       Patch(facecolor='red', alpha=0.7, label='FP'),
                       Patch(facecolor='none', edgecolor='gray', linestyle='--',
                             linewidth=1.5, label='FN')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # --- Panels 5+: Per-finger signal detail ---
    for result in finger_results:
        ax = axes[panel]; panel += 1
        finger = result['finger']
        signal = result['signal']
        gt_events = result['gt_events']

        # GT shading
        for start, end in gt_events:
            ax.axvspan(start, end, alpha=0.15, color='steelblue')

        # Pred shading (TP green, FP red)
        for pred_s, pred_e in result['pred_events']:
            is_tp = any(intervals_overlap((pred_s, pred_e), gt) for gt in gt_events)
            ax.axvspan(pred_s, pred_e, alpha=0.25,
                       color='green' if is_tp else 'red')

        # Signal line
        ax.plot(all_frames, signal, color=colors[finger], alpha=0.8, linewidth=1)
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)

        tp, fp, fn = result['tp'], result['fp'], result['fn']
        ax.set_ylabel(f'{finger}\nz-norm')
        ax.set_title(f'{finger} — TP={tp} FP={fp} FN={fn}  '
                     f'P={result["precision"]:.2f} R={result["recall"]:.2f} '
                     f'F1={result["f1"]:.2f}', fontsize=10, loc='left')

    axes[-1].set_xlabel('Frame')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Event-level flex detection metrics from ranking reference output')

    parser.add_argument('--pkl-file', type=str, nargs='+', required=True,
                        help='PKL file(s) from classify_with_ranking_references.py')
    parser.add_argument('--signal', type=str, default='p_vs_rest',
                        choices=['p_vs_rest', 'norm_scores'],
                        help='Signal to use (default: p_vs_rest)')

    # Detection hyperparameters
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-duration', type=int, default=10,
                        help='Min predicted flex interval in frames (default: 10)')
    parser.add_argument('--max-gap', type=int, default=16,
                        help='Max gap in frames to merge intervals (default: 16)')

    # GT flex label parameters
    parser.add_argument('--vel-threshold-deg-s', type=float, default=80.0)
    parser.add_argument('--angle-pct-threshold', type=float, default=0.1)
    parser.add_argument('--thumb-vel-threshold-deg-s', type=float, default=50.0)
    parser.add_argument('--thumb-angle-pct-threshold', type=float, default=0.1)

    # Optimization
    parser.add_argument('--optimize', action='store_true',
                        help='Run grid search')
    parser.add_argument('--target-metric', type=str, default='f1',
                        choices=['f1', 'accuracy', 'recall', 'precision'])
    parser.add_argument('--threshold-range', type=float, nargs='+',
                        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument('--min-duration-range', type=int, nargs='+',
                        default=[4, 6, 8, 10, 14, 20])
    parser.add_argument('--max-gap-range', type=int, nargs='+',
                        default=[0, 4, 8, 12, 16, 24, 32])

    # Signal normalization
    parser.add_argument('--normalize-signal', action='store_true',
                        help='Z-normalize signal per file per finger before thresholding')

    # Fixed labels
    parser.add_argument('--no-fixed-labels', dest='use_fixed_labels',
                        action='store_false', default=True,
                        help='Disable fixed (manually corrected) GT labels; use auto-computed instead')

    # Finger selection
    parser.add_argument('--fingers', type=str, nargs='+', default=None,
                        help='Evaluate only these fingers (default: all in PKL)')

    # Output
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--plot', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load PKL file(s) and merge
    pkl_data = {'fingers_with_refs': None, 'per_file': []}
    for pkl_path in args.pkl_file:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        if pkl_data['fingers_with_refs'] is None:
            pkl_data['fingers_with_refs'] = data['fingers_with_refs']
        pkl_data['per_file'].extend(data['per_file'])

    fingers_with_refs = pkl_data['fingers_with_refs']
    fingers = args.fingers if args.fingers else fingers_with_refs
    fingers = [f for f in fingers if f in fingers_with_refs]

    normalize = args.normalize_signal

    print(f"PKL files: {args.pkl_file}")
    print(f"Signal: {args.signal}" + (" (z-normalized)" if normalize else ""))
    print(f"Fingers: {fingers}")
    print(f"Files: {len(pkl_data['per_file'])}")

    # 2. Infer eval_stride
    sample_log = pkl_data['per_file'][0]['comparison_log']
    if len(sample_log) >= 2:
        eval_stride = sample_log[1]['frame'] - sample_log[0]['frame']
    else:
        eval_stride = 8
    print(f"Eval stride: {eval_stride}")

    # 3. Pre-load H5 files and compute GT flex labels
    h5_cache = {}
    for file_data in pkl_data['per_file']:
        h5_path = file_data['h5_file']
        joint_angles_rad, num_frames = load_joint_angles(h5_path)
        gt_flex_labels = None
        label_src = 'auto'
        if args.use_fixed_labels:
            h5_stem = Path(h5_path).stem
            gt_flex_labels = load_fixed_labels(h5_stem, num_frames)
            if gt_flex_labels is not None:
                label_src = 'fixed'
        if gt_flex_labels is None:
            gt_flex_labels = compute_flex_intervals(
                joint_angles_rad, fps=FPS,
                vel_threshold_deg_s=args.vel_threshold_deg_s,
                angle_pct_threshold=args.angle_pct_threshold,
                thumb_vel_threshold_deg_s=args.thumb_vel_threshold_deg_s,
                thumb_angle_pct_threshold=args.thumb_angle_pct_threshold,
            )
        h5_cache[h5_path] = (joint_angles_rad, num_frames, gt_flex_labels)
        print(f"  Loaded {Path(h5_path).stem}: {num_frames} frames, "
              f"flex rate: {gt_flex_labels.mean():.3f} [{label_src}]")

    print()

    if args.optimize:
        # Grid search mode
        best, all_results = run_grid_search(
            pkl_data, h5_cache, eval_stride,
            signal_key=args.signal,
            target_metric=args.target_metric,
            threshold_range=args.threshold_range,
            min_duration_range=args.min_duration_range,
            max_gap_range=args.max_gap_range,
            fingers=fingers,
            normalize=normalize,
        )

        print(f"\n{'='*70}")
        print(f"Top 15 configurations (by micro_{args.target_metric}):")
        print(f"{'='*70}")
        print(f"{'Thresh':>7} {'MinDur':>7} {'MaxGap':>7} "
              f"{'Prec':>6} {'Recall':>6} {'F1':>6} {'Acc':>6} "
              f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
        print('-' * 70)
        for entry in all_results[:15]:
            print(f"{entry['threshold']:>7.2f} {entry['min_duration']:>7d} "
                  f"{entry['max_gap']:>7d} "
                  f"{entry['micro_precision']:>6.3f} "
                  f"{entry['micro_recall']:>6.3f} "
                  f"{entry['micro_f1']:>6.3f} "
                  f"{entry['micro_accuracy']:>6.3f} "
                  f"{entry['micro_tp']:>5d} {entry['micro_fp']:>5d} "
                  f"{entry['micro_fn']:>5d} {entry['micro_tn']:>5d}")

        print(f"\nBest: threshold={best['threshold']}, "
              f"min_duration={best['min_duration']}, "
              f"max_gap={best['max_gap']}")
        print(f"  micro: P={best['micro_precision']:.3f} "
              f"R={best['micro_recall']:.3f} "
              f"F1={best['micro_f1']:.3f} "
              f"Acc={best['micro_accuracy']:.3f}")

        # Use best params for plotting
        threshold = best['threshold']
        min_duration = best['min_duration']
        max_gap = best['max_gap']

        if args.output:
            output_data = {
                'signal': args.signal,
                'target_metric': args.target_metric,
                'best_params': {
                    'threshold': best['threshold'],
                    'min_duration': best['min_duration'],
                    'max_gap': best['max_gap'],
                },
                'best_metrics': {
                    k: v for k, v in best.items()
                    if k.startswith('micro_') or k.startswith('macro_')
                },
                'top_15': all_results[:15],
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        # Single evaluation mode
        threshold = args.threshold
        min_duration = args.min_duration
        max_gap = args.max_gap

    # Run evaluation (or re-run with best params after grid search)
    all_finger_results = []
    per_file_results = []

    print(f"\nEvaluating with: threshold={threshold}, "
          f"min_duration={min_duration}, max_gap={max_gap}")
    print(f"{'='*70}")

    for file_data in pkl_data['per_file']:
        h5_path = file_data['h5_file']
        h5_stem = Path(h5_path).stem
        comparison_log = file_data['comparison_log']
        joint_angles_rad, num_frames, gt_flex_labels = h5_cache[h5_path]

        print(f"\n{h5_stem}:")
        file_finger_results = []

        for finger_idx, finger in enumerate(FINGER_NAMES):
            if finger not in fingers:
                continue

            result = evaluate_finger(
                comparison_log, finger, finger_idx,
                gt_flex_labels, num_frames, eval_stride,
                args.signal, threshold, min_duration, max_gap,
                normalize=normalize)

            file_finger_results.append(result)
            all_finger_results.append(result)

            print(f"  {finger:>8s}: TP={result['tp']:2d} FP={result['fp']:2d} "
                  f"FN={result['fn']:2d} TN={result['tn']:2d}  "
                  f"P={result['precision']:.3f} R={result['recall']:.3f} "
                  f"F1={result['f1']:.3f} Acc={result['accuracy']:.3f}  "
                  f"(GT={result['num_gt_events']} pred={result['num_pred_events']})")

        per_file_results.append({
            'h5_file': h5_path,
            'h5_stem': h5_stem,
            'finger_results': file_finger_results,
        })

        # Per-file plot
        if args.plot:
            plot_dir = Path(args.pkl_file[0]).parent / 'event_detection_plots'
            plot_dir.mkdir(exist_ok=True)
            plot_path = plot_dir / f'{h5_stem}_events.png'
            plot_event_detection(
                comparison_log, file_finger_results,
                gt_flex_labels, num_frames, eval_stride,
                args.signal, threshold, h5_stem, str(plot_path),
                joint_angles_rad=joint_angles_rad)
            print(f"  Plot saved: {plot_path}")

    # Aggregate
    overall = aggregate_results(all_finger_results)

    print(f"\n{'='*70}")
    print("Overall (micro-averaged):")
    m = overall['micro']
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print(f"  Precision={m['precision']:.3f} Recall={m['recall']:.3f} "
          f"F1={m['f1']:.3f} Accuracy={m['accuracy']:.3f}")

    print("\nPer-finger (aggregated across files):")
    for finger in FINGER_NAMES:
        if finger in overall['per_finger']:
            fm = overall['per_finger'][finger]
            print(f"  {finger:>8s}: TP={fm['tp']:3d} FP={fm['fp']:3d} "
                  f"FN={fm['fn']:3d} TN={fm['tn']:3d}  "
                  f"P={fm['precision']:.3f} R={fm['recall']:.3f} "
                  f"F1={fm['f1']:.3f} Acc={fm['accuracy']:.3f}")

    print(f"\nMacro-averaged: P={overall['macro']['precision']:.3f} "
          f"R={overall['macro']['recall']:.3f} "
          f"F1={overall['macro']['f1']:.3f} "
          f"Acc={overall['macro']['accuracy']:.3f}")


if __name__ == '__main__':
    main()

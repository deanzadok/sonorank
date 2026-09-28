"""
Fair comparison of flex embedding methods with coverage-based event matching.

Per-finger thresholds are optimized from the ranking backbone method, then
applied identically to all methods. This ensures a fair comparison at the
same operating point.

Usage:
    python -m evaluation.evaluate_methods_fair \
        --methods-dir outputs/experiments/methods

    # With z-normalization and stricter coverage:
    python -m evaluation.evaluate_methods_fair \
        --methods-dir outputs/experiments/methods \
        --normalize --min-gt-coverage 0.5 --min-pred-duration 10

Reader's guide: fold predictions are loaded first, each method follows the
threshold strategy encoded below, and event matching is performed for every
session and finger before aggregation. Inspect ``METHODS_OWN_THRESHOLDS`` and
the threshold-selection loop in ``main`` when auditing an operating point.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.event_detection_metrics import (
    znorm_signal,
    binary_to_events,
    compute_rest_intervals,
    load_joint_angles,
)
from trainer.load import compute_flex_intervals, load_fixed_labels

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
METHODS = ['flex_embedding_ranking_backbone', 'flex_embedding_no_backbone', 'flex_embedding_no_rest', 'multi_finger_ranking', 'bimbraw', 'mcintosh', 'castellini', 'huang', 'cbmf']
METHOD_LABELS = {
    'flex_embedding_ranking_backbone': 'Ranking Backbone',
    'flex_embedding_no_backbone': 'No Backbone',
    'flex_embedding_no_rest': 'No Rest Ref.',
    'multi_finger_ranking': 'Ranking (no FT)',
    'bimbraw': 'Bimbraw',
    'mcintosh': 'McIntosh',
    'castellini': 'Castellini',
    'huang': 'Huang',
    'cbmf': 'CBMF',
}
# Methods that need per-session z-normalization (uncalibrated outputs)
METHODS_ZNORM = {'multi_finger_ranking'}
# Methods that optimize their own thresholds (instead of using ranking backbone's)
METHODS_OWN_THRESHOLDS = {'multi_finger_ranking'}
# Methods that use argmax + per-finger biases instead of independent binary thresholding.
# Biases are optimized via coordinate descent: adjusted[f] = znorm[f] - bias[f],
# then winner = argmax(adjusted) if max(adjusted) > global_threshold.
# NOTE: tested empirically - argmax+biases hurts (0.306 vs 0.358 F1 with per-finger sweep).
# Root cause: p_vs_rest has near-zero variance so the argmax winner at rest frames is
# random noise; bias tuning cannot fix this. Keep empty.
METHODS_ARGMAX_BIASES: set = set()

# Methods that use diagonal p_vs_flex[f][f] values instead of p_vs_rest.
# NOTE: tested empirically - diagonal values hurt performance (0.314 vs 0.358 F1)
# because P(query > flex_ref) ≈ 0.5 during flex (not reliably above threshold).
# Diagonal values only work with argmax classification (old script approach), not
# with independent binary thresholding + event-level F1. Keep empty.
METHODS_DIAGONAL_SIGNAL: set = set()


# ---------------------------------------------------------------------------
# Ground truth labeling
# ---------------------------------------------------------------------------

def compute_velocity_gt_labels(
    joint_angles_rad: np.ndarray,
    fps: float = 20.0,
    vel_threshold_deg_s: float = 40.0,
    smooth_window: int = 5,
    min_event_duration: int = 3,
    gap_fill_frames: int = 10,
) -> np.ndarray:
    """Compute per-frame GT labels based on smoothed absolute angular velocity.

    Unlike compute_flex_intervals (hybrid velocity-onset + angle-offset), this
    uses pure velocity thresholding: a frame is labeled as active for a finger
    if its smoothed absolute velocity exceeds the threshold.

    Gap-filling bridges brief pauses between flexion and extension phases
    (the hold-at-full-flex period where velocity is near zero).

    Args:
        joint_angles_rad: [num_frames, 5] summed joint angles in radians.
        fps: Recording frame rate in Hz.
        vel_threshold_deg_s: Minimum |velocity| in deg/s to count as active.
            Default 40.0 matches the old classify_with_ranking_references approach.
        smooth_window: Window size in frames for box smoothing of velocity.
        min_event_duration: Remove events shorter than this (frames).
        gap_fill_frames: Fill gaps shorter than this between events for the
            same finger (bridges the hold phase).

    Returns:
        flex_labels: [num_frames, 5] binary float32 array (1.0 = active).
    """
    num_frames, num_fingers = joint_angles_rad.shape
    flex_labels = np.zeros((num_frames, num_fingers), dtype=np.float32)

    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1

    for finger_idx in range(num_fingers):
        angles_deg = np.rad2deg(joint_angles_rad[:, finger_idx])

        # Per-frame velocity in deg/s
        raw_velocity = np.concatenate([[0.0], np.diff(angles_deg) * fps])
        abs_velocity = np.abs(raw_velocity)
        smoothed = uniform_filter1d(abs_velocity, size=w, mode='nearest')

        active = smoothed >= vel_threshold_deg_s

        # Fill short gaps (bridges hold-at-full-flex between flex and extend)
        if gap_fill_frames > 0:
            runs = _find_runs(active)
            for i in range(len(runs) - 1):
                gap = runs[i + 1][0] - runs[i][1]
                if 0 < gap <= gap_fill_frames:
                    active[runs[i][1]:runs[i + 1][0]] = True

        # Remove short events
        runs = _find_runs(active)
        for s, e in runs:
            if (e - s) < min_event_duration:
                active[s:e] = False

        flex_labels[:, finger_idx] = active.astype(np.float32)

    return flex_labels


def apply_argmax_gt(
    flex_labels: np.ndarray,
    joint_angles_rad: np.ndarray,
    fps: float = 20.0,
    smooth_window: int = 5,
) -> np.ndarray:
    """Enforce mutual exclusivity: only the fastest-moving finger stays active.

    At each frame where multiple fingers are active, keeps only the one
    with the highest smoothed absolute velocity.

    Args:
        flex_labels: [num_frames, 5] binary labels (modified in place).
        joint_angles_rad: [num_frames, 5] for computing velocity tiebreaker.
        fps: Recording frame rate.
        smooth_window: Smoothing window for velocity.

    Returns:
        flex_labels: [num_frames, 5] with at most one finger active per frame.
    """
    num_frames, num_fingers = flex_labels.shape
    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1

    # Compute smoothed absolute velocity for all fingers
    abs_vel = np.zeros_like(flex_labels)
    for finger_idx in range(num_fingers):
        angles_deg = np.rad2deg(joint_angles_rad[:, finger_idx])
        raw_velocity = np.concatenate([[0.0], np.diff(angles_deg) * fps])
        abs_vel[:, finger_idx] = uniform_filter1d(
            np.abs(raw_velocity), size=w, mode='nearest')

    # At each frame, keep only the finger with highest velocity
    for t in range(num_frames):
        active = flex_labels[t] > 0
        if active.sum() > 1:
            # Zero out all, then set the fastest one
            velocities = abs_vel[t] * active
            winner = np.argmax(velocities)
            flex_labels[t] = 0.0
            flex_labels[t, winner] = 1.0

    return flex_labels


def apply_exclusive_gt_mask(sessions):
    """Mask out frames with multi-finger co-activation from GT and predictions.

    For frames where >1 finger is active in GT:
    - GT labels for ALL fingers are set to 0 (removes co-activation events)
    - Prediction signals for ALL fingers are set to NaN (ignored by threshold_and_filter)

    Modifies sessions in-place.

    Args:
        sessions: list of session dicts from load_fold_predictions, each with
            'prob_signals' (dict finger -> array), 'gt_flex_labels' (dict finger -> array),
            'num_frames', 'h5_stem'.

    Returns:
        list of stats dicts: [{h5_stem, num_frames, num_masked, pct_masked}, ...]
    """
    stats = []
    for sess in sessions:
        num_frames = sess['num_frames']
        gt_stack = np.column_stack(
            [sess['gt_flex_labels'][f] for f in FINGER_NAMES])  # [T, 5]
        active_count = gt_stack.sum(axis=1)
        multi_mask = active_count > 1

        num_masked = int(multi_mask.sum())
        for finger in FINGER_NAMES:
            sess['gt_flex_labels'][finger][multi_mask] = 0.0
            sess['prob_signals'][finger][multi_mask] = np.nan

        stats.append({
            'h5_stem': sess['h5_stem'],
            'num_frames': num_frames,
            'num_masked': num_masked,
            'pct_masked': 100.0 * num_masked / num_frames if num_frames > 0 else 0.0,
        })
    return stats


def _find_runs(mask: np.ndarray) -> list:
    """Find contiguous True regions. Returns list of (start, end) exclusive."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def recompute_gt_labels(h5_file, gt_mode, gt_vel_threshold, argmax_gt,
                        use_fixed_labels=True):
    """Load GT labels from fixed labels or compute from kinematics.

    Args:
        h5_file: Path to H5 file.
        gt_mode: 'hybrid' (compute_flex_intervals) or 'velocity' (absolute velocity).
        gt_vel_threshold: Velocity threshold in deg/s (used by both modes).
        argmax_gt: If True, enforce mutual exclusivity across fingers.
        use_fixed_labels: If True, try manually corrected labels first.

    Returns:
        gt_flex_labels: dict finger -> np.array of binary labels per frame.
    """
    joint_angles_rad, num_frames = load_joint_angles(h5_file)

    labels_array = None
    if use_fixed_labels:
        h5_stem = Path(h5_file).stem
        labels_array = load_fixed_labels(h5_stem, num_frames)

    if labels_array is None:
        if gt_mode == 'velocity':
            labels_array = compute_velocity_gt_labels(
                joint_angles_rad, fps=20.0,
                vel_threshold_deg_s=gt_vel_threshold,
            )
        else:
            labels_array = compute_flex_intervals(
                joint_angles_rad, fps=20.0,
                vel_threshold_deg_s=gt_vel_threshold,
            )

    if argmax_gt:
        labels_array = apply_argmax_gt(labels_array, joint_angles_rad)

    gt_flex = {}
    for i, finger in enumerate(FINGER_NAMES):
        gt_flex[finger] = labels_array[:, i]

    return gt_flex


# ---------------------------------------------------------------------------
# Coverage-based event matching
# ---------------------------------------------------------------------------

def match_events_coverage(
    pred_events: list,
    gt_events: list,
    rest_intervals: list,
    min_coverage: float = 0.3,
) -> dict:
    """Match predicted events to GT events using coverage-based matching.

    TP: GT event where predictions cover >= min_coverage of its duration.
    FN: GT event with insufficient coverage.
    FP: Predicted event that does not overlap any GT event.
    TN: Rest interval with no false-alarm predictions.
    """
    tp = 0
    fn = 0

    for gt_start, gt_end in gt_events:
        gt_duration = gt_end - gt_start
        if gt_duration <= 0:
            continue

        # Count unique frames in [gt_start, gt_end) covered by any prediction
        covered = np.zeros(gt_duration, dtype=bool)
        for pred_start, pred_end in pred_events:
            ov_start = max(gt_start, pred_start)
            ov_end = min(gt_end, pred_end)
            if ov_end > ov_start:
                covered[ov_start - gt_start:ov_end - gt_start] = True

        coverage = covered.sum() / gt_duration
        if coverage >= min_coverage:
            tp += 1
        else:
            fn += 1

    # FP: predicted events that don't overlap any GT event
    false_alarm_preds = []
    for pred in pred_events:
        overlaps_gt = False
        for gt in gt_events:
            if pred[0] < gt[1] and gt[0] < pred[1]:
                overlaps_gt = True
                break
        if not overlaps_gt:
            false_alarm_preds.append(pred)
    fp = len(false_alarm_preds)

    # TN: rest intervals with no false alarm predictions
    tn = 0
    for rest in rest_intervals:
        has_fa = any(fa[0] < rest[1] and rest[0] < fa[1] for fa in false_alarm_preds)
        if not has_fa:
            tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
        'num_gt_events': len(gt_events), 'num_pred_events': len(pred_events),
    }


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def threshold_and_filter(signal, threshold, min_pred_duration, max_merge_gap=0):
    """Threshold signal, optionally merge gaps, remove short events.

    Args:
        signal: numpy array of probabilities (may contain NaN)
        threshold: detection threshold
        min_pred_duration: remove events shorter than this (frames)
        max_merge_gap: merge events with gap <= this (0 = no merging)

    Returns:
        binary numpy array
    """
    pred = np.zeros(len(signal), dtype=bool)
    valid = ~np.isnan(signal)
    pred[valid] = signal[valid] >= threshold

    # Merge small gaps
    if max_merge_gap > 0:
        events = binary_to_events(pred.astype(float))
        for i in range(len(events) - 1):
            gap = events[i + 1][0] - events[i][1]
            if gap <= max_merge_gap:
                pred[events[i][1]:events[i + 1][0]] = True

    # Remove short events
    events = binary_to_events(pred.astype(float))
    for s, e in events:
        if (e - s) < min_pred_duration:
            pred[s:e] = False

    return pred


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_diagonal_signals(d):
    """Extract diagonal p_vs_flex[f][f] values from raw prediction JSON data.

    For each finger f, the diagonal value P(query > flex_f_ref)[f] compares
    the query against a known flex example from that same finger. This signal
    has more meaningful per-finger variance than p_vs_rest for the un-finetuned
    ranking backbone, because it anchors similarity to a real flex event.

    Values are stored at eval_stride intervals in d['predictions']; this
    function expands them to full frame resolution via sample-and-hold.

    Args:
        d: Parsed JSON dict with 'predictions', 'num_frames' fields.

    Returns:
        dict finger -> np.ndarray of shape [num_frames] with diagonal values.
    """
    num_frames = d['num_frames']
    predictions = d['predictions']

    signals = {f: np.full(num_frames, np.nan) for f in FINGER_NAMES}

    for i, pred in enumerate(predictions):
        frame = pred['frame']
        p_vs_flex = pred.get('p_vs_flex', {})

        # Sample-and-hold range: [frame, next_frame)
        next_frame = predictions[i + 1]['frame'] if (i + 1 < len(predictions)) else num_frames

        for finger in FINGER_NAMES:
            if finger in p_vs_flex and finger in p_vs_flex[finger]:
                val = p_vs_flex[finger][finger]
                end = min(next_frame, num_frames)
                signals[finger][frame:end] = val

    return signals


def find_fold_dirs(method_dir):
    """Find all fold directories (checkpoint dirs with event_detection/)."""
    method_dir = Path(method_dir)
    folds = []
    for d in sorted(method_dir.iterdir()):
        if d.is_dir() and (d / 'event_detection').is_dir():
            folds.append(d)
    return folds


def extract_test_subject(fold_dir):
    """Extract test subject ID from fold directory name."""
    name = fold_dir.name
    # Pattern: *_testNN_val*
    import re
    m = re.search(r'test(\d+)', name)
    return int(m.group(1)) if m else None


def load_fold_predictions(fold_dir, gt_mode='hybrid', gt_vel_threshold=80.0,
                          argmax_gt=False, use_diagonal=False,
                          use_fixed_labels=True):
    """Load all raw prediction JSONs for a fold's test subject.

    Args:
        fold_dir: Path to fold directory containing event_detection/.
        gt_mode: 'hybrid' uses compute_flex_intervals (velocity-onset + angle-offset),
                 'velocity' uses absolute velocity thresholding (old-style).
        gt_vel_threshold: Velocity threshold in deg/s for GT labeling.
        argmax_gt: If True, enforce mutual exclusivity (one finger per frame).
        use_diagonal: If True, use diagonal p_vs_flex[f][f] values instead of
                      p_vs_rest as the per-finger signal. Only supported for
                      methods in METHODS_DIAGONAL_SIGNAL (multi_finger_ranking).
        use_fixed_labels: If True, use manually corrected labels when available.

    Returns list of dicts, each with:
        'h5_stem', 'prob_signals' (dict finger -> np array),
        'gt_flex_labels' (dict finger -> np array), 'num_frames'
    """
    event_dir = Path(fold_dir) / 'event_detection'
    sessions = []
    for f in sorted(event_dir.glob('raw_predictions_*.json')):
        with open(f) as fp:
            d = json.load(fp)

        if use_diagonal:
            prob_signals = extract_diagonal_signals(d)
        else:
            prob_signals = {}
            for finger in FINGER_NAMES:
                sig = np.array(d['prob_signals'][finger], dtype=float)
                # JSON null → NaN
                sig = np.where(np.equal(sig, None), np.nan, sig)
                prob_signals[finger] = sig

        # GT labels: use fixed labels, or recompute from H5, or use saved
        if use_fixed_labels:
            h5_file = d['h5_file']
            gt_flex = recompute_gt_labels(
                h5_file, gt_mode, gt_vel_threshold, argmax_gt,
                use_fixed_labels=True)
        else:
            use_saved_gt = (gt_mode == 'hybrid'
                            and gt_vel_threshold == 80.0
                            and not argmax_gt)
            if use_saved_gt:
                gt_flex = {}
                for finger in FINGER_NAMES:
                    gt_flex[finger] = np.array(d['gt_flex_labels'][finger], dtype=float)
            else:
                h5_file = d['h5_file']
                gt_flex = recompute_gt_labels(
                    h5_file, gt_mode, gt_vel_threshold, argmax_gt,
                    use_fixed_labels=False)

        sessions.append({
            'h5_stem': Path(d['h5_file']).stem,
            'prob_signals': prob_signals,
            'gt_flex_labels': gt_flex,
            'num_frames': d['num_frames'],
        })
    return sessions


# ---------------------------------------------------------------------------
# Per-finger threshold optimization
# ---------------------------------------------------------------------------

def optimize_per_finger_thresholds(
    sessions,
    min_gt_coverage,
    min_pred_duration,
    max_merge_gap,
    normalize,
    max_pred_coverage=0.5,
    threshold_candidates=None,
):
    """Find optimal threshold per finger using coverage-based matching.

    Args:
        sessions: list of session dicts from load_fold_predictions
        min_gt_coverage, min_pred_duration, max_merge_gap: matching params
        normalize: whether to z-normalize signals
        max_pred_coverage: reject thresholds where predicted flex frames
            exceed this fraction of total frames (averaged across sessions)
        threshold_candidates: list of thresholds to try

    Returns:
        dict finger -> best_threshold
    """
    if threshold_candidates is None:
        threshold_candidates = [round(0.10 + i * 0.05, 2) for i in range(17)]  # 0.10..0.90

    best_thresholds = {}

    for finger in FINGER_NAMES:
        best_f1 = -1
        best_thresh = 0.5

        for thresh in threshold_candidates:
            all_results = []
            total_pred_frames = 0
            total_frames = 0

            for sess in sessions:
                signal = sess['prob_signals'][finger].copy()
                if normalize:
                    signal = znorm_signal(signal)

                pred_binary = threshold_and_filter(
                    signal, thresh, min_pred_duration, max_merge_gap)

                total_pred_frames += pred_binary.sum()
                total_frames += sess['num_frames']

                pred_events = binary_to_events(pred_binary.astype(float))
                gt_events = binary_to_events(sess['gt_flex_labels'][finger])
                rest_intervals = compute_rest_intervals(
                    gt_events, sess['num_frames'])

                result = match_events_coverage(
                    pred_events, gt_events, rest_intervals, min_gt_coverage)
                all_results.append(result)

            # Skip thresholds that predict too much flex
            pred_coverage = total_pred_frames / total_frames if total_frames > 0 else 0
            if pred_coverage > max_pred_coverage:
                continue

            # Aggregate across sessions: micro F1
            total_tp = sum(r['tp'] for r in all_results)
            total_fp = sum(r['fp'] for r in all_results)
            total_fn = sum(r['fn'] for r in all_results)
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        best_thresholds[finger] = best_thresh

    return best_thresholds


def optimize_six_class_threshold(
    sessions,
    min_gt_coverage,
    min_pred_duration,
    max_merge_gap,
    normalize,
    max_pred_coverage=0.5,
    threshold_candidates=None,
):
    """Find optimal global threshold for 6-class (argmax + rest) classification.

    At each frame: winner = argmax(signals). If signals[winner] > threshold,
    predict winner; else predict rest. Only one finger can be active per frame.

    Args:
        sessions: list of session dicts from load_fold_predictions
        min_gt_coverage, min_pred_duration, max_merge_gap: matching params
        normalize: whether to z-normalize signals before argmax
        max_pred_coverage: reject thresholds producing too many predictions
        threshold_candidates: list of thresholds to try

    Returns:
        best_threshold: float
    """
    if threshold_candidates is None:
        threshold_candidates = [round(0.10 + i * 0.05, 2) for i in range(17)]  # 0.10..0.90

    best_f1 = -1.0
    best_thresh = 0.5

    for thresh in threshold_candidates:
        all_results = []
        total_pred_frames = 0
        total_frames = 0

        for sess in sessions:
            num_frames = sess['num_frames']
            sig_stack = np.full((num_frames, 5), np.nan)
            for fi, finger in enumerate(FINGER_NAMES):
                signal = sess['prob_signals'][finger].copy()
                if normalize:
                    signal = znorm_signal(signal)
                sig_stack[:, fi] = signal

            valid = ~np.isnan(sig_stack).any(axis=1)
            safe_stack = np.where(np.isnan(sig_stack), -np.inf, sig_stack)
            max_sig = np.where(valid, np.max(safe_stack, axis=1), -np.inf)
            winner = np.where(valid, np.argmax(safe_stack, axis=1), -1)

            for fi, finger in enumerate(FINGER_NAMES):
                pred_raw = valid & (max_sig > thresh) & (winner == fi)
                pred_binary = threshold_and_filter(
                    pred_raw.astype(float), 0.5, min_pred_duration, max_merge_gap)

                total_pred_frames += pred_binary.sum()
                total_frames += num_frames

                pred_events = binary_to_events(pred_binary.astype(float))
                gt_events = binary_to_events(sess['gt_flex_labels'][finger])
                rest_intervals = compute_rest_intervals(gt_events, num_frames)
                result = match_events_coverage(
                    pred_events, gt_events, rest_intervals, min_gt_coverage)
                all_results.append(result)

        pred_coverage = total_pred_frames / total_frames if total_frames > 0 else 0
        if pred_coverage > max_pred_coverage:
            continue

        total_tp = sum(r['tp'] for r in all_results)
        total_fp = sum(r['fp'] for r in all_results)
        total_fn = sum(r['fn'] for r in all_results)
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh


def optimize_argmax_biases(
    sessions,
    min_gt_coverage,
    min_pred_duration,
    max_merge_gap,
    bias_candidates=None,
    threshold_candidates=None,
    max_iter=3,
):
    """Find per-finger biases and a global threshold for argmax classification.

    At each frame: adjusted[f] = znorm_signal[f] - bias[f].
    Predict finger with highest adjusted score if max(adjusted) > global_threshold,
    else predict rest. Biases tune relative sensitivity between fingers; the
    global threshold controls the rest vs. flex decision.

    Uses coordinate descent: alternates between optimizing each bias (one at a
    time) and optimizing the global threshold, for max_iter full passes.

    Args:
        sessions: list of session dicts from load_fold_predictions
        min_gt_coverage, min_pred_duration, max_merge_gap: matching params
        bias_candidates: list of bias values to try (default: -3.0 to 3.0, step 0.5)
        threshold_candidates: global threshold values (default: 0.5 to 4.0, step 0.25)
        max_iter: number of full coordinate descent passes

    Returns:
        biases: np.array of shape [5], one per finger
        global_threshold: float
    """
    if bias_candidates is None:
        bias_candidates = [round(-3.0 + i * 0.5, 1) for i in range(13)]  # -3.0..3.0
    if threshold_candidates is None:
        threshold_candidates = [round(0.5 + i * 0.25, 2) for i in range(15)]  # 0.5..4.0

    def apply_argmax_and_score(biases, threshold):
        """Apply argmax-biases to all sessions and return micro F1."""
        all_results = []
        for sess in sessions:
            num_frames = sess['num_frames']
            # Stack and z-normalize all finger signals
            sig_stack = np.full((num_frames, 5), np.nan)
            for fi, finger in enumerate(FINGER_NAMES):
                sig_stack[:, fi] = znorm_signal(sess['prob_signals'][finger].copy())

            adjusted = sig_stack - biases[np.newaxis, :]  # [T, 5]
            any_valid = ~np.isnan(adjusted).any(axis=1)
            max_adj = np.where(any_valid, np.nanmax(adjusted, axis=1), -np.inf)
            winner_idx = np.where(any_valid, np.nanargmax(adjusted, axis=1), -1)

            for fi, finger in enumerate(FINGER_NAMES):
                pred_raw = any_valid & (max_adj > threshold) & (winner_idx == fi)
                pred_binary = threshold_and_filter(
                    pred_raw.astype(float), 0.5, min_pred_duration, max_merge_gap)
                pred_events = binary_to_events(pred_binary.astype(float))
                gt_events = binary_to_events(sess['gt_flex_labels'][finger])
                rest_intervals = compute_rest_intervals(gt_events, num_frames)
                result = match_events_coverage(
                    pred_events, gt_events, rest_intervals, min_gt_coverage)
                all_results.append(result)

        total_tp = sum(r['tp'] for r in all_results)
        total_fp = sum(r['fp'] for r in all_results)
        total_fn = sum(r['fn'] for r in all_results)
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    biases = np.zeros(5, dtype=float)
    global_threshold = 1.0

    for _ in range(max_iter):
        # Optimize each bias independently
        for fi in range(5):
            best_f1 = -1.0
            best_val = biases[fi]
            for val in bias_candidates:
                trial = biases.copy()
                trial[fi] = val
                f1 = apply_argmax_and_score(trial, global_threshold)
                if f1 > best_f1:
                    best_f1 = f1
                    best_val = val
            biases[fi] = best_val

        # Optimize global threshold
        best_f1 = -1.0
        best_thresh = global_threshold
        for thresh in threshold_candidates:
            f1 = apply_argmax_and_score(biases, thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        global_threshold = best_thresh

    return biases, global_threshold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(sessions, finger_thresholds, min_gt_coverage,
                  min_pred_duration, max_merge_gap, normalize,
                  argmax_pred=False, argmax_biases=None,
                  six_class=False, six_class_threshold=None):
    """Evaluate one fold with given per-finger thresholds.

    Args:
        argmax_pred: If True, at each frame where multiple fingers are predicted
            active, keep only the one with the highest signal value. This
            prevents multi-finger false positives.
        argmax_biases: If not None, dict with 'biases' (np.array [5]) and
            'threshold' (float). Replaces independent thresholding with argmax
            classification: adjusted[f] = znorm[f] - biases[f], predict the
            finger with max adjusted score if it exceeds threshold.
        six_class: If True, use 6-class (5 fingers + rest) classification.
            At each frame, predict the finger with the highest signal if it
            exceeds six_class_threshold; else predict rest.
        six_class_threshold: Global rest threshold for 6-class mode.

    Returns:
        per_session: list of per-session results
        overall: aggregated metrics (micro + macro + per_finger)
    """
    per_session = []
    all_finger_results = []  # flat list for overall aggregation

    for sess in sessions:
        num_frames = sess['num_frames']
        pred_binaries = {}

        if six_class:
            # 6-class path: argmax over all fingers + rest threshold
            rest_thresh = six_class_threshold if six_class_threshold is not None else 0.5

            sig_stack = np.full((num_frames, 5), np.nan)
            for fi, finger in enumerate(FINGER_NAMES):
                signal = sess['prob_signals'][finger].copy()
                if normalize:
                    signal = znorm_signal(signal)
                sig_stack[:, fi] = signal

            valid = ~np.isnan(sig_stack).any(axis=1)
            safe_stack = np.where(np.isnan(sig_stack), -np.inf, sig_stack)
            max_sig = np.where(valid, np.max(safe_stack, axis=1), -np.inf)
            winner_idx = np.where(valid, np.argmax(safe_stack, axis=1), -1)

            for fi, finger in enumerate(FINGER_NAMES):
                pred_raw = valid & (max_sig > rest_thresh) & (winner_idx == fi)
                pred_binary = threshold_and_filter(
                    pred_raw.astype(float), 0.5, min_pred_duration, max_merge_gap)
                pred_binaries[finger] = pred_binary.astype(bool)

        elif argmax_biases is not None:
            # Argmax-biases path: adjusted[f] = znorm[f] - bias[f], then argmax
            biases_arr = argmax_biases['biases']
            threshold = argmax_biases['threshold']

            sig_stack = np.full((num_frames, 5), np.nan)
            for fi, finger in enumerate(FINGER_NAMES):
                sig_stack[:, fi] = znorm_signal(sess['prob_signals'][finger].copy())

            adjusted = sig_stack - biases_arr[np.newaxis, :]  # [T, 5]
            any_valid = ~np.isnan(adjusted).any(axis=1)
            max_adj = np.where(any_valid, np.nanmax(adjusted, axis=1), -np.inf)
            winner_idx = np.where(any_valid, np.nanargmax(adjusted, axis=1), -1)

            for fi, finger in enumerate(FINGER_NAMES):
                pred_raw = any_valid & (max_adj > threshold) & (winner_idx == fi)
                pred_binary = threshold_and_filter(
                    pred_raw.astype(float), 0.5, min_pred_duration, max_merge_gap)
                pred_binaries[finger] = pred_binary.astype(bool)

        else:
            # Independent per-finger thresholding (default)
            signals = {}
            for finger in FINGER_NAMES:
                signal = sess['prob_signals'][finger].copy()
                if normalize:
                    signal = znorm_signal(signal)
                signals[finger] = signal

                pred_binary = threshold_and_filter(
                    signal, finger_thresholds[finger], min_pred_duration, max_merge_gap)
                pred_binaries[finger] = pred_binary

            # Optionally resolve multi-finger conflicts
            if argmax_pred:
                sig_stack = np.column_stack([signals[f] for f in FINGER_NAMES])
                bin_stack = np.column_stack([pred_binaries[f] for f in FINGER_NAMES])

                multi_active = bin_stack.sum(axis=1) > 1
                for t in np.where(multi_active)[0]:
                    active_signals = sig_stack[t] * bin_stack[t]
                    winner = np.argmax(active_signals)
                    bin_stack[t] = 0
                    bin_stack[t, winner] = 1

                for fi, finger in enumerate(FINGER_NAMES):
                    pred_binaries[finger] = bin_stack[:, fi].astype(bool)

        # Step 3: Compute metrics per finger
        session_finger_results = []
        for finger in FINGER_NAMES:
            pred_events = binary_to_events(pred_binaries[finger].astype(float))
            gt_events = binary_to_events(sess['gt_flex_labels'][finger])
            rest_intervals = compute_rest_intervals(gt_events, num_frames)

            result = match_events_coverage(
                pred_events, gt_events, rest_intervals, min_gt_coverage)
            result['finger'] = finger
            session_finger_results.append(result)
            all_finger_results.append(result)

        # Per-session aggregate
        sess_agg = aggregate_finger_results(session_finger_results)
        per_session.append({
            'h5_stem': sess['h5_stem'],
            **sess_agg,
        })

    overall = aggregate_finger_results(all_finger_results)
    return per_session, overall


def aggregate_finger_results(finger_results):
    """Aggregate a list of per-finger result dicts into micro/macro/per_finger."""
    # Group by finger
    by_finger = {}
    for r in finger_results:
        finger = r['finger']
        if finger not in by_finger:
            by_finger[finger] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                                 'num_gt_events': 0, 'num_pred_events': 0}
        for k in ['tp', 'fp', 'fn', 'tn', 'num_gt_events', 'num_pred_events']:
            by_finger[finger][k] += r[k]

    # Per-finger metrics
    per_finger = {}
    for finger, counts in by_finger.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        tn = counts['tn']
        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total > 0 else 0.0
        per_finger[finger] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': p, 'recall': rec, 'f1': f1, 'accuracy': acc,
            'num_gt_events': counts['num_gt_events'],
            'num_pred_events': counts['num_pred_events'],
        }

    # Micro: sum counts across all fingers
    total_tp = sum(c['tp'] for c in by_finger.values())
    total_fp = sum(c['fp'] for c in by_finger.values())
    total_fn = sum(c['fn'] for c in by_finger.values())
    total_tn = sum(c['tn'] for c in by_finger.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    micro_total = total_tp + total_fp + total_fn + total_tn
    micro_acc = (total_tp + total_tn) / micro_total if micro_total > 0 else 0.0

    micro = {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn,
        'precision': micro_p, 'recall': micro_r, 'f1': micro_f1, 'accuracy': micro_acc,
    }

    # Macro: mean of per-finger metrics
    finger_metrics = list(per_finger.values())
    macro = {
        'precision': np.mean([m['precision'] for m in finger_metrics]),
        'recall': np.mean([m['recall'] for m in finger_metrics]),
        'f1': np.mean([m['f1'] for m in finger_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in finger_metrics]),
    }

    return {'micro': micro, 'macro': macro, 'per_finger': per_finger}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fair methods comparison with coverage-based event matching')
    parser.add_argument('--methods-dir', type=str, required=True,
                        help='Path to outputs/experiments/methods')
    parser.add_argument('--normalize', action='store_true',
                        help='Z-normalize signals before thresholding')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=False)
    parser.add_argument('--normalize-backbone', action='store_true', default=False,
                        help='Z-normalize both ranking backbone models (with and without FT). '
                             'multi_finger_ranking is always z-normed; this additionally '
                             'z-normalizes flex_embedding_ranking_backbone with optimized thresholds.')
    parser.add_argument('--min-gt-coverage', type=float, default=0.3,
                        help='Min fraction of GT event covered by predictions for TP (default: 0.3)')
    parser.add_argument('--min-pred-duration', type=int, default=10,
                        help='Remove predicted events shorter than N frames (default: 10 = 0.5s)')
    parser.add_argument('--max-merge-gap', type=int, default=0,
                        help='Merge predicted events with gap <= N frames (default: 0 = no merging)')
    parser.add_argument('--max-pred-coverage', type=float, default=0.5,
                        help='Reject thresholds where predicted flex exceeds this fraction of recording (default: 0.5)')
    parser.add_argument('--gt-mode', type=str, default='velocity',
                        choices=['hybrid', 'velocity'],
                        help='GT labeling mode: "velocity" = absolute velocity thresholding '
                             '(old-style, default), "hybrid" = velocity-onset + angle-offset '
                             '(compute_flex_intervals)')
    parser.add_argument('--gt-vel-threshold', type=float, default=None,
                        help='Velocity threshold in deg/s for GT labeling. '
                             'Default: 40.0 for velocity mode, 80.0 for hybrid mode.')
    parser.add_argument('--argmax-gt', action='store_true', default=False,
                        help='Enforce mutual exclusivity in GT: only one finger active '
                             'per frame (the fastest-moving one). Default: off.')
    parser.add_argument('--argmax-pred', action='store_true', default=False,
                        help='Enforce mutual exclusivity in predictions: at each frame '
                             'where multiple fingers are predicted active, keep only the '
                             'one with the highest signal. Default: off.')
    parser.add_argument('--six-class', action='store_true', default=False,
                        help='6-class mode: each frame is one of 5 fingers or rest. '
                             'Forces argmax-exclusive GT and argmax predictions with a '
                             'global rest threshold. Default: off (independent per-finger).')
    parser.add_argument('--exclusive-gt', action='store_true', default=False,
                        help='Mask out co-activation frames (>1 finger active in GT). '
                             'Sets GT to 0 and predictions to NaN at those frames so '
                             'only single-finger intervals are evaluated. Default: off.')
    parser.add_argument('--no-fixed-labels', dest='use_fixed_labels',
                        action='store_false', default=True,
                        help='Disable fixed (manually corrected) GT labels; use auto-computed instead')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (auto-generated if not specified)')
    args = parser.parse_args()

    # Set default velocity threshold based on GT mode
    if args.gt_vel_threshold is None:
        args.gt_vel_threshold = 40.0 if args.gt_mode == 'velocity' else 80.0

    # 6-class mode forces exclusive GT labels
    if args.six_class:
        args.argmax_gt = True

    # --normalize-backbone: z-normalize only the ranking backbone
    if args.normalize_backbone:
        METHODS_ZNORM.add('flex_embedding_ranking_backbone')
        METHODS_OWN_THRESHOLDS.add('flex_embedding_ranking_backbone')

    methods_dir = Path(args.methods_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"{'='*70}")
    print(f"Fair Methods Comparison")
    print(f"{'='*70}")
    print(f"  fixed_labels:     {args.use_fixed_labels}")
    print(f"  gt_mode:          {args.gt_mode}")
    print(f"  gt_vel_threshold: {args.gt_vel_threshold} deg/s")
    print(f"  argmax_gt:        {args.argmax_gt}")
    print(f"  argmax_pred:      {args.argmax_pred}")
    print(f"  six_class:        {args.six_class}")
    print(f"  exclusive_gt:     {args.exclusive_gt}")
    print(f"  normalize:        {args.normalize}")
    print(f"  normalize_bb:     {args.normalize_backbone}")
    print(f"  znorm methods:    {METHODS_ZNORM}")
    print(f"  argmax-bias methods: {METHODS_ARGMAX_BIASES}")
    print(f"  diagonal methods: {METHODS_DIAGONAL_SIGNAL}")
    print(f"  min_gt_coverage:  {args.min_gt_coverage}")
    print(f"  min_pred_duration: {args.min_pred_duration} frames ({args.min_pred_duration/20:.2f}s)")
    print(f"  max_merge_gap:    {args.max_merge_gap} frames ({args.max_merge_gap/20:.2f}s)")
    print(f"  max_pred_coverage: {args.max_pred_coverage:.0%}")
    print(f"{'='*70}\n")

    # Step 1a: Fixed threshold=0.5 for fine-tuned classifiers
    # These models were trained with BCE loss, so 0.5 is the natural decision boundary.
    fixed_threshold = 0.5
    fixed_thresholds = {f: fixed_threshold for f in FINGER_NAMES}
    print(f"Fine-tuned classifiers: fixed threshold = {fixed_threshold} (all fingers)\n")

    # Step 1b: Optimize parameters for methods with uncalibrated outputs.
    own_thresholds = {}   # method_name -> {subj -> {finger: threshold}}
    six_class_thresholds = {}  # method_name -> {subj -> float}
    argmax_biases_store = {}  # method_name -> {subj -> {'biases': np.array, 'threshold': float}}

    for method_name in METHODS_OWN_THRESHOLDS:
        method_dir = methods_dir / method_name
        if not method_dir.exists():
            continue
        label = METHOD_LABELS.get(method_name, method_name)
        use_argmax_biases = method_name in METHODS_ARGMAX_BIASES
        use_diagonal = method_name in METHODS_DIAGONAL_SIGNAL
        use_znorm = method_name in METHODS_ZNORM
        method_fold_dirs = find_fold_dirs(method_dir)

        if args.six_class:
            # 6-class mode: optimize a single global rest threshold with argmax
            znorm_candidates = [round(0.5 + i * 0.25, 2) for i in range(15)]  # 0.50..4.00
            print(f"Optimizing 6-class threshold for {label} (znorm={use_znorm})...")
            six_class_thresholds[method_name] = {}
            for fold_dir in method_fold_dirs:
                subj = extract_test_subject(fold_dir)
                sessions = load_fold_predictions(
                    fold_dir, gt_mode=args.gt_mode,
                    gt_vel_threshold=args.gt_vel_threshold,
                    argmax_gt=args.argmax_gt,
                    use_diagonal=use_diagonal,
                    use_fixed_labels=args.use_fixed_labels)
                if args.exclusive_gt:
                    apply_exclusive_gt_mask(sessions)
                thresh = optimize_six_class_threshold(
                    sessions,
                    min_gt_coverage=args.min_gt_coverage,
                    min_pred_duration=args.min_pred_duration,
                    max_merge_gap=args.max_merge_gap,
                    normalize=use_znorm,
                    max_pred_coverage=args.max_pred_coverage,
                    threshold_candidates=znorm_candidates if use_znorm else None,
                )
                six_class_thresholds[method_name][subj] = thresh
                print(f"  S{subj:02d}: threshold={thresh:.2f}")
        elif use_argmax_biases:
            print(f"Optimizing argmax biases for {label} (coord. descent)...")
            argmax_biases_store[method_name] = {}
            for fold_dir in method_fold_dirs:
                subj = extract_test_subject(fold_dir)
                sessions = load_fold_predictions(
                    fold_dir, gt_mode=args.gt_mode,
                    gt_vel_threshold=args.gt_vel_threshold,
                    argmax_gt=args.argmax_gt,
                    use_diagonal=use_diagonal,
                    use_fixed_labels=args.use_fixed_labels)
                if args.exclusive_gt:
                    apply_exclusive_gt_mask(sessions)
                biases, threshold = optimize_argmax_biases(
                    sessions,
                    min_gt_coverage=args.min_gt_coverage,
                    min_pred_duration=args.min_pred_duration,
                    max_merge_gap=args.max_merge_gap,
                )
                argmax_biases_store[method_name][subj] = {
                    'biases': biases,
                    'threshold': threshold,
                }
                bias_str = ', '.join(
                    f'{f}={biases[fi]:.1f}' for fi, f in enumerate(FINGER_NAMES))
                print(f"  S{subj:02d}: biases=[{bias_str}] threshold={threshold:.2f}")
        else:
            print(f"Optimizing per-finger thresholds for {label}"
                  f" (znorm={use_znorm})...")
            own_thresholds[method_name] = {}
            # Z-normed signals: mean=0, std=1, flex peaks at 2-7 stds
            znorm_candidates = [round(0.5 + i * 0.25, 2) for i in range(15)]  # 0.50..4.00
            for fold_dir in method_fold_dirs:
                subj = extract_test_subject(fold_dir)
                sessions = load_fold_predictions(
                    fold_dir, gt_mode=args.gt_mode,
                    gt_vel_threshold=args.gt_vel_threshold,
                    argmax_gt=args.argmax_gt,
                    use_diagonal=use_diagonal,
                    use_fixed_labels=args.use_fixed_labels)
                if args.exclusive_gt:
                    apply_exclusive_gt_mask(sessions)
                thresholds = optimize_per_finger_thresholds(
                    sessions,
                    min_gt_coverage=args.min_gt_coverage,
                    min_pred_duration=args.min_pred_duration,
                    max_merge_gap=args.max_merge_gap,
                    normalize=use_znorm,
                    max_pred_coverage=args.max_pred_coverage,
                    threshold_candidates=znorm_candidates if use_znorm else None,
                )
                own_thresholds[method_name][subj] = thresholds
                thresh_str = ', '.join(f'{f}={t:.2f}' for f, t in thresholds.items())
                print(f"  S{subj:02d}: {thresh_str}")
        print()

    # Step 2: Evaluate all methods
    all_results = {}
    all_mask_stats = []  # (method, subject, stats_list) for exclusive_gt CSV

    for method_name in METHODS:
        method_dir = methods_dir / method_name
        if not method_dir.exists():
            print(f"WARNING: {method_name} not found, skipping")
            continue

        label = METHOD_LABELS.get(method_name, method_name)
        fold_dirs = find_fold_dirs(method_dir)
        use_znorm = method_name in METHODS_ZNORM
        use_own_thresh = method_name in METHODS_OWN_THRESHOLDS
        use_argmax_biases = method_name in METHODS_ARGMAX_BIASES
        use_diagonal = method_name in METHODS_DIAGONAL_SIGNAL
        norm_tag = " [znorm]" if use_znorm else ""
        mode_tag = " [argmax-biases]" if use_argmax_biases else (" [own thresh]" if use_own_thresh else "")
        diag_tag = " [diag]" if use_diagonal else ""
        print(f"--- {label}{norm_tag}{mode_tag}{diag_tag} ({len(fold_dirs)} folds) ---")

        method_folds = []

        for fold_dir in fold_dirs:
            subj = extract_test_subject(fold_dir)

            # Determine thresholds / biases for this fold
            ab = None  # argmax_biases dict or None
            sc_thresh = None  # six_class threshold or None
            if args.six_class:
                # 6-class mode: global rest threshold
                if use_own_thresh and method_name in six_class_thresholds:
                    if subj not in six_class_thresholds[method_name]:
                        print(f"  S{subj:02d}: no 6-class threshold found, skipping")
                        continue
                    sc_thresh = six_class_thresholds[method_name][subj]
                else:
                    sc_thresh = fixed_threshold  # 0.5 for fine-tuned classifiers
                thresholds = fixed_thresholds  # unused in 6-class mode
            elif use_argmax_biases:
                if method_name not in argmax_biases_store or subj not in argmax_biases_store[method_name]:
                    print(f"  S{subj:02d}: no argmax biases found, skipping")
                    continue
                ab = argmax_biases_store[method_name][subj]
                thresholds = fixed_thresholds  # unused in argmax-biases mode
            elif use_own_thresh:
                if method_name not in own_thresholds or subj not in own_thresholds[method_name]:
                    print(f"  S{subj:02d}: no thresholds found, skipping")
                    continue
                thresholds = own_thresholds[method_name][subj]
            else:
                thresholds = fixed_thresholds

            sessions = load_fold_predictions(
                fold_dir, gt_mode=args.gt_mode,
                gt_vel_threshold=args.gt_vel_threshold,
                argmax_gt=args.argmax_gt,
                use_diagonal=use_diagonal,
                use_fixed_labels=args.use_fixed_labels)

            if args.exclusive_gt:
                mask_stats = apply_exclusive_gt_mask(sessions)
                all_mask_stats.append((method_name, subj, mask_stats))

            per_session, overall = evaluate_fold(
                sessions, thresholds,
                min_gt_coverage=args.min_gt_coverage,
                min_pred_duration=args.min_pred_duration,
                max_merge_gap=args.max_merge_gap,
                normalize=use_znorm,
                argmax_pred=args.argmax_pred,
                argmax_biases=ab,
                six_class=args.six_class,
                six_class_threshold=sc_thresh,
            )

            macro = overall['macro']
            micro = overall['micro']
            print(f"  S{subj:02d}: macro F1={macro['f1']:.4f}  "
                  f"P={macro['precision']:.4f}  R={macro['recall']:.4f}  "
                  f"Acc={macro['accuracy']:.4f}  "
                  f"(micro F1={micro['f1']:.4f})")

            fold_entry = {
                'test_subject': subj,
                'fold_dir': str(fold_dir),
                'per_session': per_session,
                'overall': overall,
            }
            if use_argmax_biases:
                fold_entry['argmax_biases'] = {
                    'biases': ab['biases'].tolist(),
                    'threshold': ab['threshold'],
                }
            else:
                fold_entry['thresholds'] = thresholds
            method_folds.append(fold_entry)

        # Aggregate across folds
        if method_folds:
            macro_f1s = [f['overall']['macro']['f1'] for f in method_folds]
            macro_precs = [f['overall']['macro']['precision'] for f in method_folds]
            macro_recs = [f['overall']['macro']['recall'] for f in method_folds]
            macro_accs = [f['overall']['macro']['accuracy'] for f in method_folds]
            micro_f1s = [f['overall']['micro']['f1'] for f in method_folds]

            print(f"\n  Summary ({len(method_folds)} folds):")
            print(f"    Macro F1:  {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")
            print(f"    Macro P:   {np.mean(macro_precs):.4f} +/- {np.std(macro_precs):.4f}")
            print(f"    Macro R:   {np.mean(macro_recs):.4f} +/- {np.std(macro_recs):.4f}")
            print(f"    Macro Acc: {np.mean(macro_accs):.4f} +/- {np.std(macro_accs):.4f}")
            print(f"    Micro F1:  {np.mean(micro_f1s):.4f} +/- {np.std(micro_f1s):.4f}")

            # Per-finger breakdown
            print(f"    Per-finger macro F1:")
            for finger in FINGER_NAMES:
                finger_f1s = []
                for fold in method_folds:
                    pf = fold['overall']['per_finger']
                    if finger in pf:
                        finger_f1s.append(pf[finger]['f1'])
                if finger_f1s:
                    print(f"      {finger:>7s}: {np.mean(finger_f1s):.4f} +/- {np.std(finger_f1s):.4f}")

        all_results[method_name] = {
            'label': label,
            'folds': method_folds,
            'summary': {
                'macro_f1_mean': float(np.mean(macro_f1s)),
                'macro_f1_std': float(np.std(macro_f1s)),
                'macro_precision_mean': float(np.mean(macro_precs)),
                'macro_precision_std': float(np.std(macro_precs)),
                'macro_recall_mean': float(np.mean(macro_recs)),
                'macro_recall_std': float(np.std(macro_recs)),
                'macro_accuracy_mean': float(np.mean(macro_accs)),
                'macro_accuracy_std': float(np.std(macro_accs)),
                'micro_f1_mean': float(np.mean(micro_f1s)),
                'micro_f1_std': float(np.std(micro_f1s)),
                'n_folds': len(method_folds),
            } if method_folds else {},
        }
        print()

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Method':<22s} {'Macro F1':>12s} {'Macro P':>12s} {'Macro R':>12s} {'Macro Acc':>12s} {'N':>4s}"
    print(header)
    print('-' * len(header))
    for method_name in METHODS:
        if method_name not in all_results:
            continue
        s = all_results[method_name].get('summary', {})
        if not s:
            continue
        label = METHOD_LABELS.get(method_name, method_name)
        f1_str = f"{s['macro_f1_mean']:.3f}+/-{s['macro_f1_std']:.3f}"
        p_str = f"{s['macro_precision_mean']:.3f}+/-{s['macro_precision_std']:.3f}"
        r_str = f"{s['macro_recall_mean']:.3f}+/-{s['macro_recall_std']:.3f}"
        a_str = f"{s['macro_accuracy_mean']:.3f}+/-{s['macro_accuracy_std']:.3f}"
        print(f"{label:<22s} {f1_str:>12s} {p_str:>12s} {r_str:>12s} {a_str:>12s} {s['n_folds']:>4d}")
    print()

    # Save JSON
    output_path = args.output
    if output_path is None:
        output_dir = Path('evaluation/plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        norm_tag = 'znorm' if args.normalize else ('znormbb' if args.normalize_backbone else 'raw')
        gt_tag = 'gtfixed' if args.use_fixed_labels else f'gt{args.gt_mode}_vel{int(args.gt_vel_threshold)}'
        if args.argmax_gt:
            gt_tag += '_argmaxgt'
        if args.argmax_pred:
            gt_tag += '_argmaxpred'
        if args.six_class:
            gt_tag += '_6class'
        if args.exclusive_gt:
            gt_tag += '_exclusive'
        output_path = str(output_dir / f'methods_fair_{gt_tag}_{norm_tag}_cov{args.min_gt_coverage}_'
                          f'mindur{args.min_pred_duration}_gap{args.max_merge_gap}_{timestamp}.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_data = {
        'config': {
            'use_fixed_labels': args.use_fixed_labels,
            'gt_mode': args.gt_mode,
            'gt_vel_threshold': args.gt_vel_threshold,
            'argmax_gt': args.argmax_gt,
            'argmax_pred': args.argmax_pred,
            'six_class': args.six_class,
            'exclusive_gt': args.exclusive_gt,
            'normalize': args.normalize,
            'normalize_backbone': args.normalize_backbone,
            'znorm_methods': sorted(METHODS_ZNORM),
            'own_threshold_methods': sorted(METHODS_OWN_THRESHOLDS),
            'argmax_bias_methods': sorted(METHODS_ARGMAX_BIASES),
            'diagonal_signal_methods': sorted(METHODS_DIAGONAL_SIGNAL),
            'fixed_threshold': fixed_threshold,
            'min_gt_coverage': args.min_gt_coverage,
            'min_pred_duration': args.min_pred_duration,
            'max_merge_gap': args.max_merge_gap,
            'max_pred_coverage': args.max_pred_coverage,
            'threshold_source': f'fixed={fixed_threshold} for classifiers, optimized for znorm methods',
            'timestamp': timestamp,
        },
        'fixed_thresholds': fixed_thresholds,
        'own_thresholds': {
            mk: {str(s): t for s, t in sv.items()}
            for mk, sv in own_thresholds.items()
        },
        'methods': {},
    }
    for method_name, mdata in all_results.items():
        fold_records = []
        for f in mdata.get('folds', []):
            rec = {
                'test_subject': f['test_subject'],
                'overall': f['overall'],
            }
            if 'argmax_biases' in f:
                rec['argmax_biases'] = f['argmax_biases']
            else:
                rec['thresholds'] = f.get('thresholds', {})
            fold_records.append(rec)
        output_data['methods'][method_name] = {
            'label': mdata['label'],
            'summary': mdata.get('summary', {}),
            'folds': fold_records,
        }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=convert)
    print(f"Results saved to: {output_path}")

    # Save exclusive-GT masking statistics as CSV
    if args.exclusive_gt and all_mask_stats:
        csv_path = Path(output_path).with_name(
            Path(output_path).stem + '_exclusive_gt_stats.csv')
        with open(csv_path, 'w') as csvf:
            csvf.write('method,subject,session,total_frames,masked_frames,pct_masked\n')
            for method_name, subj, stats_list in all_mask_stats:
                for st in stats_list:
                    csvf.write(f"{method_name},{subj},{st['h5_stem']},"
                               f"{st['num_frames']},{st['num_masked']},"
                               f"{st['pct_masked']:.1f}\n")
        print(f"Masking stats saved to: {csv_path}")

        # Print summary table
        print(f"\nExclusive-GT masking summary (first method only):")
        first_method = all_mask_stats[0][0]
        total_f, total_m = 0, 0
        for method_name, subj, stats_list in all_mask_stats:
            if method_name != first_method:
                continue
            for st in stats_list:
                total_f += st['num_frames']
                total_m += st['num_masked']
                print(f"  S{subj:02d}  {st['h5_stem']:>55s}  "
                      f"{st['num_masked']:>5d}/{st['num_frames']:>5d}  "
                      f"({st['pct_masked']:.1f}%)")
        pct = 100.0 * total_m / total_f if total_f > 0 else 0.0
        print(f"  {'TOTAL':>58s}  {total_m:>5d}/{total_f:>5d}  ({pct:.1f}%)")


if __name__ == '__main__':
    main()

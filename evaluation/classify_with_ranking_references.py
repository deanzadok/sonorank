"""
Classify finger flex events using ranking model reference comparisons.

Given reference sequences (rest + per-finger flex), classifies query windows by
comparing against each reference:
  - P(query > rest)[f]       for each finger f   → high means finger f is active
  - P(query > flex_ref_i)[f] for each finger f   → ≈0.5 means similar flexion

Classification logic:
  - active finger = argmax_i P(query > flex_i_ref)[i]   (diagonal values)
  - rest = when all diagonal values are below threshold

Usage:
    # Single H5 file (auto-detect references from same file)
    python -m evaluation.classify_with_ranking_references \
        --checkpoint outputs/multi_finger_ranking/.../checkpoint_best.pth \
        --h5-files data/.../recordings_23_gestures_enrollment02_finger_flex03.h5 \
        --gpu-id 0 --plot

    # Multiple H5 files (references from first, test on all)
    python -m evaluation.classify_with_ranking_references \
        --checkpoint outputs/multi_finger_ranking/.../checkpoint_best.pth \
        --h5-files data/.../recordings_23_gestures_*_finger_flex*.h5 \
        --gpu-id 0 --eval-stride 4 --plot

Reader's guide: the script chooses rest and flex reference windows from
kinematics, compares every query window with those references, and only then
converts comparison probabilities into finger or rest decisions. Saved logs
retain raw scores for later threshold studies.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from trainer.model import MultiFingerTemporalRankingModel
from evaluation.eval_utils import (
    load_h5_data, load_multi_finger_model, MULTI_FINGER_JOINT_COLUMNS,
    create_transform, prepare_sequence, FPS,
)

FINGER_NAMES = MultiFingerTemporalRankingModel.FINGER_NAMES
CLASS_NAMES = ['rest'] + list(FINGER_NAMES)


# ============================================================
# Reference window detection
# ============================================================

def compute_window_activity(joint_angles, num_frames, span):
    """Compute per-finger range of motion for each possible window position.

    Returns dict: finger -> array of shape [num_windows] with ROM in degrees.
    """
    num_windows = num_frames - span + 1
    activity = {}
    for finger in FINGER_NAMES:
        if finger not in joint_angles:
            continue
        angles = joint_angles[finger]
        rom = np.zeros(num_windows)
        for i in range(num_windows):
            w = angles[i:i + span]
            rom[i] = np.rad2deg(np.ptp(w))
        activity[finger] = rom
    return activity


def find_reference_windows(joint_angles, num_frames, span, min_flex_deg=8.0):
    """Auto-detect rest and per-finger flex reference windows from kinematics.

    Rest:   window with minimum total ROM across all fingers.
    Flex_i: window with maximum ROM for finger i (must exceed min_flex_deg).

    Returns:
        rest_start:  int, start frame of rest window
        flex_starts: dict finger -> int, start frame of best flex window
        activity:    dict finger -> array, per-window ROM (for diagnostics)
    """
    activity = compute_window_activity(joint_angles, num_frames, span)

    # Rest: minimize total activity across all fingers
    total_act = np.zeros(num_frames - span + 1)
    for arr in activity.values():
        total_act += arr
    rest_start = int(np.argmin(total_act))

    # Per-finger flex: maximize that finger's activity
    flex_starts = {}
    for finger in FINGER_NAMES:
        if finger not in activity:
            continue
        best_idx = int(np.argmax(activity[finger]))
        if activity[finger][best_idx] >= min_flex_deg:
            flex_starts[finger] = best_idx

    return rest_start, flex_starts, activity


def find_reference_windows_multi(all_data, span, min_flex_deg=8.0):
    """Find best reference windows across multiple H5 files.

    For each finger, picks the file+window with maximum ROM.
    Rest comes from the file with the quietest window overall.

    Returns:
        rest_info:  (h5_path, start_frame)
        flex_info:  dict finger -> (h5_path, start_frame)
    """
    best_rest_score = float('inf')
    rest_info = None

    # finger -> (h5_path, start_frame, rom)
    best_flex = {}

    for h5_path, data in all_data.items():
        activity = compute_window_activity(
            data['joint_angles'], data['num_frames'], span)

        # Check rest
        total_act = np.zeros(data['num_frames'] - span + 1)
        for arr in activity.values():
            total_act += arr
        min_total = float(np.min(total_act))
        if min_total < best_rest_score:
            best_rest_score = min_total
            rest_info = (h5_path, int(np.argmin(total_act)))

        # Check per-finger flex
        for finger in FINGER_NAMES:
            if finger not in activity:
                continue
            best_idx = int(np.argmax(activity[finger]))
            rom = float(activity[finger][best_idx])
            if rom >= min_flex_deg:
                if finger not in best_flex or rom > best_flex[finger][2]:
                    best_flex[finger] = (h5_path, best_idx, rom)

    flex_info = {f: (path, start) for f, (path, start, _) in best_flex.items()}
    return rest_info, flex_info


# ============================================================
# Ground truth labeling
# ============================================================

def label_window_gt(joint_angles, window_start, span, threshold_deg=40.0):
    """Determine ground-truth label for a window from kinematics using velocity.

    Computes mean absolute angular velocity (deg/s) per finger within the window.
    The finger with the highest velocity above threshold is labeled as active.
    If all fingers are below threshold → 'rest'.

    Also returns signed velocity (mean of diffs * FPS, in deg/s) per finger,
    where negative = flexion (angle reducing), positive = extension.

    Args:
        joint_angles: dict finger -> array of joint angles (radians)
        window_start: start frame of the window
        span: number of frames in the window
        threshold_deg: min mean |velocity| in deg/s to count as active (default: 40.0)

    Returns:
        (label, velocities_dict, signed_vel_dict)
        velocities_dict: finger -> mean |vel| in deg/s
        signed_vel_dict: finger -> mean signed vel in deg/s (negative = flexion)
    """
    velocities = {}
    signed_vel = {}
    for finger in FINGER_NAMES:
        if finger not in joint_angles:
            continue
        w = joint_angles[finger][window_start:window_start + span]
        if len(w) < 2:
            velocities[finger] = 0.0
            signed_vel[finger] = 0.0
            continue
        diffs = np.diff(w)
        velocities[finger] = float(np.rad2deg(np.mean(np.abs(diffs))) * FPS)
        signed_vel[finger] = float(np.rad2deg(np.mean(diffs)) * FPS)

    if not velocities:
        return 'rest', {}, {}

    best_finger = max(velocities, key=velocities.get)
    if velocities[best_finger] < threshold_deg:
        return 'rest', velocities, signed_vel

    return best_finger, velocities, signed_vel


def fill_gt_gaps(comparison_log, verbose=True):
    """Fill gaps in GT labels using motion direction (flex → hold → extend).

    During a flex gesture, the sequence is:
      1. Flexion: angle reducing (negative signed velocity) → labeled as finger X
      2. Hold at full flexion: near-zero velocity → labeled 'rest' (the gap)
      3. Extension: angle increasing (positive signed velocity) → may be labeled
         as finger X or something else

    This function fills the gap if:
      - The finger labeled before the gap was flexing (negative signed velocity)
      - The same finger after the gap was extending (positive signed velocity)
    No gap-length threshold needed — the direction pattern is the criterion.

    Args:
        comparison_log: list of dicts with 'gt' and 'gt_signed_vel' keys
        verbose: if True, print diagnostics for each gap

    Returns:
        Number of windows relabeled.
    """
    labels = [e['gt'] for e in comparison_log]
    n = len(labels)
    filled = 0

    i = 0
    while i < n:
        if labels[i] != 'rest':
            i += 1
            continue

        # Find extent of this rest gap
        gap_start = i
        gap_end = i
        while gap_end < n and labels[gap_end] == 'rest':
            gap_end += 1

        gap_len = gap_end - gap_start
        before_frame = comparison_log[gap_start - 1]['frame'] if gap_start > 0 else None
        after_frame = comparison_log[gap_end]['frame'] if gap_end < n else None

        if gap_start > 0 and gap_end < n:
            before_entry = comparison_log[gap_start - 1]
            after_entry = comparison_log[gap_end]
            finger = before_entry['gt']

            if finger != 'rest':
                before_sv = before_entry.get('gt_signed_vel', {}).get(finger, 0)
                after_sv = after_entry.get('gt_signed_vel', {}).get(finger, 0)
                after_label = after_entry['gt']

                # Flexion before (negative) and extension after (positive)
                if before_sv < 0 and after_sv > 0:
                    for j in range(gap_start, gap_end):
                        labels[j] = finger
                        comparison_log[j]['gt'] = finger
                        filled += 1
                    if verbose:
                        print(f"    FILLED gap frames {before_frame}..{after_frame} "
                              f"({gap_len} windows) with {finger} "
                              f"(before_sv={before_sv:+.1f}, after_sv={after_sv:+.1f})")
                elif verbose:
                    print(f"    SKIP  gap frames {before_frame}..{after_frame} "
                          f"({gap_len} windows): before={finger}(sv={before_sv:+.1f}), "
                          f"after={after_label}(sv={after_sv:+.1f} for {finger})")
            elif verbose:
                print(f"    SKIP  gap frames {before_frame}..{after_frame} "
                      f"({gap_len} windows): before=rest")
        elif verbose:
            print(f"    SKIP  gap at boundary ({gap_len} windows)")

        i = gap_end if gap_end > i else i + 1

    return filled


def smooth_gt_labels(comparison_log, verbose=True):
    """Smooth GT labels by removing single-window interruptions from coupled motion.

    If a window is labeled finger A but both immediate neighbors are finger B,
    relabel it as finger B. This handles coupled motion where e.g. ring briefly
    wins over middle by 1-2 deg/s during a middle gesture.

    Runs iteratively until no more changes (handles adjacent interruptions).

    Returns:
        Number of windows relabeled.
    """
    labels = [e['gt'] for e in comparison_log]
    n = len(labels)
    total_smoothed = 0

    changed = True
    while changed:
        changed = False
        for i in range(1, n - 1):
            if labels[i] != labels[i - 1] and labels[i - 1] == labels[i + 1] \
                    and labels[i - 1] != 'rest':
                old = labels[i]
                new = labels[i - 1]
                if verbose:
                    frame = comparison_log[i]['frame']
                    print(f"    SMOOTH frame {frame}: {old} → {new} "
                          f"(neighbors both {new})")
                labels[i] = new
                comparison_log[i]['gt'] = new
                total_smoothed += 1
                changed = True

    return total_smoothed


# ============================================================
# Classification
# ============================================================

def classify_window(p_vs_rest, p_vs_flex, flex_threshold=0.35):
    """Classify a window from ranking comparison values (raw, no normalization).

    Uses the "diagonal" values: P(query > flex_i_ref)[i] for each finger i.
    The finger whose reference the query most resembles gets the highest
    diagonal value (≈0.5), non-matching fingers get lower values.

    If the max diagonal value < flex_threshold → classify as rest.

    Returns:
        (predicted_class, scores_dict)
    """
    # Diagonal values: P(query > flex_i_ref)[i]
    diag_scores = {}
    for ref_finger in p_vs_flex:
        diag_scores[ref_finger] = p_vs_flex[ref_finger][ref_finger]

    if diag_scores:
        max_diag = max(diag_scores.values())
        if max_diag < flex_threshold:
            return 'rest', diag_scores

        predicted = max(diag_scores, key=diag_scores.get)
        return predicted, diag_scores

    # Fallback if no flex references: use P(query > rest)
    max_p_rest = max(p_vs_rest.values())
    if max_p_rest < 0.5:
        return 'rest', p_vs_rest

    predicted = max(p_vs_rest, key=p_vs_rest.get)
    return predicted, p_vs_rest


# ============================================================
# Per-finger normalization (post-hoc)
# ============================================================

def normalize_and_reclassify(comparison_log, fingers_with_refs, method='zscore',
                             rest_z_threshold=0.0):
    """Re-classify all windows after per-finger normalization of diagonal values.

    Each finger head has a different baseline P value. Z-score normalization
    puts all fingers on the same scale: a spike in a normally-quiet finger
    (e.g. thumb) gets the same normalized score as a spike in a normally-loud
    finger (e.g. ring).

    Args:
        comparison_log: list of dicts with 'p_vs_flex', 'p_vs_rest', 'gt' keys
        fingers_with_refs: list of finger names that have flex references
        method: 'zscore' or 'minmax'
        rest_z_threshold: z-score below which max → classify as rest
            (0.0 = below mean, try negative values to be stricter)

    Returns:
        stats: dict finger -> {mean, std} or {min, range}
        Updated comparison_log entries get 'pred_norm' and 'norm_scores' keys.
    """
    # Collect all diagonal values per finger
    per_finger_values = {f: [] for f in fingers_with_refs}
    for entry in comparison_log:
        for f in fingers_with_refs:
            per_finger_values[f].append(entry['p_vs_flex'][f][f])

    # Compute per-finger statistics
    stats = {}
    for f in fingers_with_refs:
        values = np.array(per_finger_values[f])
        if method == 'zscore':
            stats[f] = {'mean': float(np.mean(values)),
                        'std': float(np.std(values)) + 1e-8}
        elif method == 'minmax':
            stats[f] = {'min': float(np.min(values)),
                        'range': float(np.ptp(values)) + 1e-8}

    # Normalize and re-classify each window
    for entry in comparison_log:
        norm_scores = {}
        for f in fingers_with_refs:
            raw = entry['p_vs_flex'][f][f]
            if method == 'zscore':
                norm_scores[f] = (raw - stats[f]['mean']) / stats[f]['std']
            elif method == 'minmax':
                norm_scores[f] = (raw - stats[f]['min']) / stats[f]['range']

        entry['norm_scores'] = norm_scores

        if norm_scores:
            max_z = max(norm_scores.values())
            if max_z < rest_z_threshold:
                entry['pred_norm'] = 'rest'
            else:
                entry['pred_norm'] = max(norm_scores, key=norm_scores.get)
        else:
            entry['pred_norm'] = 'rest'

    return stats


def sweep_rest_threshold(comparison_log, fingers_with_refs, stats, method='zscore',
                         thresholds=None):
    """Sweep rest_z_threshold values and return (threshold, accuracy) pairs.

    Re-classifies using pre-computed norm_scores (must call normalize_and_reclassify first).
    """
    if thresholds is None:
        thresholds = np.arange(-3.0, 3.1, 0.25)

    gt_labels = [e['gt'] for e in comparison_log]
    results = []

    for thresh in thresholds:
        preds = []
        for entry in comparison_log:
            ns = entry.get('norm_scores', {})
            if ns:
                max_z = max(ns.values())
                if max_z < thresh:
                    preds.append('rest')
                else:
                    preds.append(max(ns, key=ns.get))
            else:
                preds.append('rest')

        correct = sum(1 for p, g in zip(preds, gt_labels) if p == g)
        acc = correct / len(gt_labels) if gt_labels else 0
        results.append((float(thresh), acc, correct))

    return results


def _classify_with_biases(norm_scores_list, gt_labels, fingers, biases,
                          rest_threshold=0.0):
    """Classify all windows using per-finger biases. Returns (accuracy, preds).

    adjusted[f] = norm_score[f] - bias[f]
    If max adjusted < rest_threshold → rest, else argmax.
    """
    preds = []
    for ns in norm_scores_list:
        adjusted = {f: ns.get(f, -999) - biases[f] for f in fingers}
        max_adj = max(adjusted.values())
        if max_adj < rest_threshold:
            preds.append('rest')
        else:
            preds.append(max(adjusted, key=adjusted.get))

    correct = sum(1 for p, g in zip(preds, gt_labels) if p == g)
    acc = correct / len(gt_labels) if gt_labels else 0
    return acc, preds


def _normalized_accuracy(preds, gt_labels, class_names):
    """Compute mean per-class accuracy (balanced accuracy)."""
    classes = set(gt_labels)
    per_class_acc = []
    for cls in class_names:
        total = sum(1 for g in gt_labels if g == cls)
        if total == 0:
            continue
        correct = sum(1 for p, g in zip(preds, gt_labels) if g == cls and p == cls)
        per_class_acc.append(correct / total)
    return np.mean(per_class_acc) if per_class_acc else 0.0


def optimize_per_finger_thresholds(comparison_log, fingers_with_refs,
                                   metric='normalized', max_iters=20,
                                   verbose=True):
    """Find per-finger bias thresholds that maximize classification accuracy.

    Uses coordinate descent: for each finger, sweep its bias while keeping
    others fixed. Iterate until convergence or max_iters.

    The biases shift each finger's z-score: adjusted[f] = z_score[f] - bias[f].
    A positive bias makes a finger harder to activate (penalizes it).
    A negative bias makes it easier to activate (boosts it).

    Args:
        comparison_log: list with 'norm_scores' and 'gt' keys
        fingers_with_refs: list of finger names
        metric: 'accuracy' (overall) or 'normalized' (mean per-class accuracy)
        max_iters: max coordinate descent iterations
        verbose: print progress

    Returns:
        biases: dict finger -> optimal bias value
        best_acc: best accuracy achieved
        preds: list of predicted labels with optimal biases
    """
    norm_scores_list = [e.get('norm_scores', {}) for e in comparison_log]
    gt_labels = [e['gt'] for e in comparison_log]
    class_names = ['rest'] + list(fingers_with_refs)

    # Sweep range for each finger
    sweep_values = np.arange(-3.0, 3.01, 0.1)

    # Initialize biases to 0
    biases = {f: 0.0 for f in fingers_with_refs}

    def score_fn(biases):
        acc, preds = _classify_with_biases(
            norm_scores_list, gt_labels, fingers_with_refs, biases)
        if metric == 'normalized':
            return _normalized_accuracy(preds, gt_labels, class_names)
        return acc

    best_score = score_fn(biases)
    if verbose:
        print(f"\n  Per-finger threshold optimization ({metric}):")
        print(f"    Initial: {best_score:.1%} (all biases = 0)")

    for iteration in range(max_iters):
        improved = False
        for finger in fingers_with_refs:
            best_val = biases[finger]
            best_finger_score = best_score

            for val in sweep_values:
                biases[finger] = val
                s = score_fn(biases)
                if s > best_finger_score:
                    best_finger_score = s
                    best_val = val

            if best_val != biases[finger] or best_finger_score > best_score:
                biases[finger] = best_val
                if best_finger_score > best_score:
                    best_score = best_finger_score
                    improved = True

        if verbose:
            bias_str = ', '.join(f'{f[:3]}={biases[f]:+.1f}'
                                 for f in fingers_with_refs)
            print(f"    Iter {iteration + 1}: {best_score:.1%}  [{bias_str}]")

        if not improved:
            if verbose:
                print(f"    Converged after {iteration + 1} iterations")
            break

    # Final classification with optimal biases
    _, preds = _classify_with_biases(
        norm_scores_list, gt_labels, fingers_with_refs, biases)

    # Also apply to comparison_log
    for entry, pred in zip(comparison_log, preds):
        entry['pred_optimized'] = pred
        adjusted = {f: entry.get('norm_scores', {}).get(f, -999) - biases[f]
                    for f in fingers_with_refs}
        entry['optimized_scores'] = adjusted

    return biases, best_score, preds


def print_classification_report(comparison_log, pred_key, label, fingers_with_refs):
    """Print accuracy, per-class accuracy, and confusion matrix."""
    predictions = [e[pred_key] for e in comparison_log]
    gt_labels = [e['gt'] for e in comparison_log]

    correct = sum(1 for p, g in zip(predictions, gt_labels) if p == g)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    print(f"\n  [{label}] Accuracy: {correct}/{total} = {accuracy:.1%}")

    classes_seen = sorted(set(gt_labels + predictions),
                          key=lambda c: CLASS_NAMES.index(c) if c in CLASS_NAMES else 99)
    print(f"\n  {'Class':>10}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}  {'Pred':>5}")
    print(f"  {'-'*42}")
    for cls in classes_seen:
        cls_total = sum(1 for g in gt_labels if g == cls)
        cls_correct = sum(1 for p, g in zip(predictions, gt_labels)
                          if g == cls and p == cls)
        cls_pred = sum(1 for p in predictions if p == cls)
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0
        print(f"  {cls:>10}  {cls_correct:>7}  {cls_total:>5}  {cls_acc:>6.1%}  {cls_pred:>5}")

    print(f"\n  Confusion matrix (rows=GT, cols=Pred):")
    header = f"  {'':>10}"
    for cls in classes_seen:
        header += f"  {cls:>8}"
    print(header)
    for gt_cls in classes_seen:
        row = f"  {gt_cls:>10}"
        for pred_cls in classes_seen:
            count = sum(1 for p, g in zip(predictions, gt_labels)
                        if g == gt_cls and p == pred_cls)
            row += f"  {count:>8}"
        print(row)

    return accuracy


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify finger flex using ranking model reference comparisons')

    # Model / data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (default: auto-detect from checkpoint dir)')
    parser.add_argument('--h5-files', type=str, nargs='+', required=True,
                        help='H5 files for reference extraction and testing')
    parser.add_argument('--ref-h5', type=str, default=None,
                        help='Specific H5 file for reference extraction '
                             '(default: search across all files)')
    parser.add_argument('--gpu-id', type=int, default=0)

    # Evaluation
    parser.add_argument('--eval-stride', type=int, default=8,
                        help='Stride between evaluation windows in frames (default: 8)')
    parser.add_argument('--gt-threshold-deg', type=float, default=40.0,
                        help='Min mean |velocity| in deg/s to count as active '
                             'for GT labeling (default: 40.0)')
    parser.add_argument('--flex-threshold', type=float, default=0.35,
                        help='Max diagonal P below this → classify as rest (default: 0.35)')
    parser.add_argument('--min-flex-deg', type=float, default=8.0,
                        help='Min ROM for auto-detecting flex reference windows')
    parser.add_argument('--normalize', type=str, default='zscore',
                        choices=['none', 'zscore', 'minmax'],
                        help='Per-finger normalization method (default: zscore)')
    parser.add_argument('--rest-z-threshold', type=float, default=0.0,
                        help='Z-score threshold for rest detection after normalization '
                             '(default: 0.0, i.e. below mean = rest)')
    parser.add_argument('--biases', type=float, nargs=5, default=None,
                        metavar=('IDX', 'MID', 'RING', 'PIN', 'THU'),
                        help='Pre-set per-finger biases (index middle ring pinky thumb). '
                             'Skips per-subject optimization and uses these shared biases.')
    parser.add_argument('--static-rest-frame', type=int, default=None,
                        metavar='N',
                        help='Use a single duplicated frame as rest reference instead of '
                             'a real sequence. Frame N from the first H5 file is repeated '
                             'seq_len times to create a zero-motion reference. '
                             'Recommended: 3 or 4 (avoids first-frame artifacts).')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Save results JSON to this path')
    parser.add_argument('--plot', action='store_true',
                        help='Generate diagnostic plot')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    config_path = args.config or str(Path(args.checkpoint).parent / 'config.json')
    model, config = load_multi_finger_model(args.checkpoint, config_path, device)
    seq_len = config['sequence_length']
    image_size = config.get('image_size', 224)
    stride = config.get('stride', 1)
    span = (seq_len - 1) * stride + 1

    transform = create_transform(image_size)

    # Load all H5 files
    all_data = {}
    for h5_path in args.h5_files:
        images, joint_angles, num_frames = load_h5_data(h5_path)
        all_data[h5_path] = {
            'images': images, 'joint_angles': joint_angles, 'num_frames': num_frames,
        }

    # --------------------------------------------------------
    # Find reference windows
    # --------------------------------------------------------
    if args.ref_h5:
        ref_data = all_data[args.ref_h5]
        rest_start, flex_starts, _ = find_reference_windows(
            ref_data['joint_angles'], ref_data['num_frames'], span,
            min_flex_deg=args.min_flex_deg)
        rest_info = (args.ref_h5, rest_start)
        flex_info = {f: (args.ref_h5, s) for f, s in flex_starts.items()}
    else:
        rest_info, flex_info = find_reference_windows_multi(
            all_data, span, min_flex_deg=args.min_flex_deg)

    # Flex references (needed regardless of rest mode)
    if rest_info is None and args.static_rest_frame is None:
        print("Error: could not find a rest reference window")
        sys.exit(1)

    print(f"\n--- References ---")

    # Static rest: duplicate a single frame to create a zero-motion reference
    use_static_rest = args.static_rest_frame is not None
    if use_static_rest:
        static_frame_idx = args.static_rest_frame
        static_h5 = args.h5_files[0]
        static_images = all_data[static_h5]['images']
        assert static_frame_idx < len(static_images), \
            f"--static-rest-frame {static_frame_idx} out of range " \
            f"(file has {len(static_images)} frames)"
        print(f"  Rest : STATIC frame {static_frame_idx} from "
              f"{Path(static_h5).name} (duplicated x{seq_len})")
        rest_h5, rest_start = static_h5, static_frame_idx
    else:
        rest_h5, rest_start = rest_info
        print(f"  Rest : {Path(rest_h5).name} frames "
              f"{rest_start}-{rest_start + span - 1}")

    for finger, (h5, start) in flex_info.items():
        print(f"  {finger:>6}: {Path(h5).name} frames {start}-{start + span - 1}")

    fingers_with_refs = sorted(flex_info.keys(), key=FINGER_NAMES.index)
    if not fingers_with_refs:
        print("Error: no flex references found. Try lowering --min-flex-deg")
        sys.exit(1)

    # Prepare reference tensors
    if use_static_rest:
        # Build rest reference by duplicating a single frame
        img = static_images[static_frame_idx].copy()
        img = img.astype(np.float32) / 255.0
        result = transform(image=img)
        frame_tensor = result['image']  # [3, H, W]
        ref_rest = frame_tensor.unsqueeze(0).repeat(
            seq_len, 1, 1, 1).unsqueeze(0).to(device)  # [1, seq_len, 3, H, W]
    else:
        ref_rest = prepare_sequence(
            all_data[rest_h5]['images'], rest_start, seq_len, stride, transform
        ).to(device)

    ref_flex = {}
    for finger, (h5, start) in flex_info.items():
        ref_flex[finger] = prepare_sequence(
            all_data[h5]['images'], start, seq_len, stride, transform
        ).to(device)

    # --------------------------------------------------------
    # Evaluate each H5 file
    # --------------------------------------------------------
    all_results = []

    for h5_path in args.h5_files:
        data = all_data[h5_path]
        images = data['images']
        joint_angles = data['joint_angles']
        num_frames = data['num_frames']

        print(f"\n--- Evaluating {Path(h5_path).name} ({num_frames} frames) ---")

        predictions = []
        gt_labels = []
        comparison_log = []

        eval_positions = list(range(span - 1, num_frames, args.eval_stride))

        for frame_idx in tqdm(eval_positions, desc="Classifying"):
            window_start = frame_idx - span + 1

            # Ground truth
            gt_label, gt_activities, gt_signed_vel = label_window_gt(
                joint_angles, window_start, span,
                threshold_deg=args.gt_threshold_deg)

            # Prepare query
            query_seq = prepare_sequence(
                images, window_start, seq_len, stride, transform).to(device)

            with torch.no_grad():
                # Compare against rest
                logits_rest = model(query_seq, ref_rest)
                p_vs_rest = {
                    f: torch.sigmoid(logits_rest[f]).item() for f in FINGER_NAMES
                }

                # Compare against each flex reference
                p_vs_flex = {}
                for ref_finger, ref_seq in ref_flex.items():
                    logits_flex = model(query_seq, ref_seq)
                    p_vs_flex[ref_finger] = {
                        f: torch.sigmoid(logits_flex[f]).item() for f in FINGER_NAMES
                    }

            # Classify
            pred_label, scores = classify_window(
                p_vs_rest, p_vs_flex, flex_threshold=args.flex_threshold)

            predictions.append(pred_label)
            gt_labels.append(gt_label)
            comparison_log.append({
                'frame': frame_idx,
                'gt': gt_label,
                'pred': pred_label,
                'p_vs_rest': p_vs_rest,
                'p_vs_flex': p_vs_flex,
                'gt_activities': gt_activities,
                'gt_signed_vel': gt_signed_vel,
            })

        # ------- GT gap filling (flex → hold → extend pattern) -------
        n_filled = fill_gt_gaps(comparison_log)
        if n_filled > 0:
            print(f"  GT gap-filling: relabeled {n_filled} rest windows "
                  f"(flex→hold→extend direction check)")

        # ------- GT smoothing (remove single-window interruptions) -------
        n_smoothed = smooth_gt_labels(comparison_log)
        if n_smoothed > 0:
            print(f"  GT smoothing: relabeled {n_smoothed} windows "
                  f"(single-window interruptions)")

        if n_filled > 0 or n_smoothed > 0:
            gt_counts = {}
            for e in comparison_log:
                gt_counts[e['gt']] = gt_counts.get(e['gt'], 0) + 1
            print(f"  GT after post-processing: {gt_counts}")

        # ------- Raw results -------
        raw_acc = print_classification_report(
            comparison_log, 'pred', 'Raw', fingers_with_refs)

        # ------- Normalized results -------
        norm_acc = None
        if args.normalize != 'none':
            stats = normalize_and_reclassify(
                comparison_log, fingers_with_refs,
                method=args.normalize,
                rest_z_threshold=args.rest_z_threshold)

            print(f"\n  Per-finger diagonal stats ({args.normalize}):")
            for f in fingers_with_refs:
                s = stats[f]
                if args.normalize == 'zscore':
                    print(f"    {f:>6}: mean={s['mean']:.3f}  std={s['std']:.3f}")
                else:
                    print(f"    {f:>6}: min={s['min']:.3f}  range={s['range']:.3f}")

            norm_acc = print_classification_report(
                comparison_log, 'pred_norm', f'Normalized ({args.normalize})',
                fingers_with_refs)

            # Threshold sweep
            sweep_results = sweep_rest_threshold(
                comparison_log, fingers_with_refs, stats, method=args.normalize)
            best_thresh, best_acc, best_correct = max(sweep_results, key=lambda x: x[1])
            total = len(comparison_log)
            print(f"\n  Threshold sweep (rest_z_threshold):")
            print(f"  {'thresh':>8}  {'acc':>8}  {'correct':>8}")
            print(f"  {'-'*28}")
            for thresh, acc, correct in sweep_results:
                marker = ' <-- best' if thresh == best_thresh else ''
                print(f"  {thresh:>+8.2f}  {acc:>7.1%}  {correct:>5}/{total}{marker}")
            print(f"\n  Best threshold: {best_thresh:+.2f} → {best_acc:.1%} ({best_correct}/{total})")

            # ------- Per-finger threshold optimization -------
            if args.biases is not None:
                # Use pre-set shared biases
                biases = {f: b for f, b in zip(fingers_with_refs, args.biases)}
                bias_str = ', '.join(f'{f[:3]}={biases[f]:+.1f}'
                                     for f in fingers_with_refs)
                print(f"\n  Using pre-set biases: [{bias_str}]")

                norm_scores_list = [e.get('norm_scores', {}) for e in comparison_log]
                gt_labels = [e['gt'] for e in comparison_log]
                _, preds = _classify_with_biases(
                    norm_scores_list, gt_labels, fingers_with_refs, biases)
                for entry, pred in zip(comparison_log, preds):
                    entry['pred_optimized'] = pred
                    adjusted = {f: entry.get('norm_scores', {}).get(f, -999) - biases[f]
                                for f in fingers_with_refs}
                    entry['adjusted_scores'] = adjusted

                opt_overall_acc = print_classification_report(
                    comparison_log, 'pred_optimized',
                    'Pre-set shared biases', fingers_with_refs)
            else:
                biases, opt_acc, opt_preds = optimize_per_finger_thresholds(
                    comparison_log, fingers_with_refs, metric='normalized')

                opt_overall_acc = print_classification_report(
                    comparison_log, 'pred_optimized',
                    'Per-finger optimized (normalized metric)', fingers_with_refs)

                # Also optimize for raw accuracy
                biases_raw, opt_acc_raw, _ = optimize_per_finger_thresholds(
                    comparison_log, fingers_with_refs, metric='accuracy')
                # Don't overwrite pred_optimized — just report
                print(f"\n  Per-finger optimized (raw accuracy metric): {opt_acc_raw:.1%}")
                bias_str = ', '.join(f'{f[:3]}={biases_raw[f]:+.1f}'
                                     for f in fingers_with_refs)
                print(f"    Biases: [{bias_str}]")

        # Example comparisons
        show_norm = args.normalize != 'none'
        print(f"\n  Example comparisons (first 5 non-rest windows):")
        shown = 0
        for entry in comparison_log:
            if entry['gt'] == 'rest' and shown == 0:
                continue
            pred_str = entry['pred']
            if show_norm:
                pred_str += f" → {entry.get('pred_norm', '?')} (norm)"
            print(f"\n    Frame {entry['frame']}: GT={entry['gt']}, Pred={pred_str}")
            print(f"      P(query > rest):  ", end='')
            for f in FINGER_NAMES:
                print(f" {f[:3]}={entry['p_vs_rest'][f]:.2f}", end='')
            print()
            # Diagonal values: raw and normalized
            print(f"      Diagonal (raw):   ", end='')
            for f in fingers_with_refs:
                v = entry['p_vs_flex'][f][f]
                print(f" {f[:3]}={v:.2f}", end='')
            print()
            if show_norm and 'norm_scores' in entry:
                print(f"      Diagonal (norm):  ", end='')
                for f in fingers_with_refs:
                    z = entry['norm_scores'][f]
                    print(f" {f[:3]}={z:+.2f}", end='')
                print()
            shown += 1
            if shown >= 5:
                break

        all_results.append({
            'h5_file': str(h5_path),
            'raw_accuracy': raw_acc,
            'norm_accuracy': norm_acc,
            'comparison_log': comparison_log,
            'optimized_biases': biases if args.normalize != 'none' else None,
        })

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    raw_accs = [r['raw_accuracy'] for r in all_results]
    print(f"Overall RAW accuracy: {np.mean(raw_accs):.1%}")
    if args.normalize != 'none':
        norm_accs = [r['norm_accuracy'] for r in all_results if r['norm_accuracy'] is not None]
        print(f"Overall NORMALIZED ({args.normalize}) accuracy: {np.mean(norm_accs):.1%}")
    print(f"Fingers with references: {', '.join(fingers_with_refs)}")

    # Save results
    if args.output:
        output_data = {
            'config': {
                'checkpoint': args.checkpoint,
                'h5_files': args.h5_files,
                'rest_ref': {'h5': rest_h5, 'start': rest_start,
                             'static': use_static_rest},
                'flex_refs': {f: {'h5': h5, 'start': s}
                              for f, (h5, s) in flex_info.items()},
                'eval_stride': args.eval_stride,
                'gt_threshold_deg': args.gt_threshold_deg,
                'flex_threshold': args.flex_threshold,
                'normalize': args.normalize,
            },
            'per_file': [{
                'h5_file': r['h5_file'],
                'raw_accuracy': r['raw_accuracy'],
                'norm_accuracy': r['norm_accuracy'],
            } for r in all_results],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

        # Also save comparison_log with norm_scores for cross-subject optimization
        import pickle
        pkl_path = args.output.replace('.json', '_log.pkl')
        pkl_data = {
            'fingers_with_refs': fingers_with_refs,
            'per_file': [{
                'h5_file': r['h5_file'],
                'comparison_log': r['comparison_log'],
            } for r in all_results],
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        print(f"Comparison log saved to: {pkl_path}")

    # --------------------------------------------------------
    # Diagnostic plot
    # --------------------------------------------------------
    show_norm = args.normalize != 'none'

    if args.plot and all_results:
        for result in all_results:
            log = result['comparison_log']
            if not log:
                continue

            h5_stem = Path(result['h5_file']).stem
            frames = [e['frame'] for e in log]

            colors = {'index': 'C0', 'middle': 'C1', 'ring': 'C2',
                      'pinky': 'C3', 'thumb': 'C4'}

            # Number of panels: joint angles + rest + GT/pred + normalized diag + per-ref panels
            n_panels = 3 + len(fingers_with_refs)
            if show_norm:
                n_panels += 2  # normalized diagonal + normalized GT/pred
            fig, axes = plt.subplots(n_panels, 1,
                                     figsize=(16, 3 * n_panels),
                                     sharex=True)

            panel = 0

            # Panel: Joint angles per finger (degrees)
            ax = axes[panel]; panel += 1
            file_data = all_data[result['h5_file']]
            all_frames_range = np.arange(file_data['num_frames'])
            for finger in FINGER_NAMES:
                if finger in file_data['joint_angles']:
                    angles_deg = np.rad2deg(file_data['joint_angles'][finger])
                    ax.plot(all_frames_range, angles_deg, label=finger,
                            color=colors[finger], alpha=0.8, linewidth=1)
            ax.set_ylabel('Joint angle (deg)')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title(f'{h5_stem} — Raw Joint Angles')

            # Panel: P(query > rest) per finger
            ax = axes[panel]; panel += 1
            for finger in FINGER_NAMES:
                values = [e['p_vs_rest'][finger] for e in log]
                ax.plot(frames, values, label=finger, color=colors[finger], alpha=0.8)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('P(query > rest)')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title(f'{h5_stem} — Comparison vs Rest')

            # Panel: GT labels and raw predictions
            ax = axes[panel]; panel += 1
            gt_numeric = [CLASS_NAMES.index(e['gt']) if e['gt'] in CLASS_NAMES else -1
                          for e in log]
            pred_numeric = [CLASS_NAMES.index(e['pred']) if e['pred'] in CLASS_NAMES else -1
                            for e in log]
            ax.scatter(frames, gt_numeric, c='green', s=12, alpha=0.6, label='GT')
            ax.scatter(frames, pred_numeric, c='red', s=12, alpha=0.6, marker='x', label='Pred (raw)')
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_yticklabels(CLASS_NAMES, fontsize=8)
            ax.set_ylabel('Class')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title('Ground Truth vs Raw Predictions')

            # Panel: Normalized diagonal z-scores (all fingers on one plot)
            if show_norm:
                ax = axes[panel]; panel += 1
                for finger in fingers_with_refs:
                    values = [e.get('norm_scores', {}).get(finger, np.nan) for e in log]
                    ax.plot(frames, values, label=finger, color=colors[finger],
                            alpha=0.8, linewidth=1.5)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='mean')
                ax.axhline(y=args.rest_z_threshold, color='red', linestyle=':',
                           alpha=0.4, label=f'rest thresh={args.rest_z_threshold}')
                ax.set_ylabel(f'Z-score')
                ax.legend(loc='upper right', fontsize=8)
                ax.set_title(f'Normalized Diagonal Scores ({args.normalize}) — per finger')

                # Panel: GT vs normalized predictions
                ax = axes[panel]; panel += 1
                pred_norm_numeric = [
                    CLASS_NAMES.index(e.get('pred_norm', 'rest'))
                    if e.get('pred_norm', 'rest') in CLASS_NAMES else -1
                    for e in log]
                ax.scatter(frames, gt_numeric, c='green', s=12, alpha=0.6, label='GT')
                ax.scatter(frames, pred_norm_numeric, c='blue', s=12, alpha=0.6,
                           marker='x', label='Pred (norm)')
                ax.set_yticks(range(len(CLASS_NAMES)))
                ax.set_yticklabels(CLASS_NAMES, fontsize=8)
                ax.set_ylabel('Class')
                ax.legend(loc='upper right', fontsize=8)
                ax.set_title(f'Ground Truth vs Normalized Predictions')

            # Panels: P(query > flex_i_ref) per reference finger (raw)
            for idx, ref_finger in enumerate(fingers_with_refs):
                ax = axes[panel]; panel += 1
                for finger in FINGER_NAMES:
                    values = [e['p_vs_flex'].get(ref_finger, {}).get(finger, np.nan)
                              for e in log]
                    lw = 2.5 if finger == ref_finger else 1
                    ax.plot(frames, values, label=finger, color=colors[finger],
                            alpha=0.9 if finger == ref_finger else 0.5, linewidth=lw)
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.axhline(y=args.flex_threshold, color='red', linestyle=':',
                           alpha=0.4, label=f'thresh={args.flex_threshold}')
                ax.set_ylabel(f'P(q > flex_{ref_finger[:3]})')
                ax.legend(loc='upper right', fontsize=8)
                ax.set_title(f'Comparison vs {ref_finger.capitalize()} Flex Reference')

            axes[-1].set_xlabel('Frame')
            plt.tight_layout()
            plot_path = f'{h5_stem}_classify_refs.png'
            if args.output:
                plot_path = str(Path(args.output).parent / plot_path)
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved to: {plot_path}")
            plt.close()


if __name__ == '__main__':
    main()

"""
Visualize flex embedding event detection from saved raw predictions.

For each recording session, creates a multi-panel plot showing:
  - Per-finger probability signals vs GT flex intervals
  - Event detection results (TP/FP/FN shading) at specified max_gap

Usage:
    python -m evaluation.visualize_flex_embedding_events \
        --methods-dir outputs/experiments/methods_fixed_labels \
        --method ranking_backbone \
        --max-gap 8 --threshold 0.4 --min-duration 10

    # All methods side by side for one session:
    python -m evaluation.visualize_flex_embedding_events \
        --methods-dir outputs/experiments/methods_fixed_labels \
        --method all \
        --max-gap 8

Reader's guide: this is a post-inference visualization. It reads saved raw
signals, applies the requested temporal filtering, matches events, and renders
the decisions without loading a neural-network checkpoint.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from evaluation.event_detection_metrics import (
    FINGER_NAMES, extract_per_finger_signal,
    threshold_and_postprocess, binary_to_events,
    compute_rest_intervals, match_events_overlap,
    aggregate_results,
)
from evaluation.recompute_event_metrics import (
    load_checkpoint_data, evaluate_finger_from_raw,
)

FINGER_COLORS = {
    'index': '#1f77b4',
    'middle': '#ff7f0e',
    'ring': '#2ca02c',
    'pinky': '#d62728',
    'thumb': '#9467bd',
}

FPS = 20


def find_best_params_fixed_gap(data, max_gap):
    """Grid search over threshold and min_duration with fixed max_gap."""
    threshold_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    min_duration_range = [4, 6, 8, 10, 14, 20]

    best_f1 = -1
    best_params = {'threshold': 0.4, 'min_duration': 10}

    for thresh, min_dur in product(threshold_range, min_duration_range):
        all_results = []
        for file_data in data:
            for fi, finger in enumerate(FINGER_NAMES):
                r = evaluate_finger_from_raw(
                    file_data['predictions'], finger, fi,
                    file_data['gt_flex_labels'], file_data['num_frames'],
                    file_data['eval_stride'],
                    thresh, min_dur, max_gap)
                all_results.append(r)
        agg = aggregate_results(all_results)
        if agg['micro']['f1'] > best_f1:
            best_f1 = agg['micro']['f1']
            best_params = {'threshold': thresh, 'min_duration': min_dur}

    return best_params, best_f1


def plot_session(raw_path, max_gap, threshold, min_duration, output_path,
                 method_name=None):
    """Plot one recording session with GT vs predictions and event detection."""
    with open(raw_path) as f:
        raw = json.load(f)

    h5_stem = Path(raw['h5_file']).stem
    num_frames = raw['num_frames']
    eval_stride = raw['eval_stride']
    predictions = raw['predictions']

    frames = np.arange(num_frames)
    time_s = frames / FPS

    fig, axes = plt.subplots(len(FINGER_NAMES), 1, figsize=(20, 3 * len(FINGER_NAMES)),
                             sharex=True)

    title = h5_stem
    if method_name:
        title = f"{h5_stem} — {method_name}"
    title += f"\nmax_gap={max_gap} ({max_gap/FPS:.1f}s), thresh={threshold}, min_dur={min_duration}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.0)

    for fi, finger in enumerate(FINGER_NAMES):
        ax = axes[fi]
        color = FINGER_COLORS[finger]

        # GT binary labels
        gt_binary = np.array(raw['gt_flex_labels'][finger], dtype=float)

        # Probability signal (full-frame interpolated)
        prob_signal = np.array(raw['prob_signals'][finger], dtype=float)
        # Replace None with NaN
        prob_signal = np.where(np.isnan(prob_signal), 0, prob_signal)

        # Event detection
        pred_binary = threshold_and_postprocess(
            prob_signal, threshold, min_duration, max_gap)
        pred_events = binary_to_events(pred_binary)
        gt_events = binary_to_events(gt_binary)
        rest_intervals = compute_rest_intervals(gt_events, num_frames)
        metrics = match_events_overlap(pred_events, gt_events, rest_intervals)

        # Shade GT events (light green)
        for start, end in gt_events:
            ax.axvspan(start / FPS, end / FPS, alpha=0.25, color='green',
                       linewidth=0)

        # Shade predicted events with TP/FP coloring
        gt_matched = set()
        for pi, (ps, pe) in enumerate(pred_events):
            is_tp = False
            for gi, (gs, ge) in enumerate(gt_events):
                if ps < ge and pe > gs:  # overlap
                    is_tp = True
                    gt_matched.add(gi)
                    break
            if is_tp:
                ax.axvspan(ps / FPS, pe / FPS, alpha=0.15, color='blue',
                           linewidth=0)
            else:
                ax.axvspan(ps / FPS, pe / FPS, alpha=0.3, color='red',
                           linewidth=0)

        # Shade FN GT events (missed)
        for gi, (gs, ge) in enumerate(gt_events):
            if gi not in gt_matched:
                ax.axvspan(gs / FPS, ge / FPS, alpha=0.2, color='orange',
                           linewidth=0)

        # Plot probability signal
        ax.plot(time_s, prob_signal, color=color, linewidth=1.0, alpha=0.9)

        # Plot threshold line
        ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.5,
                    alpha=0.5)

        # Metrics text
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        p = metrics['precision'] if 'precision' in metrics else (tp / (tp + fp) if tp + fp > 0 else 0)
        r = metrics['recall'] if 'recall' in metrics else (tp / (tp + fn) if tp + fn > 0 else 0)
        f1 = metrics['f1'] if 'f1' in metrics else (2 * p * r / (p + r) if p + r > 0 else 0)

        ax.set_ylabel(finger, fontsize=11, fontweight='bold', color=color)
        ax.set_ylim(-0.05, 1.05)
        ax.text(0.99, 0.95,
                f"TP={tp} FP={fp} FN={fn}\nP={p:.2f} R={r:.2f} F1={f1:.2f}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Time (s)', fontsize=11)

    # Legend
    legend_patches = [
        mpatches.Patch(color='green', alpha=0.25, label='GT flex event'),
        mpatches.Patch(color='blue', alpha=0.15, label='TP (correct detection)'),
        mpatches.Patch(color='red', alpha=0.3, label='FP (false alarm)'),
        mpatches.Patch(color='orange', alpha=0.2, label='FN (missed event)'),
    ]
    axes[0].legend(handles=legend_patches, loc='upper left', fontsize=7,
                   ncol=4, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize flex embedding event detection')
    parser.add_argument('--methods-dir', type=str,
                        default='outputs/experiments/methods_fixed_labels',
                        help='Directory containing method folders')
    parser.add_argument('--method', type=str, default='all',
                        help='Method to visualize (ranking_backbone, no_backbone, no_rest, or all)')
    parser.add_argument('--subjects', type=int, nargs='+', default=[4, 10, 17, 23],
                        help='Test subjects to plot')
    parser.add_argument('--max-gap', type=int, default=8,
                        help='Max gap for event merging (frames)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed threshold (if None, grid-search per subject)')
    parser.add_argument('--min-duration', type=int, default=None,
                        help='Fixed min duration (if None, grid-search per subject)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: methods_dir/plots_gap{max_gap})')
    args = parser.parse_args()

    methods_dir = Path(args.methods_dir)

    if args.method == 'all':
        method_names = ['ranking_backbone', 'no_backbone', 'no_rest']
    else:
        method_names = [args.method]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = methods_dir / f'plots_gap{args.max_gap}'
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name in method_names:
        method_dir = methods_dir / f'flex_embedding_{method_name}'
        if not method_dir.exists():
            print(f"WARNING: {method_dir} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        for subj in args.subjects:
            # Find checkpoint folder for this subject
            folders = list(method_dir.glob(f'*test{subj}_*'))
            if not folders:
                print(f"  No folder for subject {subj}, skipping")
                continue
            folder = folders[0]
            event_dir = folder / 'event_detection'

            # Determine threshold and min_duration
            if args.threshold is not None and args.min_duration is not None:
                threshold = args.threshold
                min_duration = args.min_duration
                print(f"\n  Subject {subj}: using fixed params "
                      f"thresh={threshold}, min_dur={min_duration}, max_gap={args.max_gap}")
            else:
                # Grid search for best params at this max_gap
                data = load_checkpoint_data(event_dir, use_fixed_labels=True)
                if not data:
                    print(f"  Subject {subj}: no raw predictions, skipping")
                    continue
                best_params, best_f1 = find_best_params_fixed_gap(
                    data, args.max_gap)
                threshold = best_params['threshold']
                min_duration = best_params['min_duration']
                print(f"\n  Subject {subj}: grid-searched params "
                      f"thresh={threshold}, min_dur={min_duration}, "
                      f"max_gap={args.max_gap} -> F1={best_f1:.3f}")

            # Plot each raw prediction file
            raw_files = sorted(event_dir.glob('raw_predictions_*.json'))
            for raw_path in raw_files:
                h5_stem = raw_path.stem.replace('raw_predictions_', '')
                out_name = f"{method_name}_{h5_stem}_gap{args.max_gap}.png"
                plot_session(
                    raw_path, args.max_gap, threshold, min_duration,
                    output_dir / out_name, method_name=method_name)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

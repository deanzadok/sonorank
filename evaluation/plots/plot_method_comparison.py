"""
Visualize predictions from 4 methods on a single recording session,
using per-finger thresholds from the fair evaluation.

Panels:
  1. Joint angles (summed per finger, in degrees)
  2. GT flex labels (colored bars)
  3-6. Per method: P(flex) signals row + predicted events bar row

Usage:
    python evaluation/plots/plot_method_comparison.py \
        --methods-dir outputs/experiments/methods \
        --fair-results evaluation/plots/methods_fair_raw_cov0.3_mindur10_gap0_*.json \
        --subject 4 --session enrollment01_finger_flex01 \
        --output evaluation/plots/comparison_s04_flex01.png

Reader's guide: the fair-evaluation JSON supplies thresholds while each fold's
raw prediction file supplies time-series data. Event reconstruction here is
for visualization only and does not overwrite evaluation results.
"""

import json
import re
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
FINGER_COLORS = {
    'index': 'C0', 'middle': 'C1', 'ring': 'C2',
    'pinky': 'C3', 'thumb': 'C4',
}
JOINT_COLS = {
    'index': ['joint11', 'joint12'],
    'middle': ['joint21', 'joint22'],
    'ring': ['joint31', 'joint32'],
    'pinky': ['joint41', 'joint42'],
    'thumb': ['jointt5', 'jointt6'],
}
FPS = 20

METHOD_ORDER = [
    'flex_embedding_ranking_backbone',
    'multi_finger_ranking',
    'flex_embedding_no_rest',
    'flex_embedding_no_backbone',
]
METHOD_LABELS = {
    'flex_embedding_ranking_backbone': 'Ranking Backbone (fine-tuned)',
    'flex_embedding_no_backbone': 'No Backbone',
    'flex_embedding_no_rest': 'No Rest Ref.',
    'multi_finger_ranking': 'Ranking (no fine-tune)',
}
METHOD_COLORS = {
    'flex_embedding_ranking_backbone': '#2ca02c',
    'multi_finger_ranking': '#1f77b4',
    'flex_embedding_no_rest': '#ff7f0e',
    'flex_embedding_no_backbone': '#d62728',
}
METHODS_ZNORM = {'multi_finger_ranking'}


def znorm_signal(signal):
    """Z-normalize signal per session, ignoring NaN."""
    valid = ~np.isnan(signal)
    if valid.sum() < 2:
        return signal
    mu = np.nanmean(signal)
    sigma = np.nanstd(signal)
    if sigma < 1e-8:
        return np.zeros_like(signal)
    out = signal.copy()
    out[valid] = (signal[valid] - mu) / sigma
    return out


def binary_to_events(binary):
    """Convert binary array to list of (start, end) frame intervals."""
    events = []
    in_event = False
    for i, v in enumerate(binary):
        if v and not in_event:
            start = i
            in_event = True
        elif not v and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(binary) - 1))
    return events


def threshold_and_filter(signal, threshold, min_duration=10):
    """Threshold signal and remove short events (no gap merging)."""
    pred = np.zeros(len(signal), dtype=bool)
    valid = ~np.isnan(signal)
    pred[valid] = signal[valid] >= threshold

    # Remove short events
    events = binary_to_events(pred)
    for s, e in events:
        if (e - s + 1) < min_duration:
            pred[s:e + 1] = False
    return pred


def draw_event_bars(ax, flex_labels, finger_names, finger_colors, fps, alpha=0.6):
    """Draw horizontal event bars for each finger."""
    for fi, finger in enumerate(finger_names):
        gt = np.array(flex_labels[finger], dtype=float)
        events = binary_to_events(gt)
        for s, e in events:
            ax.barh(fi, (e - s + 1) / fps, left=s / fps, height=0.7,
                    color=finger_colors[finger], alpha=alpha)
    ax.set_yticks(range(len(finger_names)))
    ax.set_yticklabels(finger_names, fontsize=8)
    ax.set_ylim(-0.5, len(finger_names) - 0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')


def find_fold_dir(method_dir, subject):
    """Find the fold directory for a given test subject."""
    for d in sorted(Path(method_dir).iterdir()):
        if d.is_dir():
            m = re.search(r'test(\d+)', d.name)
            if m and int(m.group(1)) == subject:
                return d
    return None


def find_raw_prediction(fold_dir, session_pattern):
    """Find raw prediction JSON matching session pattern."""
    event_dir = fold_dir / 'event_detection'
    for f in sorted(event_dir.glob('raw_predictions_*.json')):
        if session_pattern in f.stem:
            return f
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods-dir', type=str, required=True)
    parser.add_argument('--fair-results', type=str, required=True,
                        help='Path to fair evaluation JSON')
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--session', type=str, required=True,
                        help='Session pattern, e.g. enrollment01_finger_flex01')
    parser.add_argument('--min-pred-duration', type=int, default=10)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    methods_dir = Path(args.methods_dir)

    # Load fair evaluation results for thresholds
    fair_data = json.load(open(args.fair_results))
    subj_key = str(args.subject)

    # Fixed thresholds for classifiers
    fixed_thresholds = fair_data.get('fixed_thresholds')
    if fixed_thresholds is None:
        # Backwards compat: older JSON format had fold_thresholds
        fixed_thresholds = fair_data.get('fold_thresholds', {}).get(subj_key, {})
    print(f"Fixed thresholds (classifiers): {fixed_thresholds}")

    # Own thresholds for z-normed methods
    own_thresholds = fair_data.get('own_thresholds', {})
    for mk, sv in own_thresholds.items():
        if subj_key in sv:
            print(f"Own thresholds for {mk}: {sv[subj_key]}")

    # Find the H5 file to load joint angles
    h5_dir = Path('data/processed_h5/h5_files')
    h5_candidates = sorted(h5_dir.glob(
        f'recordings_{args.subject:02d}_*{args.session}*.h5'))
    if not h5_candidates:
        print(f"ERROR: no H5 file found for subject {args.subject}, session {args.session}")
        sys.exit(1)
    h5_path = h5_candidates[0]
    print(f"H5: {h5_path}")

    df = pd.read_hdf(h5_path, 'df')
    num_frames = len(df)
    time = np.arange(num_frames) / FPS

    joint_angles_deg = {}
    for finger in FINGER_NAMES:
        total = np.zeros(num_frames, dtype=np.float32)
        for col in JOINT_COLS[finger]:
            if col in df.columns:
                total += df[col].values.astype(np.float32)
        joint_angles_deg[finger] = np.rad2deg(total)

    # Load predictions for each method
    pred_data = {}
    gt_flex = None
    for method_key in METHOD_ORDER:
        fold_dir = find_fold_dir(methods_dir / method_key, args.subject)
        if fold_dir is None:
            print(f"WARNING: no fold dir for {method_key} subject {args.subject}")
            continue
        raw_pred_file = find_raw_prediction(fold_dir, args.session)
        if raw_pred_file is None:
            print(f"WARNING: no raw predictions for {method_key} session {args.session}")
            continue

        d = json.load(open(raw_pred_file))
        prob_signals = {}
        for finger in FINGER_NAMES:
            sig = np.array(d['prob_signals'][finger], dtype=float)
            sig = np.where(np.equal(sig, None), np.nan, sig)
            prob_signals[finger] = sig

        pred_data[method_key] = prob_signals

        if gt_flex is None:
            gt_flex = d['gt_flex_labels']

    n_methods = len(pred_data)
    print(f"Loaded {n_methods} methods")

    # --- Layout ---
    # joint_angles(3), GT_bars(1.2), then per method: signals(2.5) + pred_bars(1.2)
    height_ratios = [3, 1.2]
    for _ in range(n_methods):
        height_ratios.extend([2.5, 1.2])

    fig = plt.figure(figsize=(18, sum(height_ratios) * 0.85))
    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios,
                           hspace=0.15)

    all_axes = [fig.add_subplot(gs[i]) for i in range(len(height_ratios))]
    for ax in all_axes[1:]:
        ax.sharex(all_axes[0])

    panel = 0

    # Panel 1: Joint angles
    ax = all_axes[panel]; panel += 1
    for finger in FINGER_NAMES:
        ax.plot(time, joint_angles_deg[finger], color=FINGER_COLORS[finger],
                label=finger, alpha=0.8, linewidth=1)
    session_short = args.session.replace('_', ' ')
    ax.set_ylabel('Joint angle (deg)')
    ax.set_title(
        f'Subject {args.subject} \u2014 {session_short}  '
        f'({num_frames} frames, {num_frames/FPS:.1f}s)',
        fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)

    # Panel 2: GT flex events
    ax = all_axes[panel]; panel += 1
    draw_event_bars(ax, gt_flex, FINGER_NAMES, FINGER_COLORS, FPS)
    ax.set_ylabel('GT', fontsize=9, fontweight='bold')
    plt.setp(ax.get_xticklabels(), visible=False)

    # Per-method panels
    method_keys_present = [k for k in METHOD_ORDER if k in pred_data]
    for mi, method_key in enumerate(method_keys_present):
        label = METHOD_LABELS[method_key]
        title_color = METHOD_COLORS[method_key]
        raw_signals = pred_data[method_key]
        use_znorm = method_key in METHODS_ZNORM

        # Get thresholds for this method
        if method_key in own_thresholds and subj_key in own_thresholds[method_key]:
            method_thresholds = own_thresholds[method_key][subj_key]
        else:
            method_thresholds = fixed_thresholds

        # Apply z-norm if needed
        plot_signals = {}
        for finger in FINGER_NAMES:
            sig = raw_signals[finger].copy()
            if use_znorm:
                sig = znorm_signal(sig)
            plot_signals[finger] = sig

        # Signal panel
        ax_sig = all_axes[panel]; panel += 1
        for finger in FINGER_NAMES:
            ax_sig.plot(time, plot_signals[finger], color=FINGER_COLORS[finger],
                        alpha=0.7, linewidth=1, label=finger)

        # Draw per-finger threshold lines
        for finger in FINGER_NAMES:
            thresh = method_thresholds[finger]
            ax_sig.axhline(y=thresh, color=FINGER_COLORS[finger],
                           linestyle='--', linewidth=0.8, alpha=0.5)

        if use_znorm:
            ax_sig.set_ylabel('z-score', fontsize=9)
            # Auto y-lim for z-normed signals
            all_vals = np.concatenate([plot_signals[f] for f in FINGER_NAMES])
            ymax = min(np.nanmax(all_vals) * 1.1, 8.0)
            ymin = max(np.nanmin(all_vals) * 1.1, -3.0)
            ax_sig.set_ylim(ymin, ymax)
        else:
            ax_sig.set_ylabel('P(flex)', fontsize=9)
            ax_sig.set_ylim(-0.05, 1.05)

        # Build threshold summary for title
        thresh_parts = [f'{f[:3]}={method_thresholds[f]}'
                        for f in FINGER_NAMES]
        znorm_tag = ' [z-norm]' if use_znorm else ''
        title = f'{label}{znorm_tag}  (thresh: {", ".join(thresh_parts)}, min_dur={args.min_pred_duration})'
        ax_sig.set_title(title, fontsize=9, color=title_color,
                         fontweight='bold', loc='left')
        ax_sig.legend(loc='upper right', fontsize=7, ncol=5)
        ax_sig.grid(True, alpha=0.3)
        plt.setp(ax_sig.get_xticklabels(), visible=False)

        # Predicted events bar panel
        ax_bar = all_axes[panel]; panel += 1
        pred_flex = {}
        for finger in FINGER_NAMES:
            pred_binary = threshold_and_filter(
                plot_signals[finger], method_thresholds[finger],
                args.min_pred_duration)
            pred_flex[finger] = pred_binary.astype(float).tolist()
        draw_event_bars(ax_bar, pred_flex, FINGER_NAMES, FINGER_COLORS, FPS, alpha=0.8)
        ax_bar.set_ylabel('Pred', fontsize=9, fontweight='bold')
        is_last = (mi == n_methods - 1)
        if not is_last:
            plt.setp(ax_bar.get_xticklabels(), visible=False)

    all_axes[-1].set_xlabel('Time (s)', fontsize=10)
    all_axes[0].set_xlim(0, num_frames / FPS)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()

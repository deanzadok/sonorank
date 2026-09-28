"""
Rollout-style figure comparing GT vs predictions from 3 methods across 5 fingers.
Shows horizontal bars per finger over time, similar to the journal paper's
sample_rollout figures.

Usage:
    python -m evaluation.plots.plot_method_rollout \
        --methods-dir outputs/experiments/methods_fixed_labels \
        --session recordings_21_gestures_enrollment02_finger_flex02 \
        --t-start 0 --t-end 80 \
        --threshold 0.5 --min-dur 9 --max-gap 16

Reader's guide: this is a qualitative timeline rather than a metric
calculation. It loads one session from each method and places ground truth and
thresholded predictions on a shared finger and time coordinate system.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

METHODS = [
    ('flex_embedding_ranking_backbone', 'SonoRank'),
    ('flex_embedding_no_backbone', 'MFT+Rest'),
    ('flex_embedding_no_rest', 'MFT'),
]

# Colors matching the bar chart: purple, blue, cyan
METHOD_COLORS = {
    'flex_embedding_ranking_backbone': '#9b7fc4',
    'flex_embedding_no_backbone': '#5b9bd5',
    'flex_embedding_no_rest': '#6dcfcf',
}
GT_COLOR = 'black'

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
FINGER_LABELS = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb']
FPS = 20.0

# Bar layout within each finger row
BAR_HEIGHT = 0.18
GT_OFFSET = 0.0
METHOD_OFFSETS = {
    'flex_embedding_ranking_backbone': -BAR_HEIGHT * 1.05,
    'flex_embedding_no_backbone': -BAR_HEIGHT * 2.10,
    'flex_embedding_no_rest': -BAR_HEIGHT * 3.15,
}


def find_raw_predictions(methods_dir, method_key, session):
    """Find the raw predictions JSON for a method and session."""
    method_dir = Path(methods_dir) / method_key
    for subj_dir in sorted(method_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        json_path = subj_dir / 'event_detection' / f'raw_predictions_{session}.json'
        if json_path.exists():
            return json_path
    return None


def threshold_and_filter(signal, threshold, min_dur, max_gap):
    """Apply threshold, merge gaps, remove short events. Return binary array."""
    pred = np.array(signal) >= threshold

    # Merge gaps
    if max_gap > 0:
        in_event = False
        gap_start = 0
        for i in range(len(pred)):
            if pred[i]:
                if not in_event and i > 0:
                    gap_len = i - gap_start
                    if gap_len <= max_gap:
                        pred[gap_start:i] = True
                in_event = True
            else:
                if in_event:
                    gap_start = i
                    in_event = False

    # Remove short events
    if min_dur > 0:
        events = []
        start = None
        for i in range(len(pred)):
            if pred[i] and start is None:
                start = i
            elif not pred[i] and start is not None:
                events.append((start, i))
                start = None
        if start is not None:
            events.append((start, len(pred)))

        for s, e in events:
            if (e - s) < min_dur:
                pred[s:e] = False

    return pred


def binary_to_intervals(binary, fps):
    """Convert binary array to list of (start_sec, end_sec) intervals."""
    intervals = []
    start = None
    for i in range(len(binary)):
        if binary[i] and start is None:
            start = i
        elif not binary[i] and start is not None:
            intervals.append((start / fps, i / fps))
            start = None
    if start is not None:
        intervals.append((start / fps, len(binary) / fps))
    return intervals


def save_figure(fig, output_path):
    """Save figure as both PNG and PDF."""
    output_path = Path(output_path)
    fig.savefig(output_path.with_suffix('.png'), dpi=200,
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(output_path.with_suffix('.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {output_path.with_suffix('.png')}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description='Rollout-style method comparison')
    parser.add_argument('--methods-dir', type=str, required=True)
    parser.add_argument('--session', type=str, required=True,
                        help='Session stem, e.g. recordings_21_gestures_enrollment02_finger_flex02')
    parser.add_argument('--t-start', type=float, default=0,
                        help='Start time in seconds')
    parser.add_argument('--t-end', type=float, default=None,
                        help='End time in seconds (default: full recording)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-dur', type=int, default=9)
    parser.add_argument('--max-gap', type=int, default=16)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # Load data for all methods
    method_data = {}
    gt_data = None
    num_frames = None
    for method_key, method_label in METHODS:
        json_path = find_raw_predictions(args.methods_dir, method_key, args.session)
        if json_path is None:
            print(f"WARNING: No data for {method_key}, skipping")
            continue
        with open(json_path) as f:
            data = json.load(f)
        method_data[method_key] = data
        if gt_data is None:
            gt_data = data['gt_flex_labels']
            num_frames = data['num_frames']

    if gt_data is None:
        print("ERROR: No data found")
        return

    total_sec = num_frames / FPS
    t_start = args.t_start
    t_end = args.t_end if args.t_end is not None else total_sec

    # Compute binary predictions per method per finger
    method_preds = {}
    for method_key, data in method_data.items():
        preds = {}
        for finger in FINGER_NAMES:
            signal = data['prob_signals'][finger]
            preds[finger] = threshold_and_filter(
                signal, args.threshold, args.min_dur, args.max_gap)
        method_preds[method_key] = preds

    # GT binary
    gt_binary = {}
    for finger in FINGER_NAMES:
        gt_binary[finger] = np.array(gt_data[finger]) >= 0.5

    # --- Plot ---
    n_fingers = len(FINGER_NAMES)
    row_height = 1.0
    row_spacing = 0.3
    total_height = n_fingers * row_height + (n_fingers - 1) * row_spacing

    fig_width = 12
    fig_height = 2.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Y positions for each finger (top to bottom: index, middle, ring, pinky, thumb)
    y_positions = []
    for i in range(n_fingers):
        y = total_height - i * (row_height + row_spacing)
        y_positions.append(y)

    # Draw horizontal grid separating fingers
    for i in range(n_fingers):
        y = y_positions[i]
        ax.axhline(y - row_height / 2 - row_spacing / 2, color='#e0e0e0',
                    linewidth=0.5, zorder=0)

    # Draw bars
    def draw_intervals(intervals, y_center, offset, color, height=BAR_HEIGHT):
        for t0, t1 in intervals:
            if t1 < t_start or t0 > t_end:
                continue
            t0c = max(t0, t_start)
            t1c = min(t1, t_end)
            rect = Rectangle((t0c, y_center + offset - height / 2),
                              t1c - t0c, height,
                              facecolor=color, edgecolor='none', zorder=3)
            ax.add_patch(rect)

    # For each finger, draw GT then methods
    for fi, finger in enumerate(FINGER_NAMES):
        y = y_positions[fi]

        # GT
        gt_intervals = binary_to_intervals(gt_binary[finger], FPS)
        draw_intervals(gt_intervals, y, GT_OFFSET, GT_COLOR, BAR_HEIGHT * 1.1)

        # Methods
        for method_key, method_label in METHODS:
            if method_key not in method_preds:
                continue
            pred_intervals = binary_to_intervals(method_preds[method_key][finger], FPS)
            draw_intervals(pred_intervals, y, METHOD_OFFSETS[method_key],
                           METHOD_COLORS[method_key])

    # Y-axis: finger labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(FINGER_LABELS, fontsize=11)
    ax.set_ylim(y_positions[-1] - row_height, y_positions[0] + row_height / 2 + 0.1)

    # X-axis: time
    ax.set_xlim(t_start, t_end)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)

    # Vertical grid lines (time markers)
    for t in np.arange(np.ceil(t_start / 10) * 10, t_end, 10):
        ax.axvline(t, color='#e0e0e0', linewidth=0.5, zorder=0)

    # Frame
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='y', length=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GT_COLOR, label='GT'),
    ]
    for method_key, method_label in METHODS:
        if method_key in method_preds:
            legend_elements.append(
                Patch(facecolor=METHOD_COLORS[method_key], label=method_label))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor='#cccccc', fancybox=False,
              handlelength=1.2, handleheight=0.8, labelspacing=0.3,
              ncol=4)

    plt.tight_layout()

    output_path = args.output
    if output_path is None:
        output_path = f'draft_papers/multi_finger_ranking_paper/figs/experiments/rollout_{args.session}.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    save_figure(fig, output_path)
    plt.close()


if __name__ == '__main__':
    main()

"""
Four-panel qualitative figure for the paper (figure*).  Each panel shows a
short (3 s) flexion interval for a different test subject:

  (a) S2  — clean single-finger detection (index)
  (b) S7  — SonoRank detects pinky, baselines miss entirely
  (c) S5  — SonoRank partially correct on coupled fingers (middle+ring)
  (d) S12 — thumb detection (hardest finger)

Layout: 2x2 grid.  Each cell has ultrasound frames on top (no gap) then
horizontal bars for GT / SonoRank / MFT+Rest / MFT across five fingers.

Reader's guide: ``PANELS`` identifies the intentionally selected sessions and
time ranges. Frame loading and event rendering then put every method on the
same finger and time coordinates.

Usage:
    python -m evaluation.plots.plot_qualitative_panels \
        --methods-dir outputs/experiments/methods_fixed_labels \
        --h5-dir data/processed_h5/h5_files
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Ellipse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

# --- Panel definitions ---------------------------------------------------
PANELS = [
    {
        'label': '(a)',
        'session': 'recordings_04_gestures_enrollment03_finger_flex04',
        't_start': 7.5, 't_end': 10.5,
        'subject_label': 'S2',
    },
    {
        'label': '(b)',
        'session': 'recordings_19_gestures_enrollment03_finger_flex05',
        't_start': 64.0, 't_end': 67.0,
        'subject_label': 'S7',
    },
    {
        'label': '(c)',
        'session': 'recordings_10_gestures_enrollment03_finger_flex04',
        't_start': 72.0, 't_end': 75.0,
        'subject_label': 'S5',
    },
    {
        'label': '(d)',
        'session': 'recordings_25_gestures_enrollment01_finger_flex01',
        't_start': 81.5, 't_end': 84.5,
        'subject_label': 'S12',
    },
]

METHODS = [
    ('flex_embedding_ranking_backbone', 'SonoRank'),
    ('flex_embedding_no_backbone', 'MFT+Rest'),
    ('flex_embedding_no_rest', 'MFT'),
]

METHOD_COLORS = {
    'flex_embedding_ranking_backbone': '#9b7fc4',
    'flex_embedding_no_backbone': '#5b9bd5',
    'flex_embedding_no_rest': '#6dcfcf',
}
GT_COLOR = '#e07c8a'

FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGER_LABELS = ['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
FPS = 20.0

N_US_FRAMES = 6  # ultrasound frames to show per panel

THRESHOLD = 0.5
MIN_DUR = 9
MAX_GAP = 16


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


def threshold_and_filter(signal, threshold=0.5, min_dur=9, max_gap=16):
    """Apply threshold, merge gaps, remove short events."""
    pred = np.array(signal) >= threshold

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


def binary_to_intervals(binary, fps, t_offset=0.0):
    """Convert binary array to list of (start_sec, end_sec) intervals."""
    intervals = []
    start = None
    for i in range(len(binary)):
        if binary[i] and start is None:
            start = i
        elif not binary[i] and start is not None:
            intervals.append((start / fps + t_offset, i / fps + t_offset))
            start = None
    if start is not None:
        intervals.append((start / fps + t_offset, len(binary) / fps + t_offset))
    return intervals


def load_us_frames(h5_dir, session, frame_indices):
    """Load ultrasound frames from H5 file as numpy arrays."""
    h5_path = Path(h5_dir) / f'{session}.h5'
    df = pd.read_hdf(h5_path, 'df')
    frames = []
    for idx in frame_indices:
        idx = min(idx, len(df) - 1)
        img = df.iloc[idx]['img']
        if isinstance(img, Image.Image):
            arr = np.array(img.convert('L'))
        else:
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
        frames.append(arr)
    return frames


def draw_panel(ax_us, ax_bars, panel_cfg, methods_dir, h5_dir,
               show_legend=False):
    """Draw one panel: US frames on top, bars below."""
    session = panel_cfg['session']
    t_start = panel_cfg['t_start']
    t_end = panel_cfg['t_end']
    frame_start = int(t_start * FPS)
    frame_end = int(t_end * FPS)

    # --- Load predictions for all methods ---
    method_preds = {}
    gt_data = None
    for method_key, _ in METHODS:
        jp = find_raw_predictions(methods_dir, method_key, session)
        if jp is None:
            continue
        with open(jp) as f:
            data = json.load(f)
        preds = {}
        for finger in FINGER_NAMES:
            full_signal = data['prob_signals'][finger]
            full_pred = threshold_and_filter(full_signal, THRESHOLD, MIN_DUR, MAX_GAP)
            preds[finger] = full_pred[frame_start:frame_end]
        method_preds[method_key] = preds
        if gt_data is None:
            gt_raw = data['gt_flex_labels']
            gt_data = {}
            for finger in FINGER_NAMES:
                gt_data[finger] = (np.array(gt_raw[finger])[frame_start:frame_end] >= 0.5)

    # --- Ultrasound frames ---
    us_indices = np.linspace(frame_start, frame_end - 1, N_US_FRAMES, dtype=int)
    us_frames = load_us_frames(h5_dir, session, us_indices)

    # Draw US frames side by side with small gaps
    combined_width = sum(f.shape[1] for f in us_frames)
    gap_px = 2
    total_width = combined_width + gap_px * (N_US_FRAMES - 1)
    canvas_h = us_frames[0].shape[0]
    canvas = np.ones((canvas_h, total_width), dtype=np.uint8) * 255

    x = 0
    frame_centers_frac = []
    for i, frame in enumerate(us_frames):
        w = frame.shape[1]
        canvas[:, x:x + w] = frame
        frame_centers_frac.append((x + w / 2) / total_width)
        x += w + gap_px

    ax_us.imshow(canvas, cmap='gray', aspect='auto')
    ax_us.set_xticks([])
    ax_us.set_yticks([])
    for spine in ax_us.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#999999')

    # Time labels inside the US frames (bottom of each frame)
    us_times = us_indices / FPS
    for i, (cx, t) in enumerate(zip(frame_centers_frac, us_times)):
        ax_us.text(cx, 0.97, f'{t:.1f}s', transform=ax_us.transAxes,
                   ha='center', va='top', fontsize=5, color='white',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                             alpha=0.5, edgecolor='none'))


    # Subject ID in a fixed-size circle in the bottom-left of bar subplot
    # Compute axes aspect ratio so the ellipse appears as a true circle
    fig = ax_bars.get_figure()
    pos = ax_bars.get_position()
    ax_aspect = (pos.width * fig.get_size_inches()[0]) / \
                (pos.height * fig.get_size_inches()[1])
    r = 0.09  # radius in axes y-fraction
    ellipse = Ellipse((0.04, 0.10), width=2 * r / ax_aspect, height=2 * r,
                       transform=ax_bars.transAxes, facecolor='#333333',
                       edgecolor='white', linewidth=0.8, zorder=10,
                       clip_on=False)
    ax_bars.add_patch(ellipse)
    ax_bars.text(0.04, 0.10, panel_cfg['subject_label'],
                 transform=ax_bars.transAxes, ha='center', va='center',
                 fontsize=6, fontweight='bold', color='white',
                 zorder=11)

    # --- Bar chart ---
    # Each finger level = 4 bar heights (GT + 3 methods).
    # ytick at center of each level (2x from bottom of level).
    # Total height = 5 levels * 4x = 20x.
    n_fingers = len(FINGER_NAMES)
    x = 1.0  # unit bar height
    level_h = 4 * x  # each finger occupies 4 bars
    total_h = n_fingers * level_h  # 20x

    # y_positions: center of each level, top finger first
    y_positions = []
    for i in range(n_fingers):
        y = total_h - (i * level_h) - 2 * x  # center of level
        y_positions.append(y)

    def draw_intervals(intervals, y_bottom, color, height=x):
        for t0, t1 in intervals:
            if t1 < t_start or t0 > t_end:
                continue
            t0c = max(t0, t_start)
            t1c = min(t1, t_end)
            rect = Rectangle((t0c, y_bottom), t1c - t0c, height,
                              facecolor=color, edgecolor='none', zorder=3)
            ax_bars.add_patch(rect)

    # 4 bars per finger: GT (top), SonoRank, MFT+Rest, MFT (bottom)
    bar_order = [
        (None, GT_COLOR),  # GT
        ('flex_embedding_ranking_backbone', METHOD_COLORS['flex_embedding_ranking_backbone']),
        ('flex_embedding_no_backbone', METHOD_COLORS['flex_embedding_no_backbone']),
        ('flex_embedding_no_rest', METHOD_COLORS['flex_embedding_no_rest']),
    ]

    for fi, finger in enumerate(FINGER_NAMES):
        level_top = total_h - fi * level_h
        for bi, (method_key, color) in enumerate(bar_order):
            bar_bottom = level_top - (bi + 1) * x
            if method_key is None:
                # GT
                gt_intervals = binary_to_intervals(gt_data[finger], FPS, t_offset=t_start)
                draw_intervals(gt_intervals, bar_bottom, color)
            else:
                if method_key not in method_preds:
                    continue
                pred_intervals = binary_to_intervals(
                    method_preds[method_key][finger], FPS, t_offset=t_start)
                draw_intervals(pred_intervals, bar_bottom, color)

    # Horizontal separators between finger levels
    for i in range(n_fingers - 1):
        sep_y = total_h - (i + 1) * level_h
        ax_bars.axhline(sep_y, color='#e0e0e0', linewidth=0.4, zorder=0)

    # Y-axis
    ax_bars.set_yticks(y_positions)
    ax_bars.set_yticklabels(FINGER_LABELS, fontsize=7)
    ax_bars.set_ylim(0, total_h)
    ax_bars.tick_params(axis='y', length=0, pad=2)

    # X-axis (hidden — timestamps are shown on US frames)
    ax_bars.set_xlim(t_start, t_end)
    ax_bars.set_xticks([])
    ax_bars.set_xlabel('')

    # Vertical grid lines
    tick_interval = 0.5
    grid_ticks = np.arange(np.ceil(t_start * 2) / 2, t_end + 0.01, tick_interval)
    for t in grid_ticks:
        ax_bars.axvline(t, color='#e8e8e8', linewidth=0.3, zorder=0)
    ax_bars.set_axisbelow(True)

    for spine in ax_bars.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#999999')

    # Legend inside this panel
    if show_legend:
        legend_elements = [
            Patch(facecolor=GT_COLOR, label='GT'),
            Patch(facecolor=METHOD_COLORS['flex_embedding_ranking_backbone'],
                  label='SonoRank'),
            Patch(facecolor=METHOD_COLORS['flex_embedding_no_backbone'],
                  label='MFT+Rest'),
            Patch(facecolor=METHOD_COLORS['flex_embedding_no_rest'],
                  label='MFT'),
        ]
        ax_bars.legend(handles=legend_elements, loc='lower right',
                       fontsize=6, framealpha=0.92, edgecolor='#cccccc',
                       fancybox=False, handlelength=1.0, handleheight=0.7,
                       labelspacing=0.25, borderpad=0.4)


def save_figure(fig, output_path):
    """Save figure as both PNG and PDF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.png'), dpi=300,
                bbox_inches='tight', pad_inches=0.025)
    fig.savefig(output_path.with_suffix('.pdf'),
                bbox_inches='tight', pad_inches=0.025)
    print(f"Saved: {output_path.with_suffix('.png')}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(
        description='Four-panel qualitative rollout figure (2x2)')
    parser.add_argument('--methods-dir', type=str, required=True)
    parser.add_argument('--h5-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # 2x2 layout, each cell = (US frames | bars) stacked vertically
    fig = plt.figure(figsize=(7.16, 3.8))  # IEEE two-column width

    # Outer grid: 2 rows x 2 cols of panels
    outer_gs = GridSpec(2, 2, figure=fig, hspace=0.05, wspace=0.12)

    # Panel grid positions: (a) top-left, (b) top-right, (c) bot-left, (d) bot-right
    grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Put legend on panel (d) — bottom-right
    legend_panel_idx = 3

    for pi, (panel_cfg, (row, col)) in enumerate(zip(PANELS, grid_positions)):
        # Inner grid for this panel: US row + bars row, no gap
        inner_gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row, col],
            height_ratios=[0.7, 1.5], hspace=0.0)

        ax_us = fig.add_subplot(inner_gs[0])
        ax_bars = fig.add_subplot(inner_gs[1])

        draw_panel(ax_us, ax_bars, panel_cfg, args.methods_dir, args.h5_dir,
                   show_legend=(pi == legend_panel_idx))

    output_path = args.output
    if output_path is None:
        output_path = 'draft_papers/multi_finger_ranking_paper/figs/experiments/qualitative_panels.png'

    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.02)
    save_figure(fig, output_path)
    plt.close()


if __name__ == '__main__':
    main()

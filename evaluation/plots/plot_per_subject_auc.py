"""
Plot per-subject, per-finger AUC as a dot/strip chart.

Reads test_metrics.json from each experiment folder to get per-finger AUC
values for all 12 test subjects.

Usage:
    python -m evaluation.plots.plot_per_subject_auc
    python -m evaluation.plots.plot_per_subject_auc --seq-len 16 --stride 2

Reader's guide: ``collect_per_subject_aucs`` maps one test-fold directory to
one subject, while ``plot_per_subject_auc`` shows finger-level values and their
subject mean. Paper subject labels are presentation aliases only.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER_FIGS_DIR = Path('draft_papers/multi_finger_ranking_paper/figs/experiments')

FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGER_LABELS = {
    'thumb': '$F_1$', 'index': '$F_2$', 'middle': '$F_3$',
    'ring': '$F_4$', 'pinky': '$F_5$',
}
ALL_SUBJECTS = [1, 4, 5, 6, 10, 17, 19, 20, 21, 23, 24, 25]

# Anonymized mapping: original ID -> paper label (1-12)
SUBJECT_LABEL = {s: i + 1 for i, s in enumerate(ALL_SUBJECTS)}

FINGER_COLORS = {
    'index': '#1f77b4',
    'middle': '#ff7f0e',
    'ring': '#2ca02c',
    'pinky': '#d62728',
    'thumb': '#9467bd',
}


def save_figure(fig, output_path, paper_name=None):
    """Save figure as both PNG and PDF with minimal whitespace.

    If paper_name is given, also copy PNG and PDF to the paper figs directory.
    """
    output_path = Path(output_path)
    fig.savefig(output_path.with_suffix('.png'), dpi=200,
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(output_path.with_suffix('.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {output_path.with_suffix('.png')}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    if paper_name and PAPER_FIGS_DIR.exists():
        paper_path = PAPER_FIGS_DIR / paper_name
        fig.savefig(paper_path.with_suffix('.png'), dpi=200,
                    bbox_inches='tight', pad_inches=0.05)
        fig.savefig(paper_path.with_suffix('.pdf'),
                    bbox_inches='tight', pad_inches=0.05)
        print(f"Saved: {paper_path.with_suffix('.png')}")
        print(f"Saved: {paper_path.with_suffix('.pdf')}")


def collect_per_subject_aucs(sweep_dir, seq_len, stride):
    """Collect per-finger AUCs from test_metrics.json for all test subjects."""
    results = {}

    for subject in ALL_SUBJECTS:
        pattern = f"seqlen_sweep_seq{seq_len}_stride{stride}_test{subject}_val*"
        folders = sorted(sweep_dir.glob(pattern))
        if not folders:
            print(f"  Warning: no folder for subject {subject}")
            continue

        folder = folders[0]
        metrics_path = folder / 'test_metrics.json'
        if not metrics_path.exists():
            print(f"  Warning: no test_metrics.json for subject {subject}")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        finger_aucs = {}
        for finger in FINGER_NAMES:
            key = f'auc_{finger}'
            if key in metrics:
                finger_aucs[finger] = metrics[key]

        results[subject] = finger_aucs
        print(f"  Subject {subject} (P{SUBJECT_LABEL[subject]}): "
              f"{len(finger_aucs)} fingers, "
              f"mean AUC = {metrics.get('mean_auc', 0):.3f}")

    return results


def plot_per_subject_auc(results, output_path, paper_name=None):
    """Dot/strip chart: per-finger AUC per subject with range lines."""
    subjects = [s for s in ALL_SUBJECTS if s in results]
    n_subjects = len(subjects)

    fig, ax = plt.subplots(figsize=(10, 5))

    # x positions: 12 subjects + 1 mean column
    x_positions = np.arange(n_subjects)
    mean_x = n_subjects + 0.5  # position for the "Mean" column (extra gap)

    # Compute grand mean per finger (across all subjects)
    grand_finger_aucs = {}
    for finger in FINGER_NAMES:
        vals = [results[s][finger] for s in subjects if finger in results[s]]
        if vals:
            grand_finger_aucs[finger] = np.mean(vals)

    # Draw vertical range lines (min to max finger AUC) per subject
    for xi, subj in enumerate(subjects):
        vals = [results[subj][f] for f in FINGER_NAMES if f in results[subj]]
        if len(vals) >= 2:
            ax.plot([x_positions[xi], x_positions[xi]],
                    [min(vals), max(vals)],
                    color='#cccccc', linewidth=2.5, zorder=1)

    # Scale factor for the Mean column: expand finger spread so closely-packed
    # dots become readable. Center the cluster at the visual midpoint of ylim.
    # Labels still show the original true values.
    grand_center = np.mean(list(grand_finger_aucs.values()))
    mean_scale = 9.0

    all_auc_vals = [v for subj in subjects for v in results[subj].values()]
    ylim_bottom = 0.35
    ylim_top_est = max(all_auc_vals) + 0.03
    ylim_center = (ylim_bottom + ylim_top_est) / 2

    def scale_mean(val):
        return ylim_center + (val - grand_center) * mean_scale

    # Range line for grand mean column (scaled)
    grand_vals = list(grand_finger_aucs.values())
    if len(grand_vals) >= 2:
        ax.plot([mean_x, mean_x],
                [scale_mean(min(grand_vals)), scale_mean(max(grand_vals))],
                color='#cccccc', linewidth=2.5, zorder=1)

    # Plot per-finger dots (all at same x, no jitter)
    for fi, finger in enumerate(FINGER_NAMES):
        xs = []
        ys = []

        for xi, subj in enumerate(subjects):
            if finger in results[subj]:
                xs.append(x_positions[xi])
                ys.append(results[subj][finger])

        ax.plot(xs, ys, 'o', color=FINGER_COLORS[finger], alpha=0.6,
                markersize=10, label=FINGER_LABELS[finger], zorder=3)

        # Add grand mean dot for this finger at SCALED position
        if finger in grand_finger_aucs:
            ax.plot(mean_x, scale_mean(grand_finger_aucs[finger]), 'o',
                    color=FINGER_COLORS[finger], alpha=0.6,
                    markersize=10, zorder=3)

    # Plot mean AUC (across fingers) as black squares with std error bars
    mean_vals = []
    std_vals = []
    for xi, subj in enumerate(subjects):
        finger_vals = list(results[subj].values())
        if finger_vals:
            mean_vals.append(np.mean(finger_vals))
            std_vals.append(np.std(finger_vals))
        else:
            mean_vals.append(np.nan)
            std_vals.append(np.nan)

    # Grand mean of means
    grand_mean = np.nanmean(mean_vals)
    grand_std = np.nanstd(mean_vals)

    # Per-subject black squares with std error bars
    ax.errorbar(x_positions, mean_vals, yerr=std_vals, fmt='s',
                capsize=5, color='#333333', ecolor='#999999',
                markersize=14, zorder=10, label='Mean',
                markeredgecolor='white', markeredgewidth=1.2)
    # Mean column black square at scaled position with error bar
    ax.errorbar([mean_x], [scale_mean(grand_mean)],
                yerr=[grand_std * mean_scale], fmt='s',
                capsize=5, color='#333333', ecolor='#999999',
                markersize=14, zorder=10,
                markeredgecolor='white', markeredgewidth=1.2)

    # Annotate per-subject mean values
    for xi, mv in zip(x_positions, mean_vals):
        if not np.isnan(mv):
            ax.annotate(f'{mv:.2f}', (xi, mv),
                        textcoords='offset points', xytext=(0, 12),
                        ha='center', va='bottom', fontsize=13,
                        color='#333333', zorder=20,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.5))
    # Annotate Mean column square with true value to the left
    ax.annotate(f'{grand_mean:.2f}', (mean_x, scale_mean(grand_mean)),
                textcoords='offset points', xytext=(-8, 0),
                ha='right', va='center', fontsize=13, color='#333333',
                zorder=20,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.5))

    # Annotate per-finger values for the Mean column only
    # Dots are at scaled positions; labels show the original true values
    for finger, val in grand_finger_aucs.items():
        ax.annotate(f'{val:.2f}', (mean_x, scale_mean(val)),
                    textcoords='offset points', xytext=(10, 0),
                    ha='left', va='center', fontsize=13,
                    color=FINGER_COLORS[finger])

    # Separator line before Mean column
    ax.axvline(x=n_subjects - 0.5, color='#999999', linewidth=2.5,
               linestyle='--', zorder=0)

    ax.set_xlabel('Test Subject', fontsize=26)

    all_ticks = np.append(x_positions, mean_x)
    all_labels = [f'S{SUBJECT_LABEL[s]}' for s in subjects] + ['Mean']
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels)
    ax.tick_params(labelsize=22)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)

    # Build legend outside the plot to the right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=20, ncol=1,
              loc='center left', bbox_to_anchor=(1.0, 0.5),
              framealpha=0.9, handlelength=1.2)

    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(-0.6, mean_x + 1.0)
    ax.set_ylim(0.35, 0.85)

    plt.tight_layout()
    save_figure(fig, output_path, paper_name=paper_name)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot per-subject, per-finger AUC from test metrics')
    parser.add_argument('--sweep-dir', type=str,
                        default='outputs/experiments/seqlen_sweep',
                        help='Root sweep directory')
    parser.add_argument('--stride', type=int, default=2,
                        help='Stride subfolder (default: 2)')
    parser.add_argument('--seq-len', type=int, default=16,
                        help='Sequence length to plot (default: 16)')
    parser.add_argument('--output-dir', type=str,
                        default='evaluation/plots/per_subject_auc',
                        help='Output directory for plots')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir) / f'stride{args.stride}'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {sweep_dir}")
    print(f"Sequence length: {args.seq_len}, stride: {args.stride}")
    print()

    results = collect_per_subject_aucs(sweep_dir, args.seq_len, args.stride)
    if not results:
        print("No results found!")
        return

    suffix = f"seq{args.seq_len}_stride{args.stride}"

    plot_per_subject_auc(
        results,
        output_dir / f'per_subject_auc_{suffix}.png',
        paper_name=f'per_subject_auc_{suffix}',
    )

    # Print summary table
    print(f"\n{'Subject':>8} ", end='')
    for finger in FINGER_NAMES:
        print(f"{'  ' + finger:>10}", end='')
    print(f"{'  mean':>10}")
    print('-' * 72)

    all_finger_vals = {f: [] for f in FINGER_NAMES}
    all_means = []

    for subj in ALL_SUBJECTS:
        if subj not in results:
            continue
        label = f'S{SUBJECT_LABEL[subj]}'
        print(f"{label:>8} ", end='')
        finger_avgs = []
        for finger in FINGER_NAMES:
            if finger in results[subj]:
                v = results[subj][finger]
                print(f"  {v:.3f}  ", end='')
                finger_avgs.append(v)
                all_finger_vals[finger].append(v)
            else:
                print(f"{'N/A':>10}", end='')
        if finger_avgs:
            overall = np.mean(finger_avgs)
            print(f"  {overall:.3f}")
            all_means.append(overall)
        else:
            print()

    # Print grand averages
    print('-' * 72)
    print(f"{'Mean':>8} ", end='')
    for finger in FINGER_NAMES:
        vals = all_finger_vals[finger]
        if vals:
            print(f"  {np.mean(vals):.3f}  ", end='')
    if all_means:
        print(f"  {np.mean(all_means):.3f}")

    print(f"\n{'Std':>8} ", end='')
    for finger in FINGER_NAMES:
        vals = all_finger_vals[finger]
        if vals:
            print(f"  {np.std(vals):.3f}  ", end='')
    if all_means:
        print(f"  {np.std(all_means):.3f}")

    # Print subject mapping
    print("\n\nSubject ID mapping (original -> paper):")
    for s in ALL_SUBJECTS:
        print(f"  S{s} -> P{SUBJECT_LABEL[s]}")


if __name__ == '__main__':
    main()

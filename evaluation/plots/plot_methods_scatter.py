"""
Grouped bar chart comparing 8 methods across 3 metrics (Accuracy, Precision,
Recall).  Single axes, metric groups on x-axis, colors = methods, legend on
the side.  Matches the visual style of the journal paper's cigraph_bars figure.

Our 3 methods use post-processed results (min_dur=9, gap=16); the 5 baselines
use raw results (min_dur=0, gap=0).

Usage:
    python -m evaluation.plots.plot_methods_scatter \
        --ours evaluation/plots/methods_fair_gtfixed_raw_cov0.3_mindur9_gap16_*.json \
        --baselines evaluation/plots/methods_fair_gtfixed_raw_cov0.3_mindur0_gap0_*.json

Reader's guide: numerical values come from saved fair-evaluation files. This
module only selects method order, colors, labels, and output format for the
aggregate comparison figure.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Order within each metric group (left to right):
# SonoRank, MFT+Rest, MFT | CBMF, Bimbraw, McIntosh, Huang, Castellini
OUR_METHODS = [
    'flex_embedding_ranking_backbone',
    'flex_embedding_no_backbone',
    'flex_embedding_no_rest',
]
BASELINE_METHODS = [
    'bimbraw',
    'mcintosh',
    'huang',
    'castellini',
]
METHOD_ORDER = OUR_METHODS + BASELINE_METHODS

METHOD_LABELS = {
    'flex_embedding_ranking_backbone': 'SonoRank',
    'flex_embedding_no_backbone': 'MFT+Rest',
    'flex_embedding_no_rest': 'MFT',
    'cbmf': 'CBMF [2024]',
    'bimbraw': 'Bimbraw et al. [2020]',
    'mcintosh': 'McIntosh et al. [2017]',
    'huang': 'Huang et al. [2016]',
    'castellini': 'Castellini et al. [2011]',
}

# Warm colors for baselines (oldest=red -> newest=green),
# cool colors for ours (cyan -> blue -> purple)
METHOD_COLORS = {
    'castellini': '#e07b7b',       # salmon / red
    'huang': '#f0a04b',            # orange
    'mcintosh': '#f0d264',         # gold / yellow
    'bimbraw': '#8fca8f',          # light green
    'cbmf': '#a8d8a8',            # soft green (between bimbraw and our methods)
    'flex_embedding_no_rest': '#6dcfcf',      # cyan
    'flex_embedding_no_backbone': '#5b9bd5',  # blue
    'flex_embedding_ranking_backbone': '#9b7fc4',  # purple
}

METRICS = ['accuracy', 'recall', 'precision', 'f1']
METRIC_LABELS = {'accuracy': 'Accuracy', 'recall': 'Recall', 'precision': 'Precision', 'f1': 'F1'}


def darken_color(hex_color, factor=0.6):
    """Return a darker version of the given hex color."""
    rgb = mcolors.to_rgb(hex_color)
    dark = tuple(c * factor for c in rgb)
    return mcolors.to_hex(dark)


def extract_fold_metrics(methods_data, method_key):
    """Extract per-fold macro metrics for a method."""
    mdata = methods_data[method_key]
    fold_values = {m: [] for m in METRICS}
    for fold in mdata['folds']:
        macro = fold['overall']['macro']
        for m in METRICS:
            fold_values[m].append(macro[m])
    return fold_values


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
    parser = argparse.ArgumentParser(
        description='Grouped bar chart of macro metrics across methods')
    parser.add_argument('--ours', type=str, required=True,
                        help='JSON with our methods (min_dur=9, gap=16)')
    parser.add_argument('--baselines', type=str, required=True,
                        help='JSON with baselines (min_dur=0, gap=0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: auto-generated)')
    args = parser.parse_args()

    with open(args.ours) as f:
        ours_data = json.load(f)['methods']
    with open(args.baselines) as f:
        baselines_data = json.load(f)['methods']

    # Collect per-fold metrics: our methods from ours_data, baselines from baselines_data
    method_metrics = {}
    for mk in OUR_METHODS:
        if mk in ours_data:
            method_metrics[mk] = extract_fold_metrics(ours_data, mk)
    for mk in BASELINE_METHODS:
        if mk in baselines_data:
            method_metrics[mk] = extract_fold_metrics(baselines_data, mk)

    present_methods = [m for m in METHOD_ORDER if m in method_metrics]
    n_methods = len(present_methods)
    n_metrics = len(METRICS)

    # --- Layout ---
    bar_width = 0.11
    group_gap = 0.25
    group_width = n_methods * bar_width
    group_centers = []
    for gi in range(n_metrics):
        center = gi * (group_width + group_gap)
        group_centers.append(center)

    ylim_lo, ylim_hi = 0.1, 0.95

    fig, ax = plt.subplots(figsize=(18, 3.5))

    for mi, mk in enumerate(present_methods):
        color = METHOD_COLORS[mk]
        ecolor = darken_color(color, 0.55)
        bar_positions = []
        means = []
        stds = []
        for gi, metric in enumerate(METRICS):
            vals = np.array(method_metrics[mk][metric])
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            x = group_centers[gi] - group_width / 2 + (mi + 0.5) * bar_width
            bar_positions.append(x)
            means.append(mean_val)
            stds.append(std_val)

        # Clip error bars to ylim so they don't extend beyond the frame
        clipped_stds = [max(0, min(s, ylim_hi - m)) for m, s in zip(means, stds)]
        clipped_stds_lo = [max(0, min(s, m - ylim_lo)) for m, s in zip(means, stds)]

        ax.bar(bar_positions, means, bar_width * 0.88,
               yerr=[clipped_stds_lo, clipped_stds],
               color=color, capsize=2,
               error_kw={'linewidth': 1.5, 'capthick': 1.5, 'ecolor': ecolor},
               zorder=3, edgecolor='white', linewidth=0.4,
               label=METHOD_LABELS[mk])

        # Value annotations just above the mean bar with light background
        for x, m in zip(bar_positions, means):
            ax.text(x, m + 0.012, f'{m:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=ecolor,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='none', alpha=0.8),
                    zorder=5)

    # X-axis: metric group labels, no tick marks
    ax.set_xticks(group_centers)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=18)
    ax.tick_params(axis='x', length=0)

    # Y-axis
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_yticks(np.arange(0.1, 0.96, 0.1))
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('')

    # Grid
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.set_axisbelow(True)

    # Tight x-limits around the bar groups
    ax.set_xlim(group_centers[0] - group_width / 2 - 0.05,
                group_centers[-1] + group_width / 2 + 0.12)

    # Keep all four frame edges visible
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Legend on the right side
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=12,
              framealpha=0.9, edgecolor='#cccccc', fancybox=False,
              handlelength=1.2, handleheight=0.8, labelspacing=0.4)

    plt.tight_layout()

    output_path = args.output
    if output_path is None:
        output_path = 'draft_papers/multi_finger_ranking_paper/figs/experiments/methods_scatter.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    save_figure(fig, output_path)
    plt.close()


if __name__ == '__main__':
    main()

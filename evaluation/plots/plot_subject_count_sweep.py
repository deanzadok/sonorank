"""
Plot mean test AUC vs number of training subjects for the subject count sweep.

Reads test_metrics.json from each experiment folder, aggregates across all
available test subjects per count, and plots mean +/- std with per-finger lines.

The n=10 baseline comes from the seqlen_sweep (seq16 stride2) experiments,
which used the full 10-subject training set.

Usage:
    python -m evaluation.plots.plot_subject_count_sweep
    python -m evaluation.plots.plot_subject_count_sweep --subject-counts 1 3 5 7 9

Reader's guide: counts one through nine come from the dedicated nested sweep,
while the ten-subject point comes from the full leave-one-subject-out baseline.
This module combines those saved metrics into one curve.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGER_LABELS = {
    'thumb': '$F_1$', 'index': '$F_2$', 'middle': '$F_3$',
    'ring': '$F_4$', 'pinky': '$F_5$',
}

PAPER_FIGS_DIR = Path('draft_papers/multi_finger_ranking_paper/figs/experiments')


def save_figure(fig, output_path, paper_name=None):
    """Save figure as both PNG and PDF with minimal whitespace."""
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


def collect_metrics(sweep_dir, subject_counts):
    """Collect test metrics for each subject count."""
    results = {}
    for n in subject_counts:
        pattern = f"subj_count_n{n}_test*"
        folders = sorted(sweep_dir.glob(pattern))
        if not folders:
            print(f"  Warning: no folders found for n={n}")
            continue

        metrics_list = []
        for folder in folders:
            metrics_path = folder / 'test_metrics.json'
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            metrics_list.append(metrics)

        if metrics_list:
            results[n] = metrics_list
            print(f"  n={n}: {len(metrics_list)} subjects")

    return results


def collect_baseline_n10(seqlen_dir):
    """Collect n=10 baseline from seqlen sweep (seq16 stride2)."""
    pattern = "seqlen_sweep_seq16_stride2_test*"
    folders = sorted(seqlen_dir.glob(pattern))
    if not folders:
        print("  Warning: no seqlen sweep seq16 stride2 folders found for n=10 baseline")
        return None

    metrics_list = []
    for folder in folders:
        metrics_path = folder / 'test_metrics.json'
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics_list.append(metrics)

    if metrics_list:
        print(f"  n=10 (seqlen baseline): {len(metrics_list)} subjects")
        return metrics_list
    return None


def plot_auc_vs_subject_count(results, subject_counts, output_path,
                              paper_name=None):
    """Plot per-finger and mean AUC vs number of training subjects."""
    fig, ax = plt.subplots(figsize=(4.5, 5))

    colors = {'index': '#1f77b4', 'middle': '#ff7f0e', 'ring': '#2ca02c',
              'pinky': '#d62728', 'thumb': '#9467bd', 'mean': '#333333'}

    # Evenly-spaced positions with count labels
    valid_ns = [n for n in subject_counts if n in results]
    positions = np.arange(len(valid_ns))
    n_to_pos = {n: pos for n, pos in zip(valid_ns, positions)}

    # Per-finger AUCs (lighter, thinner)
    for finger in FINGER_NAMES:
        key = f'auc_{finger}'
        x_list, m_list = [], []
        for n in valid_ns:
            vals = [m[key] for m in results[n]]
            x_list.append(n_to_pos[n])
            m_list.append(np.mean(vals))
        ax.plot(x_list, m_list, 'o--', color=colors[finger], alpha=0.5,
                linewidth=2, markersize=8, label=FINGER_LABELS[finger])

    # Mean AUC (bold, with error bars)
    x_list, m_list, s_list = [], [], []
    for n in valid_ns:
        vals = [m['mean_auc'] for m in results[n]]
        x_list.append(n_to_pos[n])
        m_list.append(np.mean(vals))
        s_list.append(np.std(vals))
    ax.errorbar(x_list, m_list, yerr=s_list, fmt='s-', capsize=7,
                linewidth=3.5, markersize=14, color=colors['mean'],
                ecolor='#999999', label='Mean', zorder=10)

    # Annotate mean AUC values
    for x, m, s in zip(x_list, m_list, s_list):
        ax.annotate(f'{m:.2f}',
                    (x, m), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=12.5, color='#333333', zorder=20,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.5))

    ax.set_xlabel('Training Subjects', fontsize=24)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in valid_ns])
    ax.tick_params(labelsize=20)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax.set_ylim(0.35, 0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, output_path, paper_name=paper_name)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot subject count sweep results (AUC vs training set size)')
    parser.add_argument('--sweep-dir', type=str,
                        default='outputs/experiments/subject_count_sweep',
                        help='Root sweep directory')
    parser.add_argument('--seqlen-dir', type=str,
                        default='outputs/experiments/seqlen_sweep/stride2',
                        help='Seqlen sweep directory for n=10 baseline')
    parser.add_argument('--subject-counts', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='Subject counts to include (default: 1..10)')
    parser.add_argument('--output-dir', type=str,
                        default='evaluation/plots/subject_count_sweep',
                        help='Output directory for plots')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    seqlen_dir = Path(args.seqlen_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {sweep_dir}")
    print(f"Subject counts: {args.subject_counts}")

    # Collect n=1..9 from subject count sweep
    counts_to_collect = [n for n in args.subject_counts if n < 10]
    results = collect_metrics(sweep_dir, counts_to_collect)

    # Collect n=10 baseline from seqlen sweep
    if 10 in args.subject_counts:
        baseline = collect_baseline_n10(seqlen_dir)
        if baseline:
            results[10] = baseline

    if not results:
        print("No results found!")
        return

    # Plot AUC vs subject count
    plot_auc_vs_subject_count(
        results, args.subject_counts,
        output_dir / 'subject_count_sweep_auc.png',
        paper_name='subject_count_sweep_auc',
    )

    # Print summary table
    print(f"\n{'N':>4} {'Subj':>6} {'Loss':>12} {'Mean AUC':>12}")
    print('-' * 38)
    for n in args.subject_counts:
        if n not in results:
            continue
        losses = [m['loss'] for m in results[n]]
        aucs = [m['mean_auc'] for m in results[n]]
        print(f"{n:>4} {len(losses):>6} "
              f"{np.mean(losses):>6.4f}\u00b1{np.std(losses):.4f} "
              f"{np.mean(aucs):>6.4f}\u00b1{np.std(aucs):.4f}")


if __name__ == '__main__':
    main()

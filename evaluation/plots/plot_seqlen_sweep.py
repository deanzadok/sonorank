"""
Plot mean test loss and AUC vs sequence length for the seqlen sweep experiment.

Reads test_metrics.json from each experiment folder, aggregates across all 12
test subjects per sequence length, and plots mean +/- std.

Usage:
    python -m evaluation.plots.plot_seqlen_sweep
    python -m evaluation.plots.plot_seqlen_sweep --stride 3
    python -m evaluation.plots.plot_seqlen_sweep --metric mean_auc

Reader's guide: collection functions read one saved experiment directory per
window length. Plotting functions render loss, ranking AUC, or event metrics
without recomputing checkpoints.
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


def collect_metrics(sweep_dir, seq_lengths):
    """Collect test metrics for each sequence length."""
    results = {}
    for seq_len in seq_lengths:
        pattern = f"seqlen_sweep_seq{seq_len}_stride*"
        folders = sorted(sweep_dir.glob(pattern))
        if not folders:
            print(f"  Warning: no folders found for seq_len={seq_len}")
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
            results[seq_len] = metrics_list
            print(f"  seq_len={seq_len}: {len(metrics_list)} subjects")

    return results


def plot_loss_vs_seqlen(results, seq_lengths, output_path):
    """Plot test loss vs sequence length with error bars."""
    seq_lens = []
    means = []
    stds = []

    for sl in seq_lengths:
        if sl not in results:
            continue
        losses = [m['loss'] for m in results[sl]]
        seq_lens.append(sl)
        means.append(np.mean(losses))
        stds.append(np.std(losses))

    seq_lens = np.array(seq_lens)
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(seq_lens, means, yerr=stds, fmt='o-', capsize=5,
                linewidth=2, markersize=8, color='#2196F3', ecolor='#90CAF9')

    # Annotate values
    for x, m, s in zip(seq_lens, means, stds):
        ax.annotate(f'{m:.4f}\n(\u00b1{s:.4f})',
                    (x, m), textcoords='offset points', xytext=(0, 14),
                    ha='center', fontsize=8, color='#333333')

    ax.set_xlabel('Sequence Length (T)', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Multi-Finger Ranking: Test Loss vs Sequence Length', fontsize=13)
    ax.set_xticks(seq_lens)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, output_path)
    plt.close()


def plot_auc_vs_seqlen(results, seq_lengths, output_path,
                       stride=2, sample_rate=20, paper_name=None):
    """Plot per-finger and mean AUC vs temporal window duration."""
    fig, ax = plt.subplots(figsize=(5.5, 5))

    colors = {'index': '#1f77b4', 'middle': '#ff7f0e', 'ring': '#2ca02c',
              'pinky': '#d62728', 'thumb': '#9467bd', 'mean': '#333333'}

    def sl_to_sec(sl):
        return ((sl - 1) * stride + 1) / sample_rate

    # Evenly-spaced positions with time labels
    valid_sls = [sl for sl in seq_lengths if sl in results]
    positions = np.arange(len(valid_sls))
    sl_to_pos = {sl: pos for sl, pos in zip(valid_sls, positions)}
    time_labels = [f'{sl_to_sec(sl):.2f}' for sl in valid_sls]

    # Per-finger AUCs (lighter, thinner)
    for finger in FINGER_NAMES:
        key = f'auc_{finger}'
        x_list, m_list = [], []
        for sl in valid_sls:
            vals = [m[key] for m in results[sl]]
            x_list.append(sl_to_pos[sl])
            m_list.append(np.mean(vals))
        ax.plot(x_list, m_list, 'o--', color=colors[finger], alpha=0.5,
                linewidth=2, markersize=8, label=FINGER_LABELS[finger])

    # Mean AUC (bold, with error bars)
    x_list, m_list, s_list = [], [], []
    for sl in valid_sls:
        vals = [m['mean_auc'] for m in results[sl]]
        x_list.append(sl_to_pos[sl])
        m_list.append(np.mean(vals))
        s_list.append(np.std(vals))
    ax.errorbar(x_list, m_list, yerr=s_list, fmt='s-', capsize=7,
                linewidth=3.5, markersize=14, color=colors['mean'],
                ecolor='#999999', label='Mean', zorder=10)

    # Annotate mean AUC values (no std, in front of grid)
    for x, m, s in zip(x_list, m_list, s_list):
        ax.annotate(f'{m:.2f}',
                    (x, m), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=14, color='#333333', zorder=20,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.5))

    ax.set_xlabel('Temporal Window (s)', fontsize=24)
    ax.set_ylabel('Test AUC', fontsize=24)

    ax.set_xticks(positions)
    ax.set_xticklabels(time_labels)
    ax.tick_params(labelsize=20)
    ax.set_xlim(positions[0] - 0.25, positions[-1] + 0.4)
    ax.set_ylim(0.35, 0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, output_path, paper_name=paper_name)
    plt.close()


def collect_event_metrics(sweep_dir, seq_lengths, filename='metrics.json'):
    """Collect event detection metrics for each sequence length."""
    results = {}
    for seq_len in seq_lengths:
        pattern = f"seqlen_sweep_seq{seq_len}_stride*"
        folders = sorted(sweep_dir.glob(pattern))
        if not folders:
            continue

        metrics_list = []
        for folder in folders:
            metrics_path = folder / 'event_detection' / filename
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            metrics_list.append(metrics)

        if metrics_list:
            results[seq_len] = metrics_list
            print(f"  seq_len={seq_len}: {len(metrics_list)} subjects ({filename})")

    return results


def plot_event_metric_vs_seqlen(event_results, seq_lengths, metric_name,
                                output_path, agg='micro'):
    """Plot an event detection metric vs sequence length with per-finger lines."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    colors = {'index': '#1f77b4', 'middle': '#ff7f0e', 'ring': '#2ca02c',
              'pinky': '#d62728', 'thumb': '#9467bd', 'mean': '#333333'}

    pretty = {'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall',
              'accuracy': 'Accuracy'}
    label = pretty.get(metric_name, metric_name)

    # Per-finger lines
    for finger in FINGER_NAMES:
        sl_list, m_list = [], []
        for sl in seq_lengths:
            if sl not in event_results:
                continue
            vals = []
            for m in event_results[sl]:
                pf = m['overall']['per_finger'].get(finger)
                if pf and metric_name in pf:
                    vals.append(pf[metric_name])
            if vals:
                sl_list.append(sl)
                m_list.append(np.mean(vals))
        if sl_list:
            ax.plot(sl_list, m_list, 'o--', color=colors[finger], alpha=0.5,
                    linewidth=1, markersize=5, label=FINGER_LABELS[finger])

    # Micro-averaged (bold, with error bars)
    sl_list, m_list, s_list = [], [], []
    for sl in seq_lengths:
        if sl not in event_results:
            continue
        vals = [m['overall'][agg][metric_name] for m in event_results[sl]]
        sl_list.append(sl)
        m_list.append(np.mean(vals))
        s_list.append(np.std(vals))
    ax.errorbar(sl_list, m_list, yerr=s_list, fmt='s-', capsize=5,
                linewidth=2.5, markersize=9, color=colors['mean'],
                ecolor='#999999', label=f'{agg}', zorder=10)

    for x, m, s in zip(sl_list, m_list, s_list):
        ax.annotate(f'{m:.3f}\n(\u00b1{s:.3f})',
                    (x, m), textcoords='offset points', xytext=(0, 14),
                    ha='center', fontsize=8, color='#333333')

    ax.set_xlabel('Sequence Length (T)', fontsize=12)
    ax.set_ylabel(f'Event Detection {label}', fontsize=12)
    ax.set_title(f'Event Detection: {label} vs Sequence Length', fontsize=13)
    ax.set_xticks(np.array(sl_list))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=3, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot seqlen sweep results (test loss and AUC vs sequence length)')
    parser.add_argument('--sweep-dir', type=str,
                        default='outputs/experiments/seqlen_sweep',
                        help='Root sweep directory')
    parser.add_argument('--stride', type=int, default=2,
                        help='Which stride subfolder to use (default: 2)')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[2, 4, 8, 16, 24],
                        help='Sequence lengths to include (default: 2 4 8 16 24)')
    parser.add_argument('--output-dir', type=str, default='evaluation/plots/seqlen_sweep',
                        help='Output directory for plots')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir) / f'stride{args.stride}'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {sweep_dir}")
    print(f"Sequence lengths: {args.seq_lengths}")

    results = collect_metrics(sweep_dir, args.seq_lengths)
    if not results:
        print("No results found!")
        return

    suffix = f"stride{args.stride}"

    # Plot test loss
    plot_loss_vs_seqlen(
        results, args.seq_lengths,
        output_dir / f'seqlen_sweep_loss_{suffix}.png',
    )

    # Plot AUC
    plot_auc_vs_seqlen(
        results, args.seq_lengths,
        output_dir / f'seqlen_sweep_auc_{suffix}.png',
        stride=args.stride,
        paper_name=f'seqlen_sweep_auc_{suffix}',
    )

    # Event detection metrics (original grid-search per checkpoint)
    print("\nCollecting event detection metrics (original)...")
    event_results = collect_event_metrics(sweep_dir, args.seq_lengths)
    if event_results:
        for metric in ['f1', 'precision', 'recall', 'accuracy']:
            plot_event_metric_vs_seqlen(
                event_results, args.seq_lengths, metric,
                output_dir / f'seqlen_sweep_event_{metric}_{suffix}.png',
            )

    # Event detection metrics (fixed thresholds, overlap matching)
    print("\nCollecting event detection metrics (fixed thresh, overlap)...")
    event_fixed = collect_event_metrics(
        sweep_dir, args.seq_lengths, 'metrics_fixed_thresh.json')
    if event_fixed:
        for metric in ['f1', 'precision', 'recall', 'accuracy']:
            plot_event_metric_vs_seqlen(
                event_fixed, args.seq_lengths, metric,
                output_dir / f'seqlen_sweep_event_{metric}_fixed_{suffix}.png',
            )

    # Event detection metrics (fixed thresholds, IoU>=0.5 matching)
    print("\nCollecting event detection metrics (fixed thresh, IoU>=0.5)...")
    event_iou = collect_event_metrics(
        sweep_dir, args.seq_lengths, 'metrics_iou50.json')
    if event_iou:
        for metric in ['f1', 'precision', 'recall', 'accuracy']:
            plot_event_metric_vs_seqlen(
                event_iou, args.seq_lengths, metric,
                output_dir / f'seqlen_sweep_event_{metric}_iou50_{suffix}.png',
            )

    # Print summary table
    print(f"\n{'SeqLen':>8} {'N':>4} {'Loss':>12} {'Mean AUC':>12}")
    print('-' * 40)
    for sl in args.seq_lengths:
        if sl not in results:
            continue
        losses = [m['loss'] for m in results[sl]]
        aucs = [m['mean_auc'] for m in results[sl]]
        print(f"{sl:>8} {len(losses):>4} "
              f"{np.mean(losses):>6.4f}\u00b1{np.std(losses):.4f} "
              f"{np.mean(aucs):>6.4f}\u00b1{np.std(aucs):.4f}")

    if event_results:
        print(f"\n{'SeqLen':>8} {'N':>4} {'F1':>12} {'Precision':>12} "
              f"{'Recall':>12} {'Accuracy':>12}")
        print('-' * 64)
        for sl in args.seq_lengths:
            if sl not in event_results:
                continue
            n = len(event_results[sl])
            f1s = [m['overall']['micro']['f1'] for m in event_results[sl]]
            precs = [m['overall']['micro']['precision'] for m in event_results[sl]]
            recs = [m['overall']['micro']['recall'] for m in event_results[sl]]
            accs = [m['overall']['micro']['accuracy'] for m in event_results[sl]]
            print(f"{sl:>8} {n:>4} "
                  f"{np.mean(f1s):>5.3f}\u00b1{np.std(f1s):.3f} "
                  f"{np.mean(precs):>5.3f}\u00b1{np.std(precs):.3f} "
                  f"{np.mean(recs):>5.3f}\u00b1{np.std(recs):.3f} "
                  f"{np.mean(accs):>5.3f}\u00b1{np.std(accs):.3f}")


if __name__ == '__main__':
    main()

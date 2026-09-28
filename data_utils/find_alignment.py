"""
Find alignment between H5 and PKL files by comparing image similarity.

This script determines the offset needed to align H5 frames with PKL frames.

Reader's guide: candidate offsets are scored by the mean pixel difference
between sampled frame pairs. The lowest-scoring offset is the proposed
alignment; this utility does not modify either input file.
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image


def find_best_offset(h5_df, pkl_df, search_range=300, sample_frames=20):
    """
    Find the best offset to align H5 frames with PKL frames.

    Args:
        h5_df: H5 DataFrame with 'img' column (PIL Images, grayscale 640x480)
        pkl_df: PKL DataFrame with 'image' column (numpy arrays, RGB 224x224)
        search_range: How many frames to search for offset
        sample_frames: Number of PKL frames to test for alignment

    Returns:
        best_offset: Integer offset where PKL frame i aligns with H5 frame (i + offset)
        mean_diff: Average pixel difference at best offset
    """
    # Sample PKL frames throughout the recording for robust matching
    pkl_indices = np.linspace(0, len(pkl_df)-1, min(sample_frames, len(pkl_df)), dtype=int)

    best_offset = None
    min_total_diff = float('inf')

    # Search both positive and negative offsets
    # Positive offset: H5 starts before PKL (H5 has extra frames at beginning)
    # Negative offset: PKL starts before H5 (PKL has extra frames at beginning)
    max_offset = min(search_range, len(h5_df) - 1)
    min_offset = max(-search_range, -(len(pkl_df) - 1))

    for offset in range(min_offset, max_offset + 1):
        total_diff = 0
        valid_comparisons = 0

        for pkl_idx in pkl_indices:
            h5_idx = pkl_idx + offset

            # Skip if out of bounds
            if h5_idx < 0 or h5_idx >= len(h5_df):
                continue

            # Get images
            pkl_img = pkl_df['image'].iloc[pkl_idx]  # Shape: (224, 224, 3)
            h5_img_pil = h5_df['img'].iloc[h5_idx]   # PIL Image (480, 640, L)

            # Convert H5 to match PKL format (resize and convert to RGB)
            h5_img_resized = h5_img_pil.convert('RGB').resize((224, 224), Image.BILINEAR)
            h5_img_array = np.array(h5_img_resized)

            # Compute difference
            diff = np.abs(h5_img_array.astype(float) - pkl_img.astype(float)).mean()
            total_diff += diff
            valid_comparisons += 1

        if valid_comparisons > 0:
            avg_diff = total_diff / valid_comparisons
            if avg_diff < min_total_diff:
                min_total_diff = avg_diff
                best_offset = offset

    return best_offset, min_total_diff


def main():
    """Find alignment for all H5/PKL file pairs."""
    project_root = Path(__file__).parent.parent
    h5_dir = project_root / 'data' / 'h5_data'
    pkl_dir = project_root / 'data' / 'processed'

    h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])

    print('='*100)
    print('FINDING ALIGNMENT BETWEEN H5 AND PKL FILES')
    print('='*100)
    print()

    results = []

    for h5_fname in h5_files:
        h5_match = re.match(r'recordings_(\d+)_playing_enrollment(\d+)_multi_playing(\d+)\.h5', h5_fname)
        if not h5_match:
            continue

        subject = int(h5_match.group(1))
        enrollment = int(h5_match.group(2))
        trial = int(h5_match.group(3))

        pkl_fname = f'subject{subject:02d}_enrollment{enrollment:02d}_trial{trial:02d}.pkl'

        h5_path = h5_dir / h5_fname
        pkl_path = pkl_dir / pkl_fname

        if not pkl_path.exists():
            print(f'SKIP: {pkl_fname} not found')
            continue

        # Load data
        h5_df = pd.read_hdf(h5_path, 'df')
        pkl_df = pd.read_pickle(pkl_path)

        if len(pkl_df) == 0:
            print(f'SKIP: Subject {subject:2d}, Enrollment {enrollment:02d}, Trial {trial:02d} - PKL is empty')
            results.append({
                'subject': subject,
                'enrollment': enrollment,
                'trial': trial,
                'h5_frames': len(h5_df),
                'pkl_frames': 0,
                'offset': None,
                'mean_diff': None
            })
            continue

        # Find alignment
        offset, mean_diff = find_best_offset(h5_df, pkl_df)

        print(f'Subject {subject:2d}, Enrollment {enrollment:02d}, Trial {trial:02d}:')
        print(f'  H5 frames: {len(h5_df):4d}, PKL frames: {len(pkl_df):4d}')
        print(f'  Best offset: {offset:4d} (PKL frame i -> H5 frame {offset} + i)')
        print(f'  Mean pixel diff: {mean_diff:.2f}')
        print()

        results.append({
            'subject': subject,
            'enrollment': enrollment,
            'trial': trial,
            'h5_frames': len(h5_df),
            'pkl_frames': len(pkl_df),
            'offset': offset,
            'mean_diff': mean_diff
        })

    # Print summary
    print('='*100)
    print('SUMMARY')
    print('='*100)
    print(f"{'Subject':>7} {'Enroll':>7} {'Trial':>5} {'H5':>6} {'PKL':>6} {'Offset':>7} {'Mean Diff':>10}")
    print('-'*100)

    for r in results:
        offset_str = f"{r['offset']:4d}" if r['offset'] is not None else 'N/A'
        diff_str = f"{r['mean_diff']:.2f}" if r['mean_diff'] is not None else 'N/A'
        print(f"{r['subject']:7d} {r['enrollment']:7d} {r['trial']:5d} {r['h5_frames']:6d} {r['pkl_frames']:6d} {offset_str:>7} {diff_str:>10}")

    print('='*100)


if __name__ == '__main__':
    main()

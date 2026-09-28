"""
Find alignment between PKL file and raw image data.

This script compares the images in a PKL file with raw timestamp-named images
to find which raw images correspond to which PKL frames.

Reader's guide: ``find_alignment`` estimates the offset, ``verify_alignment``
checks it on additional frames, and ``find_unused_raw_images`` reports raw
frames that fall outside the aligned processed sequence.
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path


def load_raw_images(raw_images_dir):
    """
    Load raw image filenames sorted by timestamp.

    Args:
        raw_images_dir: Directory containing raw PNG images named with timestamps

    Returns:
        List of (timestamp, filepath) tuples sorted by timestamp
    """
    image_files = []
    for fname in os.listdir(raw_images_dir):
        if fname.endswith('.png'):
            timestamp = fname.replace('.png', '')
            filepath = os.path.join(raw_images_dir, fname)
            image_files.append((int(timestamp), filepath))

    # Sort by timestamp
    image_files.sort(key=lambda x: x[0])
    return image_files


def compare_images(pkl_img_array, raw_img_path):
    """
    Compare a PKL image with a raw image.

    Args:
        pkl_img_array: Numpy array from PKL (224x224x3, RGB)
        raw_img_path: Path to raw PNG image

    Returns:
        Mean pixel difference, or None if image is corrupted
    """
    try:
        # Load raw image and convert to match PKL format
        raw_img = Image.open(raw_img_path).convert('L')  # Grayscale
        raw_img_rgb = raw_img.convert('RGB')
        raw_img_resized = raw_img_rgb.resize((224, 224), Image.BILINEAR)
        raw_img_array = np.array(raw_img_resized)

        # Compute difference
        diff = np.abs(pkl_img_array.astype(float) - raw_img_array.astype(float)).mean()
        return diff
    except (OSError, IOError) as e:
        # Image is corrupted or truncated
        return None


def find_alignment(pkl_path, raw_images_dir, max_offset=50, sample_every=10):
    """
    Find which raw images correspond to PKL frames.

    Args:
        pkl_path: Path to PKL file
        raw_images_dir: Directory with raw images
        max_offset: Maximum offset to search
        sample_every: Test every N-th frame for speed

    Returns:
        best_offset: Offset where PKL frame i corresponds to raw_images[i + offset]
        frame_mapping: List of (pkl_idx, raw_idx, diff) for all tested frames
    """
    # Load PKL
    pkl_df = pd.read_pickle(pkl_path)

    # Load raw images
    raw_images = load_raw_images(raw_images_dir)

    print(f"PKL frames: {len(pkl_df)}")
    print(f"Raw images: {len(raw_images)}")
    print()

    # Sample PKL frames to test
    test_indices = list(range(0, len(pkl_df), sample_every))
    if test_indices[-1] != len(pkl_df) - 1:
        test_indices.append(len(pkl_df) - 1)

    print(f"Testing {len(test_indices)} sample frames...")
    print()

    best_offset = None
    min_total_diff = float('inf')
    offset_scores = []

    # Try different offsets
    for offset in range(-max_offset, max_offset + 1):
        total_diff = 0
        valid_comparisons = 0

        for pkl_idx in test_indices:
            raw_idx = pkl_idx + offset

            # Skip if out of bounds
            if raw_idx < 0 or raw_idx >= len(raw_images):
                continue

            # Compare images
            pkl_img = pkl_df['image'].iloc[pkl_idx]
            raw_img_path = raw_images[raw_idx][1]

            diff = compare_images(pkl_img, raw_img_path)
            if diff is not None:
                total_diff += diff
                valid_comparisons += 1

        if valid_comparisons > 0:
            avg_diff = total_diff / valid_comparisons
            offset_scores.append((offset, avg_diff, valid_comparisons))

            if avg_diff < min_total_diff:
                min_total_diff = avg_diff
                best_offset = offset

    # Print top candidates
    print("Top 5 best offsets:")
    offset_scores.sort(key=lambda x: x[1])
    for offset, avg_diff, comparisons in offset_scores[:5]:
        print(f"  Offset {offset:+4d}: mean diff = {avg_diff:.2f} ({comparisons} comparisons)")

    print()
    print(f"Best offset: {best_offset}")
    print(f"Mean difference: {min_total_diff:.2f}")
    print()

    return best_offset, offset_scores


def verify_alignment(pkl_path, raw_images_dir, offset):
    """
    Verify alignment by checking all frames.

    Args:
        pkl_path: Path to PKL file
        raw_images_dir: Directory with raw images
        offset: Offset to verify

    Returns:
        List of (pkl_idx, raw_idx, timestamp, diff) for all frames
    """
    pkl_df = pd.read_pickle(pkl_path)
    raw_images = load_raw_images(raw_images_dir)

    print(f"Verifying alignment with offset={offset}...")
    print()

    results = []
    mismatches = []

    for pkl_idx in range(len(pkl_df)):
        raw_idx = pkl_idx + offset

        if raw_idx < 0 or raw_idx >= len(raw_images):
            results.append((pkl_idx, raw_idx, None, None, 'OUT_OF_BOUNDS'))
            continue

        pkl_img = pkl_df['image'].iloc[pkl_idx]
        timestamp, raw_img_path = raw_images[raw_idx]

        diff = compare_images(pkl_img, raw_img_path)

        if diff is None:
            results.append((pkl_idx, raw_idx, timestamp, None, 'CORRUPTED'))
            mismatches.append((pkl_idx, raw_idx, timestamp, 'CORRUPTED'))
        else:
            results.append((pkl_idx, raw_idx, timestamp, diff, 'OK' if diff < 5.0 else 'MISMATCH'))
            if diff >= 5.0:
                mismatches.append((pkl_idx, raw_idx, timestamp, diff))

    # Print summary
    total_ok = sum(1 for r in results if r[4] == 'OK')
    total_mismatch = sum(1 for r in results if r[4] == 'MISMATCH')
    total_oob = sum(1 for r in results if r[4] == 'OUT_OF_BOUNDS')

    print(f"Verification results:")
    print(f"  OK matches (diff < 5):  {total_ok}")
    print(f"  Mismatches (diff >= 5): {total_mismatch}")
    print(f"  Out of bounds:          {total_oob}")
    print()

    if mismatches:
        print(f"Found {len(mismatches)} mismatches:")
        for pkl_idx, raw_idx, timestamp, diff in mismatches[:10]:
            print(f"  PKL {pkl_idx:4d} -> Raw {raw_idx:4d} (timestamp {timestamp}): diff = {diff:.2f}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")

    return results


def find_unused_raw_images(pkl_path, raw_images_dir, offset):
    """
    Find which raw images are not used in the PKL alignment.

    Args:
        pkl_path: Path to PKL file
        raw_images_dir: Directory with raw images
        offset: Alignment offset

    Returns:
        List of (raw_idx, timestamp) for unused images
    """
    pkl_df = pd.read_pickle(pkl_path)
    raw_images = load_raw_images(raw_images_dir)

    # Find which raw indices are used
    used_indices = set()
    for pkl_idx in range(len(pkl_df)):
        raw_idx = pkl_idx + offset
        if 0 <= raw_idx < len(raw_images):
            used_indices.add(raw_idx)

    # Find unused
    unused = []
    for raw_idx in range(len(raw_images)):
        if raw_idx not in used_indices:
            timestamp = raw_images[raw_idx][0]
            unused.append((raw_idx, timestamp))

    return unused


def main():
    parser = argparse.ArgumentParser(description='Find alignment between PKL and raw images')
    parser.add_argument('--pkl', required=True, help='Path to PKL file')
    parser.add_argument('--raw-dir', required=True, help='Directory with raw images')
    parser.add_argument('--max-offset', type=int, default=50, help='Max offset to search')
    parser.add_argument('--verify', action='store_true', help='Verify alignment on all frames')

    args = parser.parse_args()

    print('='*80)
    print('ALIGNMENT FINDER: PKL vs RAW IMAGES')
    print('='*80)
    print(f'PKL file: {args.pkl}')
    print(f'Raw images: {args.raw_dir}')
    print()

    # Find alignment
    best_offset, offset_scores = find_alignment(args.pkl, args.raw_dir, args.max_offset)

    if args.verify:
        print()
        print('='*80)
        results = verify_alignment(args.pkl, args.raw_dir, best_offset)

        print()
        print('='*80)
        print('UNUSED RAW IMAGES:')
        print('='*80)
        unused = find_unused_raw_images(args.pkl, args.raw_dir, best_offset)
        if unused:
            print(f"Found {len(unused)} unused raw images:")
            for raw_idx, timestamp in unused:
                print(f"  Raw image {raw_idx:4d}: {timestamp}.png")
        else:
            print("All raw images are used in the alignment.")


if __name__ == '__main__':
    main()

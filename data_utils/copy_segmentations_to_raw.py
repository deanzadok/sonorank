"""
Copy segmentation masks from raw_seg_data to raw_data folder structure.

This script matches segmentation masks with raw ultrasound images and copies
them to a new 'masks' folder within the raw data directory structure.

Reader's guide: filenames provide the correspondence between masks and raw
frames. The script validates counts before copying and supports a dry run so
the proposed mapping can be inspected without modifying the dataset.
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def verify_counts(seg_masks_dir, raw_images_dir):
    """
    Verify that the number of segmentation masks matches the number of raw images.

    Args:
        seg_masks_dir: Directory containing segmentation masks
        raw_images_dir: Directory containing raw images

    Returns:
        tuple: (seg_count, raw_count, matches)
    """
    seg_masks = sorted(seg_masks_dir.glob('*.png'))
    raw_images = sorted(raw_images_dir.glob('*.png'))

    seg_count = len(seg_masks)
    raw_count = len(raw_images)

    return seg_count, raw_count, seg_count == raw_count


def get_frame_number_from_mask(mask_filename):
    """
    Extract frame number from segmentation mask filename.

    Args:
        mask_filename: Name like 'frame_000000.png'

    Returns:
        int: Frame number
    """
    # Extract frame number from 'frame_XXXXXX.png'
    frame_str = mask_filename.stem.split('_')[1]
    return int(frame_str)


def get_timestamp_from_raw(raw_filename):
    """
    Extract timestamp from raw image filename.

    Args:
        raw_filename: Name like '457971973630.png'

    Returns:
        int: Timestamp
    """
    return int(raw_filename.stem)


def copy_segmentations(seg_dir, raw_session_dir, output_subdir='masks', dry_run=False):
    """
    Copy segmentation masks to raw data directory structure.

    Args:
        seg_dir: Path to segmentation directory (contains 'masks' folder)
        raw_session_dir: Path to raw session directory (contains 'images' folder)
        output_subdir: Name of subdirectory to create for masks (default: 'masks')
        dry_run: If True, only print what would be done without copying

    Returns:
        dict: Summary statistics
    """
    seg_dir = Path(seg_dir)
    raw_session_dir = Path(raw_session_dir)

    # Source directories
    seg_masks_dir = seg_dir / 'masks'
    seg_images_dir = seg_dir / 'images'
    raw_images_dir = raw_session_dir / 'images'

    # Verify all directories exist
    if not seg_masks_dir.exists():
        raise ValueError(f"Segmentation masks directory not found: {seg_masks_dir}")
    if not seg_images_dir.exists():
        raise ValueError(f"Segmentation images directory not found: {seg_images_dir}")
    if not raw_images_dir.exists():
        raise ValueError(f"Raw images directory not found: {raw_images_dir}")

    # Verify counts match
    seg_count, raw_count, matches = verify_counts(seg_masks_dir, raw_images_dir)

    print(f"Segmentation masks: {seg_count}")
    print(f"Raw images: {raw_count}")

    if not matches:
        raise ValueError(f"Count mismatch: {seg_count} masks vs {raw_count} raw images")

    print("✓ Counts match!")
    print()

    # Create output directory
    output_dir = raw_session_dir / output_subdir

    if not dry_run:
        output_dir.mkdir(exist_ok=True)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"[DRY RUN] Would create directory: {output_dir}")

    # Get sorted lists
    seg_masks = sorted(seg_masks_dir.glob('*.png'))
    raw_images = sorted(raw_images_dir.glob('*.png'))

    # Copy masks with same ordering
    copied_count = 0

    print(f"Copying {len(seg_masks)} segmentation masks...")
    print()

    for seg_mask, raw_image in tqdm(zip(seg_masks, raw_images), total=len(seg_masks), desc="Copying"):
        # Get frame number and timestamp
        frame_num = get_frame_number_from_mask(seg_mask)
        timestamp = get_timestamp_from_raw(raw_image)

        # Create output filename using timestamp
        output_filename = f"{timestamp}.png"
        output_path = output_dir / output_filename

        # Copy file
        if not dry_run:
            shutil.copy2(seg_mask, output_path)
            copied_count += 1

    print()
    if dry_run:
        print(f"[DRY RUN] Would copy {len(seg_masks)} masks")
    else:
        print(f"✓ Successfully copied {copied_count} masks to: {output_dir}")

    # Print sample mapping
    print()
    print("Sample mapping (first 5):")
    for i, (seg_mask, raw_image) in enumerate(zip(seg_masks[:5], raw_images[:5])):
        frame_num = get_frame_number_from_mask(seg_mask)
        timestamp = get_timestamp_from_raw(raw_image)
        print(f"  {seg_mask.name} (frame {frame_num}) -> {timestamp}.png")

    return {
        'seg_count': seg_count,
        'raw_count': raw_count,
        'copied_count': copied_count if not dry_run else 0,
        'output_dir': str(output_dir)
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Copy segmentation masks to raw data directory structure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--seg-dir',
        type=str,
        required=True,
        help='Path to segmentation directory (e.g., data/raw_seg_data/recordings_10_enrollment01_multi_playing01_compatible)'
    )
    parser.add_argument(
        '--raw-session',
        type=str,
        required=True,
        help='Path to raw session directory (e.g., data/raw_data/recordings_10/playing/enrollment01/multi_playing01)'
    )
    parser.add_argument(
        '--output-subdir',
        type=str,
        default='masks',
        help='Name of subdirectory to create for masks'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without actually copying files'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("COPY SEGMENTATIONS TO RAW DATA")
    print("=" * 80)
    print(f"Segmentation directory: {args.seg_dir}")
    print(f"Raw session directory: {args.raw_session}")
    print(f"Output subdirectory: {args.output_subdir}")
    if args.dry_run:
        print("[DRY RUN MODE - No files will be copied]")
    print("=" * 80)
    print()

    try:
        result = copy_segmentations(
            args.seg_dir,
            args.raw_session,
            output_subdir=args.output_subdir,
            dry_run=args.dry_run
        )

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Segmentation masks: {result['seg_count']}")
        print(f"Raw images: {result['raw_count']}")
        if not args.dry_run:
            print(f"Copied masks: {result['copied_count']}")
            print(f"Output directory: {result['output_dir']}")
        print("=" * 80)

    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {str(e)}")
        print("=" * 80)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

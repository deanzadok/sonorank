"""
Data preparation script for muscle ultrasound semantic segmentation.

This script processes raw ultrasound recordings with segmentation masks,
creating structured dataframes for training. Each recording session is saved
as a separate pickle file.

Reader's guide: recording names establish subject and session provenance,
image and mask loaders enforce the requested resolution, and
``process_recording`` assembles the frame table written by the CLI.
"""

import os
import json
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pickle


def parse_recording_name(folder_name):
    """
    Parse recording folder name to extract subject ID, enrollment, and trial.

    Example: recordings_20_enrollment02_multi_playing02_compatible
    Returns: subject_id=20, enrollment=2, trial=2

    Args:
        folder_name: Name of the recording folder

    Returns:
        dict with 'subject_id', 'enrollment', 'trial'
    """
    # Pattern: recordings_{subject_id}_enrollment{enrollment}_multi_playing{trial}_compatible
    pattern = r'recordings_(\d+)_enrollment(\d+)_multi_playing(\d+)_compatible'
    match = re.match(pattern, folder_name)

    if not match:
        raise ValueError(f"Folder name '{folder_name}' doesn't match expected pattern")

    subject_id = int(match.group(1))
    enrollment = int(match.group(2))
    trial = int(match.group(3))

    return {
        'subject_id': subject_id,
        'enrollment': enrollment,
        'trial': trial
    }


def load_image(image_path, target_size=None):
    """
    Load and convert image to numpy array.

    Args:
        image_path: Path to image file
        target_size: Optional tuple (height, width) to resize image

    Returns:
        numpy array with shape (H, W, C) or (target_size[0], target_size[1], C)
    """
    img = Image.open(image_path)

    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    return np.array(img)


def load_mask(mask_path, target_size=None):
    """
    Load segmentation mask and ensure it contains class indices.

    Args:
        mask_path: Path to mask PNG file
        target_size: Optional tuple (height, width) to resize mask

    Returns:
        numpy array with shape (H, W) containing integer class labels
    """
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # If mask is RGB, convert to single-channel class indices
    if len(mask_array.shape) == 3:
        # Assuming the mask uses the first channel or needs conversion
        # Check if it's already encoded as indices in one channel
        if np.array_equal(mask_array[:, :, 0], mask_array[:, :, 1]) and \
           np.array_equal(mask_array[:, :, 0], mask_array[:, :, 2]):
            mask_array = mask_array[:, :, 0]
        else:
            # If it's a color-coded mask, we need to convert colors to indices
            # This shouldn't happen based on our analysis, but handle it just in case
            raise ValueError(f"Unexpected RGB mask at {mask_path}")

    # Resize mask using nearest neighbor to preserve class indices
    if target_size is not None:
        mask_pil = Image.fromarray(mask_array)
        mask_pil = mask_pil.resize((target_size[1], target_size[0]), Image.NEAREST)
        mask_array = np.array(mask_pil)

    return mask_array.astype(np.uint8)


def process_recording(recording_path, output_dir, resize_to=None):
    """
    Process a single recording session.

    Args:
        recording_path: Path to recording folder
        output_dir: Directory to save processed pickle files
        resize_to: Optional tuple (height, width) to resize images and masks

    Returns:
        str: Path to saved pickle file
    """
    recording_path = Path(recording_path)
    folder_name = recording_path.name

    # Parse recording metadata
    metadata = parse_recording_name(folder_name)

    # Load info.json
    info_path = recording_path / 'info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    metadata['num_objects'] = info['num_objects']
    metadata['reference_frames'] = info['references']

    # Get image and mask paths
    images_dir = recording_path / 'images'
    masks_dir = recording_path / 'masks'

    # Get sorted list of image files
    image_files = sorted(images_dir.glob('frame_*.jpg'))

    # Prepare data lists
    data_records = []

    print(f"Processing {folder_name}...")
    print(f"  Subject ID: {metadata['subject_id']}, "
          f"Enrollment: {metadata['enrollment']}, "
          f"Trial: {metadata['trial']}")
    print(f"  Total frames: {len(image_files)}")
    if resize_to:
        print(f"  Resizing to: {resize_to[0]}x{resize_to[1]}")

    for img_path in tqdm(image_files, desc="Loading frames"):
        # Get corresponding mask path
        frame_name = img_path.stem  # e.g., 'frame_000000'
        mask_path = masks_dir / f"{frame_name}.png"

        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_path.name}, skipping...")
            continue

        # Load image and mask (with optional resizing)
        image = load_image(img_path, target_size=resize_to)
        mask = load_mask(mask_path, target_size=resize_to)

        # Verify mask contains valid class indices
        unique_classes = np.unique(mask)
        if unique_classes.max() > 7 or unique_classes.min() < 0:
            print(f"Warning: Unexpected class values in {mask_path.name}: {unique_classes}")

        # Create record
        record = {
            'frame_id': frame_name,
            'frame_number': int(frame_name.split('_')[1]),
            'image': image,
            'mask': mask,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'is_reference': int(frame_name.split('_')[1]) in info['references']
        }

        data_records.append(record)

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Add metadata columns
    for key, value in metadata.items():
        if key != 'reference_frames':  # Don't add the list as a column
            df[key] = value

    # Create output filename
    output_filename = f"subject{metadata['subject_id']:02d}_enrollment{metadata['enrollment']:02d}_trial{metadata['trial']:02d}.pkl"
    output_path = output_dir / output_filename

    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Saved to: {output_path}")
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Print sample class distribution from first frame
    if len(df) > 0:
        sample_mask = df.iloc[0]['mask']
        unique, counts = np.unique(sample_mask, return_counts=True)
        print(f"  Sample class distribution (frame 0):")
        for cls, count in zip(unique, counts):
            percentage = 100 * count / sample_mask.size
            print(f"    Class {cls}: {count:6d} pixels ({percentage:5.2f}%)")

    print()

    return str(output_path)


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description='Process ultrasound recordings for semantic segmentation training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--resize',
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        default=[224, 224],
        help='Target size for resizing images and masks (height width)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/deanz/neural-hs/muscle_attention/data/raw_data',
        help='Directory containing raw data recordings'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/deanz/neural-hs/muscle_attention/data/processed',
        help='Directory to save processed pickle files'
    )
    parser.add_argument(
        '--no-resize',
        action='store_true',
        help='Disable resizing, keep original image dimensions'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Determine resize target
    resize_to = None if args.no_resize else tuple(args.resize)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all recording folders
    recording_folders = sorted([f for f in data_dir.iterdir()
                               if f.is_dir() and f.name.startswith('recordings_')])

    print(f"Found {len(recording_folders)} recording sessions")
    print(f"Output directory: {output_dir}")
    if resize_to:
        print(f"Resizing images to: {resize_to[0]}x{resize_to[1]}")
    else:
        print("Resizing: Disabled (using original dimensions)")
    print("=" * 80)
    print()

    # Process each recording
    processed_files = []
    for recording_folder in recording_folders:
        try:
            output_path = process_recording(recording_folder, output_dir, resize_to=resize_to)
            processed_files.append(output_path)
        except Exception as e:
            print(f"ERROR processing {recording_folder.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            print()

    # Summary
    print("=" * 80)
    print(f"Processing complete!")
    print(f"Successfully processed: {len(processed_files)} / {len(recording_folders)} recordings")
    print(f"Output files saved to: {output_dir}")

    # Create a summary file
    summary_path = output_dir / 'processing_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Data Processing Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total recordings processed: {len(processed_files)}\n")
        f.write(f"Output directory: {output_dir}\n")
        if resize_to:
            f.write(f"Image size: {resize_to[0]}x{resize_to[1]}\n")
        else:
            f.write("Image size: Original (no resizing)\n")
        f.write("\nProcessed files:\n")
        for path in sorted(processed_files):
            f.write(f"  - {Path(path).name}\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

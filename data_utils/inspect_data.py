"""
Example script to load and inspect processed pickle files.

This script demonstrates how to load the processed data and visualize
various properties of the dataset.

Reader's guide: the inspection stages print the table schema, validate image
and mask contents, summarize class balance, and optionally render examples.
It is a diagnostic tool and does not feed the training pipeline directly.
"""

import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def inspect_dataframe(df):
    """Print detailed information about the DataFrame."""
    print_separator()
    print("DATAFRAME STRUCTURE")
    print_separator()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)

    print("\n" + "=" * 80)
    print("METADATA")
    print_separator()
    if len(df) > 0:
        print(f"Subject ID: {df['subject_id'].iloc[0]}")
        print(f"Enrollment: {df['enrollment'].iloc[0]}")
        print(f"Trial: {df['trial'].iloc[0]}")
        print(f"Number of objects: {df['num_objects'].iloc[0]}")
        print(f"Total frames: {len(df)}")
        print(f"Reference frames: {df['is_reference'].sum()}")

    print("\n" + "=" * 80)
    print("FIRST FEW ROWS (metadata only)")
    print_separator()
    # Display first few rows without image/mask arrays
    display_cols = ['frame_id', 'frame_number', 'subject_id',
                   'enrollment', 'trial', 'is_reference']
    print(df[display_cols].head(10))


def inspect_images_and_masks(df, num_samples=5):
    """Print information about images and masks."""
    if len(df) == 0:
        print("\nDataFrame is empty!")
        return

    print("\n" + "=" * 80)
    print("IMAGE AND MASK PROPERTIES")
    print_separator()

    # Sample a few random frames
    sample_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)

    for idx in sample_indices:
        row = df.iloc[idx]
        image = row['image']
        mask = row['mask']

        print(f"\nFrame: {row['frame_id']}")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min()}, {image.max()}]")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Unique classes: {sorted(np.unique(mask))}")


def analyze_class_distribution(df):
    """Analyze class distribution across all frames."""
    if len(df) == 0:
        print("\nDataFrame is empty!")
        return

    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print_separator()

    # Collect class statistics
    all_class_counts = []

    for idx in range(len(df)):
        mask = df.iloc[idx]['mask']
        unique, counts = np.unique(mask, return_counts=True)
        class_dict = dict(zip(unique, counts))
        all_class_counts.append(class_dict)

    # Compute statistics for each class
    print("\nClass statistics across all frames:")
    print(f"{'Class':<8} {'Avg %':<10} {'Std %':<10} {'Min %':<10} {'Max %':<10}")
    print("-" * 50)

    num_pixels = df.iloc[0]['mask'].size
    for cls in range(8):  # Classes 0-7
        percentages = []
        for class_dict in all_class_counts:
            count = class_dict.get(cls, 0)
            pct = 100 * count / num_pixels
            percentages.append(pct)

        avg_pct = np.mean(percentages)
        std_pct = np.std(percentages)
        min_pct = np.min(percentages)
        max_pct = np.max(percentages)

        print(f"{cls:<8} {avg_pct:<10.2f} {std_pct:<10.2f} {min_pct:<10.2f} {max_pct:<10.2f}")


def visualize_samples(df, num_samples=3, save_path=None):
    """Visualize sample images with their masks."""
    if len(df) == 0:
        print("\nDataFrame is empty!")
        return

    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print_separator()

    # Create custom colormap for masks (8 classes)
    colors = [
        [0, 0, 0],         # Class 0: Black (background)
        [1, 0, 0],         # Class 1: Red
        [0.8, 0.8, 0],     # Class 2: Yellow/Olive
        [0, 0.8, 0.8],     # Class 3: Cyan
        [0.8, 0, 0.8],     # Class 4: Magenta
        [0, 0, 1],         # Class 5: Blue
        [0, 1, 0],         # Class 6: Green
        [0.6, 0.6, 0.6],   # Class 7: Gray
    ]
    cmap = ListedColormap(colors)

    # Sample frames
    sample_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        image = row['image']
        mask = row['mask']

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image_display = np.stack([image] * 3, axis=-1)
        else:
            image_display = image

        # Create overlay
        overlay = image_display.copy().astype(float) / 255.0

        # Plot original image
        axes[i, 0].imshow(image_display, cmap='gray' if len(image.shape) == 2 else None)
        axes[i, 0].set_title(f"Frame {row['frame_id']}\nOriginal Image")
        axes[i, 0].axis('off')

        # Plot mask
        im = axes[i, 1].imshow(mask, cmap=cmap, vmin=0, vmax=7, interpolation='nearest')
        axes[i, 1].set_title("Segmentation Mask")
        axes[i, 1].axis('off')

        # Create colorbar for first row
        if i == 0:
            cbar = plt.colorbar(im, ax=axes[i, 1], ticks=range(8))
            cbar.set_label('Class')

        # Plot overlay
        axes[i, 2].imshow(image_display, cmap='gray' if len(image.shape) == 2 else None)
        mask_overlay = np.ma.masked_where(mask == 0, mask)
        axes[i, 2].imshow(mask_overlay, cmap=cmap, vmin=0, vmax=7, alpha=0.5, interpolation='nearest')
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')

        # Print class distribution for this frame
        unique, counts = np.unique(mask, return_counts=True)
        print(f"\nFrame {row['frame_id']} class distribution:")
        for cls, count in zip(unique, counts):
            pct = 100 * count / mask.size
            print(f"  Class {cls}: {count:5d} pixels ({pct:5.2f}%)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        print("\nShowing visualization (close window to continue)...")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Inspect processed pickle files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'pickle_file',
        type=str,
        nargs='?',
        default='data/processed/subject20_enrollment02_trial02.pkl',
        help='Path to pickle file to inspect'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--save-viz',
        type=str,
        default=None,
        help='Path to save visualization (if not provided, will display)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization'
    )

    args = parser.parse_args()

    # Load pickle file
    pickle_path = Path(args.pickle_file)

    if not pickle_path.exists():
        print(f"Error: File not found: {pickle_path}")
        return

    print(f"\nLoading pickle file: {pickle_path}")
    print_separator()

    with open(pickle_path, 'rb') as f:
        df = pickle.load(f)

    print(f"Loaded successfully!")

    # Inspect DataFrame
    inspect_dataframe(df)

    if len(df) > 0:
        # Inspect images and masks
        inspect_images_and_masks(df, num_samples=min(5, len(df)))

        # Analyze class distribution
        analyze_class_distribution(df)

        # Visualize samples
        if not args.no_viz:
            visualize_samples(df, num_samples=args.num_samples, save_path=args.save_viz)

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()

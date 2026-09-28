"""Batch process the enumerated ``multi_playing`` sessions into HDF5 files.

Reader's guide: ``SESSIONS`` is the authoritative batch list in this legacy
utility. The command validates that each raw directory exists, delegates one
session at a time to ``prepare_data_session``, and records a summary without
changing the list automatically.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils.process_h5 import prepare_data_session


# List of all multi_playing sessions to process
# Format: (subject, enrollment, trial)
SESSIONS = [
    # Subject 01
    (1, 1, 1),
    (1, 2, 2),
    (1, 3, 3),
    # Subject 04
    (4, 1, 1),
    (4, 2, 2),
    # Subject 05
    (5, 1, 1),
    (5, 2, 2),
    # Subject 06
    (6, 1, 1),
    (6, 2, 2),
    # Subject 10
    (10, 1, 1),
    (10, 2, 2),
    # Subject 17
    (17, 1, 1),
    (17, 2, 2),
    # Subject 19
    (19, 1, 1),
    (19, 2, 2),
    # Subject 20
    (20, 1, 1),
    (20, 2, 2),
    # Subject 21
    (21, 1, 1),
    (21, 2, 2),
    # Subject 23
    (23, 1, 1),
    (23, 2, 2),
    # Subject 24
    (24, 1, 1),
    (24, 2, 2),
    # Subject 25
    (25, 1, 1),
    (25, 2, 2),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch process all multi_playing sessions into H5 files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default='data/raw_data',
        help='Root directory containing raw data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed_h5',
        help='Output directory for H5 files'
    )
    parser.add_argument(
        '--img-res',
        type=int,
        default=224,
        help='Target resolution for images'
    )
    parser.add_argument(
        '--use-imgs',
        action='store_true',
        default=True,
        help='Include ultrasound images'
    )
    parser.add_argument(
        '--no-imgs',
        dest='use_imgs',
        action='store_false',
        help='Skip ultrasound images'
    )
    parser.add_argument(
        '--no-print-joints',
        action='store_true',
        help='Skip joint angle visualization plots'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Lists to track processing
    all_stats = []
    successful = []
    failed = []
    skipped = []

    total_sessions = len(SESSIONS)
    verbose = not args.quiet

    print("="*80)
    print("BATCH H5 PROCESSING")
    print("="*80)
    print(f"Total sessions to process: {total_sessions}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image resolution: {args.img_res}x{args.img_res}")
    print(f"Include images: {args.use_imgs}")
    print("="*80)
    print()

    for idx, (subject, enrollment, trial) in enumerate(SESSIONS, 1):
        # Construct paths
        subject_padded = f"{subject:02d}"
        data_dir = os.path.join(
            args.raw_data_dir,
            f"recordings_{subject_padded}",
            "playing",
            f"enrollment{enrollment:02d}",
            f"multi_playing{trial:02d}"
        )

        session_name = f"Subject {subject_padded}, Enrollment {enrollment:02d}, Trial {trial:02d}"

        print(f"[{idx}/{total_sessions}] Processing: {session_name}")
        print(f"  Data dir: {data_dir}")

        # Check if directory exists
        if not os.path.isdir(data_dir):
            print(f"  ⚠ SKIPPED: Directory does not exist")
            skipped.append(session_name)
            print()
            continue

        # Process the session
        try:
            stat_dict = prepare_data_session(
                data_dir=data_dir,
                output_dir=args.output_dir,
                img_res=args.img_res,
                use_imgs=args.use_imgs,
                resize_imgs=True,
                right_handed=True,
                modified=True,
                print_joints=not args.no_print_joints,
                verbose=verbose,
                use_calib=False
            )

            all_stats.append(stat_dict)
            successful.append(session_name)
            print(f"  ✓ SUCCESS")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append((session_name, str(e)))

        print()

    # Save combined statistics
    print("="*80)
    print("SAVING COMBINED STATISTICS")
    print("="*80)

    if all_stats:
        combined_stats_df = pd.DataFrame(all_stats)
        combined_stats_path = os.path.join(args.output_dir, "all_sessions_stats.csv")
        combined_stats_df.to_csv(combined_stats_path, index=False)
        print(f"Combined statistics saved to: {combined_stats_path}")
        print(f"Total rows: {len(combined_stats_df)}")
    else:
        print("No statistics to save (no sessions processed successfully)")

    # Print summary
    print()
    print("="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total sessions: {total_sessions}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")
    print(f"  ⚠ Skipped: {len(skipped)}")
    print("="*80)

    if failed:
        print()
        print("Failed sessions:")
        for session_name, error in failed:
            print(f"  - {session_name}")
            print(f"    Error: {error}")

    if skipped:
        print()
        print("Skipped sessions:")
        for session_name in skipped:
            print(f"  - {session_name}")

    print()
    print("Done!")

    # Exit with error code if any failed
    sys.exit(1 if failed else 0)

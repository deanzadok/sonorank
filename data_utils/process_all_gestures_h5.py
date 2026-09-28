"""
Batch process all gestures sessions from raw data into H5 files.

Discovers and processes all gestures folders:
- finger_flex*: Finger flexion recordings
- wrist*: Wrist movement recordings
- pinch*: Pinch gesture recordings

Reader's guide: this is a batch wrapper around ``prepare_data_session``.
Discovery is read-only; conversion begins only after argument parsing and can
be previewed with ``--dry-run``.
"""

import os
import sys
import argparse
import re
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils.process_h5 import prepare_data_session


# All 12 subjects
SUBJECTS = [1, 4, 5, 6, 10, 17, 19, 20, 21, 23, 24, 25]

# Gestures session types to process
GESTURE_TYPES = ['finger_flex', 'wrist', 'pinch']


def discover_gestures_sessions(raw_data_dir, subjects=None, gesture_types=None):
    """
    Auto-discover all gestures sessions in raw data directory.

    Args:
        raw_data_dir: Root directory containing raw data
        subjects: List of subject IDs to process (None = all)
        gesture_types: List of gesture types to process (None = all)

    Returns:
        List of tuples: (subject, enrollment, session_name, data_dir)
    """
    if subjects is None:
        subjects = SUBJECTS
    if gesture_types is None:
        gesture_types = GESTURE_TYPES

    sessions = []

    for subject in subjects:
        subject_padded = f"{subject:02d}"
        gestures_dir = Path(raw_data_dir) / f"recordings_{subject_padded}" / "gestures"

        if not gestures_dir.exists():
            continue

        # Find all enrollment directories
        for enrollment_dir in sorted(gestures_dir.iterdir()):
            if not enrollment_dir.is_dir() or not enrollment_dir.name.startswith('enrollment'):
                continue

            # Extract enrollment number
            match = re.match(r'enrollment(\d+)', enrollment_dir.name)
            if not match:
                continue
            enrollment = int(match.group(1))

            # Find all session directories matching gesture types
            for session_dir in sorted(enrollment_dir.iterdir()):
                if not session_dir.is_dir():
                    continue

                # Check if session matches any gesture type
                for gesture_type in gesture_types:
                    if session_dir.name.startswith(gesture_type):
                        sessions.append((
                            subject,
                            enrollment,
                            session_dir.name,
                            str(session_dir)
                        ))
                        break

    return sessions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch process all gestures sessions into H5 files',
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
        default='data/processed_h5/processed_gestures',
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
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        default=None,
        help='Specific subjects to process (default: all 12 subjects)'
    )
    parser.add_argument(
        '--gesture-types',
        type=str,
        nargs='+',
        default=None,
        choices=['finger_flex', 'wrist', 'pinch'],
        help='Specific gesture types to process (default: all types)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List sessions without processing'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all sessions
    print("="*80)
    print("DISCOVERING GESTURES SESSIONS")
    print("="*80)

    sessions = discover_gestures_sessions(
        args.raw_data_dir,
        subjects=args.subjects,
        gesture_types=args.gesture_types
    )

    print(f"Found {len(sessions)} gestures sessions")

    # Count by type
    type_counts = {}
    for _, _, session_name, _ in sessions:
        for gtype in GESTURE_TYPES:
            if session_name.startswith(gtype):
                type_counts[gtype] = type_counts.get(gtype, 0) + 1
                break

    print("Sessions by type:")
    for gtype in GESTURE_TYPES:
        print(f"  {gtype}: {type_counts.get(gtype, 0)}")
    print()

    if args.dry_run:
        print("DRY RUN - Sessions that would be processed:")
        print("-"*80)
        for subject, enrollment, session_name, data_dir in sessions:
            print(f"  Subject {subject:02d}, Enrollment {enrollment:02d}, {session_name}")
            print(f"    {data_dir}")
        print("-"*80)
        print(f"Total: {len(sessions)} sessions")
        sys.exit(0)

    # Lists to track processing
    all_stats = []
    successful = []
    failed = []
    skipped = []

    total_sessions = len(sessions)
    verbose = not args.quiet

    print("="*80)
    print("BATCH H5 PROCESSING - GESTURES")
    print("="*80)
    print(f"Total sessions to process: {total_sessions}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image resolution: {args.img_res}x{args.img_res}")
    print(f"Include images: {args.use_imgs}")
    print("="*80)
    print()

    for idx, (subject, enrollment, session_name, data_dir) in enumerate(sessions, 1):
        session_label = f"Subject {subject:02d}, Enrollment {enrollment:02d}, {session_name}"

        print(f"[{idx}/{total_sessions}] Processing: {session_label}")
        print(f"  Data dir: {data_dir}")

        # Check if directory exists
        if not os.path.isdir(data_dir):
            print(f"  WARNING: SKIPPED - Directory does not exist")
            skipped.append(session_label)
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
            successful.append(session_label)
            print(f"  SUCCESS")

        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((session_label, str(e)))

        print()

    # Save combined statistics
    print("="*80)
    print("SAVING COMBINED STATISTICS")
    print("="*80)

    if all_stats:
        combined_stats_df = pd.DataFrame(all_stats)
        combined_stats_path = os.path.join(args.output_dir, "all_gestures_sessions_stats.csv")
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
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    print("="*80)

    if failed:
        print()
        print("Failed sessions:")
        for session_label, error in failed:
            print(f"  - {session_label}")
            print(f"    Error: {error}")

    if skipped:
        print()
        print("Skipped sessions:")
        for session_label in skipped:
            print(f"  - {session_label}")

    print()
    print("Done!")

    # Exit with error code if any failed
    sys.exit(1 if failed else 0)

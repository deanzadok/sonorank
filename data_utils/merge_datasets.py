"""
Merge H5 kinematics data with processed pickle segmentation data.

This script combines:
- H5 files: Raw ultrasound images (640x480) + hand kinematics (joint angles, velocities, etc.)
- PKL files: Processed ultrasound images (224x224x3) + segmentation masks (224x224)

Output: Enhanced pickle files with kinematics data added.

Reader's guide: ``find_matching_files`` pairs sessions by identifiers,
``merge_single_file`` performs the frame-level join, and ``main`` applies the
operation to a directory while keeping each recording as a separate output.
"""

import os
import pickle
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm


def find_matching_files(h5_dir, pkl_dir):
    """
    Find matching H5 and PKL files based on subject/enrollment/trial.

    Returns:
        List of tuples: (h5_path, pkl_path, subject, enrollment, trial)
    """
    h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
    pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])

    matches = []

    for h5_fname in h5_files:
        # Parse H5 filename: recordings_{subject}_playing_enrollment{XX}_multi_playing{YY}.h5
        h5_match = re.match(r'recordings_(\d+)_playing_enrollment(\d+)_multi_playing(\d+)\.h5', h5_fname)
        if not h5_match:
            continue

        subject = int(h5_match.group(1))
        enrollment = int(h5_match.group(2))
        trial = int(h5_match.group(3))

        # Find matching PKL file: subject{subject}_enrollment{XX}_trial{YY}.pkl
        pkl_fname = f'subject{subject:02d}_enrollment{enrollment:02d}_trial{trial:02d}.pkl'

        if pkl_fname in pkl_files:
            h5_path = os.path.join(h5_dir, h5_fname)
            pkl_path = os.path.join(pkl_dir, pkl_fname)
            matches.append((h5_path, pkl_path, subject, enrollment, trial))
        else:
            print(f"Warning: No matching PKL file for {h5_fname}")

    return matches


def merge_single_file(h5_path, pkl_path, output_path):
    """
    Merge a single H5 and PKL file pair.

    Args:
        h5_path: Path to H5 file with kinematics
        pkl_path: Path to PKL file with segmentation
        output_path: Path to save merged PKL file
    """
    # Load H5 data
    h5_df = pd.read_hdf(h5_path, 'df')

    # Load PKL data
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    # Get frame numbers from PKL (these index into H5)
    frame_numbers = pkl_data['frame_number'].values

    # Extract kinematics for only the frames that exist in PKL
    # PKL has frames [0, 1, 2, ..., 1335] which correspond to H5 rows [0, 1, 2, ..., 1335]
    kinematics_subset = h5_df.iloc[frame_numbers]

    # Define kinematic columns to extract
    # Joint angles (primary features)
    joint_cols = [
        'joint41', 'joint31', 'joint21', 'joint11',
        'joint42', 'joint32', 'joint22', 'joint12',
        'joint43', 'joint33', 'joint23', 'joint13',
        'jointt5', 'jointt6', 'jointwr', 'jointwp', 'jointwy'
    ]

    # Joint velocities (first derivatives)
    joint_vel_cols = [col + '_der' for col in joint_cols]

    # Joint accelerations (second derivatives)
    joint_acc_cols = [col + '_der_der' for col in joint_cols]

    # Finger/thumb angles
    finger_cols = [
        'thumb3', 'thumb4', 'thumb5', 'thumb6', 'thumb7',
        'finger11', 'finger12', 'finger13', 'finger14',
        'finger21', 'finger22', 'finger23', 'finger24',
        'finger31', 'finger32', 'finger33', 'finger34',
        'finger41', 'finger42', 'finger43', 'finger44'
    ]

    # Probe and palm orientation
    probe_cols = ['probe1', 'probe2', 'probe3']
    palm_cols = ['palm_normal', 'palm_horizontal']

    # Temporal information
    time_cols = ['time_x', 'rtime', 'datetime']

    # Combine all columns to extract
    all_kinematic_cols = (
        joint_cols + joint_vel_cols + joint_acc_cols +
        finger_cols + probe_cols + palm_cols + time_cols
    )

    # Filter to only columns that exist
    available_cols = [col for col in all_kinematic_cols if col in kinematics_subset.columns]

    # Extract kinematics and reset index to match pkl_data indices
    kinematics_extracted = kinematics_subset[available_cols].reset_index(drop=True)

    # Add each kinematic column to the pkl_data dictionary
    for col in available_cols:
        pkl_data[col] = kinematics_extracted[col]

    # Save merged data
    with open(output_path, 'wb') as f:
        pickle.dump(pkl_data, f)

    return len(frame_numbers), len(available_cols)


def main():
    """Main function to merge all matching files."""
    # Paths
    project_root = Path(__file__).parent.parent
    h5_dir = project_root / 'data' / 'h5_data'
    pkl_dir = project_root / 'data' / 'processed'
    output_dir = project_root / 'data' / 'processed_with_kinematics'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("MERGING H5 KINEMATICS WITH PKL SEGMENTATION DATA")
    print("="*80)
    print(f"H5 directory:  {h5_dir}")
    print(f"PKL directory: {pkl_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Find matching files
    matches = find_matching_files(str(h5_dir), str(pkl_dir))
    print(f"Found {len(matches)} matching file pairs")
    print()

    # Merge each pair
    for h5_path, pkl_path, subject, enrollment, trial in tqdm(matches, desc="Merging files"):
        output_fname = f'subject{subject:02d}_enrollment{enrollment:02d}_trial{trial:02d}.pkl'
        output_path = output_dir / output_fname

        try:
            num_frames, num_kinematic_features = merge_single_file(h5_path, pkl_path, str(output_path))
            # print(f"✓ Subject {subject:2d}, enrollment {enrollment:02d}, trial {trial:02d}: "
            #       f"{num_frames} frames, {num_kinematic_features} kinematic features")
        except Exception as e:
            print(f"✗ Error processing subject {subject}, enrollment {enrollment}, trial {trial}: {e}")

    print()
    print("="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"Merged files saved to: {output_dir}")
    print()
    print("You can now use these files for multi-modal training!")
    print("Each file contains:")
    print("  - Original data: images, masks, metadata")
    print("  - Kinematics: 17 joint angles + velocities + accelerations")
    print("  - Hand features: 21 finger/thumb angles")
    print("  - Context: probe orientation, palm angles, timestamps")


if __name__ == '__main__':
    main()

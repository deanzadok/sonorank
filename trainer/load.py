"""
Data loading and preparation module for SonoRank paper.

This is a trimmed version of the full data loading module, containing only
the modes relevant to the SonoRank paper:
  - embedding: single-frame autoencoder training
  - multi_finger_temporal_ranking: multi-finger temporal ranking with 5 parallel heads
  - flex_embedding: embedding-based flex classification with majority-vote labels
  - cbmf: CBMF baseline (Zadok et al. 2025, UNet+GRU)

Removed modes: segmentation, joint_regression, joint_motion_classification,
optical_flow, finger_ranking, temporal_ranking, temporal_delta_regression,
flow_ranking, contrastive_embedding, motion_transformer, flex_classification,
velocity_regression.

Reader's guide: start with ``create_subject_split`` and
``create_dataloaders`` at the bottom to see how files enter a run. Those
functions instantiate ``SegmentationDataset``, whose mode selects one of the
three SonoRank sampling paths. Fixed-label loading and derived-label creation
are kept above the dataset so their frame conventions are shared by all modes.
"""

import json
import re
import random
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from scipy.ndimage import uniform_filter1d


# Mask color palette for 8-class segmentation (extracted from raw_seg_data visualizations)
MASK_COLOR_PALETTE = np.array([
    [  0,   0,   0],  # Class 0: Black - Background
    [128,   0,   0],  # Class 1: Dark Red
    [  0, 128,   0],  # Class 2: Dark Green
    [128, 128,   0],  # Class 3: Dark Yellow
    [  0,   0, 128],  # Class 4: Dark Blue
    [128,   0, 128],  # Class 5: Dark Magenta
    [  0, 128, 128],  # Class 6: Dark Cyan
    [128, 128, 128],  # Class 7: Gray
], dtype=np.uint8)


def apply_mask_color_palette(mask: np.ndarray) -> np.ndarray:
    """
    Convert a segmentation mask with class indices (0-7) to a colorful RGB image.

    Args:
        mask: 2D array of shape (H, W) with integer values 0-7 representing class indices

    Returns:
        RGB image of shape (H, W, 3) with uint8 values, where each class has a distinct color
    """
    # Ensure mask is 2D
    if len(mask.shape) != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    # Convert class indices to RGB colors using the palette
    # This uses advanced indexing: for each pixel, look up the RGB color from the palette
    rgb_mask = MASK_COLOR_PALETTE[mask.astype(np.int32)]

    return rgb_mask

# Finger ranking constants
FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
# Two joints per finger (used for summed angle computation in flow_ranking)
FINGER_JOINTS = {
    'index': ['joint11', 'joint12'],
    'middle': ['joint21', 'joint22'],
    'ring': ['joint31', 'joint32'],
    'pinky': ['joint41', 'joint42'],
    'thumb': ['jointt5', 'jointt6'],
}
# Main joint for each finger (used for acceleration-based ranking)
FINGER_MAIN_JOINT = {
    'index': 'joint11',     # Index finger main joint
    'middle': 'joint21',    # Middle finger main joint
    'ring': 'joint31',      # Ring finger main joint
    'pinky': 'joint41',     # Pinky finger main joint
    'thumb': 'jointt5',     # Thumb main joint
}
# Acceleration column for each finger's main joint
FINGER_ACCEL_COLUMN = {
    'index': 'joint11_der_der',
    'middle': 'joint21_der_der',
    'ring': 'joint31_der_der',
    'pinky': 'joint41_der_der',
    'thumb': 'jointt5_der_der',
}

# Dataset mode configuration
# Modes that use on-the-fly sample generation with virtual_dataset_size
# (rather than pre-generating all samples at init time)
#
# IMPORTANT: When adding a new training mode that generates samples on-the-fly
# (i.e., samples pairs/sequences in __getitem__ rather than in __init__),
# you MUST add it to this list. Otherwise __len__ will return len(self.data)
# instead of virtual_dataset_size, breaking --samples-per-epoch functionality.
ON_THE_FLY_MODES = [
    'multi_finger_temporal_ranking',
]

# Temporal ranking constants
TEMPORAL_SEQUENCE_LENGTH = 10  # 10 frames = 0.5 seconds at 20Hz
TEMPORAL_RANKING_MIN_DELTA_THRESHOLD = 0.05  # radians (~3 degrees) minimum motion for valid comparisons
TEMPORAL_RANKING_MIN_DELTA_DIFF_DEG = 3.0  # Minimum |delta_A - delta_B| for a valid pair (degrees)
TEMPORAL_RANKING_MIN_DELTA_DIFF_RAD = np.deg2rad(TEMPORAL_RANKING_MIN_DELTA_DIFF_DEG)
TEMPORAL_RANKING_NUM_BINS = 4  # Number of stratified bins for delta difference sampling
TEMPORAL_RANKING_BIN_WEIGHTS = [0.1, 0.2, 0.3, 0.4]  # Sampling probability per bin (low-to-high delta diff); favors larger differences
TEMPORAL_RANKING_ANALYSIS_SAMPLES = 10000  # Samples to analyze for bin edge computation

# Motion threshold for ranking modes (used in stratified sampling)
RANKING_MIN_MOTION_THRESHOLD_RAD = 0.05  # radians (~3 degrees) minimum motion for valid comparisons


def _find_contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True regions in a boolean array.

    Returns list of (start, end) tuples where end is exclusive.
    """
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


FIXED_LABELS_DIR = Path(__file__).resolve().parent.parent / 'labeling' / 'fixed_labels'


def load_fixed_labels(h5_stem: str, num_frames: int) -> Optional[np.ndarray]:
    """Load manually corrected flex labels from JSON.

    Returns [num_frames, 5] binary float32 array, or None if no JSON exists.
    """
    json_path = FIXED_LABELS_DIR / f'{h5_stem}.json'
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    labels = np.zeros((num_frames, 5), dtype=np.float32)
    for j, finger in enumerate(FINGER_NAMES):
        for start, end in data['fingers'].get(finger, []):
            labels[start:end, j] = 1.0
    return labels


FINGER_TO_CLASS = {name: i + 1 for i, name in enumerate(FINGER_NAMES)}


def load_dominant_labels(h5_stem: str, num_frames: int) -> Optional[np.ndarray]:
    """Load dominant finger labels from JSON.

    Returns [num_frames] int64 array (0=rest, 1-5=fingers), or None if no JSON.
    """
    json_path = FIXED_LABELS_DIR / f'{h5_stem}.json'
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    labels = np.zeros(num_frames, dtype=np.int64)
    for start, end, finger in data.get('dominant_sequence', []):
        labels[start:end] = FINGER_TO_CLASS[finger]
    return labels


def compute_flex_intervals(
    joint_angles_rad: np.ndarray,
    fps: float = 20.0,
    vel_threshold_deg_s: float = 80.0,
    smooth_window: int = 5,
    min_flex_duration_frames: int = 3,
    angle_pct_threshold: float = 0.1,
    thumb_vel_threshold_deg_s: float = 50.0,
    thumb_angle_pct_threshold: float = 0.1,
) -> np.ndarray:
    """Compute hybrid velocity-onset + angle-offset flex interval labels.

    Uses velocity to detect flexion onset, then angle position to detect
    when the finger returns to rest. This handles slow extensions that
    produce little velocity signal.

    Algorithm per finger:
    1. Compute smoothed angular velocity and find flexion onset runs
       (contiguous regions where velocity < -vel_threshold).
    2. Compute per-finger angle threshold: median - pct * (median - p5),
       where p5 is the 5th percentile (near max flexion).
    3. For each velocity onset, confirm the angle actually crosses below
       the angle threshold. If not, skip (noise).
    4. Label from velocity onset until the angle returns above the
       threshold (finger back near rest), regardless of extension velocity.

    The thumb (finger_idx=4) uses separate, lower velocity and angle pct
    thresholds since thumb flexion is typically slower than other fingers.

    Args:
        joint_angles_rad: [num_frames, 5] summed joint angles in radians.
                          Lower angle = more flexed.
        fps: Recording frame rate in Hz.
        vel_threshold_deg_s: Velocity threshold in deg/s for detecting
                             flexion onset (velocity < -threshold).
        smooth_window: Window size in frames for box smoothing of velocity.
        min_flex_duration_frames: Minimum frames for a flexion event to be kept.
        angle_pct_threshold: Fraction of the finger's range (median to p5)
                             that the angle must cross to confirm a flex.
                             0.1 = 10% of the way from median toward max flex.
        thumb_vel_threshold_deg_s: Velocity threshold in deg/s for the thumb
                                   (finger_idx=4). Lower than other fingers
                                   because thumb flexion is typically slower.
        thumb_angle_pct_threshold: Angle pct threshold for the thumb. Lower
                                   than other fingers so labels end sooner.

    Returns:
        flex_labels: [num_frames, 5] binary float32 array (1.0 = flexing).
    """
    num_frames, num_fingers = joint_angles_rad.shape
    flex_labels = np.zeros((num_frames, num_fingers), dtype=np.float32)

    # Ensure odd window for centered smoothing
    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1

    for finger_idx in range(num_fingers):
        angles_deg = np.rad2deg(joint_angles_rad[:, finger_idx])

        # Per-frame velocity in deg/s (prepend 0 for frame 0)
        raw_velocity = np.concatenate([[0.0], np.diff(angles_deg) * fps])

        # Smooth velocity
        velocity = uniform_filter1d(raw_velocity, size=w, mode='nearest')

        # Angle threshold: median - pct * (median - p5)
        # Thumb uses a separate, lower pct so labels end sooner
        median_angle = np.median(angles_deg)
        p5_angle = np.percentile(angles_deg, 5)
        finger_pct = thumb_angle_pct_threshold if finger_idx == 4 else angle_pct_threshold
        angle_thresh = median_angle - finger_pct * (median_angle - p5_angle)
        below_thresh = angles_deg < angle_thresh

        # Use lower velocity threshold for thumb (finger_idx=4)
        finger_vel_thresh = thumb_vel_threshold_deg_s if finger_idx == 4 else vel_threshold_deg_s

        # Find flexion onset runs (velocity < -threshold, i.e. angle decreasing)
        flex_runs = _find_contiguous_runs(velocity < -finger_vel_thresh)
        flex_runs = [(s, e) for s, e in flex_runs if (e - s) >= min_flex_duration_frames]

        for flex_start, _flex_end in flex_runs:
            # Confirm: angle must cross below threshold after velocity onset
            search_region = below_thresh[flex_start:]
            if not search_region.any():
                continue  # Velocity trigger but angle never crossed — skip

            confirm_idx = flex_start + np.argmax(search_region)

            # End: first frame after confirmation where angle returns above threshold
            end_region = ~below_thresh[confirm_idx:]
            if end_region.any():
                end_idx = confirm_idx + int(np.argmax(end_region))
            else:
                end_idx = num_frames  # Never returns to rest

            flex_labels[flex_start:end_idx, finger_idx] = 1.0

    return flex_labels


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for SonoRank paper.

    Supports modes: embedding, multi_finger_temporal_ranking, flex_embedding, cbmf.
    Loads preprocessed data from H5 files and applies transformations.
    """

    def __init__(
        self,
        h5_files: List[Union[str, Path]],
        transform: Optional[A.Compose] = None,
        normalize: bool = True,
        mode: str = 'multi_finger_temporal_ranking',
        min_frame_gap: int = 1,
        max_frame_gap: int = 5,
        finger_name: Optional[str] = None,
        use_masks_only: bool = False,
        flow_method: str = 'farneback',
        flow_params: Optional[Dict] = None,
        denoise: Optional[str] = None,
        denoise_strength: str = 'medium',
        split_mode: str = 'train',
        sequence_length: int = 32,
        stride: int = 1,
        samples_per_epoch: Optional[int] = None,
        samples_per_sequence: Optional[int] = None,
        inactive_fraction: float = 0.0,
        ranking_seq_len: int = 16,
        ranking_stride: int = 1,
        flex_threshold_deg: float = 10.0,
        rest_threshold_deg: float = 5.0,
        vel_threshold_deg_s: float = 80.0,
        vel_smooth_window: int = 5,
        angle_pct_threshold: float = 0.1,
        thumb_vel_threshold_deg_s: float = 50.0,
        thumb_angle_pct_threshold: float = 0.1,
        use_fixed_labels: bool = False,
        six_class: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            h5_files: List of paths to H5 files
            transform: Albumentations transform pipeline
            normalize: Whether to normalize images to [0, 1]
            mode: 'embedding', 'multi_finger_temporal_ranking', 'flex_embedding', or 'cbmf'
            min_frame_gap: Minimum frame gap for temporal pairs
            max_frame_gap: Maximum frame gap for temporal pairs
            finger_name: Finger name (not used in kept modes)
            use_masks_only: If True, use segmentation masks instead of images
            flow_method: Optical flow method (not used in kept modes)
            flow_params: Optional dict of parameters for optical flow (not used in kept modes)
            denoise: Denoising method (not used in kept modes)
            denoise_strength: Denoising strength (not used in kept modes)
            split_mode: Dataset split ('train', 'val', or 'test')
            sequence_length: Number of frames per sequence
            stride: Stride for sliding window
            ranking_seq_len: Sequence length of the frozen ranking model (flex_embedding mode)
            ranking_stride: Stride of the frozen ranking model (flex_embedding mode)
            flex_threshold_deg: (deprecated, unused)
            rest_threshold_deg: Max summed motion across all fingers for a rest window
            vel_threshold_deg_s: Velocity threshold in deg/s for flex onset detection
            vel_smooth_window: Smoothing window in frames for velocity computation
            angle_pct_threshold: Fraction of finger range for angle-based offset detection
            thumb_vel_threshold_deg_s: Velocity threshold in deg/s for the thumb
            thumb_angle_pct_threshold: Angle pct threshold for the thumb
            six_class: If True, use 6-class dominant finger labels
        """
        self.h5_files = [Path(f) for f in h5_files]
        self.transform = transform
        self.normalize = normalize
        self.mode = mode
        self.min_frame_gap = min_frame_gap
        self.max_frame_gap = max_frame_gap
        self.finger_name = finger_name
        self.use_masks_only = use_masks_only
        self.flow_method = flow_method
        self.flow_params = flow_params if flow_params is not None else {}
        self.denoise = denoise
        self.denoise_strength = denoise_strength
        self.split_mode = split_mode
        self.sequence_length = sequence_length
        self.stride = stride
        self.inactive_fraction = inactive_fraction
        self.ranking_seq_len = ranking_seq_len
        self.ranking_stride = ranking_stride
        self.flex_threshold_deg = flex_threshold_deg
        self.rest_threshold_deg = rest_threshold_deg
        self.vel_threshold_deg_s = vel_threshold_deg_s
        self.vel_smooth_window = vel_smooth_window
        self.angle_pct_threshold = angle_pct_threshold
        self.thumb_vel_threshold_deg_s = thumb_vel_threshold_deg_s
        self.thumb_angle_pct_threshold = thumb_angle_pct_threshold
        self.use_fixed_labels = use_fixed_labels
        self.six_class = six_class

        # Load all data into memory
        self.data = []
        self.metadata = []

        print(f"Loading {len(self.h5_files)} H5 files in {mode} mode...")
        for h5_file in self.h5_files:
            df = pd.read_hdf(h5_file, 'df')

            if len(df) == 0:
                print(f"Warning: {h5_file.name} is empty, skipping...")
                continue

            # Parse metadata from filename - support both playing and gestures patterns
            # Playing pattern: recordings_{subject:02d}_playing_enrollment{enrollment:02d}_multi_playing{trial:02d}.h5
            # Gestures pattern: recordings_{subject:02d}_gestures_enrollment{enrollment:02d}_{gesture_type}{number}.h5
            match_playing = re.match(r'recordings_(\d+)_playing_enrollment(\d+)_multi_playing(\d+)\.h5', h5_file.name)
            match_gestures = re.match(r'recordings_(\d+)_gestures_enrollment(\d+)_(\w+?)(\d+)\.h5', h5_file.name)

            if match_playing:
                subject_id = int(match_playing.group(1))
                enrollment = int(match_playing.group(2))
                trial = int(match_playing.group(3))
                gesture_type = None
            elif match_gestures:
                subject_id = int(match_gestures.group(1))
                enrollment = int(match_gestures.group(2))
                gesture_type = match_gestures.group(3)  # e.g., 'finger_flex', 'wrist', 'pinch'
                trial = int(match_gestures.group(4))
            else:
                print(f"Warning: {h5_file.name} does not match expected pattern, skipping...")
                continue

            # Extract metadata
            meta = {
                'subject_id': subject_id,
                'enrollment': enrollment,
                'trial': trial,
                'gesture_type': gesture_type,  # None for playing files, string for gestures files
                'num_objects': df['num_objects'].iloc[0] if 'num_objects' in df.columns else 8,
                'h5_file': h5_file.name,
            }

            # Convert PIL Images to NumPy arrays (vectorized operation)
            df['image'] = df['img'].apply(lambda x: np.array(x))

            if self.mode == 'multi_finger_temporal_ranking':
                # Multi-finger temporal ranking: load sequences with ALL 5 finger angles
                span = (self.sequence_length - 1) * self.stride + 1
                min_required_frames = 2 * span + 1
                if len(df) < min_required_frames:
                    print(f"  Warning: {h5_file.name} has {len(df)} frames, need at least {min_required_frames} for multi-finger temporal ranking, skipping...")
                    continue

                # Load joint angles for all 5 fingers (sum of both joints per finger)
                all_finger_angles = {}
                missing_fingers = []
                for finger in FINGER_NAMES:
                    joint_cols = FINGER_JOINTS[finger]
                    summed_angles = np.zeros(len(df), dtype=np.float32)
                    found_any = False
                    for jc in joint_cols:
                        if jc in df.columns:
                            summed_angles += df[jc].values.astype(np.float32)
                            found_any = True
                    if found_any:
                        all_finger_angles[finger] = summed_angles
                    else:
                        missing_fingers.append(finger)

                if missing_fingers:
                    print(f"  Warning: missing joints for {missing_fingers} in {h5_file.name}, skipping...")
                    continue

                sequence_info = {
                    'images': df['image'].values,
                    'masks': df['mask'].values if self.use_masks_only else None,
                    'all_finger_angles': all_finger_angles,
                    'h5_file_stem': h5_file.stem,
                    'num_frames': len(df),
                }
                self.data.append(sequence_info)
                self.metadata.append(meta)

            elif self.mode == 'embedding':
                # Embedding mode: load single frames for autoencoder training
                # Works with both playing and gestures files (no mask required)
                # Generate frame identifiers (vectorized)
                df['frame_id'] = [f"{h5_file.stem}_frame_{i}" for i in range(len(df))]
                df['frame_number'] = range(len(df))

                # Store frames using list comprehension (much faster than iterrows)
                frame_dicts = df[['image', 'frame_id', 'frame_number']].to_dict('records')
                self.data.extend(frame_dicts)

                # Store metadata for each frame
                self.metadata.extend([meta] * len(df))

            elif self.mode == 'flex_embedding':
                # Flex embedding mode: sliding windows of strided frames + random rest references
                # Returns per-sequence majority-vote flex labels for end-to-end fine-tuning
                # of the ranking model backbone with a simple MLP decoder.
                # Query frames are sampled at ranking_stride to match ranking model training.
                ranking_span = (self.ranking_seq_len - 1) * self.ranking_stride + 1

                min_required = ranking_span
                if len(df) < min_required:
                    print(f"  Warning: {h5_file.name} has {len(df)} frames, need at least {min_required} for flex_embedding, skipping...")
                    continue

                # Store all images for this recording
                all_images = [np.array(img) for img in df['image'].values]

                # Compute summed joint angles per finger (both joints, matching multi-finger ranking)
                joint_angles = np.zeros((len(df), 5), dtype=np.float32)
                finger_order = ['index', 'middle', 'ring', 'pinky', 'thumb']
                missing = False
                for j, finger in enumerate(finger_order):
                    joint_cols = FINGER_JOINTS[finger]
                    for jc in joint_cols:
                        if jc in df.columns:
                            joint_angles[:, j] += df[jc].values.astype(np.float32)
                        else:
                            print(f"  Warning: {jc} not found in {h5_file.name}")
                            missing = True
                if missing:
                    continue

                # Compute flex interval labels (fixed or auto-computed)
                if self.use_fixed_labels:
                    flex_labels = load_fixed_labels(h5_file.stem, len(df))
                    if flex_labels is None:
                        print(f"  SKIPPING {h5_file.name}: no fixed labels JSON found")
                        continue
                else:
                    flex_labels = compute_flex_intervals(
                        joint_angles,
                        fps=20.0,
                        vel_threshold_deg_s=self.vel_threshold_deg_s,
                        smooth_window=self.vel_smooth_window,
                        angle_pct_threshold=self.angle_pct_threshold,
                        thumb_vel_threshold_deg_s=self.thumb_vel_threshold_deg_s,
                        thumb_angle_pct_threshold=self.thumb_angle_pct_threshold,
                    )

                # Static rest frame: use frame 3 (or last available) as default rest reference.
                # This frame is duplicated ranking_seq_len times during __getitem__.
                static_rest_idx = min(3, len(df) - 1)

                # Optionally find rest candidate start frames for variety during training
                rest_threshold_rad = np.deg2rad(self.rest_threshold_deg)
                rest_candidates = []
                for start in range(len(df) - ranking_span + 1):
                    end = start + ranking_span - 1
                    total_delta = 0.0
                    for j in range(5):
                        total_delta += abs(joint_angles[end, j] - joint_angles[start, j])
                    if total_delta < rest_threshold_rad:
                        rest_candidates.append(start)

                # Load dominant labels for six-class mode
                dominant_labels = None
                if self.six_class:
                    dominant_labels = load_dominant_labels(h5_file.stem, len(df))
                    if dominant_labels is None:
                        print(f"  SKIPPING {h5_file.name}: no dominant labels JSON found (required for six_class)")
                        continue

                # Store per-recording data
                if not hasattr(self, 'recordings'):
                    self.recordings = []
                recording_idx = len(self.recordings)
                rec = {
                    'images': all_images,
                    'joint_angles': joint_angles,
                    'flex_labels': flex_labels,
                    'rest_candidates': rest_candidates,
                    'static_rest_idx': static_rest_idx,
                    'num_frames': len(df),
                }
                if dominant_labels is not None:
                    rec['dominant_labels'] = dominant_labels
                self.recordings.append(rec)

                # Build sliding window index — windows cover ranking_span raw frames
                for start in range(0, len(df) - ranking_span + 1, self.stride):
                    self.data.append({
                        'recording_idx': recording_idx,
                        'start': start,
                        'frame_id': f"{h5_file.stem}_flexemb_{start}",
                    })
                    self.metadata.append(meta)

                rest_info = f"{len(rest_candidates)} rest candidates" if rest_candidates else f"static rest frame {static_rest_idx}"
                label_src = "fixed" if self.use_fixed_labels else "auto"
                print(f"  {h5_file.name}: {len(df)} frames, {rest_info}, "
                      f"flex rate: {flex_labels.mean():.1%} [{label_src}]")

            elif self.mode == 'cbmf':
                # CBMF baseline mode: sliding windows with per-frame joint angles + flex labels.
                # No rest reference (CBMF predicts directly from the query sequence).
                from trainer.model import CBMFModel
                cbmf_joint_cols = CBMFModel.JOINT_COLUMNS

                ranking_span = (self.ranking_seq_len - 1) * self.ranking_stride + 1

                min_required = ranking_span
                if len(df) < min_required:
                    print(f"  Warning: {h5_file.name} has {len(df)} frames, need at least {min_required} for cbmf, skipping...")
                    continue

                all_images = [np.array(img) for img in df['image'].values]

                # Per-frame individual joint angles (10 joints, in radians)
                num_joints = len(cbmf_joint_cols)
                raw_joint_angles = np.zeros((len(df), num_joints), dtype=np.float32)
                missing = False
                for j, jc in enumerate(cbmf_joint_cols):
                    if jc in df.columns:
                        raw_joint_angles[:, j] = df[jc].values.astype(np.float32)
                    else:
                        print(f"  Warning: {jc} not found in {h5_file.name}")
                        missing = True
                if missing:
                    continue

                # Normalize joint angles to [0, 1] (range [0, pi])
                joint_angles_norm = raw_joint_angles / np.pi

                # Summed joint angles per finger (for flex label computation)
                joint_angles_summed = np.zeros((len(df), 5), dtype=np.float32)
                finger_order = ['index', 'middle', 'ring', 'pinky', 'thumb']
                for j, finger in enumerate(finger_order):
                    for jc in FINGER_JOINTS[finger]:
                        if jc in df.columns:
                            joint_angles_summed[:, j] += df[jc].values.astype(np.float32)

                # Compute flex labels
                if self.use_fixed_labels:
                    flex_labels = load_fixed_labels(h5_file.stem, len(df))
                    if flex_labels is None:
                        print(f"  SKIPPING {h5_file.name}: no fixed labels JSON found")
                        continue
                else:
                    flex_labels = compute_flex_intervals(
                        joint_angles_summed,
                        fps=20.0,
                        vel_threshold_deg_s=self.vel_threshold_deg_s,
                        smooth_window=self.vel_smooth_window,
                        angle_pct_threshold=self.angle_pct_threshold,
                        thumb_vel_threshold_deg_s=self.thumb_vel_threshold_deg_s,
                        thumb_angle_pct_threshold=self.thumb_angle_pct_threshold,
                    )

                if not hasattr(self, 'recordings'):
                    self.recordings = []
                recording_idx = len(self.recordings)
                self.recordings.append({
                    'images': all_images,
                    'joint_angles_norm': joint_angles_norm,  # [N, 10] normalized
                    'flex_labels': flex_labels,               # [N, 5] binary
                    'num_frames': len(df),
                })

                for start in range(0, len(df) - ranking_span + 1, self.stride):
                    self.data.append({
                        'recording_idx': recording_idx,
                        'start': start,
                        'frame_id': f"{h5_file.stem}_cbmf_{start}",
                    })
                    self.metadata.append(meta)

                label_src = "fixed" if self.use_fixed_labels else "auto"
                print(f"  {h5_file.name}: {len(df)} frames, "
                      f"flex rate: {flex_labels.mean():.1%} [{label_src}]")

        # Print loaded data summary
        if self.mode in ('flex_embedding', 'cbmf'):
            total_recordings = len(self.recordings) if hasattr(self, 'recordings') else 0
            ranking_span = (self.ranking_seq_len - 1) * self.ranking_stride + 1
            print(f"Loaded {len(self.data)} windows (span={ranking_span}, "
                  f"seq_len={self.ranking_seq_len}, ranking_stride={self.ranking_stride}, "
                  f"window_stride={self.stride}) from {total_recordings} recordings "
                  f"across {len(self.h5_files)} files")
        elif self.mode in ON_THE_FLY_MODES:
            print(f"Loaded {len(self.data)} sequences from {len(self.h5_files)} files")
            # Compute virtual dataset size (samples will be generated on-the-fly)
            total_frames = sum(seq['num_frames'] for seq in self.data)
            print(f"Total frames across all sequences: {total_frames:,}")
            # Use explicit samples_per_epoch if provided, otherwise compute from per-sequence count
            if samples_per_epoch is not None:
                self.virtual_dataset_size = samples_per_epoch
            else:
                self.samples_per_sequence_per_epoch = samples_per_sequence if samples_per_sequence is not None else 10000
                self.virtual_dataset_size = len(self.data) * self.samples_per_sequence_per_epoch
            # Round up to multiple of 6 for balanced mode sampling
            if self.mode == 'multi_finger_temporal_ranking':
                self.virtual_dataset_size = ((self.virtual_dataset_size + 5) // 6) * 6
            print(f"Virtual dataset size (samples per epoch): {self.virtual_dataset_size:,}")
        else:
            print(f"Loaded {len(self.data)} frames from {len(self.h5_files)} files")

        # Initialize stratified sampling for multi_finger_temporal_ranking mode (training only)
        if self.mode == 'multi_finger_temporal_ranking' and self.split_mode == 'train' and len(self.data) > 0:
            self._init_stratified_multi_finger_sampling()

    def _init_stratified_multi_finger_sampling(self):
        """
        Initialize stratified sampling for multi_finger_temporal_ranking mode.

        Pools delta_diff distributions across all 5 fingers to compute shared bin edges.
        Uses the same weighted bin sampling as temporal_ranking.
        """
        print("\nInitializing stratified sampling for multi_finger_temporal_ranking...")

        seq_len = self.sequence_length
        stride = self.stride
        span = (seq_len - 1) * stride + 1

        print(f"  Analyzing delta difference distribution ({TEMPORAL_RANKING_ANALYSIS_SAMPLES:,} samples across all fingers)...")
        delta_diffs = []

        samples_per_seq = max(1, TEMPORAL_RANKING_ANALYSIS_SAMPLES // len(self.data))
        for sequence in self.data:
            all_finger_angles = sequence['all_finger_angles']
            num_frames = sequence['num_frames']
            max_start = num_frames - span

            if max_start < 0:
                continue

            for _ in range(samples_per_seq):
                t1 = random.randint(0, max_start)
                t2 = random.randint(0, max_start)

                if abs(t1 - t2) < span:
                    continue

                # Pool delta_diffs across all 5 fingers
                for finger in FINGER_NAMES:
                    angles = all_finger_angles[finger]
                    delta_A = abs(angles[t1 + (seq_len - 1) * stride] - angles[t1])
                    delta_B = abs(angles[t2 + (seq_len - 1) * stride] - angles[t2])
                    delta_diff = abs(delta_A - delta_B)

                    if delta_diff >= TEMPORAL_RANKING_MIN_DELTA_DIFF_RAD:
                        delta_diffs.append(delta_diff)

        delta_diffs = np.array(delta_diffs)
        print(f"  Found {len(delta_diffs):,} valid samples for analysis (pooled across 5 fingers)")

        # Compute percentile-based bin edges
        num_bins = TEMPORAL_RANKING_NUM_BINS
        percentiles = np.linspace(0, 100, num_bins + 1)
        self.bin_edges = np.percentile(delta_diffs, percentiles)
        self.bin_edges[0] = TEMPORAL_RANKING_MIN_DELTA_DIFF_RAD
        self.bin_edges[-1] = np.inf

        print(f"  Bin edges (degrees): {[f'{np.rad2deg(e):.1f}' for e in self.bin_edges[:-1]]} + [inf]")

        self.bin_counts_current_epoch = np.zeros(num_bins, dtype=np.int64)
        self.target_bin = 0

        self.bin_weights = np.array(TEMPORAL_RANKING_BIN_WEIGHTS, dtype=np.float64)
        self.bin_weights /= self.bin_weights.sum()

        weights_str = ', '.join(f'bin{i}: {w:.0%}' for i, w in enumerate(self.bin_weights))
        print(f"  Bin weights: [{weights_str}]")
        print("  Stratified sampling initialized (weighted rejection sampling mode).\n")

    def _get_target_bin_for_sample(self):
        """
        Get the target bin for the next sample.

        Samples bin proportional to bin_weights, favoring higher bins
        (larger delta differences).

        Returns:
            int: Target bin index (0 to NUM_BINS-1)
        """
        if hasattr(self, 'bin_weights') and self.bin_weights is not None:
            # Weighted random selection
            return int(np.random.choice(len(self.bin_weights), p=self.bin_weights))
        else:
            # Round-robin fallback
            num_bins = len(self.bin_counts_current_epoch)
            target = self.target_bin
            self.target_bin = (self.target_bin + 1) % num_bins
            return target

    def _assign_bin(self, delta_diff):
        """
        Assign a delta_diff value to a bin.

        Args:
            delta_diff: Absolute difference between delta_A and delta_B

        Returns:
            int: Bin index (0 to NUM_BINS-1)
        """
        return int(np.digitize(delta_diff, self.bin_edges[1:]))

    def __len__(self):
        if self.mode in ON_THE_FLY_MODES:
            return self.virtual_dataset_size
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns depend on mode:
            embedding: dict with 'image' tensor and metadata
            multi_finger_temporal_ranking: dict with sequence_A, sequence_B, labels, active_mask, etc.
            flex_embedding: dict with image_sequence, rest_sequence, flex_labels, etc.
            cbmf: dict with image_sequence, joint_angle_targets, flex_labels, etc.
        """
        if self.mode == 'multi_finger_temporal_ranking':
            # Multi-finger temporal ranking: 6 sampling modes rotating via idx % 6
            sequence_idx = idx % len(self.data)
            sequence = self.data[sequence_idx]
            meta = self.metadata[sequence_idx]

            images = sequence['images']
            all_finger_angles = sequence['all_finger_angles']
            num_frames = sequence['num_frames']
            h5_file_stem = sequence['h5_file_stem']
            seq_len = self.sequence_length
            stride = self.stride
            span = (seq_len - 1) * stride + 1

            # Determine sampling mode: rotate through 6 modes
            modes = FINGER_NAMES + ['rest']
            sampling_mode = modes[idx % 6]

            max_start = num_frames - span
            max_attempts = 100
            use_stratified = hasattr(self, 'bin_edges') and self.bin_edges is not None

            if sampling_mode == 'rest':
                # Rest mode: require ALL 5 fingers to have small delta_diff
                for attempt in range(max_attempts):
                    t1 = random.randint(0, max_start)
                    t2 = random.randint(0, max_start)

                    if abs(t1 - t2) < span:
                        continue

                    # Check all 5 fingers have small delta_diff
                    all_rest = True
                    for finger in FINGER_NAMES:
                        angles = all_finger_angles[finger]
                        dA = abs(angles[t1 + (seq_len - 1) * stride] - angles[t1])
                        dB = abs(angles[t2 + (seq_len - 1) * stride] - angles[t2])
                        if abs(dA - dB) >= TEMPORAL_RANKING_MIN_DELTA_DIFF_RAD:
                            all_rest = False
                            break

                    if all_rest:
                        break

                    # On last few attempts, accept any non-overlapping pair
                    if attempt >= max_attempts - 10:
                        break

                primary_finger_idx = -1  # No primary finger for rest mode
            else:
                # Finger mode: stratified sampling on the primary finger
                primary_finger = sampling_mode
                primary_finger_idx = FINGER_NAMES.index(primary_finger)
                primary_angles = all_finger_angles[primary_finger]

                if use_stratified:
                    target_bin = self._get_target_bin_for_sample()

                    for attempt in range(max_attempts):
                        t1 = random.randint(0, max_start)
                        t2 = random.randint(0, max_start)

                        if abs(t1 - t2) < span:
                            continue

                        delta_A_primary = abs(primary_angles[t1 + (seq_len - 1) * stride] - primary_angles[t1])
                        delta_B_primary = abs(primary_angles[t2 + (seq_len - 1) * stride] - primary_angles[t2])
                        delta_diff = abs(delta_A_primary - delta_B_primary)

                        if delta_diff < TEMPORAL_RANKING_MIN_DELTA_DIFF_RAD:
                            continue

                        bin_idx = self._assign_bin(delta_diff)
                        if bin_idx == target_bin:
                            break

                        if attempt >= max_attempts - 10:
                            break
                else:
                    # Uniform sampling (val/test)
                    for attempt in range(max_attempts):
                        t1 = random.randint(0, max_start)
                        t2 = random.randint(0, max_start)

                        if abs(t1 - t2) < span:
                            continue

                        delta_A_primary = abs(primary_angles[t1 + (seq_len - 1) * stride] - primary_angles[t1])
                        delta_B_primary = abs(primary_angles[t2 + (seq_len - 1) * stride] - primary_angles[t2])
                        delta_diff = abs(delta_A_primary - delta_B_primary)

                        if delta_diff >= TEMPORAL_RANKING_MIN_DELTA_DIFF_RAD:
                            break

            # Compute deltas, labels, and active_mask for ALL 5 fingers
            labels = np.zeros(5, dtype=np.float32)
            deltas_A = np.zeros(5, dtype=np.float32)
            deltas_B = np.zeros(5, dtype=np.float32)
            active_mask = np.zeros(5, dtype=np.float32)

            for i, finger in enumerate(FINGER_NAMES):
                angles = all_finger_angles[finger]
                dA = abs(angles[t1 + (seq_len - 1) * stride] - angles[t1])
                dB = abs(angles[t2 + (seq_len - 1) * stride] - angles[t2])
                deltas_A[i] = dA
                deltas_B[i] = dB
                labels[i] = 1.0 if dA > dB else 0.0
                active_mask[i] = 1.0 if abs(dA - dB) >= TEMPORAL_RANKING_MIN_DELTA_DIFF_RAD else 0.0

            # Load sequences of images with stride
            if self.use_masks_only:
                masks = sequence['masks']
                images_A = [apply_mask_color_palette(masks[t1 + i * stride]).astype(np.float32) for i in range(seq_len)]
                images_B = [apply_mask_color_palette(masks[t2 + i * stride]).astype(np.float32) for i in range(seq_len)]
            else:
                images_A = [images[t1 + i * stride].copy() for i in range(seq_len)]
                images_B = [images[t2 + i * stride].copy() for i in range(seq_len)]

                if self.normalize:
                    images_A = [img.astype(np.float32) / 255.0 for img in images_A]
                    images_B = [img.astype(np.float32) / 255.0 for img in images_B]

            # Apply transformations (same ReplayCompose as temporal_ranking)
            if self.transform is not None:
                transformed_A = []
                replay_A = None
                for i, img in enumerate(images_A):
                    if i == 0:
                        result = self.transform(image=img)
                        transformed_A.append(result['image'])
                        replay_A = result.get('replay', None)
                    else:
                        if replay_A is not None:
                            result = A.ReplayCompose.replay(replay_A, image=img)
                        else:
                            result = self.transform(image=img)
                        transformed_A.append(result['image'])

                transformed_B = []
                replay_B = None
                for i, img in enumerate(images_B):
                    if i == 0:
                        result = self.transform(image=img)
                        transformed_B.append(result['image'])
                        replay_B = result.get('replay', None)
                    else:
                        if replay_B is not None:
                            result = A.ReplayCompose.replay(replay_B, image=img)
                        else:
                            result = self.transform(image=img)
                        transformed_B.append(result['image'])

                images_A = transformed_A
                images_B = transformed_B
            else:
                images_A = [torch.from_numpy(img).permute(2, 0, 1).float() for img in images_A]
                images_B = [torch.from_numpy(img).permute(2, 0, 1).float() for img in images_B]

            sequence_A = torch.stack(images_A, dim=0)
            sequence_B = torch.stack(images_B, dim=0)

            return {
                'sequence_A': sequence_A,  # [seq_len, 3, H, W]
                'sequence_B': sequence_B,  # [seq_len, 3, H, W]
                'labels': torch.from_numpy(labels),  # [5]
                'active_mask': torch.from_numpy(active_mask),  # [5]
                'primary_finger_idx': primary_finger_idx,  # int: 0-4 or -1 for rest
                'deltas_A': torch.from_numpy(deltas_A),  # [5]
                'deltas_B': torch.from_numpy(deltas_B),  # [5]
                't1': int(t1),
                't2': int(t2),
                'frame_id': f"{h5_file_stem}_mf_temporal_{idx}",
                'subject_id': meta['subject_id'],
                'enrollment': meta['enrollment'],
                'trial': meta['trial'],
            }

        # For all other modes, use pre-loaded data
        item = self.data[idx]
        meta = self.metadata[idx]

        if self.mode == 'embedding':
            image = item['image'].copy()

            # Normalize image if requested
            if self.normalize:
                image = image.astype(np.float32) / 255.0

            # Apply transformations
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Convert to tensors manually if no transform
                image = torch.from_numpy(image).permute(2, 0, 1).float()

            return {
                'image': image,
                'frame_id': item['frame_id'],
                'frame_number': item['frame_number'],
                'subject_id': meta['subject_id'],
                'enrollment': meta['enrollment'],
                'trial': meta['trial'],
                'gesture_type': meta.get('gesture_type') or 'playing',  # 'playing' for playing files, gesture type for gestures files
            }

        elif self.mode == 'flex_embedding':
            # Flex embedding mode: return strided query + strided rest + majority-vote flex label
            recording = self.recordings[item['recording_idx']]
            window_start = item['start']
            ranking_span = (self.ranking_seq_len - 1) * self.ranking_stride + 1

            # Extract query frames at ranking_stride spacing
            query_indices = list(range(window_start, window_start + ranking_span, self.ranking_stride))
            query_images = [recording['images'][i] for i in query_indices]

            # Rest reference: use static rest frame by default, rest candidates if available
            if recording['rest_candidates']:
                rest_start = random.choice(recording['rest_candidates'])
                rest_end = rest_start + ranking_span
                rest_images = recording['images'][rest_start:rest_end:self.ranking_stride]
            else:
                rest_image = recording['images'][recording['static_rest_idx']]
                rest_images = [rest_image] * self.ranking_seq_len

            # Compute labels over the full span of raw frames
            if self.six_class:
                # Six-class: mode (most frequent class) over the window
                raw_dominant = recording['dominant_labels'][window_start:window_start + ranking_span]
                dominant_class = int(np.bincount(raw_dominant).argmax())
            else:
                # Binary per-finger: majority vote
                raw_flex_labels = recording['flex_labels'][window_start:window_start + ranking_span]
                majority_labels = (raw_flex_labels.mean(axis=0) >= 0.5).astype(np.float32)

            # Process images: normalize
            processed_query = []
            for img in query_images:
                img_np = img.copy()
                if self.normalize:
                    img_np = img_np.astype(np.float32) / 255.0
                processed_query.append(img_np)

            processed_rest = []
            for img in rest_images:
                img_np = img.copy()
                if self.normalize:
                    img_np = img_np.astype(np.float32) / 255.0
                processed_rest.append(img_np)

            # Apply augmentations with ReplayCompose
            if self.transform is not None:
                # Query: apply same augmentation to all frames
                first_transformed = self.transform(image=processed_query[0])
                transformed_query = [first_transformed['image']]
                for img in processed_query[1:]:
                    replayed = A.ReplayCompose.replay(first_transformed['replay'], image=img)
                    transformed_query.append(replayed['image'])
                processed_query = transformed_query

                # Rest: apply different augmentation (fresh call)
                first_rest_transformed = self.transform(image=processed_rest[0])
                transformed_rest = [first_rest_transformed['image']]
                for img in processed_rest[1:]:
                    replayed = A.ReplayCompose.replay(first_rest_transformed['replay'], image=img)
                    transformed_rest.append(replayed['image'])
                processed_rest = transformed_rest
            else:
                processed_query = [
                    torch.from_numpy(img).permute(2, 0, 1).float()
                    for img in processed_query
                ]
                processed_rest = [
                    torch.from_numpy(img).permute(2, 0, 1).float()
                    for img in processed_rest
                ]

            image_sequence = torch.stack(processed_query)            # [T, 3, H, W]
            rest_sequence = torch.stack(processed_rest)              # [T, 3, H, W]

            result = {
                'image_sequence': image_sequence,       # [ranking_seq_len, 3, H, W]
                'rest_sequence': rest_sequence,          # [ranking_seq_len, 3, H, W]
                'frame_id': item['frame_id'],
                'subject_id': meta['subject_id'],
                'enrollment': meta['enrollment'],
                'trial': meta['trial'],
                'gesture_type': meta.get('gesture_type') or 'playing',
            }

            if self.six_class:
                result['dominant_label'] = torch.tensor(dominant_class, dtype=torch.long)  # scalar
            else:
                result['flex_labels'] = torch.from_numpy(majority_labels).float()  # [5]

            return result

        elif self.mode == 'cbmf':
            # CBMF mode: return strided query + per-frame joint angles + per-frame flex labels
            recording = self.recordings[item['recording_idx']]
            window_start = item['start']
            ranking_span = (self.ranking_seq_len - 1) * self.ranking_stride + 1

            # Extract query frames at ranking_stride spacing
            query_indices = list(range(window_start, window_start + ranking_span, self.ranking_stride))
            query_images = [recording['images'][i] for i in query_indices]

            # Per-frame joint angles for the strided frames (normalized [0,1])
            joint_targets = recording['joint_angles_norm'][query_indices]  # [T, 10]

            # Per-frame flex labels for the strided frames
            flex_per_frame = recording['flex_labels'][query_indices]  # [T, 5]

            # Majority-vote window label (for evaluation compatibility)
            raw_flex_labels = recording['flex_labels'][window_start:window_start + ranking_span]
            majority_labels = (raw_flex_labels.mean(axis=0) >= 0.5).astype(np.float32)  # [5]

            # Process images: normalize
            processed_query = []
            for img in query_images:
                img_np = img.copy()
                if self.normalize:
                    img_np = img_np.astype(np.float32) / 255.0
                processed_query.append(img_np)

            # Apply augmentations with ReplayCompose
            if self.transform is not None:
                first_transformed = self.transform(image=processed_query[0])
                transformed_query = [first_transformed['image']]
                for img in processed_query[1:]:
                    replayed = A.ReplayCompose.replay(first_transformed['replay'], image=img)
                    transformed_query.append(replayed['image'])
                processed_query = transformed_query
            else:
                processed_query = [
                    torch.from_numpy(img).permute(2, 0, 1).float()
                    for img in processed_query
                ]

            image_sequence = torch.stack(processed_query)  # [T, 3, H, W]

            return {
                'image_sequence': image_sequence,                              # [T, 3, H, W]
                'joint_angle_targets': torch.from_numpy(joint_targets).float(),  # [T, 10]
                'flex_labels_per_frame': torch.from_numpy(flex_per_frame).float(),  # [T, 5]
                'flex_labels': torch.from_numpy(majority_labels).float(),       # [5]
                'frame_id': item['frame_id'],
                'subject_id': meta['subject_id'],
                'enrollment': meta['enrollment'],
                'trial': meta['trial'],
                'gesture_type': meta.get('gesture_type') or 'playing',
            }


def get_temporal_ranking_transforms(mode='train', image_size=224, use_masks_only=False):
    """
    Get augmentation pipeline for temporal ranking with sequences.

    Uses ReplayCompose to enable applying the SAME random augmentation
    to all frames within a sequence (temporal consistency), while each
    sequence (A vs B) gets its own independent random augmentation.

    Augmentations match finger_ranking mode for consistency.

    Args:
        mode: 'train' or 'val'
        image_size: Image size (assumes square images)
        use_masks_only: If True, removes brightness/contrast augmentations for mask-only training

    Returns:
        Albumentations ReplayCompose object that supports replay for consistent sequence augmentation
    """
    if mode == 'train':
        augmentations = [
            # Geometric transformations (same as finger_ranking)
            # Keep bilinear interpolation to help model learn fuzzy boundaries and color uncertainty
            A.Rotate(
                limit=15,
                border_mode=0,
                crop_border=True,
                p=0.5
            ),

            # Resize back to target size after rotation crop
            A.Resize(height=image_size, width=image_size, interpolation=1, always_apply=True),

            # Additional zoom and shift (only zoom in to avoid black borders)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=(0.0, 0.15),
                rotate_limit=0,
                border_mode=0,
                p=0.3
            ),
        ]

        # Intensity transformations - skip for masks as they corrupt discrete color palette
        if not use_masks_only:
            augmentations.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
            ])

        # Noise and blur - apply to both images and masks
        augmentations.extend([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            # Blur and sharpening
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),

            # Convert to tensor
            ToTensorV2(),
        ])

        # Use ReplayCompose to enable replaying the same augmentation on all frames
        # within a sequence. Each sequence (A vs B) gets its own independent augmentation.
        transform = A.ReplayCompose(augmentations)
    else:  # validation/test
        # For validation, still use ReplayCompose for consistency (though deterministic)
        transform = A.ReplayCompose([
            A.Resize(height=image_size, width=image_size, interpolation=1),
            ToTensorV2(),
        ])

    return transform


def get_embedding_transforms(mode='train', image_size=224):
    """
    Get augmentation pipeline for embedding/autoencoder training.

    Uses similar augmentations to segmentation mode but without mask handling.

    Args:
        mode: 'train' or 'val'
        image_size: Image size (assumes square images)

    Returns:
        Albumentations Compose object
    """
    if mode == 'train':
        transform = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),

            # Rotation with automatic zoom to avoid black borders
            A.Rotate(
                limit=15,
                border_mode=0,
                crop_border=True,
                p=0.5
            ),

            # Resize back to target size after rotation crop
            A.Resize(height=image_size, width=image_size, interpolation=1, always_apply=True),

            # Additional zoom and shift (only zoom in to avoid black borders)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=(0.0, 0.15),
                rotate_limit=0,
                border_mode=0,
                p=0.3
            ),

            # Intensity transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            # Blur and sharpening
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),

            # Convert to tensor
            ToTensorV2(),
        ])
    else:  # validation/test
        transform = A.Compose([
            ToTensorV2(),
        ])

    return transform


def get_motion_transformer_transforms(mode='train', image_size=224):
    """
    Get augmentation pipeline for motion transformer training.

    Uses ReplayCompose to apply the same geometric augmentation consistently
    across all frames in a sequence. This is critical for temporal consistency.
    Similar augmentations to embedding mode but with ReplayCompose for sequences.

    Args:
        mode: 'train' or 'val'
        image_size: Image size (assumes square images)

    Returns:
        Albumentations ReplayCompose object
    """
    if mode == 'train':
        transform = A.ReplayCompose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),

            # Rotation with automatic zoom to avoid black borders
            A.Rotate(
                limit=15,
                border_mode=0,
                crop_border=True,
                p=0.5
            ),

            # Resize back to target size after rotation crop
            A.Resize(height=image_size, width=image_size, interpolation=1),

            # Additional zoom and shift (only zoom in to avoid black borders)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=(0.0, 0.15),
                rotate_limit=0,
                border_mode=0,
                p=0.3
            ),

            # Intensity transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(p=0.3),

            # Blur and sharpening
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),

            # Convert to tensor
            ToTensorV2(),
        ])
    else:  # validation/test
        transform = A.ReplayCompose([
            A.Resize(height=image_size, width=image_size, interpolation=1),
            ToTensorV2(),
        ])

    return transform


def load_h5_files(
    data_dir: Union[str, Path],
    subject_ids: Optional[List[int]] = None,
    exclude_empty: bool = True,
    session_types: Optional[List[str]] = None,
) -> List[Path]:
    """
    Load H5 file paths from directory.

    Args:
        data_dir: Directory containing H5 files
        subject_ids: Optional list of subject IDs to include
        exclude_empty: Whether to exclude empty H5 files
        session_types: Optional list of session types to include.
            Options: 'playing', 'finger_flex', 'wrist', 'pinch'
            If None, all session types are included.

    Returns:
        List of H5 file paths
    """
    data_dir = Path(data_dir)

    # Get all H5 files
    h5_files = sorted(data_dir.glob('recordings_*.h5'))

    # Filter by session type and subject ID
    filtered_files = []
    for h5_file in h5_files:
        # Parse filename - support both playing and gestures patterns
        # Playing: recordings_{subject:02d}_playing_enrollment{enrollment:02d}_multi_playing{trial:02d}.h5
        # Gestures: recordings_{subject:02d}_gestures_enrollment{enrollment:02d}_{gesture_type}{number}.h5
        match_playing = re.match(r'recordings_(\d+)_playing_enrollment(\d+)_multi_playing(\d+)\.h5', h5_file.name)
        match_gestures = re.match(r'recordings_(\d+)_gestures_enrollment(\d+)_(\w+?)(\d+)\.h5', h5_file.name)

        if match_playing:
            subject_id = int(match_playing.group(1))
            file_session_type = 'playing'
        elif match_gestures:
            subject_id = int(match_gestures.group(1))
            file_session_type = match_gestures.group(3)  # e.g., 'finger_flex', 'wrist', 'pinch'
        else:
            continue  # Skip files that don't match either pattern

        # Check session type filter
        if session_types is not None and file_session_type not in session_types:
            continue

        # Check subject ID filter
        if subject_ids is not None and subject_id not in subject_ids:
            continue

        filtered_files.append(h5_file)

    h5_files = filtered_files

    # Exclude empty files if requested
    if exclude_empty:
        non_empty_files = []
        for h5_file in h5_files:
            df = pd.read_hdf(h5_file, 'df')
            if len(df) > 0:
                non_empty_files.append(h5_file)
        h5_files = non_empty_files

    return h5_files


def create_subject_split(
    data_dir: Union[str, Path],
    train_subjects: List[int],
    val_subjects: List[int],
    test_subjects: Optional[List[int]] = None,
    session_types: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    """
    Create train/val/test splits based on subject IDs.

    Args:
        data_dir: Directory containing H5 files
        train_subjects: List of subject IDs for training
        val_subjects: List of subject IDs for validation
        test_subjects: Optional list of subject IDs for testing
        session_types: Optional list of session types to include.
            Options: 'playing', 'finger_flex', 'wrist', 'pinch'
            If None, all session types are included.

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' file lists
    """
    splits = {
        'train': load_h5_files(data_dir, subject_ids=train_subjects, session_types=session_types),
        'val': load_h5_files(data_dir, subject_ids=val_subjects, session_types=session_types),
    }

    if test_subjects is not None:
        splits['test'] = load_h5_files(data_dir, subject_ids=test_subjects, session_types=session_types)

    # Print split information
    print("\nDataset splits:")
    for split_name, files in splits.items():
        num_frames = 0
        for h5_file in files:
            df = pd.read_hdf(h5_file, 'df')
            num_frames += len(df)

        print(f"  {split_name:5s}: {len(files):2d} files, {num_frames:5d} frames")

    return splits


def create_dataloaders(
    splits: Dict[str, List[Path]],
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    normalize: bool = True,
    mode: str = 'multi_finger_temporal_ranking',
    min_frame_gap: int = 1,
    max_frame_gap: int = 5,
    finger_name: Optional[str] = None,
    use_masks_only: bool = False,
    flow_method: str = 'farneback',
    flow_params: Optional[Dict] = None,
    denoise: Optional[str] = None,
    denoise_strength: str = 'medium',
    sequence_length: int = 32,
    stride: int = 1,
    weighted_sampling: bool = False,
    samples_per_epoch: Optional[int] = None,
    samples_per_sequence: Optional[int] = None,
    ranking_seq_len: int = 16,
    ranking_stride: int = 1,
    flex_threshold_deg: float = 10.0,
    rest_threshold_deg: float = 5.0,
    vel_threshold_deg_s: float = 80.0,
    vel_smooth_window: int = 5,
    angle_pct_threshold: float = 0.1,
    thumb_vel_threshold_deg_s: float = 50.0,
    thumb_angle_pct_threshold: float = 0.1,
    use_fixed_labels: bool = False,
    six_class: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits.

    Args:
        splits: Dictionary with file lists for each split
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size (for augmentations)
        normalize: Whether to normalize images
        mode: 'embedding', 'multi_finger_temporal_ranking', 'flex_embedding', or 'cbmf'
        min_frame_gap: Minimum frame gap for temporal pairs
        max_frame_gap: Maximum frame gap for temporal pairs
        finger_name: Finger name (not used in kept modes)
        use_masks_only: If True, use segmentation masks instead of images
        flow_method: Optical flow method (not used in kept modes)
        flow_params: Optional dict of parameters for optical flow (not used in kept modes)
        denoise: Denoising method (not used in kept modes)
        denoise_strength: Denoising strength (not used in kept modes)
        sequence_length: Number of frames per sequence
        stride: Stride for sliding window
        weighted_sampling: If True, use weighted sampling for training

    Returns:
        Dictionary of DataLoaders
    """
    # Validate that samples_per_epoch or samples_per_sequence is provided for on-the-fly modes
    if mode in ON_THE_FLY_MODES and samples_per_epoch is None and samples_per_sequence is None:
        raise ValueError(
            f"samples_per_epoch or samples_per_sequence is required for mode '{mode}' (on-the-fly generation mode). "
            f"Please specify --samples-per-epoch or --samples-per-sequence argument in the training script. "
            f"Recommended values: 500-5000 for training, 100-500 for quick testing."
        )

    dataloaders = {}

    for split_name, files in splits.items():
        # Get appropriate transform based on mode
        split_mode = 'train' if split_name == 'train' else 'val'

        if mode == 'multi_finger_temporal_ranking':
            # Same transforms as temporal_ranking (ReplayCompose for temporal consistency)
            transform = get_temporal_ranking_transforms(mode=split_mode, image_size=image_size, use_masks_only=use_masks_only)
        elif mode == 'embedding':
            transform = get_embedding_transforms(mode=split_mode, image_size=image_size)
        elif mode == 'flex_embedding':
            transform = get_motion_transformer_transforms(mode=split_mode, image_size=image_size)
        elif mode == 'cbmf':
            transform = get_motion_transformer_transforms(mode=split_mode, image_size=image_size)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create dataset
        dataset = SegmentationDataset(
            h5_files=files,
            transform=transform,
            normalize=normalize,
            mode=mode,
            min_frame_gap=min_frame_gap,
            max_frame_gap=max_frame_gap,
            finger_name=finger_name,
            use_masks_only=use_masks_only,
            flow_method=flow_method,
            flow_params=flow_params,
            denoise=denoise,
            denoise_strength=denoise_strength,
            split_mode=split_mode,
            sequence_length=sequence_length,
            stride=stride,
            samples_per_epoch=samples_per_epoch,
            samples_per_sequence=samples_per_sequence,
            ranking_seq_len=ranking_seq_len,
            ranking_stride=ranking_stride,
            flex_threshold_deg=flex_threshold_deg,
            rest_threshold_deg=rest_threshold_deg,
            vel_threshold_deg_s=vel_threshold_deg_s,
            vel_smooth_window=vel_smooth_window,
            angle_pct_threshold=angle_pct_threshold,
            thumb_vel_threshold_deg_s=thumb_vel_threshold_deg_s,
            thumb_angle_pct_threshold=thumb_angle_pct_threshold,
            use_fixed_labels=use_fixed_labels,
            six_class=six_class,
        )

        # Create dataloader
        shuffle = (split_name == 'train')
        sampler = None

        # For multi_finger_temporal_ranking, use sequential indices (no shuffle)
        # to ensure balanced sampling mode distribution per batch via idx % 6
        if mode == 'multi_finger_temporal_ranking':
            shuffle = False

        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle if sampler is None else False,
            'sampler': sampler,
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': (split_name == 'train'),
        }

        # Only add multiprocessing args if num_workers > 0
        if num_workers > 0:
            dataloader_kwargs['persistent_workers'] = True
            dataloader_kwargs['prefetch_factor'] = 2

        dataloader = DataLoader(dataset, **dataloader_kwargs)

        dataloaders[split_name] = dataloader
        print(f"\n{split_name} dataloader: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloaders

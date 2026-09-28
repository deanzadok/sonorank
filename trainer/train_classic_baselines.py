"""
Training script for classic baselines (sklearn-based).

Implements three external baselines from:
  Zadok et al., "Inferring Fine Finger Motions for Prosthetic Control",
  ASME J. Biomech. Eng., 2025.

Methods:
  - castellini: Castellini & Passig 2011 (spherical patch features + Ridge regression)
  - huang: Huang & Liu 2016 (column pixel sampling + linear fitting + MLP)
  - mcintosh: McIntosh et al. 2017 / EchoFlex (optical flow + avg pooling + MLP)

All methods use sklearn classifiers (no GPU needed) and produce per-finger
binary flex predictions compatible with the evaluation pipeline.

Reader's guide: recording and window construction are shared, then the code
branches into method-specific feature extraction. Every branch returns raw
per-frame probabilities through ``_save_raw_predictions`` so the downstream
event comparison remains identical across methods.
"""

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from trainer.load import (
    compute_flex_intervals, load_fixed_labels,
    FINGER_NAMES, FINGER_JOINTS,
    create_subject_split,
)
from trainer.utils import SummaryLogger


# ============================================================================
# Data loading
# ============================================================================

def load_recording(h5_file, use_fixed_labels=True,
                   vel_threshold_deg_s=80.0, vel_smooth_window=5,
                   angle_pct_threshold=0.1,
                   thumb_vel_threshold_deg_s=50.0,
                   thumb_angle_pct_threshold=0.1):
    """Load images and flex labels from a single H5 file.

    Returns dict with images [N, 224, 224, 3], flex_labels [N, 5], etc.
    Returns None if the recording should be skipped.
    """
    h5_file = Path(h5_file)
    df = pd.read_hdf(h5_file, 'df')
    if len(df) == 0:
        return None

    images = np.stack([np.array(img) for img in df['img'].values])  # [N, H, W, 3]

    # Summed joint angles for flex label computation
    joint_angles_summed = np.zeros((len(df), 5), dtype=np.float32)
    for j, finger in enumerate(FINGER_NAMES):
        for jc in FINGER_JOINTS[finger]:
            if jc in df.columns:
                joint_angles_summed[:, j] += df[jc].values.astype(np.float32)

    if use_fixed_labels:
        flex_labels = load_fixed_labels(h5_file.stem, len(df))
        if flex_labels is None:
            return None
    else:
        flex_labels = compute_flex_intervals(
            joint_angles_summed, fps=20.0,
            vel_threshold_deg_s=vel_threshold_deg_s,
            smooth_window=vel_smooth_window,
            angle_pct_threshold=angle_pct_threshold,
            thumb_vel_threshold_deg_s=thumb_vel_threshold_deg_s,
            thumb_angle_pct_threshold=thumb_angle_pct_threshold,
        )

    return {
        'images': images,
        'flex_labels': flex_labels,
        'num_frames': len(df),
        'h5_stem': h5_file.stem,
        'h5_path': str(h5_file),
    }


def load_split_data(h5_files, **label_kwargs):
    """Load all recordings for a data split."""
    recordings = []
    total_frames = 0
    skipped = 0
    for h5_file in h5_files:
        rec = load_recording(h5_file, **label_kwargs)
        if rec is None:
            skipped += 1
            continue
        recordings.append(rec)
        total_frames += rec['num_frames']
    print(f"  Loaded {len(recordings)} recordings ({total_frames} frames, skipped {skipped})")
    return recordings


# ============================================================================
# Feature extraction: Castellini 2011
# ============================================================================

def precompute_castellini_projections(image_size=224, sphere_radius=9, sphere_gap=4):
    """Precompute pseudo-inverse projection matrices for all 181 spheres.

    Returns list of (proj_matrix [3, M], pixel_indices [M, 2]) tuples.
    Features per sphere: [coef_x, coef_y, intercept].
    """
    num_along = 10
    side_gap = (image_size - (sphere_radius * 2 * num_along +
                (num_along - 1) * sphere_gap)) // 2

    outer = [side_gap + sphere_radius + (2 * sphere_radius + sphere_gap) * i
             for i in range(num_along)]
    inner = [side_gap + 2 * sphere_radius + side_gap // 2 +
             (2 * sphere_radius + sphere_gap) * i
             for i in range(num_along - 1)]

    centers = [(x, y) for x in outer for y in outer]
    centers += [(x, y) for x in inner for y in inner]

    projections = []
    for cx, cy in centers:
        indices = []
        for i in range(max(0, cx - sphere_radius), min(image_size, cx + sphere_radius + 1)):
            for j in range(max(0, cy - sphere_radius), min(image_size, cy + sphere_radius + 1)):
                if (i - cx) ** 2 + (j - cy) ** 2 <= sphere_radius ** 2:
                    indices.append([i, j])
        indices = np.array(indices)
        X = np.column_stack([indices, np.ones(len(indices))])
        proj = np.linalg.pinv(X)  # [3, M]
        projections.append((proj, indices))

    return projections


def extract_castellini_features(images, projections):
    """Extract Castellini features: per-sphere linear regression coefficients.

    Args:
        images: [N, H, W, 3] uint8 RGB
        projections: from precompute_castellini_projections()

    Returns:
        features: [N, num_spheres * 3] float64
    """
    gray = np.mean(images.astype(np.float64), axis=-1) / 255.0  # [N, H, W]
    N = len(gray)
    num_spheres = len(projections)
    features = np.zeros((N, num_spheres * 3), dtype=np.float64)

    for s_idx, (proj, indices) in enumerate(projections):
        intensities = gray[:, indices[:, 0], indices[:, 1]]  # [N, M]
        coeffs = (proj @ intensities.T).T  # [N, 3]
        features[:, s_idx * 3:(s_idx + 1) * 3] = coeffs

    return features


# ============================================================================
# Feature extraction: Huang 2016
# ============================================================================

def extract_huang_features_single(image, num_lines=5, window_size=20):
    """Extract Huang features for a single grayscale image.

    Samples num_lines uniformly-spaced vertical columns, divides each into
    segments of window_size pixels, fits a linear model per segment.

    Returns: [num_lines * num_segments * 2] float64
    """
    H, W = image.shape
    num_segments = H // window_size
    col_step = W // (num_lines + 1)
    col_indices = [col_step * (i + 1) for i in range(num_lines)]

    x_seg = np.arange(window_size, dtype=np.float64)
    x_sum = x_seg.sum()
    x2_sum = (x_seg ** 2).sum()
    denom = window_size * x2_sum - x_sum ** 2

    features = np.zeros(num_lines * num_segments * 2, dtype=np.float64)
    idx = 0
    for col in col_indices:
        column = image[:, col].astype(np.float64)
        for seg in range(num_segments):
            y = column[seg * window_size:(seg + 1) * window_size]
            y_sum = y.sum()
            xy_sum = (y * x_seg).sum()
            slope = (window_size * xy_sum - x_sum * y_sum) / denom
            intercept = (y_sum - slope * x_sum) / window_size
            features[idx] = slope
            features[idx + 1] = intercept
            idx += 2

    return features


def extract_huang_features_window(images_window, num_lines=5, window_size=20):
    """Extract Huang features for a multi-frame window, concatenated across frames.

    Args:
        images_window: [T, H, W, 3] uint8 RGB

    Returns: 1D array of concatenated per-frame features
    """
    feats = []
    for img in images_window:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feats.append(extract_huang_features_single(gray, num_lines, window_size))
    return np.concatenate(feats)


# ============================================================================
# Feature extraction: McIntosh 2017 (EchoFlex)
# ============================================================================

def extract_mcintosh_features_window(images_window, pool_size=20):
    """Extract McIntosh/EchoFlex features for a multi-frame window.

    Computes optical flow between adjacent frames, average-pools each flow
    magnitude field, and concatenates.

    Args:
        images_window: [T, H, W, 3] uint8 RGB

    Returns: 1D array of pooled flow features
    """
    T, H, W, _ = images_window.shape
    pooled_h = H // pool_size
    pooled_w = W // pool_size
    all_pooled = []

    for t in range(T - 1):
        prev = cv2.cvtColor(images_window[t], cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(images_window[t + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 5, 35, 6, 7, 1.5, 0)
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        pooled = np.zeros((pooled_h, pooled_w), dtype=np.float64)
        for i in range(pooled_h):
            for j in range(pooled_w):
                pooled[i, j] = magnitude[
                    i * pool_size:(i + 1) * pool_size,
                    j * pool_size:(j + 1) * pool_size
                ].mean()
        all_pooled.append(pooled.flatten())

    return np.concatenate(all_pooled)


# ============================================================================
# Window creation for multi-frame methods
# ============================================================================

def create_windows(recordings, seq_len, seq_stride, window_stride=1):
    """Create sliding window indices.

    Returns list of (recording_idx, frame_indices) tuples.
    """
    span = (seq_len - 1) * seq_stride + 1
    windows = []
    for rec_idx, rec in enumerate(recordings):
        for start in range(0, rec['num_frames'] - span + 1, window_stride):
            frame_indices = [start + i * seq_stride for i in range(seq_len)]
            windows.append((rec_idx, frame_indices))
    return windows


def window_label(rec, frame_indices):
    """Compute majority-vote flex label for a window."""
    labels = rec['flex_labels'][frame_indices]  # [T, 5]
    return (labels.mean(axis=0) >= 0.5).astype(np.float32)


# ============================================================================
# Training and evaluation
# ============================================================================

def train_and_evaluate(args):
    """Main training function."""
    method = args.method
    print("=" * 80)
    print(f"CLASSIC BASELINE: {method.upper()}")
    print("=" * 80)
    print(f"Method: {method}")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:35s}: {v}")
    print("=" * 80)

    # Session type filtering
    session_types = None
    if args.exclude_playing or args.exclude_pinch or args.exclude_wrist:
        all_types = ['finger_flex', 'pinch', 'wrist', 'multi_playing']
        if args.exclude_playing:
            all_types.remove('multi_playing')
        if args.exclude_pinch:
            all_types.remove('pinch')
        if args.exclude_wrist:
            all_types.remove('wrist')
        session_types = all_types
        print(f"Session filtering: {session_types}")

    # Create splits
    train_subjects = args.train_subjects
    val_subjects = args.val_subjects
    test_subjects = args.test_subjects
    print(f"\nTrain subjects: {train_subjects}")
    print(f"Val subjects: {val_subjects}")
    print(f"Test subjects: {test_subjects}")

    splits = create_subject_split(
        data_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        session_types=session_types,
    )

    # Output directory
    base_output_dir = Path(args.output_dir)
    if args.exp_name:
        run_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_str = '_'.join(str(s) for s in test_subjects)
        run_name = f"{method}_{timestamp}_test{test_str}"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = SummaryLogger(output_dir)
    summary.log_config({
        'method': method,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'use_fixed_labels': args.use_fixed_labels,
    })

    # Save config
    config = vars(args).copy()
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Label kwargs
    label_kwargs = dict(
        use_fixed_labels=args.use_fixed_labels,
        vel_threshold_deg_s=args.vel_threshold_deg_s,
        vel_smooth_window=args.vel_smooth_window,
        angle_pct_threshold=args.angle_pct_threshold,
        thumb_vel_threshold_deg_s=args.thumb_vel_threshold_deg_s,
        thumb_angle_pct_threshold=args.thumb_angle_pct_threshold,
    )

    # Load data
    print("\nLoading train data...")
    train_recs = load_split_data(splits['train'], **label_kwargs)
    print("Loading test data...")
    test_recs = load_split_data(splits['test'], **label_kwargs)

    # ----------------------------------------------------------------
    # Extract features and train
    # ----------------------------------------------------------------
    t0 = time.time()

    if method == 'castellini':
        X_train, y_train, X_test, y_test, test_pred_info = _train_castellini(
            train_recs, test_recs, args)
    elif method == 'huang':
        X_train, y_train, X_test, y_test, test_pred_info = _train_multiframe(
            train_recs, test_recs, args, feature_fn='huang')
    elif method == 'mcintosh':
        X_train, y_train, X_test, y_test, test_pred_info = _train_multiframe(
            train_recs, test_recs, args, feature_fn='mcintosh')
    else:
        raise ValueError(f"Unknown method: {method}")

    # Train classifiers
    print(f"\nTraining classifiers...")
    if method == 'castellini':
        classifiers = []
        for finger_idx in range(5):
            clf = Ridge(alpha=args.ridge_alpha)
            clf.fit(X_train, y_train[:, finger_idx])
            classifiers.append(clf)
            print(f"  {FINGER_NAMES[finger_idx]}: trained")
    else:
        def _fit_mlp(finger_idx):
            clf = MLPClassifier(
                hidden_layer_sizes=tuple(args.mlp_hidden_sizes),
                max_iter=args.mlp_max_iter,
                random_state=42,
            )
            clf.fit(X_train, y_train[:, finger_idx])
            return clf

        classifiers = Parallel(n_jobs=5)(
            delayed(_fit_mlp)(i) for i in range(5)
        )
        for finger_idx in range(5):
            print(f"  {FINGER_NAMES[finger_idx]}: trained")

    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    preds_test = np.zeros_like(y_test)
    probs_test = np.zeros_like(y_test, dtype=np.float64)
    for finger_idx in range(5):
        if method == 'castellini':
            raw = classifiers[finger_idx].predict(X_test)
            probs_test[:, finger_idx] = np.clip(raw, 0, 1)
            preds_test[:, finger_idx] = (probs_test[:, finger_idx] >= 0.5).astype(float)
        else:
            if hasattr(classifiers[finger_idx], 'predict_proba'):
                prob = classifiers[finger_idx].predict_proba(X_test)
                if prob.shape[1] == 2:
                    probs_test[:, finger_idx] = prob[:, 1]
                else:
                    probs_test[:, finger_idx] = prob[:, 0]
            else:
                probs_test[:, finger_idx] = classifiers[finger_idx].predict(X_test).astype(float)
            preds_test[:, finger_idx] = (probs_test[:, finger_idx] >= 0.5).astype(float)

    # Per-finger metrics
    print("\nTest Results:")
    per_finger_f1 = []
    per_finger_pre = []
    per_finger_rec = []
    for i, name in enumerate(FINGER_NAMES):
        f1 = f1_score(y_test[:, i], preds_test[:, i], zero_division=0)
        pre = precision_score(y_test[:, i], preds_test[:, i], zero_division=0)
        rec = recall_score(y_test[:, i], preds_test[:, i], zero_division=0)
        per_finger_f1.append(f1)
        per_finger_pre.append(pre)
        per_finger_rec.append(rec)
        print(f"  {name:8s}: F1={f1:.4f}  Pre={pre:.4f}  Rec={rec:.4f}")

    macro_f1 = np.mean(per_finger_f1)
    print(f"  Macro F1: {macro_f1:.4f}")

    summary.log_message(f"\nResults ({method}):")
    for i, name in enumerate(FINGER_NAMES):
        summary.log_message(f"  {name}: F1={per_finger_f1[i]:.4f} Pre={per_finger_pre[i]:.4f} Rec={per_finger_rec[i]:.4f}")
    summary.log_message(f"  Macro F1: {macro_f1:.4f}")
    summary.log_message(f"  Training time: {train_time:.1f}s")

    # Save classifier
    with open(output_dir / 'classifier.pkl', 'wb') as f:
        pickle.dump(classifiers, f)

    # Save raw predictions for evaluate_methods_fair.py
    _save_raw_predictions(
        output_dir, test_recs, classifiers, method, args, test_pred_info)

    print(f"\nOutputs saved to: {output_dir}")
    print("DONE")


def _train_castellini(train_recs, test_recs, args):
    """Extract Castellini features and prepare training data."""
    print("\nPrecomputing sphere projections...")
    projections = precompute_castellini_projections(
        image_size=args.image_size,
        sphere_radius=args.sphere_radius,
        sphere_gap=args.sphere_gap,
    )
    num_spheres = len(projections)
    print(f"  {num_spheres} spheres, {num_spheres * 3} features per frame")

    print("Extracting train features...")
    all_train_feats = []
    all_train_labels = []
    for rec in train_recs:
        feats = extract_castellini_features(rec['images'], projections)
        all_train_feats.append(feats)
        all_train_labels.append(rec['flex_labels'])
    X_train = np.vstack(all_train_feats)
    y_train = np.vstack(all_train_labels)
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    print("Extracting test features...")
    all_test_feats = []
    all_test_labels = []
    for rec in test_recs:
        feats = extract_castellini_features(rec['images'], projections)
        all_test_feats.append(feats)
        all_test_labels.append(rec['flex_labels'])
    X_test = np.vstack(all_test_feats)
    y_test = np.vstack(all_test_labels)
    print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # For raw predictions, store per-recording frame counts
    test_pred_info = {
        'type': 'single_frame',
        'projections': projections,
    }
    return X_train, y_train, X_test, y_test, test_pred_info


def _train_multiframe(train_recs, test_recs, args, feature_fn):
    """Extract multi-frame features (Huang or McIntosh) and prepare training data."""
    seq_len = args.sequence_length
    seq_stride = args.sequence_stride
    span = (seq_len - 1) * seq_stride + 1

    print(f"\nCreating train windows (seq_len={seq_len}, stride={seq_stride}, span={span})...")
    train_windows = create_windows(train_recs, seq_len, seq_stride,
                                   window_stride=args.window_stride)
    print(f"  {len(train_windows)} train windows")

    print(f"Extracting {feature_fn} features for train windows...")
    t0 = time.time()

    n_jobs = args.n_jobs

    # Pre-compute optical flow for McIntosh to avoid redundant computation
    if feature_fn == 'mcintosh':
        print("  Pre-computing optical flow for train recordings...")
        train_flows = _precompute_flows(train_recs, n_jobs=n_jobs)

    if feature_fn == 'huang':
        def _extract_train(rec_idx, frame_indices):
            rec = train_recs[rec_idx]
            imgs = rec['images'][frame_indices]
            feat = extract_huang_features_window(imgs, args.num_lines, args.window_size)
            label = window_label(rec, frame_indices)
            return feat, label

        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_extract_train)(rec_idx, frame_indices)
            for rec_idx, frame_indices in train_windows
        )
        X_train_list = [r[0] for r in results]
        y_train_list = [r[1] for r in results]
    else:
        # McIntosh: precomputed flows are dicts, extract sequentially
        X_train_list = []
        y_train_list = []
        for wi, (rec_idx, frame_indices) in enumerate(train_windows):
            rec = train_recs[rec_idx]
            feat = _extract_mcintosh_from_precomputed(
                train_flows[rec_idx], frame_indices, args.pool_size,
                rec['images'].shape[1], rec['images'].shape[2])
            X_train_list.append(feat)
            y_train_list.append(window_label(rec, frame_indices))
            if (wi + 1) % 5000 == 0:
                print(f"  {wi + 1}/{len(train_windows)} windows...")

    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features ({time.time()-t0:.1f}s)")

    print(f"Creating test windows...")
    test_windows = create_windows(test_recs, seq_len, seq_stride,
                                  window_stride=args.window_stride)
    print(f"  {len(test_windows)} test windows")

    print(f"Extracting {feature_fn} features for test windows...")
    t0 = time.time()

    if feature_fn == 'mcintosh':
        print("  Pre-computing optical flow for test recordings...")
        test_flows = _precompute_flows(test_recs, n_jobs=n_jobs)

    if feature_fn == 'huang':
        def _extract_test(rec_idx, frame_indices):
            rec = test_recs[rec_idx]
            imgs = rec['images'][frame_indices]
            feat = extract_huang_features_window(imgs, args.num_lines, args.window_size)
            label = window_label(rec, frame_indices)
            return feat, label

        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_extract_test)(rec_idx, frame_indices)
            for rec_idx, frame_indices in test_windows
        )
        X_test_list = [r[0] for r in results]
        y_test_list = [r[1] for r in results]
    else:
        X_test_list = []
        y_test_list = []
        for wi, (rec_idx, frame_indices) in enumerate(test_windows):
            rec = test_recs[rec_idx]
            feat = _extract_mcintosh_from_precomputed(
                test_flows[rec_idx], frame_indices, args.pool_size,
                rec['images'].shape[1], rec['images'].shape[2])
            X_test_list.append(feat)
            y_test_list.append(window_label(rec, frame_indices))
            if (wi + 1) % 5000 == 0:
                print(f"  {wi + 1}/{len(test_windows)} windows...")

    X_test = np.vstack(X_test_list)
    y_test = np.vstack(y_test_list)
    print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features ({time.time()-t0:.1f}s)")

    test_pred_info = {
        'type': 'multi_frame',
        'feature_fn': feature_fn,
        'seq_len': seq_len,
        'seq_stride': seq_stride,
    }
    return X_train, y_train, X_test, y_test, test_pred_info


def _compute_single_flow(prev_img, curr_img):
    """Compute optical flow magnitude between two RGB frames."""
    prev = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    curr = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None, 0.5, 5, 35, 6, 7, 1.5, 0)
    return np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)


def _precompute_flows(recordings, n_jobs=-1):
    """Pre-compute optical flow between all consecutive frame pairs."""
    all_flows = []
    for rec in recordings:
        n_frames = rec['num_frames']
        magnitudes = Parallel(n_jobs=n_jobs)(
            delayed(_compute_single_flow)(rec['images'][t], rec['images'][t + 1])
            for t in range(n_frames - 1)
        )
        flows = {t: mag for t, mag in enumerate(magnitudes)}
        all_flows.append(flows)
    return all_flows


def _extract_mcintosh_from_precomputed(flows_dict, frame_indices, pool_size, H, W):
    """Extract McIntosh features using precomputed flow magnitudes."""
    pooled_h = H // pool_size
    pooled_w = W // pool_size
    all_pooled = []

    for t in range(len(frame_indices) - 1):
        f0 = frame_indices[t]
        f1 = frame_indices[t + 1]
        # Average flow magnitudes between the subsampled frames
        if f1 - f0 == 1 and f0 in flows_dict:
            magnitude = flows_dict[f0]
        else:
            # For strided sequences, average the flows across the gap
            mags = []
            for k in range(f0, f1):
                if k in flows_dict:
                    mags.append(flows_dict[k])
            if mags:
                magnitude = np.mean(mags, axis=0)
            else:
                magnitude = np.zeros((H, W), dtype=np.float64)

        pooled = np.zeros((pooled_h, pooled_w), dtype=np.float64)
        for i in range(pooled_h):
            for j in range(pooled_w):
                pooled[i, j] = magnitude[
                    i * pool_size:(i + 1) * pool_size,
                    j * pool_size:(j + 1) * pool_size
                ].mean()
        all_pooled.append(pooled.flatten())

    return np.concatenate(all_pooled)


# ============================================================================
# Raw predictions for evaluate_methods_fair.py
# ============================================================================

def _save_raw_predictions(output_dir, test_recs, classifiers, method, args, pred_info):
    """Save per-frame probability signals for evaluation pipeline."""
    event_dir = output_dir / 'event_detection'
    event_dir.mkdir(exist_ok=True)

    if pred_info['type'] == 'single_frame':
        projections = pred_info['projections']
        for rec in test_recs:
            feats = extract_castellini_features(rec['images'], projections)
            prob_signals = {}
            gt_labels = {}
            for finger_idx, name in enumerate(FINGER_NAMES):
                raw = classifiers[finger_idx].predict(feats)
                prob_signals[name] = np.clip(raw, 0, 1).tolist()
                gt_labels[name] = rec['flex_labels'][:, finger_idx].tolist()

            raw_data = {
                'h5_file': rec['h5_path'],
                'num_frames': rec['num_frames'],
                'method': method,
                'prob_signals': prob_signals,
                'gt_flex_labels': gt_labels,
            }
            out_path = event_dir / f"raw_predictions_{rec['h5_stem']}.json"
            with open(out_path, 'w') as f:
                json.dump(raw_data, f)

    elif pred_info['type'] == 'multi_frame':
        seq_len = pred_info['seq_len']
        seq_stride = pred_info['seq_stride']
        span = (seq_len - 1) * seq_stride + 1

        if pred_info['feature_fn'] == 'mcintosh':
            test_flows = _precompute_flows(test_recs)

        for rec_idx, rec in enumerate(test_recs):
            num_frames = rec['num_frames']
            # Accumulate per-frame predictions
            prob_accum = np.zeros((num_frames, 5), dtype=np.float64)
            count_accum = np.zeros(num_frames, dtype=np.float64)

            for start in range(0, num_frames - span + 1, args.window_stride):
                frame_indices = [start + i * seq_stride for i in range(seq_len)]

                if pred_info['feature_fn'] == 'huang':
                    imgs = rec['images'][frame_indices]
                    feat = extract_huang_features_window(
                        imgs, args.num_lines, args.window_size)
                elif pred_info['feature_fn'] == 'mcintosh':
                    feat = _extract_mcintosh_from_precomputed(
                        test_flows[rec_idx], frame_indices, args.pool_size,
                        rec['images'].shape[1], rec['images'].shape[2])

                feat = feat.reshape(1, -1)
                for finger_idx in range(5):
                    if hasattr(classifiers[finger_idx], 'predict_proba'):
                        prob = classifiers[finger_idx].predict_proba(feat)
                        p = prob[0, 1] if prob.shape[1] == 2 else prob[0, 0]
                    else:
                        p = float(classifiers[finger_idx].predict(feat)[0])
                    # Assign to center frame of window
                    center = frame_indices[seq_len // 2]
                    prob_accum[center, finger_idx] += p
                    if finger_idx == 0:
                        count_accum[center] += 1

            # Average overlapping predictions
            mask = count_accum > 0
            prob_signals = {}
            gt_labels = {}
            for finger_idx, name in enumerate(FINGER_NAMES):
                signal = np.zeros(num_frames, dtype=np.float64)
                signal[mask] = prob_accum[mask, finger_idx] / count_accum[mask]
                # Forward-fill edges
                if mask.any():
                    first_valid = np.argmax(mask)
                    last_valid = num_frames - 1 - np.argmax(mask[::-1])
                    signal[:first_valid] = signal[first_valid]
                    signal[last_valid + 1:] = signal[last_valid]
                prob_signals[name] = signal.tolist()
                gt_labels[name] = rec['flex_labels'][:, finger_idx].tolist()

            raw_data = {
                'h5_file': rec['h5_path'],
                'num_frames': num_frames,
                'method': method,
                'prob_signals': prob_signals,
                'gt_flex_labels': gt_labels,
            }
            out_path = event_dir / f"raw_predictions_{rec['h5_stem']}.json"
            with open(out_path, 'w') as f:
                json.dump(raw_data, f)

    print(f"  Raw predictions saved to: {event_dir}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Classic baseline training (Castellini, Huang, McIntosh)')

    parser.add_argument('--method', type=str, required=True,
                        choices=['castellini', 'huang', 'mcintosh'],
                        help='Baseline method to train')

    # Data paths
    parser.add_argument('--data-dir', type=str,
                        default='./data/processed_h5/h5_files',
                        help='Directory containing H5 files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputs/{method})')

    # Subject splits
    parser.add_argument('--train-subjects', type=int, nargs='+',
                        default=[1, 4, 5, 6, 10, 17, 19, 20])
    parser.add_argument('--val-subjects', type=int, nargs='+',
                        default=[21, 23])
    parser.add_argument('--test-subjects', type=int, nargs='+',
                        default=[24, 25])

    # Flex label parameters
    parser.add_argument('--vel-threshold-deg-s', type=float, default=80.0)
    parser.add_argument('--vel-smooth-window', type=int, default=5)
    parser.add_argument('--angle-pct-threshold', type=float, default=0.1)
    parser.add_argument('--thumb-vel-threshold-deg-s', type=float, default=50.0)
    parser.add_argument('--thumb-angle-pct-threshold', type=float, default=0.1)
    parser.add_argument('--use-fixed-labels', action='store_true')

    # Session filtering
    parser.add_argument('--exclude-playing', action='store_true')
    parser.add_argument('--exclude-pinch', action='store_true')
    parser.add_argument('--exclude-wrist', action='store_true')

    # Sequence params (Huang, McIntosh)
    parser.add_argument('--sequence-length', type=int, default=16)
    parser.add_argument('--sequence-stride', type=int, default=2)
    parser.add_argument('--window-stride', type=int, default=1)

    # Castellini-specific
    parser.add_argument('--sphere-radius', type=int, default=9)
    parser.add_argument('--sphere-gap', type=int, default=4)
    parser.add_argument('--ridge-alpha', type=float, default=1.0)

    # Huang-specific
    parser.add_argument('--num-lines', type=int, default=5)
    parser.add_argument('--window-size', type=int, default=20,
                        help='Pixel segment size for Huang features')

    # McIntosh-specific
    parser.add_argument('--pool-size', type=int, default=20)

    # MLP params (Huang, McIntosh)
    parser.add_argument('--mlp-hidden-sizes', type=int, nargs='+', default=[100])
    parser.add_argument('--mlp-max-iter', type=int, default=500)

    # General
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for feature extraction (-1 = all cores)')
    parser.add_argument('--exp-name', type=str, default=None)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'./outputs/{args.method}'

    return args


if __name__ == '__main__':
    args = parse_args()
    train_and_evaluate(args)

"""Shared primitives for ranking-model evaluation.

Reader's guide: this module owns the common HDF5-to-sequence path used by the
ranking evaluators. Centralizing model loading, transforms, finger order, and
temporal sampling prevents evaluation scripts from silently disagreeing.
"""

import json
from pathlib import Path

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


FPS = 20

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']

MULTI_FINGER_JOINT_COLUMNS = {
    'index': ['joint11', 'joint12'],
    'middle': ['joint21', 'joint22'],
    'ring': ['joint31', 'joint32'],
    'pinky': ['joint41', 'joint42'],
    'thumb': ['jointt5', 'jointt6'],
}


def load_h5_data(h5_path):
    """Load images and per-finger summed joint angles from H5 file.

    Returns:
        images: list of numpy arrays (RGB, 224x224)
        joint_angles: dict finger_name -> numpy array of summed joint angles (radians)
        num_frames: int
    """
    import pandas as pd
    print(f"Loading {h5_path}...")
    df = pd.read_hdf(h5_path, 'df')
    print(f"  {len(df)} frames")

    images = [np.array(img) for img in df['img']]

    joint_angles = {}
    for finger, cols in MULTI_FINGER_JOINT_COLUMNS.items():
        angles = np.zeros(len(df), dtype=np.float32)
        for col in cols:
            if col in df.columns:
                angles += df[col].values.astype(np.float32)
        joint_angles[finger] = angles

    return images, joint_angles, len(df)


def load_multi_finger_model(checkpoint_path, config_path, device):
    """Load MultiFingerTemporalRankingModel from checkpoint.

    Returns:
        model: MultiFingerTemporalRankingModel on device, in eval mode
        config: dict loaded from config.json
    """
    from trainer.model import MultiFingerTemporalRankingModel

    with open(config_path) as f:
        config = json.load(f)

    model = MultiFingerTemporalRankingModel(
        embedding_dim=config.get('embedding_dim', 64),
        base_channels=config.get('base_channels', 64),
        hidden_dim=config.get('hidden_dim', 256),
        dropout=config.get('dropout', 0.3),
        sequence_length=config.get('sequence_length', 16),
        num_transformer_layers=config.get('num_transformer_layers', 2),
        num_heads=config.get('num_heads', 4),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    metrics = checkpoint.get('metrics', {})
    if 'mean_auc' in metrics:
        print(f"  Mean AUC: {metrics['mean_auc']:.4f}")

    return model, config


def create_transform(image_size=224):
    """Create val/test transform (resize + to tensor, no augmentation)."""
    return A.ReplayCompose([
        A.Resize(height=image_size, width=image_size, interpolation=1),
        ToTensorV2(),
    ])


def prepare_sequence(images, start_idx, seq_len, stride, transform):
    """Build a model-ready sequence tensor from raw images.

    Args:
        images: list of numpy arrays
        start_idx: first frame index
        seq_len: number of frames in sequence
        stride: frame stride
        transform: albumentations ReplayCompose transform

    Returns:
        tensor: [1, seq_len, 3, H, W]
    """
    frames = []
    replay = None
    for i in range(seq_len):
        img = images[start_idx + i * stride].copy()
        img = img.astype(np.float32) / 255.0
        if replay is None:
            result = transform(image=img)
            replay = result.get('replay', None)
        else:
            result = A.ReplayCompose.replay(replay, image=img)
        frames.append(result['image'])
    return torch.stack(frames, dim=0).unsqueeze(0)

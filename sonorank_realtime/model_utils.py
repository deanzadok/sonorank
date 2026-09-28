"""Model loading and preprocessing utilities for live SonoRank inference.

Reader's guide: ``load_model`` reconstructs architecture settings from the
checkpoint, ``preprocess_frame`` applies the training-compatible image path,
and ``build_rest_reference`` creates the fixed second input used throughout a
live session.
"""

import json
from pathlib import Path

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

SCRIPT_DIR = Path(__file__).resolve().parent

from model_definitions import FlexEmbeddingClassifier

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']

# Model parameters from BENCHMARK_REALTIME.md
DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE = 2
DEFAULT_IMAGE_SIZE = 224
TARGET_FPS = 20.0


def get_inference_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """Get the inference transform (resize + tensor conversion, no augmentation)."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=1),
        ToTensorV2(),
    ])


# Default architecture parameters from BENCHMARK_REALTIME.md
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_BASE_CHANNELS = 64
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_TRANSFORMER_LAYERS = 2
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.3
DEFAULT_MLP_HIDDEN_DIM = 128
DEFAULT_MLP_DROPOUT = 0.3


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load FlexEmbeddingClassifier from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint_best.pth
        device: torch device

    Returns:
        model: Loaded model in eval mode
        seq_len: Sequence length (16)
        stride: Ranking stride (2)
    """
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'

    with open(config_path) as f:
        config = json.load(f)

    # Try to load ranking config, fall back to defaults if not found
    ranking_config = None
    ranking_checkpoint = config.get('ranking_checkpoint')
    if ranking_checkpoint:
        ranking_config_path = config.get('ranking_config')
        if ranking_config_path is None:
            ranking_config_path = Path(ranking_checkpoint).parent / 'config.json'

        # Try relative to checkpoint_dir, then absolute
        for base in [checkpoint_dir, Path('.')]:
            try_path = base / ranking_config_path
            if try_path.exists():
                with open(try_path) as f:
                    ranking_config = json.load(f)
                break

    # Use ranking_config values if available, otherwise use defaults
    if ranking_config:
        embedding_dim = ranking_config.get('embedding_dim', DEFAULT_EMBEDDING_DIM)
        base_channels = ranking_config.get('base_channels', DEFAULT_BASE_CHANNELS)
        hidden_dim = ranking_config.get('hidden_dim', DEFAULT_HIDDEN_DIM)
        dropout = ranking_config.get('dropout', DEFAULT_DROPOUT)
        sequence_length = ranking_config.get('sequence_length', DEFAULT_SEQ_LEN)
        num_transformer_layers = ranking_config.get('num_transformer_layers', DEFAULT_NUM_TRANSFORMER_LAYERS)
        num_heads = ranking_config.get('num_heads', DEFAULT_NUM_HEADS)
    else:
        # Use defaults from BENCHMARK_REALTIME.md
        print("WARNING: Ranking config not found, using default architecture parameters")
        embedding_dim = DEFAULT_EMBEDDING_DIM
        base_channels = DEFAULT_BASE_CHANNELS
        hidden_dim = DEFAULT_HIDDEN_DIM
        dropout = DEFAULT_DROPOUT
        sequence_length = DEFAULT_SEQ_LEN
        num_transformer_layers = DEFAULT_NUM_TRANSFORMER_LAYERS
        num_heads = DEFAULT_NUM_HEADS

    # Get MLP parameters from flex embedding config
    mlp_hidden_dim = config.get('mlp_hidden_dim', DEFAULT_MLP_HIDDEN_DIM)
    mlp_dropout = config.get('mlp_dropout', DEFAULT_MLP_DROPOUT)
    no_rest = config.get('no_rest', False)

    model = FlexEmbeddingClassifier(
        embedding_dim=embedding_dim,
        base_channels=base_channels,
        hidden_dim=hidden_dim,
        dropout=dropout,
        sequence_length=sequence_length,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout=mlp_dropout,
        no_rest=no_rest,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Get stride from config if available, otherwise use default
    seq_len = config.get('ranking_seq_len', DEFAULT_SEQ_LEN)
    stride = config.get('ranking_stride', DEFAULT_STRIDE)

    return model, seq_len, stride


def preprocess_frame(img_uint8: np.ndarray, transform: A.Compose) -> torch.Tensor:
    """
    Convert a single uint8 image to a model-ready tensor.

    Args:
        img_uint8: numpy array, shape (H, W, 3), dtype uint8
        transform: albumentations Compose

    Returns:
        torch tensor, shape (3, 224, 224), dtype float32, range [0, 1]
    """
    img_f = img_uint8.copy().astype(np.float32) / 255.0
    result = transform(image=img_f)
    return result['image']


def build_rest_reference(
    rest_frame_uint8: np.ndarray,
    seq_len: int,
    transform: A.Compose,
    device: torch.device
) -> torch.Tensor:
    """
    Build rest reference from a single static rest frame.

    Args:
        rest_frame_uint8: numpy array (H, W, 3), uint8 - a frame with no finger motion
        seq_len: int, must match model's sequence_length (16)
        transform: albumentations inference transform
        device: torch device

    Returns:
        rest_seq: tensor [1, seq_len, 3, 224, 224]
    """
    rest_tensor = preprocess_frame(rest_frame_uint8, transform)
    rest_seq = rest_tensor.unsqueeze(0).repeat(seq_len, 1, 1, 1)  # [seq_len, 3, H, W]
    return rest_seq.unsqueeze(0).to(device)  # [1, seq_len, 3, H, W]


def grayscale_to_rgb(gray_array: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to 3-channel RGB.

    Args:
        gray_array: numpy array (H, W), uint8

    Returns:
        rgb_array: numpy array (H, W, 3), uint8
    """
    return np.stack([gray_array] * 3, axis=-1)

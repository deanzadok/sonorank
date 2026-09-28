"""
Real-time inference benchmark for SonoRank (FlexEmbeddingClassifier).

Feeds pre-recorded ultrasound frames from H5 files one by one, simulating a
real-time camera stream, and measures prediction throughput and latency.

Usage:
    python -m evaluation.benchmark_realtime \
        --checkpoint outputs/.../checkpoint_best.pth \
        --gpu-id 0 --test-subject 25

Reader's guide: this benchmark replays recorded frames but times only the
online preprocessing and forward-pass path. It measures runtime rather than
event-detection quality.
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from trainer.model import FlexEmbeddingClassifier

FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']
DATA_DIR = "data/processed_h5/h5_files"


def find_h5_files(subject_id, data_dir=DATA_DIR):
    """Find all finger_flex H5 files for a given subject."""
    data_dir = Path(data_dir)
    pattern = f"recordings_{subject_id:02d}_*finger_flex*.h5"
    files = sorted(data_dir.glob(pattern))
    if not files:
        pattern = f"recordings_{subject_id}_*finger_flex*.h5"
        files = sorted(data_dir.glob(pattern))
    return [str(f) for f in files]


def load_h5_images(h5_path):
    """Load images from H5 file."""
    import pandas as pd
    df = pd.read_hdf(h5_path, 'df')
    images = [np.array(img) for img in df['img']]
    return images


def load_flex_embedding_model(checkpoint_path, device):
    """Load FlexEmbeddingClassifier from checkpoint."""
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    ranking_checkpoint = config['ranking_checkpoint']
    ranking_config_path = config.get('ranking_config')
    if ranking_config_path is None:
        ranking_config_path = str(Path(ranking_checkpoint).parent / 'config.json')
    with open(ranking_config_path) as f:
        ranking_config = json.load(f)

    model = FlexEmbeddingClassifier(
        embedding_dim=ranking_config['embedding_dim'],
        base_channels=ranking_config['base_channels'],
        hidden_dim=ranking_config['hidden_dim'],
        dropout=ranking_config.get('dropout', 0.3),
        sequence_length=ranking_config['sequence_length'],
        num_transformer_layers=ranking_config['num_transformer_layers'],
        num_heads=ranking_config['num_heads'],
        mlp_hidden_dim=config['mlp_hidden_dim'],
        mlp_dropout=config['mlp_dropout'],
        no_rest=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    seq_len = ranking_config['sequence_length']
    stride = ranking_config.get('ranking_stride', config.get('ranking_stride', 2))

    print(f"Model loaded: seq_len={seq_len}, stride={stride}")
    return model, seq_len, stride


def preprocess_frame(img, transform):
    """Normalize and transform a single frame to tensor."""
    img_f = img.copy().astype(np.float32) / 255.0
    result = transform(image=img_f)
    return result['image']


def build_rest_reference(images, seq_len, stride, transform, device):
    """Build rest reference from static early frame (frame 3), duplicated seq_len times."""
    static_rest_idx = min(3, len(images) - 1)
    rest_img = images[static_rest_idx]
    rest_tensor = preprocess_frame(rest_img, transform)
    rest_seq = rest_tensor.unsqueeze(0).repeat(seq_len, 1, 1, 1)  # [seq_len, 3, H, W]
    rest_seq = rest_seq.unsqueeze(0).to(device)  # [1, seq_len, 3, H, W]
    return rest_seq


def run_benchmark(model, images, seq_len, stride, transform, device,
                  warmup_frames, max_frames):
    """Run frame-by-frame inference benchmark on one session."""
    span = (seq_len - 1) * stride + 1
    num_frames = len(images)
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)

    rest_seq = build_rest_reference(images, seq_len, stride, transform, device)

    # Sliding buffer of preprocessed frames
    frame_buffer = deque(maxlen=span)
    latencies = []
    predictions = []
    total_frames = 0

    for frame_idx in range(num_frames):
        frame_tensor = preprocess_frame(images[frame_idx], transform)
        frame_buffer.append(frame_tensor)
        total_frames += 1

        if len(frame_buffer) < span:
            continue

        # Build query sequence by striding the buffer
        buffer_list = list(frame_buffer)
        query_frames = buffer_list[::stride][:seq_len]
        query_seq = torch.stack(query_frames).unsqueeze(0).to(device)  # [1, seq_len, 3, H, W]

        # Timed inference
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = model(query_seq, rest_seq)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # [5]

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        if frame_idx >= warmup_frames:
            latencies.append(latency_ms)
            predictions.append((frame_idx, probs))

    return latencies, predictions, total_frames


def print_results(latencies, predictions, checkpoint_path, device, seq_len, stride):
    """Print benchmark summary."""
    latencies = np.array(latencies)

    print()
    print("=" * 60)
    print("  Real-Time Inference Benchmark: SonoRank")
    print("=" * 60)
    print(f"  Model:       FlexEmbeddingClassifier")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Device:      {device}")
    print(f"  Seq length:  {seq_len}, stride: {stride}, span: {(seq_len - 1) * stride + 1}")
    print(f"  Predictions: {len(latencies)}")
    print()
    print("  Latency (ms):")
    print(f"    Mean:  {latencies.mean():.2f}")
    print(f"    Std:   {latencies.std():.2f}")
    print(f"    p50:   {np.percentile(latencies, 50):.2f}")
    print(f"    p95:   {np.percentile(latencies, 95):.2f}")
    print(f"    p99:   {np.percentile(latencies, 99):.2f}")
    print(f"    Min:   {latencies.min():.2f}")
    print(f"    Max:   {latencies.max():.2f}")
    print()

    throughput = 1000.0 / latencies.mean()
    target_hz = 20.0
    status = "OK" if throughput >= target_hz else "TOO SLOW"
    print(f"  Throughput:  {throughput:.1f} Hz (target: {target_hz} Hz) [{status}]")
    print()

    # Sample predictions
    if predictions:
        print("  Sample predictions (last 5):")
        for frame_idx, probs in predictions[-5:]:
            finger_str = "  ".join(f"{f}={probs[i]:.2f}" for i, f in enumerate(FINGER_NAMES))
            print(f"    Frame {frame_idx:5d}: {finger_str}")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SonoRank real-time inference benchmark")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint_best.pth')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--test-subject', type=int, default=25)
    parser.add_argument('--warmup-frames', type=int, default=100,
                        help='Frames to skip for GPU warmup')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Max frames per session (0 = all)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    model, seq_len, stride = load_flex_embedding_model(args.checkpoint, device)

    transform = A.Compose([
        A.Resize(height=224, width=224, interpolation=1),
        ToTensorV2(),
    ])

    h5_files = find_h5_files(args.test_subject, args.data_dir)
    if not h5_files:
        print(f"No H5 files found for subject {args.test_subject}")
        return
    print(f"Found {len(h5_files)} H5 files for subject {args.test_subject}")

    all_latencies = []
    all_predictions = []

    for h5_path in h5_files:
        print(f"\nProcessing {Path(h5_path).name}...")
        images = load_h5_images(h5_path)
        print(f"  {len(images)} frames")

        latencies, predictions, total = run_benchmark(
            model, images, seq_len, stride, transform, device,
            args.warmup_frames, args.max_frames,
        )
        all_latencies.extend(latencies)
        all_predictions.extend(predictions)
        print(f"  {len(latencies)} predictions, mean latency: {np.mean(latencies):.2f} ms")

    if all_latencies:
        print_results(all_latencies, all_predictions, args.checkpoint,
                      device, seq_len, stride)
    else:
        print("No predictions made (too few frames or warmup too large)")


if __name__ == '__main__':
    main()

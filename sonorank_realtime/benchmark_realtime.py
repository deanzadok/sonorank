"""
Real-time inference benchmark for SonoRank (FlexEmbeddingClassifier).

Feeds pre-recorded ultrasound frames from H5 files one by one, simulating a
real-time camera stream, and measures prediction throughput and latency.

Usage:
    python -m evaluation.benchmark_realtime \
        --checkpoint outputs/.../checkpoint_best.pth \
        --gpu-id 0 --test-subject 25

Reader's guide: this standalone copy uses the self-contained real-time model
definitions. Recorded frames substitute for the probe, but sequence buffering,
preprocessing, and forward-pass timing match the deployment path.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from model_utils import (
    load_model, get_inference_transform, preprocess_frame,
    build_rest_reference, FINGER_NAMES,
)

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


def run_benchmark(model, images, seq_len, stride, transform, device,
                  warmup_frames, max_frames):
    """Run frame-by-frame inference benchmark on one session."""
    span = (seq_len - 1) * stride + 1
    num_frames = len(images)
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)

    # Build rest reference from early frame
    static_rest_idx = min(3, len(images) - 1)
    rest_seq = build_rest_reference(images[static_rest_idx], seq_len, transform, device)

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

    model, seq_len, stride = load_model(args.checkpoint, device)
    print(f"Model loaded: seq_len={seq_len}, stride={stride}")

    transform = get_inference_transform()

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

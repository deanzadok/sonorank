#!/usr/bin/env python
"""
Test inference pipeline with dummy noise images (no Clarius SDK).
Tests if the model inference itself causes freezing.

Reader's guide: this isolates checkpoint loading, temporal buffering, and GPU
inference from probe networking and robotic-hand serial communication. It is a
deployment smoke test rather than an accuracy evaluation.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_utils import (
    load_model, get_inference_transform, preprocess_frame,
    build_rest_reference, FINGER_NAMES,
)

DEFAULT_CHECKPOINT = str(SCRIPT_DIR / "outputs/flex_embedding_ranking_backbone/flex_embedding_lambda0.5_20260228_133401_test17_val10/checkpoint_best.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    parser.add_argument("--target-fps", type=float, default=20.0, help="Simulated frame rate")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, seq_len, stride = load_model(args.checkpoint, device)
    span = (seq_len - 1) * stride + 1
    print(f"seq_len={seq_len}, stride={stride}, span={span}")

    transform = get_inference_transform()

    # Create rest reference from random noise
    print("Creating rest reference from noise...")
    rest_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    rest_seq = build_rest_reference(rest_frame, seq_len, transform, device)

    frame_buffer = deque(maxlen=span)
    frame_count = 0
    prediction_count = 0
    start_time = time.time()
    frame_interval = 1.0 / args.target_fps

    print(f"\nRunning inference test for {args.duration}s at {args.target_fps} fps...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            loop_start = time.time()
            elapsed = loop_start - start_time

            if elapsed >= args.duration:
                break

            # Generate random noise frame (simulating ultrasound)
            rgb_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            frame_count += 1

            # Preprocess
            frame_tensor = preprocess_frame(rgb_array, transform)
            frame_buffer.append(frame_tensor)

            if len(frame_buffer) < span:
                print(f"\rBuffering: {len(frame_buffer)}/{span}", end="", flush=True)
            else:
                # Build query
                buffer_list = list(frame_buffer)
                query_frames = buffer_list[::stride][:seq_len]
                query_seq = torch.stack(query_frames).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    logits = model(query_seq, rest_seq)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]

                prediction_count += 1
                fps = frame_count / elapsed if elapsed > 0 else 0
                pred_fps = prediction_count / elapsed if elapsed > 0 else 0

                active = [FINGER_NAMES[i] for i in range(5) if probs[i] >= args.threshold]
                active_str = ", ".join(active) if active else "rest"
                prob_str = " ".join(f"{FINGER_NAMES[i][:3]}={probs[i]:.2f}" for i in range(5))

                print(f"\rFrame {frame_count:5d} | {fps:.1f} fps | pred: {prediction_count:4d} ({pred_fps:.1f}/s) | {active_str:20s} | {prob_str}", end="", flush=True)

            # Rate limit to target FPS
            loop_time = time.time() - loop_start
            sleep_time = frame_interval - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nInterrupted")

    elapsed = time.time() - start_time
    print(f"\n\n{'='*60}")
    print(f"Test completed!")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Frames: {frame_count} ({frame_count/elapsed:.1f} fps)")
    print(f"Predictions: {prediction_count} ({prediction_count/elapsed:.1f}/s)")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Real-time finger flex detection using SonoRank model and Clarius ultrasound probe.
Optionally drives a robotic hand via serial (--serial-port).

Reader's guide: the Clarius callback only converts and enqueues frames. The
main thread owns temporal buffering, model inference, automatic thresholding,
and optional serial commands. Keeping work out of the callback prevents the
SDK acquisition thread from stalling.
"""

import argparse
import os
import struct
import sys
import time
import threading
import queue
from collections import deque
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty

import numpy as np
import torch
from PIL import Image

from clarius_utils import load_clarius_sdk
from model_utils import (
    load_model, get_inference_transform, preprocess_frame,
    build_rest_reference, grayscale_to_rgb, FINGER_NAMES,
)

DEFAULT_CHECKPOINT = str(SCRIPT_DIR / "outputs/flex_embedding_ranking_backbone/flex_embedding_lambda0.5_20260228_133401_test17_val10/checkpoint_best.pth")

# Shared state accessed by both the SDK callback thread and the main loop.
model = None
device = None
transform = None
rest_seq = None
seq_len = 16
stride = 2
span = 31
threshold = 0.5
frame_count = 0
start_time = None
running = True
lock = threading.Lock()

# Warm-up and auto-threshold settings
WARMUP_INFERENCES = 25    # skip first N inferences (GPU JIT warm-up)
CALIB_INFERENCES  = 50    # observe next N frames to set per-finger thresholds
FLOOR_TARGET      = 0.5   # finger must drop below this to be considered calibrated
ACTIVATION_MARGIN = 0.30  # threshold = min_observed + this margin

# Confidence hysteresis for robotic hand
CONFIDENCE_FLEX   = 0.87  # flex when confidence rises above this
CONFIDENCE_EXTEND = 0.60  # extend when confidence drops below this
FLEX_DEBOUNCE_S   = 0.10  # finger must stay above CONFIDENCE_FLEX for this long before sending
MIN_SEND_INTERVAL = 1.50  # minimum seconds between serial messages (flex→extend cooldown)

# The bounded queue prevents unbounded allocation if inference falls behind.
frame_queue = queue.Queue(maxsize=50)

# --- Robotic hand serial protocol ---
NUM_MOTORS = 8
HAND_BAUD  = 57600

# Motor indices (must match robotic_hand_manual.py)
MOTOR_MAP = {
    'index':  0,
    'middle': 1,
    'ring':   2,
    'pinky':  3,
    'thumb':  4,
}

CLOSED_POS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
OPEN_POS   = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255}

MSG_FORMAT = '<B8Bx8H'
MSG_SIZE   = struct.calcsize(MSG_FORMAT)  # 26 bytes


def configure_keyboard():
    """Enable single-key input on POSIX terminals and return prior settings."""
    if os.name == "nt" or not sys.stdin.isatty():
        return None
    settings = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin.fileno())
    return settings


def restore_keyboard(settings):
    """Restore POSIX terminal settings changed by configure_keyboard."""
    if settings is not None:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, settings)


def quit_requested():
    """Return True when the user presses q without blocking inference."""
    if os.name == "nt":
        if not msvcrt.kbhit():
            return False
        return msvcrt.getwche().lower() == "q"

    if not sys.stdin.isatty():
        return False
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(readable) and sys.stdin.read(1).lower() == "q"


def hand_send(ser, motor_commands, pwm):
    """Send a motor command to the robotic hand."""
    motor_mask = 0
    pwms      = [0] * NUM_MOTORS
    positions = [0] * NUM_MOTORS
    for idx, pos in motor_commands.items():
        motor_mask      |= (1 << idx)
        pwms[idx]        = pwm
        positions[idx]   = pos
    msg = struct.pack(MSG_FORMAT, motor_mask, *pwms, *positions)
    ser.write(msg)


def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    """Callback - decode and queue frame."""
    global frame_count, start_time, rest_seq

    if not running:
        return

    with lock:
        frame_count += 1
        if start_time is None:
            start_time = time.time()

    # Decode
    bpp = sz / (width * height)
    if bpp == 4:
        img = Image.frombytes("RGBA", (width, height), image)
    else:
        img = Image.frombytes("L", (width, height), image)

    gray_array = np.array(img.convert("L"))
    rgb_array = grayscale_to_rgb(gray_array)

    # Frame 10 = rest reference
    if frame_count == 10:
        rest_seq = build_rest_reference(rgb_array, seq_len, transform, device)
        print(f"\n[Frame 10 = rest reference]")
        return

    if rest_seq is None:
        print(f"\rWaiting... frame {frame_count}", end="", flush=True)
        return

    # Queue frame (drop if full)
    try:
        frame_queue.put_nowait((rgb_array, frame_count))
    except queue.Full:
        pass


# The Caster API requires handlers for streams that SonoRank does not consume.
# These no-op callbacks make that part of the SDK contract explicit.
def newRawImage(*args): pass
def newSpectrumImage(*args): pass
def newImuData(*args): pass
def freezeFn(frozen): print(f"\n[{'Frozen' if frozen else 'Running'}]")
def buttonsFn(*args): pass


def main():
    """Connect the live pipeline and run until the user requests shutdown."""
    global model, device, transform, seq_len, stride, span, threshold, running

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", "-a", default="192.168.1.1")
    parser.add_argument("--port", "-p", type=int, default=5828)
    parser.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--serial-port", default=None,
                        help="COM port for robotic hand (e.g. COM3). Omit to run without hand.")
    parser.add_argument("--pwm", type=int, default=120, help="Motor PWM speed 0-255")
    args = parser.parse_args()

    try:
        pyclariuscast = load_clarius_sdk()
    except (FileNotFoundError, ImportError, OSError) as exc:
        print(f"Clarius SDK could not be loaded: {exc}", file=sys.stderr)
        print("See sonorank_realtime/README.md for SDK setup instructions.",
              file=sys.stderr)
        return 1

    threshold = args.threshold
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, seq_len, stride = load_model(args.checkpoint, device)
    span = (seq_len - 1) * stride + 1
    print(f"seq_len={seq_len}, stride={stride}, span={span}")

    transform = get_inference_transform()

    # Robotic hand (optional)
    ser = None
    if args.serial_port:
        import serial
        print(f"Connecting to robotic hand on {args.serial_port}...")
        ser = serial.Serial(args.serial_port, HAND_BAUD, timeout=1)
        time.sleep(2)  # ESP32 boot delay
        # Open all fingers at start
        hand_send(ser, {m: OPEN_POS[m] for m in MOTOR_MAP.values()}, args.pwm)
        print("Robotic hand connected.")

    # Clarius
    cast = pyclariuscast.Caster(
        newProcessedImage, newRawImage, newSpectrumImage,
        newImuData, freezeFn, buttonsFn,
    )
    cast.init(os.path.expanduser("~/"), 640, 480)

    print(f"Connecting to {args.address}:{args.port}...")
    if not cast.connect(args.address, args.port, "research"):
        print("Connection failed")
        cast.destroy()
        if ser:
            ser.close()
        return 1

    print("Connected! Press 'q' to quit\n")

    # Frame buffer for inference
    frame_buffer = deque(maxlen=span)

    # Warm-up and auto-threshold state
    inference_count = 0
    calib_probs     = [[] for _ in range(5)]
    thresholds      = [0.5] * 5
    calibrated      = [False] * 5
    calib_done      = False

    # Per-finger flexion state for robotic hand (True = flexed)
    finger_flexed = {name: False for name in FINGER_NAMES}
    # Debounce: timestamp when finger first went above CONFIDENCE_FLEX (None = not pending)
    flex_pending_since = {name: None for name in FINGER_NAMES}
    # Cooldown: timestamp of last serial message sent
    last_send_time = 0.0

    # Main loop - process frames from queue
    keyboard_settings = configure_keyboard()
    try:
        while running:
            if quit_requested():
                break

            # Get frame from queue (non-blocking)
            try:
                rgb_array, fnum = frame_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue

            # Preprocess
            frame_tensor = preprocess_frame(rgb_array, transform)
            frame_buffer.append(frame_tensor)

            if len(frame_buffer) < span:
                print(f"\rBuffering: {len(frame_buffer)}/{span}", end="", flush=True)
                continue

            # Build query
            buffer_list = list(frame_buffer)
            query_frames = buffer_list[::stride][:seq_len]
            query_seq = torch.stack(query_frames).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                logits = model(query_seq, rest_seq)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            with lock:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

            prob_str = " ".join(f"{FINGER_NAMES[i][:3]}={probs[i]:.2f}" for i in range(5))
            inference_count += 1

            # --- Phase 1: GPU warm-up ---
            if inference_count <= WARMUP_INFERENCES:
                print(f"\rWarm-up {inference_count}/{WARMUP_INFERENCES} | {prob_str}", end="", flush=True)
                continue

            # --- Phase 2: auto-threshold calibration ---
            if not calib_done:
                calib_n = inference_count - WARMUP_INFERENCES
                for i in range(5):
                    calib_probs[i].append(probs[i])
                print(f"\rCalibrating {calib_n}/{CALIB_INFERENCES} | {prob_str}", end="", flush=True)
                if calib_n >= CALIB_INFERENCES:
                    for i in range(5):
                        floor = min(calib_probs[i])
                        thresholds[i] = min(0.90, floor + ACTIVATION_MARGIN)
                        calibrated[i] = True
                    calib_done = True
                    thr_str = " ".join(
                        f"{FINGER_NAMES[i][:3]}={thresholds[i]:.2f}({'ok' if calibrated[i] else '?'})"
                        for i in range(5)
                    )
                    print(f"\nThresholds: {thr_str}")
                continue

            # --- Phase 3: single-finger, confidence-based prediction ---
            best_finger = -1
            best_confidence = 0.0
            for i in range(5):
                if calibrated[i] and probs[i] > thresholds[i]:
                    confidence = (probs[i] - thresholds[i]) / (1.0 - thresholds[i])
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_finger = i

            if best_finger >= 0:
                active_str = f"{FINGER_NAMES[best_finger]} ({best_confidence:.0%})"
            else:
                active_str = "rest"

            # --- Hand control logic (single finger, debounced, cooldown) ---
            now = time.time()
            hand_str = ""
            can_send = (now - last_send_time) >= MIN_SEND_INTERVAL

            if best_finger >= 0 and best_confidence >= CONFIDENCE_FLEX:
                target_name = FINGER_NAMES[best_finger]

                # Start or continue debounce timer for this finger
                if flex_pending_since[target_name] is None:
                    flex_pending_since[target_name] = now
                # Clear debounce for all other fingers
                for name in FINGER_NAMES:
                    if name != target_name:
                        flex_pending_since[name] = None

                # Only flex after debounce period has passed
                if now - flex_pending_since[target_name] >= FLEX_DEBOUNCE_S:
                    if not finger_flexed[target_name] and can_send:
                        motor = MOTOR_MAP[target_name]
                        if ser:
                            hand_send(ser, {motor: CLOSED_POS[motor]}, args.pwm)
                        hand_str += f" [FLEX {target_name}]" if not ser else f" +{target_name[:3]}"
                        finger_flexed[target_name] = True
                        last_send_time = now
                    # Extend all other flexed fingers (only if cooldown allows)
                    for name in FINGER_NAMES:
                        if name != target_name and finger_flexed[name] and can_send:
                            motor = MOTOR_MAP[name]
                            if ser:
                                hand_send(ser, {motor: OPEN_POS[motor]}, args.pwm)
                            hand_str += f" [EXTEND {name}]" if not ser else f" -{name[:3]}"
                            finger_flexed[name] = False
                            last_send_time = now
            else:
                # No confident finger — clear all debounce timers
                for name in FINGER_NAMES:
                    flex_pending_since[name] = None
                    if finger_flexed[name] and can_send:
                        motor = MOTOR_MAP[name]
                        if ser:
                            hand_send(ser, {motor: OPEN_POS[motor]}, args.pwm)
                        hand_str += f" [EXTEND {name}]" if not ser else f" -{name[:3]}"
                        finger_flexed[name] = False
                        last_send_time = now

            if hand_str:
                active_str += hand_str

            print(f"\rFrame {fnum:5d} | {fps:.1f} fps | {active_str:30s} | {prob_str}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted")
    finally:
        restore_keyboard(keyboard_settings)

    running = False

    # Open all fingers before shutdown
    if ser:
        hand_send(ser, {m: OPEN_POS[m] for m in MOTOR_MAP.values()}, args.pwm)
        ser.close()
        print("\nRobotic hand disconnected (fingers opened).")

    cast.destroy()

    if start_time:
        elapsed = time.time() - start_time
        print(f"\nFrames: {frame_count}, Duration: {elapsed:.1f}s, Avg FPS: {frame_count/elapsed:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

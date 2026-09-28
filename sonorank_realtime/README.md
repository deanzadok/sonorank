# SonoRank Real-Time Inference

Self-contained module for real-time finger flex detection from live Clarius ultrasound images, with optional robotic hand control via serial.

## What It Does

A `FlexEmbeddingClassifier` model processes temporal sequences of ultrasound frames and outputs per-finger flex probabilities for 5 fingers (index, middle, ring, pinky, thumb). The system connects to a Clarius ultrasound probe, receives frames in real-time, and predicts which finger is being flexed. Optionally, it drives a robotic hand over serial to mirror the detected motion.

## Module Structure

```
sonorank_realtime/
├── realtime_inference.py      # Main entry point for the probe, model, and optional hand
├── model_utils.py             # load_model, preprocess_frame, build_rest_reference
├── model_definitions.py       # All model classes (FlexEmbeddingClassifier, backbone, etc.)
├── robotic_hand_manual.py     # Manual keyboard control of robotic hand (standalone)
├── benchmark_realtime.py      # Offline latency benchmark using H5 recordings
├── test_inference_only.py     # Model throughput test with random noise (no probe)
└── clarius_utils.py           # Clarius SDK loading helpers
```

## Dependencies

```
torch >= 1.13
albumentations
numpy
Pillow
segmentation_models_pytorch
pyserial              # only if using --serial-port
pandas / tables       # only for benchmark_realtime.py (H5 loading)
```

## Running Real-Time Inference

### Prerequisites

1. Clarius App running on mobile, connected to probe
2. Cast Permission set to **"Research"** in Clarius App settings
3. PC connected to probe's WiFi (`DIRECT-...` network)
4. Probe IP + port from Clarius App status page (default port: **5828**)

### Basic usage (no robotic hand)

```bash
cd sonorank_realtime
python realtime_inference.py --address 192.168.1.1 --port 5828
```

Without `--serial-port`, predictions are printed but no hand commands are sent.

### With robotic hand

```bash
python realtime_inference.py --address 192.168.1.1 --port 5828 --serial-port COM3 --pwm 120
```

### Startup sequence

1. **Connect** to Clarius probe, start receiving frames
2. **Frame 10** is captured as the rest reference (hand must be relaxed)
3. **GPU warmup:** discard the first 25 inferences.
4. **Threshold calibration:** keep the hand relaxed for the next 50 inferences while the program estimates a threshold for each finger.
5. **Live prediction:** report the most confident finger above its threshold. If serial control is enabled, send commands with debounce and cooldown.

### Output format

```
Frame   100 | 23.5 fps | index (87%)              | idx=0.87 mid=0.32 rin=0.12 pnk=0.08 thm=0.15
Frame   105 | 23.5 fps | index (92%) [FLEX index]  | idx=0.92 mid=0.28 rin=0.10 pnk=0.07 thm=0.14
```

Press `q` to quit. Single-key quitting is supported in both Windows and Linux
terminals. `Ctrl+C` remains available as a fallback, including when standard
input is not attached to an interactive terminal.

### Key parameters (tuned for demo)

| Constant | Value | Purpose |
|----------|-------|---------|
| `CONFIDENCE_FLEX` | 0.87 | Confidence must exceed this to flex |
| `CONFIDENCE_EXTEND` | 0.60 | Confidence must drop below this to extend |
| `FLEX_DEBOUNCE_S` | 0.10 s | Finger must stay above threshold this long before sending |
| `MIN_SEND_INTERVAL` | 1.50 s | Minimum gap between serial messages (cooldown) |
| `WARMUP_INFERENCES` | 25 | GPU warmup inferences to discard |
| `CALIB_INFERENCES` | 50 | Frames used for auto-threshold calibration |
| `ACTIVATION_MARGIN` | 0.30 | Threshold = min_observed_rest + this margin |

## Model Architecture

`FlexEmbeddingClassifier` is a two-stage pipeline (~10.7M params):

1. **Backbone** (`MultiFingerTemporalRankingModel`): a ResNet-style CNN with five downsampling stages followed by a two-layer, four-head transformer. Its output is a 64-dimensional embedding.
2. **Head:** an MLP maps the query and rest embeddings to five finger logits.

The forward call is `model(query_seq, rest_seq)`, where both inputs have shape `[1, 16, 3, 224, 224]`. Applying `torch.sigmoid()` to the output logits gives the finger-flexion probabilities.

## Inference Pipeline

**Preprocessing** (must match training exactly):
- Replicate the grayscale image into three channels with `np.stack([gray] * 3, axis=-1)`.
- Resize to 224×224 (bilinear)
- Normalize with `img.astype(float32) / 255.0`. Do not apply ImageNet mean and standard deviation.

**Temporal buffer**: `span = (seq_len - 1) * stride + 1 = 31` raw frames. Sampling every second frame produces a 16-frame sequence. The first prediction is available after 31 frames, or about 1.55 seconds at 20 Hz.

**Rest reference**: Frame 10 is captured with the hand relaxed and repeated 16 times to form a tensor with shape `[1, 16, 3, 224, 224]`. It is built once per session.

**Target throughput**: 20 Hz.

## Robotic Hand Protocol

26-byte serial message at 57600 baud (matches ESP32 `MotorsMessage` struct):

```
<B8Bx8H: 1B motor_mask + 8B PWMs + 1B padding + 8×uint16 positions
```

Motor index mapping:
| Motor | Finger |
|-------|--------|
| 0 | index |
| 1 | middle |
| 2 | ring |
| 3 | pinky |
| 4 | thumb |

Positions: 0 = closed/flexed, 255 = open/extended.

### Manual hand control

```bash
python robotic_hand_manual.py --port COM3 --pwm 120
```

Keys: `a/s/d/f/g` = flex thumb/index/middle/ring/pinky, `z/x/c/v/b` = extend, `h` = extend all, `q` = quit.

## Offline Benchmark

Measures inference latency using pre-recorded H5 files (no probe required):

```bash
cd sonorank_realtime
python benchmark_realtime.py \
    --checkpoint outputs/flex_embedding_ranking_backbone/flex_embedding_lambda0.5_20260228_133401_test17_val10/checkpoint_best.pth \
    --gpu-id 0 \
    --test-subject 25
```

Reports mean/p50/p95/p99 latency and throughput in Hz. Target is 20 Hz.

## Clarius SDK Setup

The Clarius Cast SDK binaries are **not** included in this repository. To use real-time inference with a Clarius probe, you need to obtain the SDK separately:

1. Download the Clarius Cast SDK from [Clarius developer portal](https://developers.clarius.com/)
2. Place the SDK files in this directory:
   - **Windows**: `cast.dll` + `pyclariuscast.pyd`
   - **Linux**: `libcast.so` + `pyclariuscast.so`

**Linux**: Pre-load the shared library before importing:
```python
import ctypes
ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)
import pyclariuscast
```

The provided `realtime_inference.py` performs this SDK loading through
`clarius_utils.py`. It also uses platform-specific non-blocking keyboard input,
so the Windows-only `msvcrt` module is never imported on Linux.

Image callbacks run on background threads. Do not block inside a callback. Use `queue.Queue` to pass frames to the inference thread.

## Checkpoints

Trained checkpoints are not included. Point `--checkpoint` to a trained `FlexEmbeddingClassifier` checkpoint (from `run_cross_subject_flex_embedding.sh`).

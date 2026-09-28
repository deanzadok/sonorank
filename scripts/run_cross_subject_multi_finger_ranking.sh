#!/bin/bash
#
# Cross-subject validation training script for multi-finger temporal ranking
# The checked-in batch tests four subjects, with the previous subject used for validation.
# Uses pretrained EmbeddingAutoencoder for encoder initialization
#
# 5 parallel comparison heads (one per finger) sharing a single backbone.
# 6 sampling modes rotate: index, middle, ring, pinky, thumb, rest
#
# IMPORTANT: This script requires pre-trained embedding autoencoders!
# Run run_cross_subject_embedding.sh first to train the encoders.
# Reader's guide: this launches Level 2. Each fold locates its Level 1 encoder,
# constructs the circular validation split, and trains all five ranking heads
# together on the selected GPU.
#

# Configuration
PYTHON_BIN="/home/deanz/anaconda3/envs/nhst39/bin/python"
SCRIPT="trainer.train_multi_finger_ranking"
NUM_WORKERS=4
BATCH_SIZE=8  # Smaller due to sequence memory
DROPOUT=0.3
EXCLUDE_PLAYING=true  # Exclude multi_playing sessions (train on finger_flex, pinch, wrist only)
EXCLUDE_PINCH=true    # Exclude pinch sessions
EXCLUDE_WRIST=true    # Exclude wrist sessions (train on finger_flex only)

# Temporal ranking specific parameters
SEQUENCE_LENGTH=16  # 16 frames per sequence (~0.8s at 20Hz)
SEQUENCE_STRIDE=3   # Every 3rd frame (~2.4s temporal span with seq_len=16)
NUM_TRANSFORMER_LAYERS=2
NUM_HEADS=4

# Model architecture (must match EmbeddingAutoencoder)
EMBEDDING_DIM=64
BASE_CHANNELS=64
HIDDEN_DIM=256

# Multi-finger loss weights
SECONDARY_LOSS_WEIGHT=0.3
INACTIVE_LOSS_WEIGHT=0.1
DYNAMIC_SECONDARY=true  # Use min(1, secondary_delta/primary_delta) instead of fixed weight

# Pair-level uncertainty penalty
UNCERTAINTY_WEIGHT=1.0  # Weight for uncertainty penalty on ambiguous pairs
UNCERTAINTY_THRESHOLD_DEG=5.0  # Pairs with delta difference below this are ambiguous
# Per-finger rest thresholds (index middle ring pinky thumb)
PER_FINGER_REST_THRESHOLDS="17.5 17.5 17.5 12.5 7.5"

# Sampling
SAMPLES_PER_EPOCH=50000  # Number of samples per epoch (required for on-the-fly generation)

# Training hyperparameters
PHASE1_EPOCHS=0   # Skip phase 1 (frozen encoder) and go directly to full fine-tuning
PHASE2_EPOCHS=30
PHASE1_LR=1e-3
ENCODER_LR_EARLY=1e-5
ENCODER_LR_LATE=5e-5
TRANSFORMER_LR=1e-4
HEAD_LR=3e-4
SAVE_FREQ=10

# All 12 subjects (used for train/val/test splits)
ALL_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# Selected representative subjects to actually train on (4 favorable subjects)
SELECTED_TEST_SUBJECTS=(4 10 17 23)

# GPU assignment (one subject per GPU, all 4 GPUs)
GPU_IDS=(0 1 2 3)

# Path to embedding checkpoints (from run_cross_subject_embedding.sh)
EMBEDDING_DIR="outputs/embedding"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Cross-Subject Multi-Finger Ranking Training"
echo "=========================================="
echo "Fingers: ALL 5 (index, middle, ring, pinky, thumb)"
echo "Total subjects available: ${#ALL_SUBJECTS[@]}"
echo "Selected test subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})"
echo "GPUs: 2, 3 (2 trials per GPU)"
echo "Workers per session: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "Dropout: $DROPOUT"
echo "Exclude playing: $EXCLUDE_PLAYING"
echo "Exclude pinch: $EXCLUDE_PINCH"
echo "Exclude wrist: $EXCLUDE_WRIST"
echo ""
echo "Temporal Configuration:"
echo "  Sequence length: $SEQUENCE_LENGTH frames (stride: $SEQUENCE_STRIDE)"
echo "  Transformer layers: $NUM_TRANSFORMER_LAYERS"
echo "  Attention heads: $NUM_HEADS"
echo ""
echo "Model Configuration:"
echo "  Embedding dim: $EMBEDDING_DIM"
echo "  Base channels: $BASE_CHANNELS"
echo "  Hidden dim: $HIDDEN_DIM"
echo ""
echo "Multi-Finger Loss Weights:"
echo "  Secondary active: $SECONDARY_LOSS_WEIGHT (dynamic=$DYNAMIC_SECONDARY)"
echo "  Inactive penalty: $INACTIVE_LOSS_WEIGHT"
echo ""
echo "Pair-Level Uncertainty Penalty:"
echo "  Weight: $UNCERTAINTY_WEIGHT"
echo "  Uncertainty threshold: $UNCERTAINTY_THRESHOLD_DEG degrees (pair delta diff)"
echo "  Per-finger rest thresholds: $PER_FINGER_REST_THRESHOLDS (idx mid ring pnk thb)"
echo ""
echo "Training:"
echo "  Phase 1 epochs: $PHASE1_EPOCHS (frozen encoder)"
echo "  Phase 2 epochs: $PHASE2_EPOCHS (full fine-tuning)"
echo "  Phase 1 LR: $PHASE1_LR"
echo "  Encoder LR (early/late): $ENCODER_LR_EARLY / $ENCODER_LR_LATE"
echo "  Transformer LR: $TRANSFORMER_LR"
echo "  Head LR: $HEAD_LR"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs/cross_subject_multi_finger_ranking
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cross_subject_multi_finger_ranking/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Check for embedding checkpoints
echo "Checking for embedding checkpoints..."
missing_checkpoints=0
for test_subject in "${SELECTED_TEST_SUBJECTS[@]}"; do
    # Find checkpoint for this test subject
    checkpoint_pattern="${EMBEDDING_DIR}/*test${test_subject}_val*/checkpoint_best.pth"
    checkpoint=$(ls $checkpoint_pattern 2>/dev/null | head -1)

    if [ -z "$checkpoint" ]; then
        echo -e "${RED}  Missing checkpoint for test subject $test_subject${NC}"
        missing_checkpoints=$((missing_checkpoints + 1))
    else
        echo -e "${GREEN}  Found checkpoint for test subject $test_subject: $(basename $(dirname $checkpoint))${NC}"
    fi
done

if [ $missing_checkpoints -gt 0 ]; then
    echo ""
    echo -e "${RED}Error: Missing $missing_checkpoints embedding checkpoint(s)!${NC}"
    echo "Please run run_cross_subject_embedding.sh first to train the encoders."
    exit 1
fi
echo ""

# Array to store background process IDs
pids=()
gpu_idx=0

# Launch training sessions for each subject as test set
for i in "${!ALL_SUBJECTS[@]}"; do
    test_subject=${ALL_SUBJECTS[$i]}

    # Check if this test subject is in the selected list
    is_selected=0
    for selected_subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
        if [ $test_subject -eq $selected_subj ]; then
            is_selected=1
            break
        fi
    done

    # Skip if not selected
    if [ $is_selected -eq 0 ]; then
        echo -e "${BLUE}[Subject $((i+1))/${#ALL_SUBJECTS[@]}]${NC} Test=${GREEN}$test_subject${NC} - ${YELLOW}SKIPPED${NC} (not in selected subjects)"
        continue
    fi

    # Validation subject is the previous one in the list (wrap around for first subject)
    if [ $i -eq 0 ]; then
        val_idx=$((${#ALL_SUBJECTS[@]} - 1))
    else
        val_idx=$((i - 1))
    fi
    val_subject=${ALL_SUBJECTS[$val_idx]}

    # Training subjects are all except test and val
    train_subjects=()
    for subj in "${ALL_SUBJECTS[@]}"; do
        if [ $subj -ne $test_subject ] && [ $subj -ne $val_subject ]; then
            train_subjects+=($subj)
        fi
    done

    # Convert array to space-separated string
    train_str="${train_subjects[*]}"

    # Assign GPU from selected GPU list using gpu_idx counter
    gpu_id=${GPU_IDS[$gpu_idx]}

    # Find embedding checkpoint for this test subject
    checkpoint_pattern="${EMBEDDING_DIR}/*test${test_subject}_val*/checkpoint_best.pth"
    encoder_checkpoint=$(ls $checkpoint_pattern 2>/dev/null | head -1)

    # Create experiment name
    exp_name="multi_finger_ranking_${TIMESTAMP}_test${test_subject}_val${val_subject}"

    # Log file for this session
    log_file="${LOG_DIR}/test_subject_${test_subject}.log"

    # Build command
    cmd="$PYTHON_BIN -m $SCRIPT \
        --train-subjects $train_str \
        --val-subjects $val_subject \
        --test-subjects $test_subject \
        --pretrained-embedding-checkpoint $encoder_checkpoint \
        --encoder-init embedding \
        --embedding-dim $EMBEDDING_DIM \
        --base-channels $BASE_CHANNELS \
        --hidden-dim $HIDDEN_DIM \
        --sequence-length $SEQUENCE_LENGTH \
        --sequence-stride $SEQUENCE_STRIDE \
        --num-transformer-layers $NUM_TRANSFORMER_LAYERS \
        --num-heads $NUM_HEADS \
        --secondary-loss-weight $SECONDARY_LOSS_WEIGHT \
        --inactive-loss-weight $INACTIVE_LOSS_WEIGHT \
        --uncertainty-weight $UNCERTAINTY_WEIGHT \
        --uncertainty-threshold-deg $UNCERTAINTY_THRESHOLD_DEG \
        --per-finger-rest-thresholds-deg $PER_FINGER_REST_THRESHOLDS \
        $([ "$DYNAMIC_SECONDARY" = true ] && echo '--dynamic-secondary') \
        --samples-per-epoch $SAMPLES_PER_EPOCH \
        --gpu-id $gpu_id \
        --num-workers $NUM_WORKERS \
        --batch-size $BATCH_SIZE \
        --dropout $DROPOUT \
        --phase1-epochs $PHASE1_EPOCHS \
        --phase2-epochs $PHASE2_EPOCHS \
        --phase1-lr $PHASE1_LR \
        --encoder-lr-early $ENCODER_LR_EARLY \
        --encoder-lr-late $ENCODER_LR_LATE \
        --transformer-lr $TRANSFORMER_LR \
        --head-lr $HEAD_LR \
        --save-freq $SAVE_FREQ \
        --exp-name $exp_name \
        $([ "$EXCLUDE_PLAYING" = true ] && echo '--exclude-playing') \
        $([ "$EXCLUDE_PINCH" = true ] && echo '--exclude-pinch') \
        $([ "$EXCLUDE_WRIST" = true ] && echo '--exclude-wrist')"

    # Launch in background
    echo -e "${BLUE}[Subject $((i+1))/${#ALL_SUBJECTS[@]}]${NC} Test=${GREEN}$test_subject${NC}, Val=${YELLOW}$val_subject${NC}, Train=[${train_str}]"
    echo "  GPU: $gpu_id"
    echo "  Encoder: $(basename $(dirname $encoder_checkpoint))"
    echo "  Log: $log_file"

    # Run in background and redirect output to log file
    $cmd > "$log_file" 2>&1 &

    # Store PID
    pids+=($!)

    echo "  PID: $!"
    echo ""

    # Increment GPU index for next selected subject
    gpu_idx=$((gpu_idx + 1))

    # Small delay to avoid overwhelming system
    sleep 3
done

echo "=========================================="
echo -e "${GREEN}All training sessions launched!${NC}"
echo "=========================================="
echo ""
echo "Process IDs:"
pid_idx=0
for selected_subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
    if [ $pid_idx -lt ${#pids[@]} ]; then
        gpu_id=${GPU_IDS[$pid_idx]}
        echo "  Test subject $selected_subj (GPU $gpu_id): ${pids[$pid_idx]}"
        pid_idx=$((pid_idx + 1))
    fi
done
echo ""

# Save PIDs to file for later reference
echo "${pids[@]}" > "${LOG_DIR}/pids.txt"
echo "PIDs saved to: ${LOG_DIR}/pids.txt"
echo ""

# Save configuration details
cat > "${LOG_DIR}/config.txt" << EOF
Cross-Subject Multi-Finger Ranking Training Configuration
==========================================================
Timestamp: $TIMESTAMP
Fingers: ALL 5 (index, middle, ring, pinky, thumb)
Total subjects available: ${#ALL_SUBJECTS[@]}
Selected test subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})
Batch size: $BATCH_SIZE
Dropout: $DROPOUT
Workers per session: $NUM_WORKERS

Temporal Configuration:
  Sequence length: $SEQUENCE_LENGTH frames (stride: $SEQUENCE_STRIDE)
  Transformer layers: $NUM_TRANSFORMER_LAYERS
  Attention heads: $NUM_HEADS

Model Configuration:
  Embedding dim: $EMBEDDING_DIM
  Base channels: $BASE_CHANNELS
  Hidden dim: $HIDDEN_DIM
  Encoder: Pretrained EmbeddingAutoencoder

Multi-Finger Loss Weights:
  Secondary active: $SECONDARY_LOSS_WEIGHT (dynamic=$DYNAMIC_SECONDARY)
  Inactive penalty: $INACTIVE_LOSS_WEIGHT

Pair-Level Uncertainty Penalty:
  Weight: $UNCERTAINTY_WEIGHT
  Uncertainty threshold: $UNCERTAINTY_THRESHOLD_DEG degrees (pair delta diff)
  Per-finger rest thresholds (idx/mid/ring/pnk/thb): $PER_FINGER_REST_THRESHOLDS

Training:
  Phase 1 epochs: $PHASE1_EPOCHS (frozen encoder)
  Phase 2 epochs: $PHASE2_EPOCHS (full fine-tuning)
  Phase 1 LR: $PHASE1_LR
  Encoder LR (early/late): $ENCODER_LR_EARLY / $ENCODER_LR_LATE
  Transformer LR: $TRANSFORMER_LR
  Head LR: $HEAD_LR
  Save frequency: Every $SAVE_FREQ epochs

Subject splits:
EOF

for i in "${!ALL_SUBJECTS[@]}"; do
    test_subject=${ALL_SUBJECTS[$i]}
    if [ $i -eq 0 ]; then
        val_idx=$((${#ALL_SUBJECTS[@]} - 1))
    else
        val_idx=$((i - 1))
    fi
    val_subject=${ALL_SUBJECTS[$val_idx]}

    train_subjects=()
    for subj in "${ALL_SUBJECTS[@]}"; do
        if [ $subj -ne $test_subject ] && [ $subj -ne $val_subject ]; then
            train_subjects+=($subj)
        fi
    done
    train_str="${train_subjects[*]}"

    echo "  Subject $test_subject: Val=$val_subject, Train=[$train_str]" >> "${LOG_DIR}/config.txt"
done

echo ""
echo "Configuration saved to: ${LOG_DIR}/config.txt"
echo ""

# Monitor option
echo "=========================================="
echo "Monitoring Options:"
echo "=========================================="
echo "1. Watch all logs:"
echo "   tail -f ${LOG_DIR}/*.log"
echo ""
echo "2. Watch specific subject:"
echo "   tail -f ${LOG_DIR}/test_subject_4.log"
echo ""
echo "3. Check GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "4. Kill all sessions:"
echo "   kill \$(cat ${LOG_DIR}/pids.txt)"
echo ""
echo "5. Check progress:"
echo "   grep -r 'PHASE 1\|PHASE 2\|Test Metrics' ${LOG_DIR}/*.log"
echo ""
echo "6. Check final results:"
echo "   grep -r 'New best mean AUC\|Test Metrics' ${LOG_DIR}/*.log"
echo ""
echo "7. Wait for all to complete:"
echo "   ./$(basename $0) --wait"
echo ""

# Wait option
if [ "$1" == "--wait" ]; then
    echo "=========================================="
    echo -e "${YELLOW}Waiting for all training sessions to complete...${NC}"
    echo "=========================================="
    echo ""

    # Wait for all background processes
    failed=0
    pid_idx=0
    for selected_subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
        if [ $pid_idx -lt ${#pids[@]} ]; then
            pid=${pids[$pid_idx]}
            gpu_id=${GPU_IDS[$pid_idx]}

            echo -n "Waiting for Test subject $selected_subj (GPU $gpu_id, PID $pid)... "

            if wait $pid; then
                echo -e "${GREEN}Done${NC}"
            else
                echo -e "${RED}Failed (exit code $?)${NC}"
                failed=$((failed + 1))
            fi
            pid_idx=$((pid_idx + 1))
        fi
    done

    echo ""
    echo "=========================================="
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All training sessions completed successfully!${NC}"
        echo ""
        echo "Collecting results..."
        echo ""
        for selected_subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
            log_file="${LOG_DIR}/test_subject_${selected_subj}.log"

            echo -e "${BLUE}Subject $selected_subj:${NC}"
            grep "Test Metrics:" -A 5 "$log_file" | tail -6 || echo "  No results found"
            echo ""
        done
    else
        echo -e "${RED}$failed session(s) failed!${NC}"
    fi
    echo "=========================================="
else
    echo "Training sessions running in background."
    echo "Run with --wait flag to wait for completion:"
    echo "  ./$(basename $0) --wait"
    echo ""
fi

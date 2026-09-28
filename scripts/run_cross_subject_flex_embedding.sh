#!/bin/bash
#
# Cross-subject SonoRank training (Level 3: flex classification).
# Fine-tunes the ranking backbone from Level 2 with a classification head
# that predicts per-finger flex/no-flex using a rest reference.
#
# This produces the SonoRank model from the paper.
# For ablation baselines (MFT, MFT+Rest), see the flags below.
#
# IMPORTANT: Requires completed Level 2 checkpoints from
#   run_cross_subject_multi_finger_ranking.sh
# Reader's guide: the launcher resolves the matching ranking checkpoint for
# each fold, then passes the model-variant flags below to produce SonoRank or
# one of its direct-classification ablations.
#

# Configuration
PYTHON_BIN="/home/deanz/anaconda3/envs/nhst39/bin/python"
SCRIPT="trainer.train_flex_embedding"
NUM_WORKERS=4
BATCH_SIZE=8

# MLP head parameters
MLP_HIDDEN_DIM=128
MLP_DROPOUT=0.3

# Window parameters (sequence length T is auto-detected from ranking model)
WINDOW_STRIDE=1
SAMPLES_PER_EPOCH=50000

# Flex label parameters
FLEX_THRESHOLD_DEG=10.0
REST_THRESHOLD_DEG=5.0
VEL_THRESHOLD_DEG_S=80.0
VEL_SMOOTH_WINDOW=5
ANGLE_PCT_THRESHOLD=0.1
THUMB_VEL_THRESHOLD_DEG_S=50.0
THUMB_ANGLE_PCT_THRESHOLD=0.1

# Training hyperparameters — Phase 1 (head only)
PHASE1_EPOCHS=0
PHASE1_LR=1e-3
FREEZE_BACKBONE=false

# Training hyperparameters — Phase 2 (full fine-tuning)
PHASE2_EPOCHS=10
ENCODER_LR_EARLY=1e-5
ENCODER_LR_LATE=5e-5
TRANSFORMER_LR=1e-4
FLEX_HEAD_LR=3e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_FREQ=10
LOSS=bce
FOCAL_GAMMA=2.0
THRESHOLD=0.5
EXCLUDE_PLAYING=true
EXCLUDE_PINCH=true
EXCLUDE_WRIST=true
FLEX_WEIGHTED_SAMPLING=true
# --- Model variant flags ---
# SonoRank (default): ENCODER_ONLY=false, NO_REST=false
#   Loads full ranking backbone (encoder + transformer + comparison heads),
#   uses rest reference for classification.
# MFT+Rest ablation: ENCODER_ONLY=true, NO_REST=false
#   Loads only CNN encoder weights, skips ranking pretraining.
# MFT ablation:      ENCODER_ONLY=true, NO_REST=true
#   Loads only CNN encoder weights, no rest reference.
ENCODER_ONLY=false
NO_REST=false
USE_FIXED_LABELS=true

# All 12 subjects (used for train/val/test splits)
ALL_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# First 8 subjects (2 per GPU, already completed):
# SELECTED_TEST_SUBJECTS=(1 4 5 6 10 17 19 20)
# Last 4 subjects (1 per GPU)
SELECTED_TEST_SUBJECTS=(21 23 24 25)
# All 12 subjects:
# SELECTED_TEST_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# GPU assignment: 1 subject per GPU
GPU_IDS=(4 5 6 7)
# All 12 subjects (3 per GPU):
# GPU_IDS=(0 0 0 1 1 1 2 2 2 3 3 3)

# Path to Level 2 (multi-finger ranking) checkpoints
RANKING_DIR="outputs/multi_finger_ranking"
RANKING_PREFIX="multi_finger_ranking"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "==========================================="
echo "SonoRank: Flex Classification (Level 3)"
echo "==========================================="
echo "Fine-tunes ranking backbone with classification head"
echo "Ranking backbone from Level 2 (multi-finger ranking)"
echo ""
echo "Total subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})"
echo "GPUs: 0-3 (3 subjects per GPU)"
echo "Workers per session: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "MLP Head:"
echo "  Hidden dim: $MLP_HIDDEN_DIM"
echo "  Dropout: $MLP_DROPOUT"
echo ""
echo "Window Configuration:"
echo "  Sequence length: auto-detected from ranking checkpoint"
echo "  Window stride: $WINDOW_STRIDE raw frames"
echo "  Samples per epoch: $SAMPLES_PER_EPOCH"
echo ""
echo "Label Configuration:"
echo "  Velocity threshold: $VEL_THRESHOLD_DEG_S deg/s"
echo "  Velocity smooth window: $VEL_SMOOTH_WINDOW frames"
echo "  Angle pct threshold: $ANGLE_PCT_THRESHOLD"
echo "  Thumb velocity threshold: $THUMB_VEL_THRESHOLD_DEG_S deg/s"
echo "  Thumb angle pct threshold: $THUMB_ANGLE_PCT_THRESHOLD"
echo "  Rest threshold: $REST_THRESHOLD_DEG degrees"
echo ""
echo "Training (matching seqlen sweep):"
echo "  Phase 1: $PHASE1_EPOCHS epochs"
echo "  Phase 2: $PHASE2_EPOCHS epochs (fine-tuning)"
echo "  Encoder LR (early/late): $ENCODER_LR_EARLY / $ENCODER_LR_LATE"
echo "  Transformer LR: $TRANSFORMER_LR"
echo "  Flex head LR: $FLEX_HEAD_LR"
echo "  Loss: $LOSS"
echo "  Encoder only: $ENCODER_ONLY"
echo "  No rest reference: $NO_REST"
echo "  Weight decay: $WEIGHT_DECAY"
echo ""
echo "Source checkpoints: ${RANKING_DIR}/${RANKING_PREFIX}_*"
echo "==========================================="
echo ""

# Create logs directory
mkdir -p logs/cross_subject_flex_embedding
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cross_subject_flex_embedding/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Check for seqlen sweep checkpoints
echo "Checking for Level 2 (ranking) checkpoints..."
missing_checkpoints=0
for test_subject in "${SELECTED_TEST_SUBJECTS[@]}"; do
    checkpoint_pattern="${RANKING_DIR}/${RANKING_PREFIX}_test${test_subject}_val*/checkpoint_best.pth"
    checkpoint=$(ls $checkpoint_pattern 2>/dev/null | head -1)

    if [ -z "$checkpoint" ]; then
        echo -e "${RED}  Missing checkpoint for test subject $test_subject${NC}"
        missing_checkpoints=$((missing_checkpoints + 1))
    else
        echo -e "${GREEN}  Found: $(basename $(dirname $checkpoint))${NC}"
    fi
done

if [ $missing_checkpoints -gt 0 ]; then
    echo ""
    echo -e "${RED}Error: Missing $missing_checkpoints checkpoint(s)!${NC}"
    echo "Please run run_cross_subject_multi_finger_ranking.sh first."
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

    # Assign GPU
    gpu_id=${GPU_IDS[$gpu_idx]}

    # Find seqlen sweep checkpoint for this test subject
    checkpoint_pattern="${RANKING_DIR}/${RANKING_PREFIX}_test${test_subject}_val*/checkpoint_best.pth"
    ranking_checkpoint=$(ls $checkpoint_pattern 2>/dev/null | head -1)

    # Create experiment name
    exp_name="flex_embedding_${TIMESTAMP}_test${test_subject}_val${val_subject}"

    # Log file for this session
    log_file="${LOG_DIR}/test_subject_${test_subject}.log"

    # Build command
    cmd="$PYTHON_BIN -m $SCRIPT \
        --train-subjects $train_str \
        --val-subjects $val_subject \
        --test-subjects $test_subject \
        --ranking-checkpoint $ranking_checkpoint \
        --mlp-hidden-dim $MLP_HIDDEN_DIM \
        --mlp-dropout $MLP_DROPOUT \
        --window-stride $WINDOW_STRIDE \
        --samples-per-epoch $SAMPLES_PER_EPOCH \
        --flex-threshold-deg $FLEX_THRESHOLD_DEG \
        --rest-threshold-deg $REST_THRESHOLD_DEG \
        --vel-threshold-deg-s $VEL_THRESHOLD_DEG_S \
        --vel-smooth-window $VEL_SMOOTH_WINDOW \
        --angle-pct-threshold $ANGLE_PCT_THRESHOLD \
        --thumb-vel-threshold-deg-s $THUMB_VEL_THRESHOLD_DEG_S \
        --thumb-angle-pct-threshold $THUMB_ANGLE_PCT_THRESHOLD \
        --phase1-epochs $PHASE1_EPOCHS \
        --phase1-lr $PHASE1_LR \
        --phase2-epochs $PHASE2_EPOCHS \
        --encoder-lr-early $ENCODER_LR_EARLY \
        --encoder-lr-late $ENCODER_LR_LATE \
        --transformer-lr $TRANSFORMER_LR \
        --flex-head-lr $FLEX_HEAD_LR \
        --weight-decay $WEIGHT_DECAY \
        --grad-clip $GRAD_CLIP \
        --batch-size $BATCH_SIZE \
        --save-freq $SAVE_FREQ \
        --loss $LOSS \
        --focal-gamma $FOCAL_GAMMA \
        --threshold $THRESHOLD \
        --gpu-id $gpu_id \
        --num-workers $NUM_WORKERS \
        --exp-name $exp_name"

    # Append boolean flags
    if [ "$EXCLUDE_PLAYING" = true ]; then
        cmd="$cmd --exclude-playing"
    fi
    if [ "$EXCLUDE_PINCH" = true ]; then
        cmd="$cmd --exclude-pinch"
    fi
    if [ "$EXCLUDE_WRIST" = true ]; then
        cmd="$cmd --exclude-wrist"
    fi
    if [ "$FLEX_WEIGHTED_SAMPLING" = true ]; then
        cmd="$cmd --flex-weighted-sampling"
    fi
    if [ "$ENCODER_ONLY" = true ]; then
        cmd="$cmd --encoder-only"
    fi
    if [ "$FREEZE_BACKBONE" = true ]; then
        cmd="$cmd --freeze-backbone"
    fi
    if [ "$NO_REST" = true ]; then
        cmd="$cmd --no-rest"
    fi
    if [ "$USE_FIXED_LABELS" = true ]; then
        cmd="$cmd --use-fixed-labels"
    fi

    # Launch in background
    echo -e "${BLUE}[Subject $((i+1))/${#ALL_SUBJECTS[@]}]${NC} Test=${GREEN}$test_subject${NC}, Val=${YELLOW}$val_subject${NC}, Train=[${train_str}]"
    echo "  GPU: $gpu_id"
    echo "  Ranking model: $(basename $(dirname $ranking_checkpoint))"
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

echo "==========================================="
echo -e "${GREEN}All training sessions launched!${NC}"
echo "==========================================="
echo ""
echo "Process IDs:"
for pid_idx in "${!pids[@]}"; do
    echo "  ${SELECTED_TEST_SUBJECTS[$pid_idx]} (GPU ${GPU_IDS[$pid_idx]}): ${pids[$pid_idx]}"
done
echo ""

# Save PIDs to file for later reference
echo "${pids[@]}" > "${LOG_DIR}/pids.txt"
echo "PIDs saved to: ${LOG_DIR}/pids.txt"
echo ""

# Save configuration details
cat > "${LOG_DIR}/config.txt" << EOF
SonoRank Flex Classification (Level 3) Training Configuration
==============================================================
Timestamp: $TIMESTAMP
Source checkpoints: ${RANKING_DIR}/${RANKING_PREFIX}_*
Encoder only: $ENCODER_ONLY
No rest reference: $NO_REST
Test subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})
Batch size: $BATCH_SIZE
Workers per session: $NUM_WORKERS

MLP Head:
  Hidden dim: $MLP_HIDDEN_DIM
  Dropout: $MLP_DROPOUT

Window Configuration:
  Sequence length: auto-detected from ranking checkpoint
  Window stride: $WINDOW_STRIDE raw frames
  Samples per epoch: $SAMPLES_PER_EPOCH

Label Configuration:
  Velocity threshold: $VEL_THRESHOLD_DEG_S deg/s
  Velocity smooth window: $VEL_SMOOTH_WINDOW frames
  Angle pct threshold: $ANGLE_PCT_THRESHOLD
  Thumb velocity threshold: $THUMB_VEL_THRESHOLD_DEG_S deg/s
  Thumb angle pct threshold: $THUMB_ANGLE_PCT_THRESHOLD
  Rest threshold: $REST_THRESHOLD_DEG degrees

Training (matching seqlen sweep):
  Phase 1: $PHASE1_EPOCHS epochs
  Phase 2: $PHASE2_EPOCHS epochs (fine-tuning)
  Encoder LR (early/late): $ENCODER_LR_EARLY / $ENCODER_LR_LATE
  Transformer LR: $TRANSFORMER_LR
  Flex head LR: $FLEX_HEAD_LR
  Weight decay: $WEIGHT_DECAY
  Loss: $LOSS
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
echo "==========================================="
echo "Monitoring Options:"
echo "==========================================="
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
echo "   grep -r 'Epoch\|New best' ${LOG_DIR}/*.log"
echo ""
echo "6. Check final results:"
echo "   grep -r 'New best model\|TRAINING COMPLETE' ${LOG_DIR}/*.log"
echo ""

# Wait option
if [ "$1" == "--wait" ]; then
    echo "==========================================="
    echo -e "${YELLOW}Waiting for all training sessions to complete...${NC}"
    echo "==========================================="
    echo ""

    # Wait for all background processes
    failed=0
    for pid_idx in "${!pids[@]}"; do
        pid=${pids[$pid_idx]}
        subj=${SELECTED_TEST_SUBJECTS[$pid_idx]}
        gpu_id=${GPU_IDS[$pid_idx]}

        echo -n "Waiting for Test subject $subj (GPU $gpu_id, PID $pid)... "

        if wait $pid; then
            echo -e "${GREEN}Done${NC}"
        else
            echo -e "${RED}Failed (exit code $?)${NC}"
            failed=$((failed + 1))
        fi
    done

    echo ""
    echo "==========================================="
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All training sessions completed successfully!${NC}"
        echo ""
        echo "Collecting results..."
        echo ""
        for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
            log_file="${LOG_DIR}/test_subject_${subj}.log"

            echo -e "${BLUE}Subject $subj:${NC}"
            grep "Best Val Mean F1\|TRAINING COMPLETE" "$log_file" | tail -3 || echo "  No results found"
            echo ""
        done
    else
        echo -e "${RED}$failed session(s) failed!${NC}"
    fi
    echo "==========================================="
else
    echo "Training sessions running in background."
    echo "Run with --wait flag to wait for completion:"
    echo "  ./$(basename $0) --wait"
    echo ""
fi

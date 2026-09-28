#!/bin/bash
#
# Cross-subject Bimbraw 2023 (Modified VGG16) baseline training.
# Single-frame CNN flex classification supporting 12-fold leave-one-subject-out.
# The checked-in subject array contains one six-fold batch from the original run.
# Reader's guide: the loop derives train and validation subjects for each
# selected test fold, launches one independent process, and records its PID and
# configuration under a timestamped log directory.
#

# Configuration
PYTHON_BIN="/home/deanz/anaconda3/envs/nhst39/bin/python"
SCRIPT="trainer.train_bimbraw"
NUM_WORKERS=4
BATCH_SIZE=32

# Training hyperparameters
EPOCHS=10
LR=0.001
DROPOUT=0.5
GRAD_CLIP=1.0
THRESHOLD=0.5
SAVE_FREQ=10
SAMPLES_PER_EPOCH=50000

# Flex label parameters
VEL_THRESHOLD_DEG_S=80.0
VEL_SMOOTH_WINDOW=5
ANGLE_PCT_THRESHOLD=0.1
THUMB_VEL_THRESHOLD_DEG_S=50.0
THUMB_ANGLE_PCT_THRESHOLD=0.1

# Data filtering
EXCLUDE_PLAYING=true
EXCLUDE_PINCH=true
EXCLUDE_WRIST=true
USE_FIXED_LABELS=true

# All 12 subjects
ALL_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# Select which subjects to run
# SELECTED_TEST_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)
# SELECTED_TEST_SUBJECTS=(1 4 5 6 10 17)
SELECTED_TEST_SUBJECTS=(19 20 21 23 24 25)

# GPU assignment
# GPU_IDS=(0 0 0 0 1 1 1 1 2 2 2 2)
GPU_IDS=(0 0 1 1 3 3)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "==========================================="
echo "Bimbraw VGG Baseline Cross-Subject Training"
echo "==========================================="
echo "Modified VGG16: single-frame flex classification"
echo ""
echo "Total subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})"
echo "GPUs: ${GPU_IDS[*]}"
echo ""
echo "Training:"
echo "  Epochs: $EPOCHS, LR: $LR, Dropout: $DROPOUT"
echo "  Batch size: $BATCH_SIZE"
echo ""
echo "Fixed labels: $USE_FIXED_LABELS"
echo "==========================================="
echo ""

# Create logs directory
mkdir -p logs/cross_subject_bimbraw
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cross_subject_bimbraw/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Array to store background process IDs
pids=()
gpu_idx=0

# Launch training sessions
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

    if [ $is_selected -eq 0 ]; then
        echo -e "${BLUE}[Subject $((i+1))/${#ALL_SUBJECTS[@]}]${NC} Test=${GREEN}$test_subject${NC} - ${YELLOW}SKIPPED${NC}"
        continue
    fi

    # Validation subject: previous in list (circular)
    if [ $i -eq 0 ]; then
        val_idx=$((${#ALL_SUBJECTS[@]} - 1))
    else
        val_idx=$((i - 1))
    fi
    val_subject=${ALL_SUBJECTS[$val_idx]}

    # Training subjects: all except test and val
    train_subjects=()
    for subj in "${ALL_SUBJECTS[@]}"; do
        if [ $subj -ne $test_subject ] && [ $subj -ne $val_subject ]; then
            train_subjects+=($subj)
        fi
    done
    train_str="${train_subjects[*]}"

    # Assign GPU
    gpu_id=${GPU_IDS[$gpu_idx]}

    # Experiment name
    exp_name="bimbraw_${TIMESTAMP}_test${test_subject}_val${val_subject}"

    # Log file
    log_file="${LOG_DIR}/test_subject_${test_subject}.log"

    # Build command
    cmd="$PYTHON_BIN -m $SCRIPT \
        --train-subjects $train_str \
        --val-subjects $val_subject \
        --test-subjects $test_subject \
        --epochs $EPOCHS \
        --lr $LR \
        --dropout $DROPOUT \
        --grad-clip $GRAD_CLIP \
        --threshold $THRESHOLD \
        --batch-size $BATCH_SIZE \
        --save-freq $SAVE_FREQ \
        --gpu-id $gpu_id \
        --num-workers $NUM_WORKERS \
        --vel-threshold-deg-s $VEL_THRESHOLD_DEG_S \
        --vel-smooth-window $VEL_SMOOTH_WINDOW \
        --angle-pct-threshold $ANGLE_PCT_THRESHOLD \
        --thumb-vel-threshold-deg-s $THUMB_VEL_THRESHOLD_DEG_S \
        --thumb-angle-pct-threshold $THUMB_ANGLE_PCT_THRESHOLD \
        --samples-per-epoch $SAMPLES_PER_EPOCH \
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
    if [ "$USE_FIXED_LABELS" = true ]; then
        cmd="$cmd --use-fixed-labels"
    fi

    echo -e "${BLUE}[Subject $((i+1))/${#ALL_SUBJECTS[@]}]${NC} Test=${GREEN}$test_subject${NC}, Val=${YELLOW}$val_subject${NC}, Train=[${train_str}]"
    echo "  GPU: $gpu_id | Log: $log_file"

    $cmd > "$log_file" 2>&1 &
    pids+=($!)
    echo "  PID: $!"
    echo ""

    gpu_idx=$((gpu_idx + 1))
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

echo "${pids[@]}" > "${LOG_DIR}/pids.txt"
echo "PIDs saved to: ${LOG_DIR}/pids.txt"
echo ""

# Save configuration
cat > "${LOG_DIR}/config.txt" << EOF
Bimbraw VGG Baseline Cross-Subject Training Configuration
============================================================
Timestamp: $TIMESTAMP
Architecture: Modified VGG16 (5 blocks: 64->128->256->512->512)
Epochs: $EPOCHS
LR: $LR, Dropout: $DROPOUT
Batch size: $BATCH_SIZE
Fixed labels: $USE_FIXED_LABELS
EOF

echo "Configuration saved to: ${LOG_DIR}/config.txt"
echo ""

echo "==========================================="
echo "Monitoring:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  grep -r 'best' ${LOG_DIR}/*.log"
echo "  kill \$(cat ${LOG_DIR}/pids.txt)"
echo "==========================================="

# Wait option
if [ "$1" == "--wait" ]; then
    echo ""
    echo -e "${YELLOW}Waiting for all training sessions to complete...${NC}"
    echo ""

    failed=0
    for pid_idx in "${!pids[@]}"; do
        pid=${pids[$pid_idx]}
        subj=${SELECTED_TEST_SUBJECTS[$pid_idx]}
        echo -n "Waiting for Test subject $subj (PID $pid)... "
        if wait $pid; then
            echo -e "${GREEN}Done${NC}"
        else
            echo -e "${RED}Failed (exit code $?)${NC}"
            failed=$((failed + 1))
        fi
    done

    echo ""
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All training sessions completed successfully!${NC}"
        echo ""
        for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
            echo -e "${BLUE}Subject $subj:${NC}"
            grep "Best Val Mean F1\|TRAINING COMPLETE" "${LOG_DIR}/test_subject_${subj}.log" | tail -3 || echo "  No results found"
            echo ""
        done
    else
        echo -e "${RED}$failed session(s) failed!${NC}"
    fi
fi

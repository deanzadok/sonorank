#!/bin/bash
#
# Cross-subject validation training script for embedding autoencoder
# Supports all 12 test subjects, with the previous subject used for validation.
# The checked-in subject array contains one six-fold batch from the original run.
# Trains autoencoder to compress 224x224 images to 64-dim embeddings
# Reader's guide: this launches Level 1. Every selected test fold trains its own
# encoder so no test-subject frames can leak into the checkpoints later loaded
# by the corresponding ranking fold.
#

# Configuration
PYTHON_BIN="/home/deanz/anaconda3/envs/nhst39/bin/python"
SCRIPT="trainer/train_embedding.py"
NUM_WORKERS=8
BATCH_SIZE=32
IMAGE_SIZE=224

# Model architecture
EMBEDDING_DIM=64
BASE_CHANNELS=64

# Training hyperparameters
EPOCHS=50
LR=1e-4
WEIGHT_DECAY=1e-4
SAVE_FREQ=10

# Session types (gestures only by default)
SESSION_TYPES="finger_flex wrist pinch"

# All 12 subjects (used for train/val/test splits)
ALL_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# Selected 6 representative subjects to actually train on
SELECTED_TEST_SUBJECTS=(1 5 6 19 20 21)

# GPU assignment (2 sessions per GPU across 3 GPUs for 6 subjects)
GPU_IDS=(1 1 2 2 3 3)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Cross-Subject Embedding Autoencoder Training"
echo "=========================================="
echo "Total subjects available: ${#ALL_SUBJECTS[@]}"
echo "Selected test subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})"
echo "GPUs: 1, 2, 3 (2 trials per GPU)"
echo "Workers per session: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Session types: $SESSION_TYPES"
echo ""
echo "Model Configuration:"
echo "  Embedding dimension: $EMBEDDING_DIM"
echo "  Base channels: $BASE_CHANNELS"
echo ""
echo "Training:"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Save frequency: Every $SAVE_FREQ epochs"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs/cross_subject_embedding
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cross_subject_embedding/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
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

    # Create experiment name
    exp_name="embedding_${TIMESTAMP}_test${test_subject}_val${val_subject}"

    # Log file for this session
    log_file="${LOG_DIR}/test_subject_${test_subject}.log"

    # Build command
    cmd="$PYTHON_BIN -m trainer.train_embedding \
        --train-subjects $train_str \
        --val-subjects $val_subject \
        --test-subjects $test_subject \
        --gpu-id $gpu_id \
        --num-workers $NUM_WORKERS \
        --batch-size $BATCH_SIZE \
        --image-size $IMAGE_SIZE \
        --embedding-dim $EMBEDDING_DIM \
        --base-channels $BASE_CHANNELS \
        --epochs $EPOCHS \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --save-freq $SAVE_FREQ \
        --session-types $SESSION_TYPES \
        --exp-name $exp_name"

    # Launch in background
    echo -e "${BLUE}[Subject $((i+1))/${#ALL_SUBJECTS[@]}]${NC} Test=${GREEN}$test_subject${NC}, Val=${YELLOW}$val_subject${NC}, Train=[${train_str}]"
    echo "  GPU: $gpu_id"
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
Cross-Subject Embedding Autoencoder Training Configuration
===========================================================
Timestamp: $TIMESTAMP
Total subjects available: ${#ALL_SUBJECTS[@]}
Selected test subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})
Batch size: $BATCH_SIZE
Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}
Workers per session: $NUM_WORKERS
Session types: $SESSION_TYPES
Epochs: $EPOCHS
Learning rate: $LR
Weight decay: $WEIGHT_DECAY
Save frequency: Every $SAVE_FREQ epochs

Model Architecture:
  Embedding dimension: $EMBEDDING_DIM
  Base channels: $BASE_CHANNELS
  Architecture: ResNet-style autoencoder (~22M params)

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
echo "5. Check progress (search for epoch metrics):"
echo "   grep -r 'Epoch.*Summary\\|Val Loss' ${LOG_DIR}/*.log"
echo ""
echo "6. Check final results:"
echo "   grep -r 'New best model\\|TRAINING COMPLETE' ${LOG_DIR}/*.log"
echo ""
echo "7. Check current epoch for each subject:"
echo "   for f in ${LOG_DIR}/*.log; do echo \"\$f:\"; grep -o 'Epoch [0-9]*/[0-9]*' \"\$f\" | tail -1; done"
echo ""
echo "8. Wait for all to complete:"
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
            grep "New best model" "$log_file" | tail -1 || echo "  No results found"
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

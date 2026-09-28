#!/bin/bash
#
# Cross-subject CBMF baseline training.
# Trains from scratch (no pretrained backbone) using the two-stage CBMF approach:
#   Phase 1: Joint angle prediction (MSE)
#   Phase 2: Flex classification (lambda*MSE + BCE)
# Reader's guide: each selected test subject is a separate LOSO fold. The GPU
# and subject arrays below describe the checked-in batch, not available
# hardware, and must be reviewed before execution.
#

# Configuration
PYTHON_BIN="/home/deanz/anaconda3/envs/nhst39/bin/python"
SCRIPT="trainer.train_cbmf"
NUM_WORKERS=4
BATCH_SIZE=8

# Architecture (matching original CBMF paper)
N_FILTERS=32
IR_DIM=128
ENCODER_DROPOUT=0.3
ENC_GRU1_HIDDEN=1024
ENC_GRU2_HIDDEN=256
DEC_GRU1_HIDDEN=256
DEC_GRU2_HIDDEN=128

# Sequence parameters (matching SonoRank for fair comparison)
SEQUENCE_LENGTH=16
SEQUENCE_STRIDE=2
WINDOW_STRIDE=1
SAMPLES_PER_EPOCH=50000

# Flex label parameters
VEL_THRESHOLD_DEG_S=80.0
VEL_SMOOTH_WINDOW=5
ANGLE_PCT_THRESHOLD=0.1
THUMB_VEL_THRESHOLD_DEG_S=50.0
THUMB_ANGLE_PCT_THRESHOLD=0.1

# Training hyperparameters (matching original CBMF paper)
PHASE1_EPOCHS=10
PHASE2_EPOCHS=10
LR=0.001
LAMBDA_CONFIG=4.0
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0
SAVE_FREQ=10
THRESHOLD=0.5

# Data filtering
EXCLUDE_PLAYING=true
EXCLUDE_PINCH=true
EXCLUDE_WRIST=true
USE_FIXED_LABELS=true

# All 12 subjects
ALL_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# Select which subjects to run (all 12 for full cross-validation)
SELECTED_TEST_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# GPU assignment: 3 GPUs, 4 subjects each
GPU_IDS=(0 0 0 0 1 1 1 1 2 2 2 2)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "==========================================="
echo "CBMF Baseline Cross-Subject Training"
echo "==========================================="
echo "Two-stage: Phase 1 (joint angles, MSE) → Phase 2 (flex, λ*MSE+BCE)"
echo "Trains from scratch (no pretrained backbone)"
echo ""
echo "Total subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})"
echo "GPUs: ${GPU_IDS[*]}"
echo ""
echo "Architecture:"
echo "  UNet encoder: n_filters=$N_FILTERS, ir_dim=$IR_DIM"
echo "  Encoder GRU: $IR_DIM → $ENC_GRU1_HIDDEN → $ENC_GRU2_HIDDEN → 10 joints"
echo "  Decoder GRU: $(($IR_DIM + 10)) → $DEC_GRU1_HIDDEN → $DEC_GRU2_HIDDEN → 5 fingers"
echo ""
echo "Training:"
echo "  Phase 1: $PHASE1_EPOCHS epochs (joint angle prediction)"
echo "  Phase 2: $PHASE2_EPOCHS epochs (flex classification, λ=$LAMBDA_CONFIG)"
echo "  LR: $LR (Adam), batch_size: $BATCH_SIZE"
echo ""
echo "Sequence: length=$SEQUENCE_LENGTH, stride=$SEQUENCE_STRIDE"
echo "Fixed labels: $USE_FIXED_LABELS"
echo "==========================================="
echo ""

# Create logs directory
mkdir -p logs/cross_subject_cbmf
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cross_subject_cbmf/${TIMESTAMP}"
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
    exp_name="cbmf_${TIMESTAMP}_test${test_subject}_val${val_subject}"

    # Log file
    log_file="${LOG_DIR}/test_subject_${test_subject}.log"

    # Build command
    cmd="$PYTHON_BIN -m $SCRIPT \
        --train-subjects $train_str \
        --val-subjects $val_subject \
        --test-subjects $test_subject \
        --n-filters $N_FILTERS \
        --ir-dim $IR_DIM \
        --encoder-dropout $ENCODER_DROPOUT \
        --enc-gru1-hidden $ENC_GRU1_HIDDEN \
        --enc-gru2-hidden $ENC_GRU2_HIDDEN \
        --dec-gru1-hidden $DEC_GRU1_HIDDEN \
        --dec-gru2-hidden $DEC_GRU2_HIDDEN \
        --sequence-length $SEQUENCE_LENGTH \
        --sequence-stride $SEQUENCE_STRIDE \
        --window-stride $WINDOW_STRIDE \
        --samples-per-epoch $SAMPLES_PER_EPOCH \
        --vel-threshold-deg-s $VEL_THRESHOLD_DEG_S \
        --vel-smooth-window $VEL_SMOOTH_WINDOW \
        --angle-pct-threshold $ANGLE_PCT_THRESHOLD \
        --thumb-vel-threshold-deg-s $THUMB_VEL_THRESHOLD_DEG_S \
        --thumb-angle-pct-threshold $THUMB_ANGLE_PCT_THRESHOLD \
        --phase1-epochs $PHASE1_EPOCHS \
        --phase2-epochs $PHASE2_EPOCHS \
        --lr $LR \
        --lambda-config $LAMBDA_CONFIG \
        --weight-decay $WEIGHT_DECAY \
        --grad-clip $GRAD_CLIP \
        --batch-size $BATCH_SIZE \
        --save-freq $SAVE_FREQ \
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
CBMF Baseline Cross-Subject Training Configuration
=====================================================
Timestamp: $TIMESTAMP
Architecture: UNet($N_FILTERS) + GRU($ENC_GRU1_HIDDEN→$ENC_GRU2_HIDDEN→10) + Decoder($DEC_GRU1_HIDDEN→$DEC_GRU2_HIDDEN→5)
IR dim: $IR_DIM
Sequence: length=$SEQUENCE_LENGTH, stride=$SEQUENCE_STRIDE
Phase 1: $PHASE1_EPOCHS epochs (MSE, lr=$LR)
Phase 2: $PHASE2_EPOCHS epochs (λ=$LAMBDA_CONFIG * MSE + BCE, lr=$LR)
Batch size: $BATCH_SIZE
Fixed labels: $USE_FIXED_LABELS
EOF

echo "Configuration saved to: ${LOG_DIR}/config.txt"
echo ""

echo "==========================================="
echo "Monitoring:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  grep -r 'New best' ${LOG_DIR}/*.log"
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

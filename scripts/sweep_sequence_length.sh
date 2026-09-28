#!/bin/bash
#
# Sequence length sweep for multi-finger temporal ranking
# Runs each seq_len value on 4 pilot subjects (one per GPU), waits for completion,
# then moves to the next seq_len.
#
# Usage:
#   ./scripts/sweep_sequence_length.sh                  # Run all seq_len values
#   ./scripts/sweep_sequence_length.sh 8 12             # Run specific values only
#   SUBJECTS=all ./scripts/sweep_sequence_length.sh     # Run on all 12 subjects
# Reader's guide: values are processed sequentially, while folds for one value
# run in parallel. The script waits for the entire batch before advancing so
# results from different window lengths do not compete for the same GPUs.
#

# Configuration (shared with run_cross_subject_multi_finger_ranking.sh)
PYTHON_BIN="/home/deanz/anaconda3/envs/nhst39/bin/python"
SCRIPT="trainer.train_multi_finger_ranking"
NUM_WORKERS=4
BATCH_SIZE=8
DROPOUT=0.3
EXCLUDE_PLAYING=true
EXCLUDE_PINCH=true
EXCLUDE_WRIST=true

# Fixed: stride=2 (chosen from prior ablation)
SEQUENCE_STRIDE=2

# Model architecture (must match EmbeddingAutoencoder)
EMBEDDING_DIM=64
BASE_CHANNELS=64
HIDDEN_DIM=256
NUM_TRANSFORMER_LAYERS=2
NUM_HEADS=4

# Multi-finger loss weights
SECONDARY_LOSS_WEIGHT=0.3
INACTIVE_LOSS_WEIGHT=0.1
DYNAMIC_SECONDARY=true

# Pair-level uncertainty penalty
UNCERTAINTY_WEIGHT=1.0
UNCERTAINTY_THRESHOLD_DEG=5.0
PER_FINGER_REST_THRESHOLDS="17.5 17.5 17.5 12.5 7.5"

# Sampling
SAMPLES_PER_EPOCH=50000

# Training
PHASE1_EPOCHS=0
PHASE2_EPOCHS=10
PHASE1_LR=1e-3
ENCODER_LR_EARLY=1e-5
ENCODER_LR_LATE=5e-5
TRANSFORMER_LR=1e-4
HEAD_LR=3e-4
SAVE_FREQ=10

# All 12 subjects
ALL_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# Subject selection: 4 pilot subjects or all 12
if [ "${SUBJECTS}" = "all" ]; then
    SELECTED_TEST_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)
else
    # Pilot subjects (stride=3, already completed):
    # SELECTED_TEST_SUBJECTS=(4 10 17 23)
    # Batch 2 (stride=2, already completed):
    # SELECTED_TEST_SUBJECTS=(1 5 6 19)
    SELECTED_TEST_SUBJECTS=(24 25)
fi

# GPU assignment
GPU_IDS=(0 1)

# Embedding checkpoints directory
EMBEDDING_DIR="outputs/embedding"

# Sequence lengths to sweep (override with positional args)
if [ $# -gt 0 ]; then
    SEQ_LENGTHS=("$@")
else
    SEQ_LENGTHS=(2 4 8 16 24)
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SWEEP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_BASE="logs/sweep_sequence_length/${SWEEP_TIMESTAMP}"
mkdir -p "$LOG_BASE"

echo "=========================================="
echo "Sequence Length Sweep"
echo "=========================================="
echo "Stride: $SEQUENCE_STRIDE (fixed)"
echo "Sequence lengths: ${SEQ_LENGTHS[*]}"
echo "Test subjects: ${SELECTED_TEST_SUBJECTS[*]}"
echo "GPUs: ${GPU_IDS[*]}"
echo "Exclude: playing=$EXCLUDE_PLAYING pinch=$EXCLUDE_PINCH wrist=$EXCLUDE_WRIST"
echo ""
echo "Temporal spans at stride=$SEQUENCE_STRIDE:"
for sl in "${SEQ_LENGTHS[@]}"; do
    span=$(( (sl - 1) * SEQUENCE_STRIDE + 1 ))
    secs=$(echo "scale=2; $span / 20.0" | bc)
    echo "  seq_len=$sl → span=${span} frames (${secs}s)"
done
echo ""
echo "Log directory: $LOG_BASE"
echo "=========================================="
echo ""

# Save sweep config
cat > "${LOG_BASE}/sweep_config.txt" << EOF
Sequence Length Sweep Configuration
====================================
Sweep timestamp: $SWEEP_TIMESTAMP
Stride: $SEQUENCE_STRIDE (fixed)
Sequence lengths: ${SEQ_LENGTHS[*]}
Test subjects: ${SELECTED_TEST_SUBJECTS[*]}
GPUs: ${GPU_IDS[*]}
Phase 2 epochs: $PHASE2_EPOCHS
Exclude: playing=$EXCLUDE_PLAYING pinch=$EXCLUDE_PINCH wrist=$EXCLUDE_WRIST
EOF

# Track overall results
declare -A RESULTS

get_val_subject() {
    local test_subject=$1
    local val_subject=""
    for i in "${!ALL_SUBJECTS[@]}"; do
        if [ "${ALL_SUBJECTS[$i]}" -eq "$test_subject" ]; then
            if [ "$i" -eq 0 ]; then
                val_subject=${ALL_SUBJECTS[$((${#ALL_SUBJECTS[@]} - 1))]}
            else
                val_subject=${ALL_SUBJECTS[$((i - 1))]}
            fi
            break
        fi
    done
    echo "$val_subject"
}

get_train_subjects() {
    local test_subject=$1
    local val_subject=$2
    local train_subjects=()
    for subj in "${ALL_SUBJECTS[@]}"; do
        if [ "$subj" -ne "$test_subject" ] && [ "$subj" -ne "$val_subject" ]; then
            train_subjects+=("$subj")
        fi
    done
    echo "${train_subjects[*]}"
}

# Main sweep loop
sweep_start_time=$(date +%s)

for seq_len in "${SEQ_LENGTHS[@]}"; do
    span=$(( (seq_len - 1) * SEQUENCE_STRIDE + 1 ))
    secs=$(echo "scale=2; $span / 20.0" | bc)

    echo ""
    echo "=========================================="
    echo -e "${CYAN}SWEEP: seq_len=$seq_len (span=${span} frames, ${secs}s)${NC}"
    echo "=========================================="

    BATCH_LOG_DIR="${LOG_BASE}/seqlen_${seq_len}"
    mkdir -p "$BATCH_LOG_DIR"
    BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    # Launch subjects in batches of GPU_COUNT
    num_gpus=${#GPU_IDS[@]}
    num_subjects=${#SELECTED_TEST_SUBJECTS[@]}
    batch_num=0

    for ((start_idx=0; start_idx<num_subjects; start_idx+=num_gpus)); do
        batch_num=$((batch_num + 1))
        end_idx=$((start_idx + num_gpus))
        if [ $end_idx -gt $num_subjects ]; then
            end_idx=$num_subjects
        fi

        batch_subjects=("${SELECTED_TEST_SUBJECTS[@]:$start_idx:$((end_idx - start_idx))}")
        echo ""
        echo -e "${YELLOW}  Batch $batch_num: subjects ${batch_subjects[*]}${NC}"

        pids=()
        gpu_idx=0

        for test_subject in "${batch_subjects[@]}"; do
            val_subject=$(get_val_subject "$test_subject")
            train_str=$(get_train_subjects "$test_subject" "$val_subject")
            gpu_id=${GPU_IDS[$gpu_idx]}

            # Find embedding checkpoint
            checkpoint_pattern="${EMBEDDING_DIR}/*test${test_subject}_val*/checkpoint_best.pth"
            encoder_checkpoint=$(ls $checkpoint_pattern 2>/dev/null | head -1)

            if [ -z "$encoder_checkpoint" ]; then
                echo -e "${RED}  Missing embedding checkpoint for test subject $test_subject — skipping${NC}"
                continue
            fi

            exp_name="seqlen_sweep_seq${seq_len}_stride${SEQUENCE_STRIDE}_test${test_subject}_val${val_subject}"
            log_file="${BATCH_LOG_DIR}/test_subject_${test_subject}.log"

            cmd="$PYTHON_BIN -m $SCRIPT \
                --train-subjects $train_str \
                --val-subjects $val_subject \
                --test-subjects $test_subject \
                --pretrained-embedding-checkpoint $encoder_checkpoint \
                --encoder-init embedding \
                --embedding-dim $EMBEDDING_DIM \
                --base-channels $BASE_CHANNELS \
                --hidden-dim $HIDDEN_DIM \
                --sequence-length $seq_len \
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

            echo "    Test=$test_subject Val=$val_subject GPU=$gpu_id"

            $cmd > "$log_file" 2>&1 &
            pids+=($!)

            gpu_idx=$((gpu_idx + 1))
            sleep 2
        done

        # Wait for this batch to complete
        echo ""
        echo -e "  ${YELLOW}Waiting for batch $batch_num (${#pids[@]} jobs)...${NC}"
        batch_failed=0
        for pid_idx in "${!pids[@]}"; do
            pid=${pids[$pid_idx]}
            subj=${batch_subjects[$pid_idx]}
            if wait $pid; then
                echo -e "    ${GREEN}Subject $subj: done${NC}"
            else
                echo -e "    ${RED}Subject $subj: FAILED (exit code $?)${NC}"
                batch_failed=$((batch_failed + 1))
            fi
        done

        if [ $batch_failed -gt 0 ]; then
            echo -e "  ${RED}$batch_failed job(s) failed in batch $batch_num${NC}"
        fi
    done

    # Collect results for this seq_len
    echo ""
    echo -e "  ${CYAN}Results for seq_len=$seq_len:${NC}"
    for test_subject in "${SELECTED_TEST_SUBJECTS[@]}"; do
        log_file="${BATCH_LOG_DIR}/test_subject_${test_subject}.log"
        if [ -f "$log_file" ]; then
            # Extract best mean AUC from summary lines
            best_auc=$(grep "New best mean AUC" "$log_file" | tail -1 | grep -oP 'AUC: \K[0-9.]+')
            if [ -n "$best_auc" ]; then
                echo "    Subject $test_subject: best mean AUC = $best_auc"
                RESULTS["${seq_len}_${test_subject}"]=$best_auc
            else
                echo "    Subject $test_subject: no AUC found in log"
            fi
        else
            echo "    Subject $test_subject: log file missing"
        fi
    done
done

sweep_end_time=$(date +%s)
sweep_duration=$(( (sweep_end_time - sweep_start_time) / 60 ))

echo ""
echo "=========================================="
echo -e "${GREEN}Sweep Complete! (${sweep_duration} minutes)${NC}"
echo "=========================================="
echo ""

# Print summary table
echo "Summary (Best Mean AUC per subject):"
echo "----------------------------------------------"
printf "%-10s" "seq_len"
for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
    printf "%-12s" "S${subj}"
done
printf "%-12s\n" "Mean"
echo "----------------------------------------------"

for seq_len in "${SEQ_LENGTHS[@]}"; do
    printf "%-10s" "$seq_len"
    sum=0
    count=0
    for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
        val=${RESULTS["${seq_len}_${subj}"]:-"N/A"}
        printf "%-12s" "$val"
        if [ "$val" != "N/A" ] && [ -n "$val" ]; then
            sum=$(echo "$sum + $val" | bc)
            count=$((count + 1))
        fi
    done
    if [ $count -gt 0 ]; then
        mean=$(echo "scale=4; $sum / $count" | bc)
        printf "%-12s" "$mean"
    else
        printf "%-12s" "N/A"
    fi
    echo ""
done
echo "----------------------------------------------"

# Save summary
{
    echo "Sequence Length Sweep Results"
    echo "============================="
    echo "Stride: $SEQUENCE_STRIDE"
    echo "Duration: ${sweep_duration} minutes"
    echo ""
    printf "%-10s" "seq_len"
    for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
        printf "%-12s" "S${subj}"
    done
    printf "%-12s\n" "Mean"
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        printf "%-10s" "$seq_len"
        sum=0
        count=0
        for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
            val=${RESULTS["${seq_len}_${subj}"]:-"N/A"}
            printf "%-12s" "$val"
            if [ "$val" != "N/A" ] && [ -n "$val" ]; then
                sum=$(echo "$sum + $val" | bc)
                count=$((count + 1))
            fi
        done
        if [ $count -gt 0 ]; then
            mean=$(echo "scale=4; $sum / $count" | bc)
            printf "%-12s" "$mean"
        fi
        echo ""
    done
} > "${LOG_BASE}/results_summary.txt"

echo ""
echo "Results saved to: ${LOG_BASE}/results_summary.txt"
echo "Logs: ${LOG_BASE}/"
echo ""
echo "To extend to all 12 subjects with the winning seq_len:"
echo "  SUBJECTS=all ./scripts/sweep_sequence_length.sh <winning_seq_len>"

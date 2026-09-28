#!/bin/bash
#
# Uncertainty weight (lambda) sweep for multi-finger temporal ranking.
# Sweeps the --uncertainty-weight parameter that controls the penalty
# on ambiguous pairs (delta diff < threshold) in the ranking loss:
#   active_loss = clear * bce_loss + ambiguous * lambda * logit^2
#
# Configuration matches seqlen sweep seq16 stride2 experiments.
#
# Usage:
#   ./scripts/sweep_uncertainty_weight.sh              # Sweep all lambda values
#   ./scripts/sweep_uncertainty_weight.sh 0.5 1.0 2.0  # Sweep specific values only
# Reader's guide: all model and data settings remain fixed while only the loss
# weight changes. Lambda batches run sequentially and their folds run in
# parallel, preserving a separate output directory for each operating point.
#

# Configuration (matching seqlen sweep seq16 stride2)
PYTHON_BIN="/home/deanz/anaconda3/envs/nhst39/bin/python"
SCRIPT="trainer.train_multi_finger_ranking"
NUM_WORKERS=4
BATCH_SIZE=8
DROPOUT=0.3
EXCLUDE_PLAYING=true
EXCLUDE_PINCH=true
EXCLUDE_WRIST=true

# Sequence length (fixed: seq16 stride2, matching seqlen sweep)
SEQUENCE_LENGTH=16
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

# Pair-level uncertainty (threshold fixed, weight is the sweep variable)
UNCERTAINTY_THRESHOLD_DEG=5.0
PER_FINGER_REST_THRESHOLDS="17.5 17.5 17.5 12.5 7.5"

# Sampling
SAMPLES_PER_EPOCH=50000

# Training (matching seqlen sweep: phase1=0, phase2=10)
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

# Test subjects (override with SUBJECTS=all)
if [ "${SUBJECTS}" = "all" ]; then
    SELECTED_TEST_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)
else
    SELECTED_TEST_SUBJECTS=(4 10 17 23)
fi

# GPU assignment
GPU_IDS=(0 1 2 3)

# Embedding checkpoints directory
EMBEDDING_DIR="outputs/embedding"

# Uncertainty weights to sweep (override with positional args)
if [ $# -gt 0 ]; then
    LAMBDA_VALUES=("$@")
else
    LAMBDA_VALUES=(0 0.1 0.25 0.5 1.0 2.0 5.0 10.0)
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SWEEP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_BASE="logs/sweep_uncertainty_weight/${SWEEP_TIMESTAMP}"
mkdir -p "$LOG_BASE"

echo "=========================================="
echo "Uncertainty Weight (Lambda) Sweep"
echo "=========================================="
echo "Sequence: length=$SEQUENCE_LENGTH stride=$SEQUENCE_STRIDE (fixed, matching seqlen sweep)"
echo "Lambda values: ${LAMBDA_VALUES[*]}"
echo "Test subjects: ${SELECTED_TEST_SUBJECTS[*]}"
echo "GPUs: ${GPU_IDS[*]}"
echo "Exclude: playing=$EXCLUDE_PLAYING pinch=$EXCLUDE_PINCH wrist=$EXCLUDE_WRIST"
echo ""
echo "Loss formula: active_loss = clear * BCE + ambiguous * lambda * logit^2"
echo "Uncertainty threshold: ${UNCERTAINTY_THRESHOLD_DEG} deg"
echo "Per-finger rest thresholds: [${PER_FINGER_REST_THRESHOLDS}]"
echo ""
echo "Log directory: $LOG_BASE"
echo "=========================================="
echo ""

# Save sweep config
cat > "${LOG_BASE}/sweep_config.txt" << EOF
Uncertainty Weight (Lambda) Sweep Configuration
====================================
Sweep timestamp: $SWEEP_TIMESTAMP
Sequence: length=$SEQUENCE_LENGTH stride=$SEQUENCE_STRIDE
Lambda values: ${LAMBDA_VALUES[*]}
Test subjects: ${SELECTED_TEST_SUBJECTS[*]}
GPUs: ${GPU_IDS[*]}
Phase 1 epochs: $PHASE1_EPOCHS
Phase 2 epochs: $PHASE2_EPOCHS
Samples per epoch: $SAMPLES_PER_EPOCH
Exclude: playing=$EXCLUDE_PLAYING pinch=$EXCLUDE_PINCH wrist=$EXCLUDE_WRIST
Uncertainty threshold: $UNCERTAINTY_THRESHOLD_DEG deg
Per-finger rest thresholds: $PER_FINGER_REST_THRESHOLDS
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

# Check for embedding checkpoints
echo "Checking for embedding checkpoints..."
missing_checkpoints=0
for test_subject in "${SELECTED_TEST_SUBJECTS[@]}"; do
    val_subject=$(get_val_subject "$test_subject")
    checkpoint_pattern="${EMBEDDING_DIR}/*test${test_subject}_val${val_subject}/checkpoint_best.pth"
    checkpoint=$(ls $checkpoint_pattern 2>/dev/null | head -1)

    if [ -z "$checkpoint" ]; then
        echo -e "${RED}  Missing embedding checkpoint for test=$test_subject val=$val_subject${NC}"
        missing_checkpoints=$((missing_checkpoints + 1))
    else
        echo -e "${GREEN}  Found: $(basename $(dirname $checkpoint))${NC}"
    fi
done

if [ $missing_checkpoints -gt 0 ]; then
    echo ""
    echo -e "${RED}Error: Missing $missing_checkpoints embedding checkpoint(s)!${NC}"
    exit 1
fi
echo ""

# Main sweep loop
sweep_start_time=$(date +%s)

for lambda_val in "${LAMBDA_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo -e "${CYAN}SWEEP: uncertainty_weight=$lambda_val${NC}"
    echo "=========================================="

    BATCH_LOG_DIR="${LOG_BASE}/lambda_${lambda_val}"
    mkdir -p "$BATCH_LOG_DIR"

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
            checkpoint_pattern="${EMBEDDING_DIR}/*test${test_subject}_val${val_subject}/checkpoint_best.pth"
            encoder_checkpoint=$(ls $checkpoint_pattern 2>/dev/null | head -1)

            if [ -z "$encoder_checkpoint" ]; then
                echo -e "${RED}  Missing embedding checkpoint for test=$test_subject — skipping${NC}"
                continue
            fi

            exp_name="lambda_sweep_w${lambda_val}_test${test_subject}_val${val_subject}"
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
                --sequence-length $SEQUENCE_LENGTH \
                --sequence-stride $SEQUENCE_STRIDE \
                --num-transformer-layers $NUM_TRANSFORMER_LAYERS \
                --num-heads $NUM_HEADS \
                --secondary-loss-weight $SECONDARY_LOSS_WEIGHT \
                --inactive-loss-weight $INACTIVE_LOSS_WEIGHT \
                --uncertainty-weight $lambda_val \
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
                --output-dir outputs/experiments/uncertainty_weight_sweep \
                --exp-name $exp_name \
                $([ "$EXCLUDE_PLAYING" = true ] && echo '--exclude-playing') \
                $([ "$EXCLUDE_PINCH" = true ] && echo '--exclude-pinch') \
                $([ "$EXCLUDE_WRIST" = true ] && echo '--exclude-wrist')"

            echo "    Test=$test_subject Val=$val_subject GPU=$gpu_id lambda=$lambda_val"

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

    # Collect results for this lambda value
    echo ""
    echo -e "  ${CYAN}Results for uncertainty_weight=$lambda_val:${NC}"
    for test_subject in "${SELECTED_TEST_SUBJECTS[@]}"; do
        log_file="${BATCH_LOG_DIR}/test_subject_${test_subject}.log"
        if [ -f "$log_file" ]; then
            best_auc=$(grep "New best mean AUC" "$log_file" | tail -1 | grep -oP 'AUC: \K[0-9.]+')
            if [ -n "$best_auc" ]; then
                echo "    Subject $test_subject: best mean AUC = $best_auc"
                RESULTS["${lambda_val}_${test_subject}"]=$best_auc
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
printf "%-10s" "lambda"
for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
    printf "%-12s" "S${subj}"
done
printf "%-12s\n" "Mean"
echo "----------------------------------------------"

for lambda_val in "${LAMBDA_VALUES[@]}"; do
    printf "%-10s" "$lambda_val"
    sum=0
    count=0
    for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
        val=${RESULTS["${lambda_val}_${subj}"]:-"N/A"}
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
    echo "Uncertainty Weight (Lambda) Sweep Results"
    echo "============================="
    echo "Sequence: length=$SEQUENCE_LENGTH stride=$SEQUENCE_STRIDE"
    echo "Duration: ${sweep_duration} minutes"
    echo ""
    printf "%-10s" "lambda"
    for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
        printf "%-12s" "S${subj}"
    done
    printf "%-12s\n" "Mean"
    for lambda_val in "${LAMBDA_VALUES[@]}"; do
        printf "%-10s" "$lambda_val"
        sum=0
        count=0
        for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
            val=${RESULTS["${lambda_val}_${subj}"]:-"N/A"}
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
echo "Checkpoints saved to: outputs/experiments/uncertainty_weight_sweep/"
echo ""
echo "To extend to all 12 subjects:"
echo "  SUBJECTS=all ./scripts/sweep_uncertainty_weight.sh <winning_lambda>"

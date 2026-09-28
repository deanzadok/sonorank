#!/bin/bash
#
# Cross-subject classic baselines training.
# Runs Castellini 2011, Huang 2016, McIntosh 2017 (EchoFlex) baselines
# using 12-fold leave-one-subject-out cross-validation.
#
# These are sklearn-based (CPU only, no GPU needed).
# Reader's guide: the outer method loop and inner test-subject loop create the
# same subject-disjoint folds for every baseline. ``METHODS`` currently records
# an original partial batch and should be expanded when all methods are needed.
#

# Configuration
PYTHON_BIN="/home/deanz/anaconda3/envs/nhst39/bin/python"
SCRIPT="trainer.train_classic_baselines"

# Methods to run
# METHODS=("castellini" "huang" "mcintosh")
METHODS=("mcintosh")

# Sequence params (for Huang, McIntosh multi-frame methods)
SEQUENCE_LENGTH=16
SEQUENCE_STRIDE=2
WINDOW_STRIDE=1

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
SELECTED_TEST_SUBJECTS=(1 4 5 6 10 17 19 20 21 23 24 25)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "==========================================="
echo "Classic Baselines Cross-Subject Training"
echo "==========================================="
echo "Methods: ${METHODS[*]}"
echo "Total subjects: ${#SELECTED_TEST_SUBJECTS[@]} (${SELECTED_TEST_SUBJECTS[*]})"
echo "Total jobs: $((${#METHODS[@]} * ${#SELECTED_TEST_SUBJECTS[@]}))"
echo ""
echo "Sequence: length=$SEQUENCE_LENGTH, stride=$SEQUENCE_STRIDE"
echo "Fixed labels: $USE_FIXED_LABELS"
echo "==========================================="
echo ""

# Create logs directory
mkdir -p logs/cross_subject_classic_baselines
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cross_subject_classic_baselines/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Array to store background process IDs
pids=()
pid_labels=()

# Launch training sessions
for method in "${METHODS[@]}"; do
    echo -e "${YELLOW}=== Method: ${method} ===${NC}"

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

        # Experiment name
        exp_name="${method}_${TIMESTAMP}_test${test_subject}_val${val_subject}"

        # Log file
        log_file="${LOG_DIR}/${method}_test_subject_${test_subject}.log"

        # Build command
        cmd="$PYTHON_BIN -m $SCRIPT \
            --method $method \
            --train-subjects $train_str \
            --val-subjects $val_subject \
            --test-subjects $test_subject \
            --sequence-length $SEQUENCE_LENGTH \
            --sequence-stride $SEQUENCE_STRIDE \
            --window-stride $WINDOW_STRIDE \
            --vel-threshold-deg-s $VEL_THRESHOLD_DEG_S \
            --vel-smooth-window $VEL_SMOOTH_WINDOW \
            --angle-pct-threshold $ANGLE_PCT_THRESHOLD \
            --thumb-vel-threshold-deg-s $THUMB_VEL_THRESHOLD_DEG_S \
            --thumb-angle-pct-threshold $THUMB_ANGLE_PCT_THRESHOLD \
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

        echo -e "  Test=${GREEN}$test_subject${NC}, Val=${YELLOW}$val_subject${NC}, Train=[${train_str}]"
        echo "    Log: $log_file"

        $cmd > "$log_file" 2>&1 &
        pids+=($!)
        pid_labels+=("${method}_test${test_subject}")
        echo "    PID: $!"

        sleep 1
    done
    echo ""
done

echo "==========================================="
echo -e "${GREEN}All training sessions launched!${NC}"
echo "==========================================="
echo ""

echo "Process IDs:"
for pid_idx in "${!pids[@]}"; do
    echo "  ${pid_labels[$pid_idx]}: ${pids[$pid_idx]}"
done
echo ""

echo "${pids[@]}" > "${LOG_DIR}/pids.txt"
echo "PIDs saved to: ${LOG_DIR}/pids.txt"
echo ""

# Save configuration
cat > "${LOG_DIR}/config.txt" << EOF
Classic Baselines Cross-Subject Training Configuration
=======================================================
Timestamp: $TIMESTAMP
Methods: ${METHODS[*]}
Sequence: length=$SEQUENCE_LENGTH, stride=$SEQUENCE_STRIDE
Fixed labels: $USE_FIXED_LABELS
Exclude: playing=$EXCLUDE_PLAYING, pinch=$EXCLUDE_PINCH, wrist=$EXCLUDE_WRIST
EOF

echo "==========================================="
echo "Monitoring:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  grep -r 'Macro F1' ${LOG_DIR}/*.log"
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
        label=${pid_labels[$pid_idx]}
        echo -n "Waiting for $label (PID $pid)... "
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
        for method in "${METHODS[@]}"; do
            echo -e "${BLUE}=== ${method} ===${NC}"
            for subj in "${SELECTED_TEST_SUBJECTS[@]}"; do
                echo -n "  Subject $subj: "
                grep "Macro F1" "${LOG_DIR}/${method}_test_subject_${subj}.log" 2>/dev/null | tail -1 || echo "No results"
            done
            echo ""
        done
    else
        echo -e "${RED}$failed session(s) failed!${NC}"
    fi
fi

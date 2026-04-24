#!/bin/bash

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_FILE="${JOBS_DIR}/job_summary_$(date +%Y%m%d_%H%M%S).txt"
MAX_CONCURRENT=4
POLL_INTERVAL=90

V36_TASKS="social_iqa"

needs_v36() {
    local task="$1"
    [[ ",$V36_TASKS," == *",$task,"* ]]
}

declare -a PENDING_JOBS
declare -A RUNNING_JOBS
declare -A FAILED_JOBS
declare -A COMPLETED_JOBS

TOTAL_JOBS=0
COMPLETED_COUNT=0
FAILED_COUNT=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

RESULTS_BASE="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results"

is_job_completed() {
    local sbatch_file="$1"
    local job_name
    job_name=$(basename "$sbatch_file" .sbatch)
    
    local model_dir
    model_dir=$(dirname "$sbatch_file")
    local model_name
    model_name=$(basename "$model_dir")
    
    # Strip model family prefix (q35-, llama32-, etc.) and size prefix (2b_, 9b_, etc.)
    local task_name="${job_name#*-}"
    task_name="${task_name#*_}"
    
    # Handle known task name mismatches between job filenames and results directories
    case "$task_name" in
        acpbench) task_name="acp_bench" ;;
        gpqa) task_name="gpqa_main_n_shot" ;;
        lambada) task_name="lambada_standard" ;;
    esac
    
    local results_dir="${RESULTS_BASE}/${model_name}/${task_name}"
    
    # Check for any results_*.json file at expected location
    if ls "$results_dir"/results_*.json 1>/dev/null 2>&1; then
        echo "SKIP (already completed): $model_name/$task_name"
        return 0
    fi
    
    # Extract pretrained model path from sbatch file to handle lm-eval's subdirectory naming
    local model_path
    model_path=$(grep -oP 'pretrained=\K[^,]+' "$sbatch_file" 2>/dev/null)
    if [ -n "$model_path" ]; then
        local model_subdir="${model_path//\//__}"
        if ls "$results_dir/$model_subdir"/results_*.json 1>/dev/null 2>&1; then
            echo "SKIP (already completed): $model_name/$task_name"
            return 0
        fi
    fi
    
    return 1
}

get_job_status() {
    local job_id="$1"
    local status
    status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
    if [ -z "$status" ]; then
        echo "COMPLETED"
    else
        echo "$status"
    fi
}

submit_job() {
    local sbatch_file="$1"
    local job_name
    job_name=$(basename "$sbatch_file" .sbatch)
    
    local job_id
    job_id=$(sbatch "$sbatch_file" 2>&1 | awk '/Submitted batch job/ {print $NF}')
    
    if [ -n "$job_id" ] && [[ "$job_id" =~ ^[0-9]+$ ]]; then
        RUNNING_JOBS["$job_id"]="$job_name"
        log "Submitted: $job_name -> Job ID: $job_id"
        return 0
    else
        log "FAILED to submit: $job_name"
        FAILED_JOBS["$job_name"]="SUBMIT_FAILED"
        ((FAILED_COUNT++))
        return 1
    fi
}

collect_pending_jobs() {
    local phase="$1"
    PENDING_JOBS=()
    local skipped_count=0
    
    for model_dir in "Qwen3.5-2B" "Qwen3.5-9B-AWQ" "Qwen3.5-27B-AWQ" "Qwen3.5-35B-A3B-AWQ" "Qwen3.5-397B-A17B-AWQ" "Llama-3.2-1B-Instruct" "Llama-3-8B-Instruct-AWQ" "Llama-3.3-70B-Instruct-AWQ" "Qwen3.5-122B-A10B-AWQ"; do
        model_path="$JOBS_DIR/$model_dir"
        
        if [ -d "$model_path" ]; then
            for sbatch_file in "$model_path"/*.sbatch; do
                if [ -f "$sbatch_file" ]; then
                    local job_name
                    job_name=$(basename "$sbatch_file" .sbatch)
                    # Strip model family prefix (q35-, llama32-, etc.) and size prefix (2b_, 9b_, etc.)
                    local task_name="${job_name#*-}"
                    task_name="${task_name#*_}"
                    
                    local should_add=false
                    if [ "$phase" = "v4" ] && ! needs_v36 "$task_name"; then
                        should_add=true
                    elif [ "$phase" = "v36" ] && needs_v36 "$task_name"; then
                        should_add=true
                    fi
                    
                    if $should_add; then
                        if is_job_completed "$sbatch_file"; then
                            ((skipped_count++))
                        else
                            PENDING_JOBS+=("$sbatch_file")
                            ((TOTAL_JOBS++))
                        fi
                    fi
                fi
            done
        fi
    done
    
    if [ $skipped_count -gt 0 ]; then
        log "Skipped $skipped_count already completed jobs"
    fi
}

print_status() {
    local running=${#RUNNING_JOBS[@]}
    local pending=${#PENDING_JOBS[@]}
    
    echo ""
    echo "=========================================="
    log "Status Report"
    echo "------------------------------------------"
    echo "  Running:  $running"
    echo "  Pending:  $pending"
    echo "  Completed: $COMPLETED_COUNT"
    echo "  Failed:    $FAILED_COUNT"
    echo "  Total:     $TOTAL_JOBS"
    echo "------------------------------------------"
    
    if [ $running -gt 0 ]; then
        echo "Running jobs:"
        for job_id in "${!RUNNING_JOBS[@]}"; do
            local job_name="${RUNNING_JOBS[$job_id]}"
            local status
            status=$(get_job_status "$job_id")
            printf "  %-25s (Job %s) - %s\n" "$job_name" "$job_id" "$status"
        done
    fi
    
    if [ $FAILED_COUNT -gt 0 ]; then
        echo ""
        echo "Failed jobs:"
        for job_name in "${!FAILED_JOBS[@]}"; do
            printf "  %-25s - %s\n" "$job_name" "${FAILED_JOBS[$job_name]}"
        done
    fi
    
    echo "=========================================="
    echo ""
}

run_phase() {
    local phase_name="$1"
    local phase_count=${#PENDING_JOBS[@]}
    
    log "Phase: $phase_name - $phase_count jobs to submit"
    echo ""
    
    log "Submitting initial batch (up to $MAX_CONCURRENT jobs)..."
    while [ ${#PENDING_JOBS[@]} -gt 0 ] && [ ${#RUNNING_JOBS[@]} -lt $MAX_CONCURRENT ]; do
        local sbatch_file="${PENDING_JOBS[0]}"
        PENDING_JOBS=("${PENDING_JOBS[@]:1}")
        submit_job "$sbatch_file"
    done
    
    print_status
    
    log "Entering monitoring loop..."
    
    while [ ${#PENDING_JOBS[@]} -gt 0 ] || [ ${#RUNNING_JOBS[@]} -gt 0 ]; do
        sleep $POLL_INTERVAL
        
        local to_remove=()
        
        for job_id in "${!RUNNING_JOBS[@]}"; do
            local job_name="${RUNNING_JOBS[$job_id]}"
            local status
            status=$(get_job_status "$job_id")
            
            if [ "$status" = "COMPLETED" ]; then
                local job_exit_code
                job_exit_code=$(sacct -j "$job_id" -n -P -o "ExitCode" 2>/dev/null | head -1 | cut -d: -f1)
                
                if [ "$job_exit_code" = "0" ]; then
                    log "Job COMPLETED: $job_name (Job ID: $job_id)"
                    COMPLETED_JOBS["$job_id"]="$job_name"
                    ((COMPLETED_COUNT++))
                else
                    log "Job FAILED: $job_name (Job ID: $job_id, Exit Code: $job_exit_code)"
                    FAILED_JOBS["$job_name"]="EXIT_CODE_$job_exit_code"
                    ((FAILED_COUNT++))
                fi
                
                to_remove+=("$job_id")
            fi
        done
        
        for job_id in "${to_remove[@]}"; do
            unset RUNNING_JOBS["$job_id"]
        done
        
        while [ ${#PENDING_JOBS[@]} -gt 0 ] && [ ${#RUNNING_JOBS[@]} -lt $MAX_CONCURRENT ]; do
            local sbatch_file="${PENDING_JOBS[0]}"
            PENDING_JOBS=("${PENDING_JOBS[@]:1}")
            submit_job "$sbatch_file"
        done
        
        print_status
    done
    
    log "Phase complete: $phase_name"
    echo ""
}

generate_summary() {
    {
        echo "========================================"
        echo "Job Summary - run_all_jobs.sh"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"
        echo ""
        echo "Configuration:"
        echo "  Total Jobs: $TOTAL_JOBS"
        echo "  Max Concurrent: $MAX_CONCURRENT"
        echo "  Polling Interval: ${POLL_INTERVAL}s"
        echo ""
        echo "Status Breakdown:"
        echo "  Completed: $COMPLETED_COUNT"
        echo "  Failed:    $FAILED_COUNT"
        echo ""
        echo "========================================"
        echo "COMPLETED JOBS:"
        echo "----------------------------------------"
        if [ ${#COMPLETED_JOBS[@]} -gt 0 ]; then
            printf "%-30s %-12s %-15s\n" "Job Name" "Status" "Job ID"
            printf "%-30s %-12s %-15s\n" "--------------------------------" "------------" "---------------"
            for job_id in "${!COMPLETED_JOBS[@]}"; do
                printf "%-30s %-12s %-15s\n" "${COMPLETED_JOBS[$job_id]}" "COMPLETED" "$job_id"
            done
        else
            echo "  None"
        fi
        echo ""
        echo "========================================"
        echo "FAILED JOBS:"
        echo "----------------------------------------"
        if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
            printf "%-30s %-15s\n" "Job Name" "Reason"
            printf "%-30s %-15s\n" "--------------------------------" "---------------"
            for job_name in "${!FAILED_JOBS[@]}"; do
                printf "%-30s %-15s\n" "$job_name" "${FAILED_JOBS[$job_name]}"
            done
        else
            echo "  None"
        fi
        echo ""
        echo "========================================"
        echo "OUTPUT LOCATIONS:"
        echo "----------------------------------------"
        echo "  Qwen3.5-2B:                     $JOBS_DIR/../results/Qwen3.5-2B/"
        echo "  Qwen3.5-9B-AWQ:                 $JOBS_DIR/../results/Qwen3.5-9B-AWQ/"
        echo "  Qwen3.5-27B-AWQ:                $JOBS_DIR/../results/Qwen3.5-27B-AWQ/"
        echo "  Qwen3.5-35B-A3B-AWQ:            $JOBS_DIR/../results/Qwen3.5-35B-A3B-AWQ/"
        echo "  Qwen3.5-397B-A17B-AWQ:          $JOBS_DIR/../results/Qwen3.5-397B-A17B-AWQ/"
        echo "  Qwen3.5-122B-A10B-AWQ:          $JOBS_DIR/../results/Qwen3.5-122B-A10B-AWQ/"
        echo "  Llama-3.2-1B-Instruct:          $JOBS_DIR/../results/Llama-3.2-1B-Instruct/"
        echo "  Llama-3-8B-Instruct-AWQ:        $JOBS_DIR/../results/Llama-3-8B-Instruct-AWQ/"
        echo "  Llama-3.3-70B-Instruct-AWQ:     $JOBS_DIR/../results/Llama-3.3-70B-Instruct-AWQ/"
        echo "========================================"
    } > "$SUMMARY_FILE"
    
    log "Summary saved to: $SUMMARY_FILE"
}

cleanup_and_exit() {
    log "Interrupt received. Generating partial summary..."
    generate_summary
    exit 1
}

trap cleanup_and_exit INT TERM

main() {
    log "Starting SLURM job manager..."
    log "Max concurrent jobs: $MAX_CONCURRENT"
    log "Polling interval: ${POLL_INTERVAL}s"
    echo ""
    echo "=========================================="
    echo "PHASE 1: Tasks using datasets>=4.0"
    echo "  Tasks: arc_challenge, arc_easy, boolq,"
    echo "         piqa, sciq, winogrande"
    echo "=========================================="
    echo ""
    
    collect_pending_jobs "v4"
    local v4_count=${#PENDING_JOBS[@]}
    log "Found $v4_count v4 jobs to submit"
    run_phase "v4 (datasets>=4.0)"
    
    echo ""
    echo "=========================================="
    echo "PHASE 2: Tasks using datasets==3.6.0"
    echo "  Tasks: social_iqa"
    echo "=========================================="
    echo ""
    
    collect_pending_jobs "v36"
    local v36_count=${#PENDING_JOBS[@]}
    log "Found $v36_count v3.6 jobs to submit"
    run_phase "v3.6 (datasets==3.6.0)"
    
    log "All jobs finished!"
    echo ""
    
    generate_summary
    
    log "Final Results:"
    echo "  Completed: $COMPLETED_COUNT"
    echo "  Failed:    $FAILED_COUNT"
    echo ""
    
    if [ $FAILED_COUNT -gt 0 ]; then
        log "WARNING: Some jobs failed. Check $SUMMARY_FILE for details."
        exit 1
    else
        log "All jobs completed successfully!"
        exit 0
    fi
}

main "$@"
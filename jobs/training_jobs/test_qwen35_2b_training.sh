#!/bin/bash
# Test script for Qwen3.5-2B with training-limited benchmarks

RESULTS_DIR="./results_training/Qwen3.5-2B"
mkdir -p "$RESULTS_DIR"

LIMITS=(
    "hellaswag_train_limited:10042"
    "winogrande_train_limited:1267"
    "arc_challenge_train_limited:1172"
    "arc_easy_train_limited:2376"
    "gsm8k_train_limited:1319"
    "boolq_train_limited:3270"
    "piqa_train_limited:3084"
    "sciq_train_limited:1000"
    "social_iqa_train_limited:1954"
    "lambada_standard_train_limited:5153"
    "gpqa_main_n_shot:0"
)

echo "Testing Qwen3.5-2B with training-limited benchmarks"
echo "===================================================="

for entry in "${LIMITS[@]}"; do
    task="${entry%%:*}"
    limit="${entry##*:}"
    
    echo ""
    echo "Running: $task (limit=$limit)"
    
    TASK_DIR="$RESULTS_DIR/$task"
    mkdir -p "$TASK_DIR"
    
    LIMIT_ARG=""
    if [ "$limit" != "0" ]; then
        LIMIT_ARG="--limit $limit"
    fi
    
    if [ "$task" = "gpqa_main_n_shot" ]; then
        lm_eval run \
            --model vllm \
            --model_args pretrained=Qwen/Qwen3.5-2B,enforce_eager=True,gdn_prefill_backend=triton,gpu_memory_utilization=0.9,trust_remote_code=True \
            --tasks "$task" \
            --batch_size 1 \
            --output_path "$TASK_DIR" \
            --log_samples
    else
        lm_eval run \
            --model vllm \
            --model_args pretrained=Qwen/Qwen3.5-2B,enforce_eager=True,gdn_prefill_backend=triton,gpu_memory_utilization=0.9,trust_remote_code=True \
            --tasks "$task" \
            --batch_size 1 \
            $LIMIT_ARG \
            --output_path "$TASK_DIR" \
            --log_samples
    fi
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Completed: $task"
    else
        echo "  ✗ Failed: $task"
    fi
done

echo ""
echo "===================================================="
echo "All training-limited benchmarks completed!"
echo "Results saved to: $RESULTS_DIR"

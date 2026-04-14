#!/bin/bash
# Test script for Qwen3.5-2B with 10 samples per benchmark
# Usage: bash jobs/test_qwen35_2b_local.sh

set -e

export HF_ALLOW_CODE_EVAL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$WORK_DIR/results/test_qwen35_2b"

MODEL_PATH="Qwen/Qwen3.5-2B"
LIMIT=1

echo "=========================================="
echo "Testing Qwen3.5-2B with $LIMIT samples per benchmark"
echo "=========================================="
echo ""

mkdir -p "$RESULTS_DIR"

TASKS=(
    # "hellaswag"
    # "winogrande"
    # "arc_challenge"
    # "mmlu"
    # "gsm8k"
    # "gpqa_main_n_shot"
    # "lambada_standard"
    # "wikitext"
    # "arc_easy"
    # "boolq"
    # "piqa"
    # "sciq"
    # "social_iqa"
    # "acp_bench"
    "aime25"
)

for task in "${TASKS[@]}"; do
    echo "-------------------------------------------"
    echo "Running: $task"
    echo "-------------------------------------------"

    TASK_DIR="$RESULTS_DIR/$task"
    mkdir -p "$TASK_DIR"

    cd "$WORK_DIR"

    if [ "$task" = "acp_bench" ]; then
        echo "Note: acp_bench requires 'pip install lm-eval[acpbench]'"
    fi

    uv run lm_eval run \
        --model vllm \
        --model_args pretrained=$MODEL_PATH,enforce_eager=True,gpu_memory_utilization=0.9,trust_remote_code=True \
        --tasks "$task" \
        --limit $LIMIT \
        --batch_size 1 \
        --output_path "$TASK_DIR" \
        --log_samples

    echo ""
done

echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
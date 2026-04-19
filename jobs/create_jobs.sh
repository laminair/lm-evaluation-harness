#!/bin/bash

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results"
WORK_DIR="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness"
IMAGE="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v3.sqsh"
MOUNT_DSS="-m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2"
MOUNT_HOME="-m /dss/dsshome1/09/\$USER:/root"

V36_TASKS="social_iqa"
CODE_EVAL_TASKS=""

needs_v36() {
    local task="$1"
    [[ ",$V36_TASKS," == *",$task,"* ]]
}

needs_code_eval() {
    local task="$1"
    [[ ",$CODE_EVAL_TASKS," == *",$task,"* ]]
}

create_sbatch_multi_gpu() {
    local model_name="$1"
    local model_path="$2"
    local task="$3"
    local time="$4"
    local job_name="$5"
    local output_dir="$6"
    local gpu_count="$7"
    local tp_size="$8"
    
    local dir="$JOBS_DIR/$model_name"
    mkdir -p "$dir"
    
    local sbatch_file="$dir/${job_name}.sbatch"
    
    local datasets_flag=""
    if needs_v36 "$task"; then
        datasets_flag="--with 'datasets==3.6.0'"
    fi
    
    local code_eval_flag=""
    if needs_code_eval "$task"; then
        code_eval_flag="-e HF_ALLOW_CODE_EVAL=1"
    fi
    
    cat > "$sbatch_file" <<EOF
#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${gpu_count}
#SBATCH --time=${time}
#SBATCH --job-name=${job_name}
#SBATCH --output=%x_%j.out

# Environment
export HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf
export HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false

# WandB key from ~/.netrc
export WANDB_API_KEY=\$(grep -A1 "machine api.wandb.ai" /dss/dsshome1/09/\$USER/.netrc 2>/dev/null | grep "password" | awk '{print \$2}')

# Paths
WORK_DIR=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v3.sqsh

# Create enroot container from squashfs image (|| true handles re-import)
enroot create --name lm_eval_container_v3 "\$IMAGE" || true

# Run evaluation in enroot container with mounts
enroot start \\
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf \\
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \\
    -m /dss/dsshome1/09/\$USER:/root \\
    ${code_eval_flag} \\
    --root lm_eval_container_v3 bash -c "
    cd \$WORK_DIR && \\
    uv run --extra vllm --extra energy ${datasets_flag:+$datasets_flag }lm_eval run \\
        --model vllm \\
        --model_args pretrained=${model_path},enforce_eager=True,gdn_prefill_backend=triton,tensor_parallel_size=${tp_size},gpu_memory_utilization=0.9,trust_remote_code=True \\
        --tasks ${task} \\
        --batch_size 1 \\
        --output_path ${output_dir} \\
        --log_samples \\
        --track_energy
"
EOF
    
    echo "Created: $sbatch_file"
}

mkdir -p "$RESULTS_DIR"

create_sbatch() {
    local model_name="$1"
    local model_path="$2"
    local task="$3"
    local time="$4"
    local job_name="$5"
    local output_dir="$6"
    
    local dir="$JOBS_DIR/$model_name"
    mkdir -p "$dir"
    
    local sbatch_file="$dir/${job_name}.sbatch"
    
    local datasets_flag=""
    if needs_v36 "$task"; then
        datasets_flag="--with 'datasets==3.6.0'"
    fi
    
    local code_eval_flag=""
    if needs_code_eval "$task"; then
        code_eval_flag="-e HF_ALLOW_CODE_EVAL=1"
    fi
    
    cat > "$sbatch_file" <<EOF
#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=${time}
#SBATCH --job-name=${job_name}
#SBATCH --output=%x_%j.out

# Environment
export HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf
export HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false

# WandB key from ~/.netrc
export WANDB_API_KEY=\$(grep -A1 "machine api.wandb.ai" /dss/dsshome1/09/\$USER/.netrc 2>/dev/null | grep "password" | awk '{print \$2}')

# Paths
WORK_DIR=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v3.sqsh

# Create enroot container from squashfs image (|| true handles re-import)
enroot create --name lm_eval_container_v3 "\$IMAGE" || true

# Run evaluation in enroot container with mounts
enroot start \\
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf \\
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \\
    -m /dss/dsshome1/09/\$USER:/root \\
    ${code_eval_flag} \\
    --root lm_eval_container_v3 bash -c "
    cd \$WORK_DIR && \\
    uv run --extra vllm --extra energy ${datasets_flag:+$datasets_flag }lm_eval run \\
        --model vllm \\
        --model_args pretrained=${model_path},enforce_eager=True,gdn_prefill_backend=triton,track_energy=true,approx_instant_energy=true,dtype=float16,gpu_memory_utilization=0.9,trust_remote_code=True \\
        --tasks ${task} \\
        --batch_size 1 \\
        --output_path ${output_dir} \\
        --log_samples \\
        --track_energy
"
EOF
    
    echo "Created: $sbatch_file"
}

echo "Creating sbatch files..."
echo ""

# Task definitions: task_name,time_limit
# 11 new benchmarks + 9 existing = 20 tasks per model

echo "=== Qwen3.5-2B (18 jobs) ==="
# New benchmarks
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "hellaswag" "01:30:00" "q35-2b_hellaswag" "$RESULTS_DIR/Qwen3.5-2B/hellaswag"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "winogrande" "01:30:00" "q35-2b_winogrande" "$RESULTS_DIR/Qwen3.5-2B/winogrande"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "arc_challenge" "01:30:00" "q35-2b_arc_challenge" "$RESULTS_DIR/Qwen3.5-2B/arc_challenge"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "mmlu" "08:30:00" "q35-2b_mmlu" "$RESULTS_DIR/Qwen3.5-2B/mmlu"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "gsm8k" "08:30:00" "q35-2b_gsm8k" "$RESULTS_DIR/Qwen3.5-2B/gsm8k"

create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "aime25" "09:30:00" "q35-2b_aime25" "$RESULTS_DIR/Qwen3.5-2B/aime25"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "gpqa_main_n_shot" "01:30:00" "q35-2b_gpqa" "$RESULTS_DIR/Qwen3.5-2B/gpqa_main_n_shot"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "lambada_standard" "01:30:00" "q35-2b_lambada" "$RESULTS_DIR/Qwen3.5-2B/lambada_standard"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "wikitext" "04:00:00" "q35-2b_wikitext" "$RESULTS_DIR/Qwen3.5-2B/wikitext"
# Existing benchmarks
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "arc_easy" "01:30:00" "q35-2b_arc_easy" "$RESULTS_DIR/Qwen3.5-2B/arc_easy"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "boolq" "01:30:00" "q35-2b_boolq" "$RESULTS_DIR/Qwen3.5-2B/boolq"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "piqa" "01:30:00" "q35-2b_piqa" "$RESULTS_DIR/Qwen3.5-2B/piqa"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "sciq" "01:30:00" "q35-2b_sciq" "$RESULTS_DIR/Qwen3.5-2B/sciq"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "social_iqa" "01:30:00" "q35-2b_social_iqa" "$RESULTS_DIR/Qwen3.5-2B/social_iqa"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "acp_bench" "08:00:00" "q35-2b_acpbench" "$RESULTS_DIR/Qwen3.5-2B/acp_bench"

echo ""
echo "=== Qwen3.5-9B-AWQ (18 jobs) ==="
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "hellaswag" "01:30:00" "q35-9b_hellaswag" "$RESULTS_DIR/Qwen3.5-9B-AWQ/hellaswag"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "winogrande" "01:30:00" "q35-9b_winogrande" "$RESULTS_DIR/Qwen3.5-9B-AWQ/winogrande"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "arc_challenge" "01:30:00" "q35-9b_arc_challenge" "$RESULTS_DIR/Qwen3.5-9B-AWQ/arc_challenge"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "mmlu" "08:00:00" "q35-9b_mmlu" "$RESULTS_DIR/Qwen3.5-9B-AWQ/mmlu"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "gsm8k" "08:00:00" "q35-9b_gsm8k" "$RESULTS_DIR/Qwen3.5-9B-AWQ/gsm8k"

create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "aime25" "09:30:00" "q35-9b_aime25" "$RESULTS_DIR/Qwen3.5-9B-AWQ/aime25"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "gpqa_main_n_shot" "01:30:00" "q35-9b_gpqa" "$RESULTS_DIR/Qwen3.5-9B-AWQ/gpqa_main_n_shot"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "lambada_standard" "01:30:00" "q35-9b_lambada" "$RESULTS_DIR/Qwen3.5-9B-AWQ/lambada_standard"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "wikitext" "04:00:00" "q35-9b_wikitext" "$RESULTS_DIR/Qwen3.5-9B-AWQ/wikitext"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "arc_easy" "01:30:00" "q35-9b_arc_easy" "$RESULTS_DIR/Qwen3.5-9B-AWQ/arc_easy"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "boolq" "01:30:00" "q35-9b_boolq" "$RESULTS_DIR/Qwen3.5-9B-AWQ/boolq"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "piqa" "01:30:00" "q35-9b_piqa" "$RESULTS_DIR/Qwen3.5-9B-AWQ/piqa"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "sciq" "01:30:00" "q35-9b_sciq" "$RESULTS_DIR/Qwen3.5-9B-AWQ/sciq"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "social_iqa" "01:30:00" "q35-9b_social_iqa" "$RESULTS_DIR/Qwen3.5-9B-AWQ/social_iqa"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "acp_bench" "08:30:00" "q35-9b_acpbench" "$RESULTS_DIR/Qwen3.5-9B-AWQ/acp_bench"

echo ""
echo "=== Qwen3.5-27B-AWQ (18 jobs) ==="
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "hellaswag" "01:30:00" "q35-27b_hellaswag" "$RESULTS_DIR/Qwen3.5-27B-AWQ/hellaswag"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "winogrande" "01:30:00" "q35-27b_winogrande" "$RESULTS_DIR/Qwen3.5-27B-AWQ/winogrande"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "arc_challenge" "01:30:00" "q35-27b_arc_challenge" "$RESULTS_DIR/Qwen3.5-27B-AWQ/arc_challenge"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "mmlu" "08:00:00" "q35-27b_mmlu" "$RESULTS_DIR/Qwen3.5-27B-AWQ/mmlu"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "gsm8k" "08:00:00" "q35-27b_gsm8k" "$RESULTS_DIR/Qwen3.5-27B-AWQ/gsm8k"

create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "aime25" "09:30:00" "q35-27b_aime25" "$RESULTS_DIR/Qwen3.5-27B-AWQ/aime25"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "gpqa_main_n_shot" "01:30:00" "q35-27b_gpqa" "$RESULTS_DIR/Qwen3.5-27B-AWQ/gpqa_main_n_shot"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "lambada_standard" "02:00:00" "q35-27b_lambada" "$RESULTS_DIR/Qwen3.5-27B-AWQ/lambada_standard"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "wikitext" "04:00:00" "q35-27b_wikitext" "$RESULTS_DIR/Qwen3.5-27B-AWQ/wikitext"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "arc_easy" "02:00:00" "q35-27b_arc_easy" "$RESULTS_DIR/Qwen3.5-27B-AWQ/arc_easy"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "boolq" "01:00:00" "q35-27b_boolq" "$RESULTS_DIR/Qwen3.5-27B-AWQ/boolq"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "piqa" "01:00:00" "q35-27b_piqa" "$RESULTS_DIR/Qwen3.5-27B-AWQ/piqa"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "sciq" "01:00:00" "q35-27b_sciq" "$RESULTS_DIR/Qwen3.5-27B-AWQ/sciq"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "social_iqa" "02:00:00" "q35-27b_social_iqa" "$RESULTS_DIR/Qwen3.5-27B-AWQ/social_iqa"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "acp_bench" "08:30:00" "q35-27b_acpbench" "$RESULTS_DIR/Qwen3.5-27B-AWQ/acp_bench"

echo ""
echo "=== Qwen3.5-35B-A3B-AWQ (18 jobs) ==="
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "hellaswag" "02:00:00" "q35-35b_hellaswag" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/hellaswag"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "winogrande" "02:00:00" "q35-35b_winogrande" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/winogrande"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "arc_challenge" "02:00:00" "q35-35b_arc_challenge" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/arc_challenge"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "mmlu" "04:00:00" "q35-35b_mmlu" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/mmlu"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "gsm8k" "08:00:00" "q35-35b_gsm8k" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/gsm8k"

create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "aime25" "09:30:00" "q35-35b_aime25" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/aime25"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "gpqa_main_n_shot" "01:30:00" "q35-35b_gpqa" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/gpqa_main_n_shot"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "lambada_standard" "02:00:00" "q35-35b_lambada" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/lambada_standard"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "wikitext" "02:00:00" "q35-35b_wikitext" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/wikitext"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "arc_easy" "02:00:00" "q35-35b_arc_easy" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/arc_easy"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "boolq" "01:00:00" "q35-35b_boolq" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/boolq"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "piqa" "01:00:00" "q35-35b_piqa" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/piqa"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "sciq" "01:00:00" "q35-35b_sciq" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/sciq"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "social_iqa" "02:00:00" "q35-35b_social_iqa" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/social_iqa"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "acp_bench" "08:30:00" "q35-35b_acpbench" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/acp_bench"

echo ""
echo "=== Qwen3.5-122B-A10B-AWQ (15 jobs, 2 GPUs) ==="
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "hellaswag" "02:00:00" "q35-122b_hellaswag" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/hellaswag" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "winogrande" "02:00:00" "q35-122b_winogrande" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/winogrande" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "arc_challenge" "02:00:00" "q35-122b_arc_challenge" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/arc_challenge" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "mmlu" "05:00:00" "q35-122b_mmlu" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/mmlu" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "gsm8k" "05:00:00" "q35-122b_gsm8k" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/gsm8k" 2 2

create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "aime25" "09:30:00" "q35-122b_aime25" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/aime25" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "gpqa_main_n_shot" "01:30:00" "q35-122b_gpqa" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/gpqa_main_n_shot" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "lambada_standard" "02:00:00" "q35-122b_lambada" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/lambada_standard" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "wikitext" "02:00:00" "q35-122b_wikitext" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/wikitext" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "arc_easy" "02:00:00" "q35-122b_arc_easy" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/arc_easy" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "boolq" "01:00:00" "q35-122b_boolq" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/boolq" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "piqa" "01:00:00" "q35-122b_piqa" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/piqa" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "sciq" "01:00:00" "q35-122b_sciq" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/sciq" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "social_iqa" "02:00:00" "q35-122b_social_iqa" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/social_iqa" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "acp_bench" "08:30:00" "q35-122b_acpbench" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/acp_bench" 2 2

echo ""
echo "=== Qwen3.5-397B-A17B-AWQ (15 jobs, 4 GPUs) ==="
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "hellaswag" "02:30:00" "q35-397b_hellaswag" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/hellaswag" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "winogrande" "02:30:00" "q35-397b_winogrande" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/winogrande" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "arc_challenge" "02:30:00" "q35-397b_arc_challenge" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/arc_challenge" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "mmlu" "06:00:00" "q35-397b_mmlu" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/mmlu" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "gsm8k" "06:00:00" "q35-397b_gsm8k" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/gsm8k" 4 4

create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "aime25" "10:30:00" "q35-397b_aime25" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/aime25" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "gpqa_main_n_shot" "03:30:00" "q35-397b_gpqa" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/gpqa_main_n_shot" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "lambada_standard" "02:30:00" "q35-397b_lambada" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/lambada_standard" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "wikitext" "05:30:00" "q35-397b_wikitext" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/wikitext" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "arc_easy" "02:30:00" "q35-397b_arc_easy" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/arc_easy" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "boolq" "02:30:00" "q35-397b_boolq" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/boolq" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "piqa" "02:30:00" "q35-397b_piqa" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/piqa" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "sciq" "02:30:00" "q35-397b_sciq" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/sciq" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "social_iqa" "02:30:00" "q35-397b_social_iqa" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/social_iqa" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "acp_bench" "10:30:00" "q35-397b_acpbench" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/acp_bench" 4 4

echo ""
echo "=== Llama-3.2-1B-Instruct (18 jobs) ==="
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "hellaswag" "01:30:00" "llama32-1b_hellaswag" "$RESULTS_DIR/Llama-3.2-1B-Instruct/hellaswag"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "winogrande" "01:30:00" "llama32-1b_winogrande" "$RESULTS_DIR/Llama-3.2-1B-Instruct/winogrande"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "arc_challenge" "01:30:00" "llama32-1b_arc_challenge" "$RESULTS_DIR/Llama-3.2-1B-Instruct/arc_challenge"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "mmlu" "04:30:00" "llama32-1b_mmlu" "$RESULTS_DIR/Llama-3.2-1B-Instruct/mmlu"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "gsm8k" "04:30:00" "llama32-1b_gsm8k" "$RESULTS_DIR/Llama-3.2-1B-Instruct/gsm8k"

create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "aime25" "08:30:00" "llama32-1b_aime25" "$RESULTS_DIR/Llama-3.2-1B-Instruct/aime25"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "gpqa_main_n_shot" "01:30:00" "llama32-1b_gpqa" "$RESULTS_DIR/Llama-3.2-1B-Instruct/gpqa_main_n_shot"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "lambada_standard" "01:30:00" "llama32-1b_lambada" "$RESULTS_DIR/Llama-3.2-1B-Instruct/lambada_standard"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "wikitext" "04:00:00" "llama32-1b_wikitext" "$RESULTS_DIR/Llama-3.2-1B-Instruct/wikitext"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "arc_easy" "01:30:00" "llama32-1b_arc_easy" "$RESULTS_DIR/Llama-3.2-1B-Instruct/arc_easy"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "boolq" "01:30:00" "llama32-1b_boolq" "$RESULTS_DIR/Llama-3.2-1B-Instruct/boolq"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "piqa" "01:30:00" "llama32-1b_piqa" "$RESULTS_DIR/Llama-3.2-1B-Instruct/piqa"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "sciq" "01:30:00" "llama32-1b_sciq" "$RESULTS_DIR/Llama-3.2-1B-Instruct/sciq"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "social_iqa" "01:30:00" "llama32-1b_social_iqa" "$RESULTS_DIR/Llama-3.2-1B-Instruct/social_iqa"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "acp_bench" "04:30:00" "llama32-1b_acpbench" "$RESULTS_DIR/Llama-3.2-1B-Instruct/acp_bench"

echo ""
echo "=== Llama-3-8B-Instruct-AWQ (18 jobs) ==="
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "hellaswag" "01:30:00" "llama3-8b_hellaswag" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/hellaswag"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "winogrande" "01:30:00" "llama3-8b_winogrande" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/winogrande"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "arc_challenge" "01:30:00" "llama3-8b_arc_challenge" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/arc_challenge"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "mmlu" "05:30:00" "llama3-8b_mmlu" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/mmlu"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "gsm8k" "08:00:00" "llama3-8b_gsm8k" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/gsm8k"

create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "aime25" "09:30:00" "llama3-8b_aime25" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/aime25"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "gpqa_main_n_shot" "01:30:00" "llama3-8b_gpqa" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/gpqa_main_n_shot"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "lambada_standard" "01:30:00" "llama3-8b_lambada" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/lambada_standard"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "wikitext" "04:00:00" "llama3-8b_wikitext" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/wikitext"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "arc_easy" "01:30:00" "llama3-8b_arc_easy" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/arc_easy"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "boolq" "01:30:00" "llama3-8b_boolq" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/boolq"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "piqa" "01:30:00" "llama3-8b_piqa" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/piqa"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "sciq" "01:30:00" "llama3-8b_sciq" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/sciq"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "social_iqa" "01:30:00" "llama3-8b_social_iqa" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/social_iqa"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "acp_bench" "05:00:00" "llama3-8b_acpbench" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/acp_bench"

echo ""
echo "=== Llama-3.3-70B-Instruct-AWQ (18 jobs) ==="
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "hellaswag" "01:30:00" "llama33-70b_hellaswag" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/hellaswag"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "winogrande" "01:30:00" "llama33-70b_winogrande" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/winogrande"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "arc_challenge" "01:30:00" "llama33-70b_arc_challenge" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/arc_challenge"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "mmlu" "06:30:00" "llama33-70b_mmlu" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/mmlu"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "gsm8k" "08:00:00" "llama33-70b_gsm8k" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/gsm8k"

create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "aime25" "09:30:00" "llama33-70b_aime25" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/aime25"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "gpqa_main_n_shot" "01:30:00" "llama33-70b_gpqa" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/gpqa_main_n_shot"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "lambada_standard" "01:30:00" "llama33-70b_lambada" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/lambada_standard"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "wikitext" "04:30:00" "llama33-70b_wikitext" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/wikitext"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "arc_easy" "01:30:00" "llama33-70b_arc_easy" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/arc_easy"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "boolq" "01:30:00" "llama33-70b_boolq" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/boolq"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "piqa" "01:30:00" "llama33-70b_piqa" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/piqa"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "sciq" "01:30:00" "llama33-70b_sciq" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/sciq"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "social_iqa" "01:30:00" "llama33-70b_social_iqa" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/social_iqa"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "acp_bench" "06:30:00" "llama33-70b_acpbench" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/acp_bench"

echo ""
echo "=========================================="
echo "Done! Created all sbatch files."
echo ""
echo "Job summary:"
echo "  Qwen3.5-2B:                     15 jobs (1 GPU each)"
echo "  Qwen3.5-9B-AWQ:                 15 jobs (1 GPU each)"
echo "  Qwen3.5-27B-AWQ:                15 jobs (1 GPU each)"
echo "  Qwen3.5-35B-A3B-AWQ:            15 jobs (1 GPU each)"
echo "  Qwen3.5-122B-A10B-AWQ:          15 jobs (2 GPUs each)"
echo "  Qwen3.5-397B-A17B-AWQ:          15 jobs (4 GPUs each)"
echo "  Llama-3.2-1B-Instruct:          15 jobs (1 GPU each)"
echo "  Llama-3-8B-Instruct-AWQ:        15 jobs (1 GPU each)"
echo "  Llama-3.3-70B-Instruct-AWQ:     15 jobs (1 GPU each)"
echo "  Total:                          135 jobs"
echo ""
echo "NEW BENCHMARKS (11):"
echo "  hellaswag, winogrande, arc_challenge, mmlu"
echo "  gsm8k, aime25, gpqa_main_n_shot"
echo "  lambada_standard, wikitext, acp_bench (11 benchmarks)"
echo ""
echo "EXISTING BENCHMARKS (7):"
echo "  arc_easy, boolq, piqa, sciq, social_iqa"
echo ""
echo "Tasks requiring datasets==3.6.0: social_iqa"
echo ""
echo "Run './run_all_jobs.sh' to submit all jobs to SLURM."

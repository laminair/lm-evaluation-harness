#!/bin/bash

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results_train_data"
WORK_DIR="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness"
IMAGE="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v3.sqsh"
MOUNT_DSS="-m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2"
MOUNT_HOME="-m /dss/dsshome1/09/\$USER:/root"

V36_TASKS="social_iqa"
CODE_EVAL_TASKS=""

needs_v36() {
    local task="$1"
    # Strip _train_limited suffix for v36 check
    task="${task%_train_limited}"
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
    local limit="$9"
    
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
    
    local limit_arg=""
    if [ -n "$limit" ]; then
        limit_arg="--limit ${limit}"
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
#SBATCH --exclude=lrz-hgx-h100-020

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
        ${limit_arg} \\
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
    local limit="$7"
    
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
    
    local limit_arg=""
    if [ -n "$limit" ]; then
        limit_arg="--limit ${limit}"
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
#SBATCH --exclude=lrz-hgx-h100-020

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
        ${limit_arg} \\
        --output_path ${output_dir} \\
        --log_samples \\
        --track_energy
"
EOF
    
    echo "Created: $sbatch_file"
}

echo "Creating training sbatch files..."
echo ""

# Task definitions with limits matching original eval split sizes
# (lm-eval will cap to actual training split size if train < limit)

echo "=== Qwen3.5-2B (11 jobs) ==="
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "hellaswag_train_limited" "01:30:00" "q35-2b_hellaswag_train_limited" "$RESULTS_DIR/Qwen3.5-2B/hellaswag_train_limited" "10042"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "winogrande_train_limited" "01:30:00" "q35-2b_winogrande_train_limited" "$RESULTS_DIR/Qwen3.5-2B/winogrande_train_limited" "1267"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "arc_challenge_train_limited" "01:30:00" "q35-2b_arc_challenge_train_limited" "$RESULTS_DIR/Qwen3.5-2B/arc_challenge_train_limited" "1172"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "arc_easy_train_limited" "01:30:00" "q35-2b_arc_easy_train_limited" "$RESULTS_DIR/Qwen3.5-2B/arc_easy_train_limited" "2376"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "gsm8k_train_limited" "08:30:00" "q35-2b_gsm8k_train_limited" "$RESULTS_DIR/Qwen3.5-2B/gsm8k_train_limited" "1319"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "boolq_train_limited" "01:30:00" "q35-2b_boolq_train_limited" "$RESULTS_DIR/Qwen3.5-2B/boolq_train_limited" "3270"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "piqa_train_limited" "01:30:00" "q35-2b_piqa_train_limited" "$RESULTS_DIR/Qwen3.5-2B/piqa_train_limited" "3084"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "sciq_train_limited" "01:30:00" "q35-2b_sciq_train_limited" "$RESULTS_DIR/Qwen3.5-2B/sciq_train_limited" "1000"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "social_iqa_train_limited" "01:30:00" "q35-2b_social_iqa_train_limited" "$RESULTS_DIR/Qwen3.5-2B/social_iqa_train_limited" "1954"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "lambada_standard_train_limited" "01:30:00" "q35-2b_lambada_train_limited" "$RESULTS_DIR/Qwen3.5-2B/lambada_standard_train_limited" "5153"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "gpqa_main_n_shot" "01:30:00" "q35-2b_gpqa" "$RESULTS_DIR/Qwen3.5-2B/gpqa_main_n_shot"

echo ""
echo "=== Qwen3.5-9B-AWQ (11 jobs) ==="
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "hellaswag_train_limited" "01:30:00" "q35-9b_hellaswag_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/hellaswag_train_limited" "10042"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "winogrande_train_limited" "01:30:00" "q35-9b_winogrande_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/winogrande_train_limited" "1267"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "arc_challenge_train_limited" "01:30:00" "q35-9b_arc_challenge_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/arc_challenge_train_limited" "1172"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "arc_easy_train_limited" "01:30:00" "q35-9b_arc_easy_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/arc_easy_train_limited" "2376"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "gsm8k_train_limited" "08:00:00" "q35-9b_gsm8k_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/gsm8k_train_limited" "1319"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "boolq_train_limited" "01:30:00" "q35-9b_boolq_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/boolq_train_limited" "3270"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "piqa_train_limited" "01:30:00" "q35-9b_piqa_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/piqa_train_limited" "3084"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "sciq_train_limited" "01:30:00" "q35-9b_sciq_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/sciq_train_limited" "1000"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "social_iqa_train_limited" "01:30:00" "q35-9b_social_iqa_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/social_iqa_train_limited" "1954"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "lambada_standard_train_limited" "01:30:00" "q35-9b_lambada_train_limited" "$RESULTS_DIR/Qwen3.5-9B-AWQ/lambada_standard_train_limited" "5153"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "gpqa_main_n_shot" "01:30:00" "q35-9b_gpqa" "$RESULTS_DIR/Qwen3.5-9B-AWQ/gpqa_main_n_shot"

echo ""
echo "=== Qwen3.5-27B-AWQ (11 jobs) ==="
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "hellaswag_train_limited" "01:30:00" "q35-27b_hellaswag_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/hellaswag_train_limited" "10042"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "winogrande_train_limited" "01:30:00" "q35-27b_winogrande_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/winogrande_train_limited" "1267"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "arc_challenge_train_limited" "01:30:00" "q35-27b_arc_challenge_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/arc_challenge_train_limited" "1172"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "arc_easy_train_limited" "02:00:00" "q35-27b_arc_easy_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/arc_easy_train_limited" "2376"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "gsm8k_train_limited" "08:00:00" "q35-27b_gsm8k_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/gsm8k_train_limited" "1319"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "boolq_train_limited" "01:00:00" "q35-27b_boolq_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/boolq_train_limited" "3270"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "piqa_train_limited" "01:00:00" "q35-27b_piqa_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/piqa_train_limited" "3084"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "sciq_train_limited" "01:00:00" "q35-27b_sciq_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/sciq_train_limited" "1000"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "social_iqa_train_limited" "02:00:00" "q35-27b_social_iqa_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/social_iqa_train_limited" "1954"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "lambada_standard_train_limited" "02:00:00" "q35-27b_lambada_train_limited" "$RESULTS_DIR/Qwen3.5-27B-AWQ/lambada_standard_train_limited" "5153"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "gpqa_main_n_shot" "01:30:00" "q35-27b_gpqa" "$RESULTS_DIR/Qwen3.5-27B-AWQ/gpqa_main_n_shot"

echo ""
echo "=== Qwen3.5-35B-A3B-AWQ (11 jobs) ==="
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "hellaswag_train_limited" "02:00:00" "q35-35b_hellaswag_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/hellaswag_train_limited" "10042"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "winogrande_train_limited" "02:00:00" "q35-35b_winogrande_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/winogrande_train_limited" "1267"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "arc_challenge_train_limited" "02:00:00" "q35-35b_arc_challenge_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/arc_challenge_train_limited" "1172"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "arc_easy_train_limited" "02:00:00" "q35-35b_arc_easy_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/arc_easy_train_limited" "2376"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "gsm8k_train_limited" "08:00:00" "q35-35b_gsm8k_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/gsm8k_train_limited" "1319"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "boolq_train_limited" "01:00:00" "q35-35b_boolq_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/boolq_train_limited" "3270"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "piqa_train_limited" "01:00:00" "q35-35b_piqa_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/piqa_train_limited" "3084"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "sciq_train_limited" "01:00:00" "q35-35b_sciq_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/sciq_train_limited" "1000"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "social_iqa_train_limited" "02:00:00" "q35-35b_social_iqa_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/social_iqa_train_limited" "1954"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "lambada_standard_train_limited" "02:00:00" "q35-35b_lambada_train_limited" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/lambada_standard_train_limited" "5153"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "gpqa_main_n_shot" "01:30:00" "q35-35b_gpqa" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/gpqa_main_n_shot"

echo ""
echo "=== Qwen3.5-122B-A10B-AWQ (11 jobs, 2 GPUs) ==="
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "hellaswag_train_limited" "02:00:00" "q35-122b_hellaswag_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/hellaswag_train_limited" 2 2 "10042"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "winogrande_train_limited" "02:00:00" "q35-122b_winogrande_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/winogrande_train_limited" 2 2 "1267"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "arc_challenge_train_limited" "02:00:00" "q35-122b_arc_challenge_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/arc_challenge_train_limited" 2 2 "1172"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "arc_easy_train_limited" "02:00:00" "q35-122b_arc_easy_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/arc_easy_train_limited" 2 2 "2376"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "gsm8k_train_limited" "05:00:00" "q35-122b_gsm8k_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/gsm8k_train_limited" 2 2 "1319"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "boolq_train_limited" "01:00:00" "q35-122b_boolq_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/boolq_train_limited" 2 2 "3270"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "piqa_train_limited" "01:00:00" "q35-122b_piqa_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/piqa_train_limited" 2 2 "3084"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "sciq_train_limited" "01:00:00" "q35-122b_sciq_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/sciq_train_limited" 2 2 "1000"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "social_iqa_train_limited" "02:00:00" "q35-122b_social_iqa_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/social_iqa_train_limited" 2 2 "1954"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "lambada_standard_train_limited" "02:00:00" "q35-122b_lambada_train_limited" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/lambada_standard_train_limited" 2 2 "5153"
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "gpqa_main_n_shot" "01:30:00" "q35-122b_gpqa" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/gpqa_main_n_shot" 2 2

echo ""
echo "=== Llama-3.2-1B-Instruct (11 jobs) ==="
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "hellaswag_train_limited" "01:30:00" "llama32-1b_hellaswag_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/hellaswag_train_limited" "10042"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "winogrande_train_limited" "01:30:00" "llama32-1b_winogrande_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/winogrande_train_limited" "1267"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "arc_challenge_train_limited" "01:30:00" "llama32-1b_arc_challenge_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/arc_challenge_train_limited" "1172"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "arc_easy_train_limited" "01:30:00" "llama32-1b_arc_easy_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/arc_easy_train_limited" "2376"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "gsm8k_train_limited" "04:30:00" "llama32-1b_gsm8k_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/gsm8k_train_limited" "1319"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "boolq_train_limited" "01:30:00" "llama32-1b_boolq_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/boolq_train_limited" "3270"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "piqa_train_limited" "01:30:00" "llama32-1b_piqa_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/piqa_train_limited" "3084"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "sciq_train_limited" "01:30:00" "llama32-1b_sciq_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/sciq_train_limited" "1000"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "social_iqa_train_limited" "01:30:00" "llama32-1b_social_iqa_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/social_iqa_train_limited" "1954"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "lambada_standard_train_limited" "01:30:00" "llama32-1b_lambada_train_limited" "$RESULTS_DIR/Llama-3.2-1B-Instruct/lambada_standard_train_limited" "5153"
create_sbatch "Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "gpqa_main_n_shot" "01:30:00" "llama32-1b_gpqa" "$RESULTS_DIR/Llama-3.2-1B-Instruct/gpqa_main_n_shot"

echo ""
echo "=== Llama-3-8B-Instruct-AWQ (11 jobs) ==="
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "hellaswag_train_limited" "01:30:00" "llama3-8b_hellaswag_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/hellaswag_train_limited" "10042"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "winogrande_train_limited" "01:30:00" "llama3-8b_winogrande_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/winogrande_train_limited" "1267"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "arc_challenge_train_limited" "01:30:00" "llama3-8b_arc_challenge_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/arc_challenge_train_limited" "1172"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "arc_easy_train_limited" "01:30:00" "llama3-8b_arc_easy_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/arc_easy_train_limited" "2376"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "gsm8k_train_limited" "08:00:00" "llama3-8b_gsm8k_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/gsm8k_train_limited" "1319"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "boolq_train_limited" "01:30:00" "llama3-8b_boolq_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/boolq_train_limited" "3270"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "piqa_train_limited" "01:30:00" "llama3-8b_piqa_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/piqa_train_limited" "3084"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "sciq_train_limited" "01:30:00" "llama3-8b_sciq_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/sciq_train_limited" "1000"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "social_iqa_train_limited" "01:30:00" "llama3-8b_social_iqa_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/social_iqa_train_limited" "1954"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "lambada_standard_train_limited" "01:30:00" "llama3-8b_lambada_train_limited" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/lambada_standard_train_limited" "5153"
create_sbatch "Llama-3-8B-Instruct-AWQ" "casperhansen/llama-3-8b-instruct-awq" "gpqa_main_n_shot" "01:30:00" "llama3-8b_gpqa" "$RESULTS_DIR/Llama-3-8B-Instruct-AWQ/gpqa_main_n_shot"

echo ""
echo "=== Llama-3.3-70B-Instruct-AWQ (11 jobs) ==="
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "hellaswag_train_limited" "01:30:00" "llama33-70b_hellaswag_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/hellaswag_train_limited" "10042"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "winogrande_train_limited" "01:30:00" "llama33-70b_winogrande_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/winogrande_train_limited" "1267"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "arc_challenge_train_limited" "01:30:00" "llama33-70b_arc_challenge_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/arc_challenge_train_limited" "1172"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "arc_easy_train_limited" "01:30:00" "llama33-70b_arc_easy_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/arc_easy_train_limited" "2376"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "gsm8k_train_limited" "08:00:00" "llama33-70b_gsm8k_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/gsm8k_train_limited" "1319"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "boolq_train_limited" "01:30:00" "llama33-70b_boolq_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/boolq_train_limited" "3270"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "piqa_train_limited" "01:30:00" "llama33-70b_piqa_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/piqa_train_limited" "3084"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "sciq_train_limited" "01:30:00" "llama33-70b_sciq_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/sciq_train_limited" "1000"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "social_iqa_train_limited" "01:30:00" "llama33-70b_social_iqa_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/social_iqa_train_limited" "1954"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "lambada_standard_train_limited" "01:30:00" "llama33-70b_lambada_train_limited" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/lambada_standard_train_limited" "5153"
create_sbatch "Llama-3.3-70B-Instruct-AWQ" "casperhansen/llama-3.3-70b-instruct-awq" "gpqa_main_n_shot" "01:30:00" "llama33-70b_gpqa" "$RESULTS_DIR/Llama-3.3-70B-Instruct-AWQ/gpqa_main_n_shot"

echo ""
echo "=========================================="
echo "Done! Created all training sbatch files."
echo ""
echo "Job summary:"
echo "  Qwen3.5-2B:                     11 jobs (1 GPU each)"
echo "  Qwen3.5-9B-AWQ:                 11 jobs (1 GPU each)"
echo "  Qwen3.5-27B-AWQ:                11 jobs (1 GPU each)"
echo "  Qwen3.5-35B-A3B-AWQ:            11 jobs (1 GPU each)"
echo "  Qwen3.5-122B-A10B-AWQ:          11 jobs (2 GPUs each)"
echo "  Llama-3.2-1B-Instruct:          11 jobs (1 GPU each)"
echo "  Llama-3-8B-Instruct-AWQ:        11 jobs (1 GPU each)"
echo "  Llama-3.3-70B-Instruct-AWQ:     11 jobs (1 GPU each)"
echo "  Total:                          88 jobs"
echo ""
echo "TRAIN-LIMITED BENCHMARKS (10):"
echo "  hellaswag, winogrande, arc_challenge, arc_easy, gsm8k"
echo "  boolq, piqa, sciq, social_iqa, lambada_standard"
echo ""
echo "KEPT AS-IS (1):"
echo "  gpqa_main_n_shot"
echo ""
echo "EXCLUDED (4):"
echo "  wikitext, aime25, mmlu, acp_bench"
echo ""
echo "Tasks requiring datasets==3.6.0: social_iqa_train_limited"
echo ""
echo "Run './run_training_jobs.sh' to submit all jobs to SLURM."

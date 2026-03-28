#!/bin/bash

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results"
WORK_DIR="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness"
IMAGE="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v2.sqsh"
MOUNT_DSS="-m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2"
MOUNT_HOME="-m /dss/dsshome1/09/\$USER:/root"

V36_TASKS="logiqa,logiqa2,social_iqa"

needs_v36() {
    local task="$1"
    [[ ",$V36_TASKS," == *",$task,"* ]]
}

# Multi-GPU model sbatch creator (for tensor parallel)
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
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v2.sqsh

# Create enroot container from squashfs image (|| true handles re-import)
enroot create --name lm_eval_container_v2 "\$IMAGE" || true

# Run evaluation in enroot container with mounts
enroot start \\
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf \\
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \\
    -m /dss/dsshome1/09/\$USER:/root \\
    --root lm_eval_container_v2 bash -c "
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
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v2.sqsh

# Create enroot container from squashfs image (|| true handles re-import)
enroot create --name lm_eval_container_v2 "\$IMAGE" || true

# Run evaluation in enroot container with mounts
enroot start \\
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf \\
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \\
    -m /dss/dsshome1/09/\$USER:/root \\
    --root lm_eval_container_v2 bash -c "
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

# Qwen3.5-2B (9 individual jobs)
echo "=== Qwen3.5-2B ==="
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "arc_challenge" "02:00:00" "q35-2b_arc_challenge" "$RESULTS_DIR/Qwen3.5-2B/arc_challenge"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "arc_easy" "02:00:00" "q35-2b_arc_easy" "$RESULTS_DIR/Qwen3.5-2B/arc_easy"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "boolq" "01:00:00" "q35-2b_boolq" "$RESULTS_DIR/Qwen3.5-2B/boolq"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "logiqa" "02:00:00" "q35-2b_logiqa" "$RESULTS_DIR/Qwen3.5-2B/logiqa"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "logiqa2" "02:00:00" "q35-2b_logiqa2" "$RESULTS_DIR/Qwen3.5-2B/logiqa2"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "piqa" "01:00:00" "q35-2b_piqa" "$RESULTS_DIR/Qwen3.5-2B/piqa"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "sciq" "01:00:00" "q35-2b_sciq" "$RESULTS_DIR/Qwen3.5-2B/sciq"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "social_iqa" "02:00:00" "q35-2b_social_iqa" "$RESULTS_DIR/Qwen3.5-2B/social_iqa"
create_sbatch "Qwen3.5-2B" "Qwen/Qwen3.5-2B" "winogrande" "01:00:00" "q35-2b_winogrande" "$RESULTS_DIR/Qwen3.5-2B/winogrande"

echo ""
echo "=== Qwen3.5-9B-AWQ ==="
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "arc_challenge" "02:00:00" "q35-9b_arc_challenge" "$RESULTS_DIR/Qwen3.5-9B-AWQ/arc_challenge"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "arc_easy" "02:00:00" "q35-9b_arc_easy" "$RESULTS_DIR/Qwen3.5-9B-AWQ/arc_easy"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "boolq" "01:00:00" "q35-9b_boolq" "$RESULTS_DIR/Qwen3.5-9B-AWQ/boolq"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "logiqa" "02:00:00" "q35-9b_logiqa" "$RESULTS_DIR/Qwen3.5-9B-AWQ/logiqa"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "logiqa2" "02:00:00" "q35-9b_logiqa2" "$RESULTS_DIR/Qwen3.5-9B-AWQ/logiqa2"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "piqa" "01:00:00" "q35-9b_piqa" "$RESULTS_DIR/Qwen3.5-9B-AWQ/piqa"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "sciq" "01:00:00" "q35-9b_sciq" "$RESULTS_DIR/Qwen3.5-9B-AWQ/sciq"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "social_iqa" "02:00:00" "q35-9b_social_iqa" "$RESULTS_DIR/Qwen3.5-9B-AWQ/social_iqa"
create_sbatch "Qwen3.5-9B-AWQ" "QuantTrio/Qwen3.5-9B-AWQ" "winogrande" "01:00:00" "q35-9b_winogrande" "$RESULTS_DIR/Qwen3.5-9B-AWQ/winogrande"

echo ""
echo "=== Qwen3.5-27B-AWQ ==="
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "arc_challenge" "02:00:00" "q35-27b_arc_challenge" "$RESULTS_DIR/Qwen3.5-27B-AWQ/arc_challenge"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "arc_easy" "02:00:00" "q35-27b_arc_easy" "$RESULTS_DIR/Qwen3.5-27B-AWQ/arc_easy"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "boolq" "01:00:00" "q35-27b_boolq" "$RESULTS_DIR/Qwen3.5-27B-AWQ/boolq"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "logiqa" "02:00:00" "q35-27b_logiqa" "$RESULTS_DIR/Qwen3.5-27B-AWQ/logiqa"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "logiqa2" "02:00:00" "q35-27b_logiqa2" "$RESULTS_DIR/Qwen3.5-27B-AWQ/logiqa2"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "piqa" "01:00:00" "q35-27b_piqa" "$RESULTS_DIR/Qwen3.5-27B-AWQ/piqa"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "sciq" "01:00:00" "q35-27b_sciq" "$RESULTS_DIR/Qwen3.5-27B-AWQ/sciq"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "social_iqa" "02:00:00" "q35-27b_social_iqa" "$RESULTS_DIR/Qwen3.5-27B-AWQ/social_iqa"
create_sbatch "Qwen3.5-27B-AWQ" "QuantTrio/Qwen3.5-27B-AWQ" "winogrande" "01:00:00" "q35-27b_winogrande" "$RESULTS_DIR/Qwen3.5-27B-AWQ/winogrande"

echo ""
echo "=== Qwen3.5-35B-A3B-AWQ ==="
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "arc_challenge" "02:00:00" "q35-35b_arc_challenge" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/arc_challenge"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "arc_easy" "02:00:00" "q35-35b_arc_easy" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/arc_easy"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "boolq" "01:00:00" "q35-35b_boolq" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/boolq"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "logiqa" "02:00:00" "q35-35b_logiqa" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/logiqa"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "logiqa2" "02:00:00" "q35-35b_logiqa2" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/logiqa2"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "piqa" "01:00:00" "q35-35b_piqa" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/piqa"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "sciq" "01:00:00" "q35-35b_sciq" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/sciq"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "social_iqa" "02:00:00" "q35-35b_social_iqa" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/social_iqa"
create_sbatch "Qwen3.5-35B-A3B-AWQ" "QuantTrio/Qwen3.5-35B-A3B-AWQ" "winogrande" "01:00:00" "q35-35b_winogrande" "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/winogrande"

echo ""
echo "=== Qwen3.5-122B-A10B-AWQ (2 GPUs) ==="
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "arc_challenge" "02:00:00" "q35-122b_arc_challenge" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/arc_challenge" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "arc_easy" "02:00:00" "q35-122b_arc_easy" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/arc_easy" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "boolq" "01:00:00" "q35-122b_boolq" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/boolq" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "logiqa" "02:00:00" "q35-122b_logiqa" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/logiqa" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "logiqa2" "02:00:00" "q35-122b_logiqa2" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/logiqa2" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "piqa" "01:00:00" "q35-122b_piqa" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/piqa" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "sciq" "01:00:00" "q35-122b_sciq" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/sciq" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "social_iqa" "02:00:00" "q35-122b_social_iqa" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/social_iqa" 2 2
create_sbatch_multi_gpu "Qwen3.5-122B-A10B-AWQ" "QuantTrio/Qwen3.5-122B-A10B-AWQ" "winogrande" "01:00:00" "q35-122b_winogrande" "$RESULTS_DIR/Qwen3.5-122B-A10B-AWQ/winogrande" 2 2

echo ""
echo "=== Qwen3.5-397B-A17B-AWQ (4 GPUs) ==="
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "arc_challenge" "03:00:00" "q35-397b_arc_challenge" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/arc_challenge" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "arc_easy" "03:00:00" "q35-397b_arc_easy" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/arc_easy" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "boolq" "02:00:00" "q35-397b_boolq" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/boolq" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "logiqa" "03:00:00" "q35-397b_logiqa" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/logiqa" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "logiqa2" "03:00:00" "q35-397b_logiqa2" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/logiqa2" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "piqa" "02:00:00" "q35-397b_piqa" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/piqa" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "sciq" "02:00:00" "q35-397b_sciq" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/sciq" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "social_iqa" "03:00:00" "q35-397b_social_iqa" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/social_iqa" 4 4
create_sbatch_multi_gpu "Qwen3.5-397B-A17B-AWQ" "QuantTrio/Qwen3.5-397B-A17B-AWQ" "winogrande" "02:00:00" "q35-397b_winogrande" "$RESULTS_DIR/Qwen3.5-397B-A17B-AWQ/winogrande" 4 4

echo ""
echo "=========================================="
echo "Done! Created all sbatch files."
echo ""
echo "Job summary:"
echo "  Qwen3.5-2B:            9 jobs (1 GPU each)"
echo "  Qwen3.5-9B-AWQ:       9 jobs (1 GPU each)"
echo "  Qwen3.5-27B-AWQ:      9 jobs (1 GPU each)"
echo "  Qwen3.5-35B-A3B-AWQ:  9 jobs (1 GPU each)"
echo "  Qwen3.5-122B-A10B-AWQ: 9 jobs (2 GPUs each)"
echo "  Qwen3.5-397B-A17B-AWQ: 9 jobs (4 GPUs each)"
echo "  Total:                54 jobs"
echo ""
echo "Tasks requiring datasets==3.6.0: logiqa, logiqa2, social_iqa"
echo "Tasks using datasets>=4.0: arc_challenge, arc_easy, boolq, piqa, sciq, winogrande"
echo ""
echo "Run './run_all_jobs.sh' to submit all jobs to SLURM."
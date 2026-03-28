#!/bin/bash

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results"
WORK_DIR="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness"
IMAGE="/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus-v2/messplus_v2.sqsh"
MOUNT_DSS="-m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2"
MOUNT_HOME="-m /dss/dsshome1/09/\$USER:/root"

mkdir -p "$RESULTS_DIR"

create_sbatch() {
    local model_name="$1"
    local model_path="$2"
    local tasks="$3"
    local time="$4"
    local job_name="$5"
    local output_dir="$6"
    
    local dir="$JOBS_DIR/$model_name"
    mkdir -p "$dir"
    
    local sbatch_file="$dir/${job_name}.sbatch"
    
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
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus-v2/messplus_v2.sqsh

# Create enroot container from squashfs image (|| true handles re-import)
enroot create --name messplus_v2 "\$IMAGE" || true

# Run evaluation in enroot container with mounts
enroot start \\
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf \\
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \\
    -m /dss/dsshome1/09/\$USER:/root \\
    --root messplus_v2 bash -c "
    cd \$WORK_DIR && \\
    uv run --extra vllm --extra energy lm_eval run \\
        --model vllm \\
        --model_args pretrained=${model_path},enforce_eager=True,gdn_prefill_backend=triton,track_energy=true,approx_instant_energy=true,dtype=float16,gpu_memory_utilization=0.9,trust_remote_code=True \\
        --tasks ${tasks} \\
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

# Qwen3.5-2B (2 jobs, ~4h total)
echo "=== Qwen3.5-2B ==="
create_sbatch \
    "Qwen3.5-2B" \
    "Qwen/Qwen3.5-2B" \
    "arc_challenge,winogrande,piqa,sciq" \
    "02:00:00" \
    "q35-2b_g1" \
    "$RESULTS_DIR/Qwen3.5-2B/group_a"

create_sbatch \
    "Qwen3.5-2B" \
    "Qwen/Qwen3.5-2B" \
    "arc_easy,boolq,logiqa,social_iqa" \
    "02:00:00" \
    "q35-2b_g2" \
    "$RESULTS_DIR/Qwen3.5-2B/group_b"

# Qwen3.5-9B-AWQ (4 jobs, ~11.5h total)
echo ""
echo "=== Qwen3.5-9B-AWQ ==="
create_sbatch \
    "Qwen3.5-9B-AWQ" \
    "QuantTrio/Qwen3.5-9B-AWQ" \
    "arc_challenge,logiqa" \
    "01:30:00" \
    "q35-9b_g1" \
    "$RESULTS_DIR/Qwen3.5-9B-AWQ/group_a"

create_sbatch \
    "Qwen3.5-9B-AWQ" \
    "QuantTrio/Qwen3.5-9B-AWQ" \
    "boolq,arc_easy,social_iqa" \
    "01:30:00" \
    "q35-9b_g2" \
    "$RESULTS_DIR/Qwen3.5-9B-AWQ/group_b"

create_sbatch \
    "Qwen3.5-9B-AWQ" \
    "QuantTrio/Qwen3.5-9B-AWQ" \
    "winogrande,sciq" \
    "01:30:00" \
    "q35-9b_g3" \
    "$RESULTS_DIR/Qwen3.5-9B-AWQ/group_c"

create_sbatch \
    "Qwen3.5-9B-AWQ" \
    "QuantTrio/Qwen3.5-9B-AWQ" \
    "piqa" \
    "01:00:00" \
    "q35-9b_g4" \
    "$RESULTS_DIR/Qwen3.5-9B-AWQ/group_d"

# Qwen3.5-27B-AWQ (8 jobs, ~44h total)
echo ""
echo "=== Qwen3.5-27B-AWQ ==="
create_sbatch \
    "Qwen3.5-27B-AWQ" \
    "QuantTrio/Qwen3.5-27B-AWQ" \
    "arc_challenge" \
    "08:00:00" \
    "q35-27b_arc_challenge" \
    "$RESULTS_DIR/Qwen3.5-27B-AWQ/arc_challenge"

create_sbatch \
    "Qwen3.5-27B-AWQ" \
    "QuantTrio/Qwen3.5-27B-AWQ" \
    "arc_easy" \
    "04:00:00" \
    "q35-27b_arc_easy" \
    "$RESULTS_DIR/Qwen3.5-27B-AWQ/arc_easy"

create_sbatch \
    "Qwen3.5-27B-AWQ" \
    "QuantTrio/Qwen3.5-27B-AWQ" \
    "winogrande" \
    "03:00:00" \
    "q35-27b_winogrande" \
    "$RESULTS_DIR/Qwen3.5-27B-AWQ/winogrande"

create_sbatch \
    "Qwen3.5-27B-AWQ" \
    "QuantTrio/Qwen3.5-27B-AWQ" \
    "boolq" \
    "03:00:00" \
    "q35-27b_boolq" \
    "$RESULTS_DIR/Qwen3.5-27B-AWQ/boolq"

create_sbatch \
    "Qwen3.5-27B-AWQ" \
    "QuantTrio/Qwen3.5-27B-AWQ" \
    "logiqa" \
    "04:00:00" \
    "q35-27b_logiqa" \
    "$RESULTS_DIR/Qwen3.5-27B-AWQ/logiqa"

create_sbatch \
    "Qwen3.5-27B-AWQ" \
    "QuantTrio/Qwen3.5-27B-AWQ" \
    "piqa" \
    "03:00:00" \
    "q35-27b_piqa" \
    "$RESULTS_DIR/Qwen3.5-27B-AWQ/piqa"

create_sbatch \
    "Qwen3.5-27B-AWQ" \
    "QuantTrio/Qwen3.5-27B-AWQ" \
    "sciq" \
    "03:00:00" \
    "q35-27b_sciq" \
    "$RESULTS_DIR/Qwen3.5-27B-AWQ/sciq"

create_sbatch \
    "Qwen3.5-27B-AWQ" \
    "QuantTrio/Qwen3.5-27B-AWQ" \
    "social_iqa" \
    "04:00:00" \
    "q35-27b_social_iqa" \
    "$RESULTS_DIR/Qwen3.5-27B-AWQ/social_iqa"

# Qwen3.5-35B-A3B-AWQ (8 jobs, ~64h total)
echo ""
echo "=== Qwen3.5-35B-A3B-AWQ ==="
create_sbatch \
    "Qwen3.5-35B-A3B-AWQ" \
    "QuantTrio/Qwen3.5-35B-A3B-AWQ" \
    "arc_challenge" \
    "12:00:00" \
    "q35-35b_arc_challenge" \
    "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/arc_challenge"

create_sbatch \
    "Qwen3.5-35B-A3B-AWQ" \
    "QuantTrio/Qwen3.5-35B-A3B-AWQ" \
    "arc_easy" \
    "04:00:00" \
    "q35-35b_arc_easy" \
    "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/arc_easy"

create_sbatch \
    "Qwen3.5-35B-A3B-AWQ" \
    "QuantTrio/Qwen3.5-35B-A3B-AWQ" \
    "winogrande" \
    "04:00:00" \
    "q35-35b_winogrande" \
    "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/winogrande"

create_sbatch \
    "Qwen3.5-35B-A3B-AWQ" \
    "QuantTrio/Qwen3.5-35B-A3B-AWQ" \
    "boolq" \
    "04:00:00" \
    "q35-35b_boolq" \
    "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/boolq"

create_sbatch \
    "Qwen3.5-35B-A3B-AWQ" \
    "QuantTrio/Qwen3.5-35B-A3B-AWQ" \
    "logiqa" \
    "06:00:00" \
    "q35-35b_logiqa" \
    "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/logiqa"

create_sbatch \
    "Qwen3.5-35B-A3B-AWQ" \
    "QuantTrio/Qwen3.5-35B-A3B-AWQ" \
    "piqa" \
    "04:00:00" \
    "q35-35b_piqa" \
    "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/piqa"

create_sbatch \
    "Qwen3.5-35B-A3B-AWQ" \
    "QuantTrio/Qwen3.5-35B-A3B-AWQ" \
    "sciq" \
    "04:00:00" \
    "q35-35b_sciq" \
    "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/sciq"

create_sbatch \
    "Qwen3.5-35B-A3B-AWQ" \
    "QuantTrio/Qwen3.5-35B-A3B-AWQ" \
    "social_iqa" \
    "06:00:00" \
    "q35-35b_social_iqa" \
    "$RESULTS_DIR/Qwen3.5-35B-A3B-AWQ/social_iqa"

echo ""
echo "=========================================="
echo "Done! Created all sbatch files."
echo ""
echo "Job summary:"
echo "  Qwen3.5-2B:          2 jobs"
echo "  Qwen3.5-9B-AWQ:      4 jobs"
echo "  Qwen3.5-27B-AWQ:     8 jobs"
echo "  Qwen3.5-35B-A3B-AWQ: 8 jobs"
echo "  Total:               22 jobs"
echo ""
echo "Run './run_all_jobs.sh' to submit all jobs to SLURM."
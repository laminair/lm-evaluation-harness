#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=q35-27b_arc_easy
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Environment
export HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf
export HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false

# WandB key from ~/.netrc
export WANDB_API_KEY=$(grep -A1 "machine api.wandb.ai" /dss/dsshome1/09/$USER/.netrc 2>/dev/null | grep "password" | awk '{print $2}')

# Paths
WORK_DIR=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus-v2/messplus_v2.sqsh

# Create enroot container from squashfs image (|| true handles re-import)
enroot create --name messplus_v2 "$IMAGE" || true

# Run evaluation in enroot container with mounts
srun enroot start \
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \
    -m /dss/dsshome1/09/$USER:/root \
    --root messplus_v2 bash -c "
    cd $WORK_DIR && \
    uv pip install ray && \
    uv run lm_eval \
        --model vllm \
        --model_args pretrained=QuantTrio/Qwen3.5-27B-AWQ,dtype=float16,gpu_memory_utilization=0.9,trust_remote_code=True \
        --tasks arc_easy \
        --batch_size 1 \
        --output_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results/Qwen3.5-27B-AWQ/arc_easy \
        --log_samples \
        --track_energy
"

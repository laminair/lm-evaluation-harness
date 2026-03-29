#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=llama4-scout_logiqa2
#SBATCH --output=%x_%j.out

# Environment
export HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf
export HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false

# WandB key from ~/.netrc
export WANDB_API_KEY=$(grep -A1 "machine api.wandb.ai" /dss/dsshome1/09/$USER/.netrc 2>/dev/null | grep "password" | awk '{print $2}')

# Paths
WORK_DIR=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v2.sqsh

# Create enroot container from squashfs image (|| true handles re-import)
enroot create --name lm_eval_container_v2 "$IMAGE" || true

# Run evaluation in enroot container with mounts
enroot start \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf \
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \
    -m /dss/dsshome1/09/$USER:/root \
    --root lm_eval_container_v2 bash -c "
    cd $WORK_DIR && \
    uv run --extra vllm --extra energy --with 'datasets==3.6.0' lm_eval run \
        --model vllm \
        --model_args pretrained=kishizaki-sci/Llama-4-Scout-17B-16E-Instruct-AWQ,enforce_eager=True,gdn_prefill_backend=triton,track_energy=true,approx_instant_energy=true,dtype=float16,gpu_memory_utilization=0.9,trust_remote_code=True \
        --tasks logiqa2 \
        --batch_size 1 \
        --output_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results/Llama-4-Scout-17B-16E-Instruct-AWQ/logiqa2 \
        --log_samples \
        --track_energy
"

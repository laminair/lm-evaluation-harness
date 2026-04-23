#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=q35-35b_hellaswag_train_limited
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
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/lm_eval_container_v3.sqsh

# Create enroot container from squashfs image (|| true handles re-import)
enroot create --name lm_eval_container_v3 "$IMAGE" || true

# Run evaluation in enroot container with mounts
enroot start \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf \
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \
    -m /dss/dsshome1/09/$USER:/root \
     \
    --root lm_eval_container_v3 bash -c "
    cd $WORK_DIR && \
    uv run --extra vllm --extra energy lm_eval run \
        --model vllm \
        --model_args pretrained=QuantTrio/Qwen3.5-35B-A3B-AWQ,enforce_eager=True,gdn_prefill_backend=triton,track_energy=true,approx_instant_energy=true,dtype=float16,gpu_memory_utilization=0.9,trust_remote_code=True \
        --tasks hellaswag_train_limited \
        --batch_size 1 \
        --limit 10042 \
        --output_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results_train_data/Qwen3.5-35B-A3B-AWQ/hellaswag_train_limited \
        --log_samples \
        --track_energy
"

#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --exclude=lrz-hgx-h100-002,lrz-hgx-h100-014
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=llama3-8b_lambada
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
        --model_args pretrained=casperhansen/llama-3-8b-instruct-awq,enforce_eager=True,gdn_prefill_backend=triton,track_energy=true,approx_instant_energy=true,dtype=float16,gpu_memory_utilization=0.9,trust_remote_code=True \
        --tasks lambada_standard \
        --batch_size 1 \
        --output_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results/Llama-3-8B-Instruct-AWQ/lambada_standard \
        --log_samples \
        --track_energy
"

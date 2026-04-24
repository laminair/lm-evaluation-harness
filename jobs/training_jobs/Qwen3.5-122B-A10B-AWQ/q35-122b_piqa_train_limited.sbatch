#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --job-name=q35-122b_piqa_train_limited
#SBATCH --output=%x_%j.out
#SBATCH --exclude=lrz-hgx-h100-020

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
        --model_args pretrained=QuantTrio/Qwen3.5-122B-A10B-AWQ,enforce_eager=True,gdn_prefill_backend=triton,tensor_parallel_size=2,gpu_memory_utilization=0.9,trust_remote_code=True \
        --tasks piqa_train_limited \
        --batch_size 1 \
        --limit 3084 \
        --output_path /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness/results_train_data/Qwen3.5-122B-A10B-AWQ/piqa_train_limited \
        --log_samples \
        --track_energy
"

#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=download_models
#SBATCH --output=%x_%j.out

# Environment
export HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf
export HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false

# Paths
WORK_DIR=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/lm-evaluation-harness
IMAGE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus-v2/messplus_v2.sqsh

# Create enroot container from squashfs image
enroot create --name messplus_v2 "$IMAGE" || true

# Download models sequentially
enroot start \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf \
    -m /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2:/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2 \
    -m /dss/dsshome1/09/$USER:/root \
    --root messplus_v2 bash -c "
    cd $WORK_DIR && \
    echo '=== Downloading Qwen/Qwen3.5-2B ===' && \
    hf download Qwen/Qwen3.5-2B && \
    echo '=== Downloading QuantTrio/Qwen3.5-9B-AWQ ===' && \
    hf download QuantTrio/Qwen3.5-9B-AWQ && \
    echo '=== Downloading QuantTrio/Qwen3.5-27B-AWQ ===' && \
    hf download QuantTrio/Qwen3.5-27B-AWQ && \
    echo '=== Downloading QuantTrio/Qwen3.5-35B-A3B-AWQ ===' && \
    hf download QuantTrio/Qwen3.5-35B-A3B-AWQ && \
    echo '=== All model downloads complete ==='
"

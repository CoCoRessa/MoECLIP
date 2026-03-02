#!/usr/bin/env bash
set -euo pipefail

# Usage: GPU_ID=0 bash scripts.sh
# If GPU_ID is not set, default to 0
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

BASE_SAVE_ROOT="ckpt/baseline"
SAVE_PATH="${BASE_SAVE_ROOT}"
echo "[INFO] Using save_path=${SAVE_PATH}"

# training
python train.py --save_path "${SAVE_PATH}"

# testing
declare -a dataset=(Brain BTAD Colon_clinicDB Colon_colonDB Colon_cvc300 Colon_Kvasir DTD-Synthetic Endo headct Liver MVTec Retina RSDD)
for i in "${dataset[@]}"; do
        python test.py --dataset "$i" --save_path "${SAVE_PATH}"
done
#!/usr/bin/env bash
# Usage: ./run1.sh <exp_name> <repo_id_or_dataset_path> [config_name] [nproc]
# Example: bash run1.sh verb_color_0316 /path/to/lerobot_dataset
#          bash run1.sh my_pi0_exp /path/to/data pi0_maniskill
#          bash run1.sh my_fast_exp /path/to/data pi0_fast_maniskill 4
#
# config_name options: pi05_maniskill (default) | pi0_maniskill | pi0_fast_maniskill
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

EXP_NAME="${1:?Usage: $0 <exp_name> <repo_id_or_dataset_path> [config_name] [nproc]}"
REPO_ID="${2:?Usage: $0 <exp_name> <repo_id_or_dataset_path> [config_name] [nproc]}"
CONFIG="${3:-pi05_maniskill}"
NPROC="${4:-8}"

mkdir -p run
LOG="run/${EXP_NAME}.log"
echo "Logging to $LOG and terminal"
echo "exp_name=$EXP_NAME repo_id=$REPO_ID config=$CONFIG nproc=$NPROC"
uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    scripts/train_pytorch.py "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --data.repo_id "${REPO_ID}" \
    2>&1 | tee "${LOG}"
echo "Successfully finished training"

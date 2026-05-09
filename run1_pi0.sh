#!/usr/bin/env bash
# Usage: ./run1_pi0.sh <exp_name> <repo_id_or_dataset_path>
# Example: bash run1_pi0.sh color_spatial_random_f18_pi0 /root/workspace/demo/200/huggingface_data/color_spatial/random_f18
# Equivalent of run1.sh but uses pi0_maniskill (pi05=False) config.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

EXP_NAME="${1:?Usage: $0 <exp_name> <repo_id_or_dataset_path>}"
REPO_ID="${2:?Usage: $0 <exp_name> <repo_id_or_dataset_path>}"

mkdir -p run
LOG="run/${EXP_NAME}.log"
echo "Logging to $LOG and terminal"
echo "exp_name=$EXP_NAME repo_id=$REPO_ID"
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi0_maniskill --exp_name "$EXP_NAME" --data.repo_id "$REPO_ID" 2>&1 | tee "$LOG"
echo "Successfully finished training"

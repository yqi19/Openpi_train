#!/usr/bin/env bash
# Usage: ./run.sh <exp_name> <repo_id_or_dataset_path>
# Example: ./run.sh verb_color_0316 /root/workspace/data_gen/0316_verb_color/lerobot_dataset
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

EXP_NAME="${1:?Usage: $0 <exp_name> <repo_id_or_dataset_path>}"
REPO_ID="${2:?Usage: $0 <exp_name> <repo_id_or_dataset_path>}"

uv run scripts/compute_norm_stats.py pi05_maniskill --data.repo_id "$REPO_ID"

mkdir -p run
LOG="run/${EXP_NAME}.log"
echo "Logging to $LOG and terminal"
echo "exp_name=$EXP_NAME repo_id=$REPO_ID"
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_maniskill --exp_name "$EXP_NAME" --data.repo_id "$REPO_ID" 2>&1 | tee "$LOG"
echo "Successfully finished training"

cd /root/workspace
python occupy_gpu.py
echo "Successfully occupied GPU"

#!/usr/bin/env bash
# Usage: ./run.sh <exp_name> <repo_id_or_dataset_path> [config_name]
# Example: bash run.sh verb_color_0316 /path/to/lerobot_dataset
#          bash run.sh my_pi0_exp /path/to/data pi0_maniskill
#
# config_name options: pi05_maniskill (default) | pi0_maniskill | pi0_fast_maniskill
# This script only computes norm stats. Use run1.sh to actually train.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

EXP_NAME="${1:?Usage: $0 <exp_name> <repo_id_or_dataset_path> [config_name]}"
REPO_ID="${2:?Usage: $0 <exp_name> <repo_id_or_dataset_path> [config_name]}"
CONFIG="${3:-pi05_maniskill}"

echo "Computing norm stats for config=$CONFIG repo_id=$REPO_ID"
uv run scripts/compute_norm_stats.py "${CONFIG}" --data.repo_id "$REPO_ID"
echo "Norm stats done. Now run: bash run1.sh $EXP_NAME $REPO_ID $CONFIG"

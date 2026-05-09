#!/usr/bin/env bash
# Usage: ./run_pi0.sh <exp_name> <repo_id_or_dataset_path>
# Example: bash run_pi0.sh color_spatial_random_f18_pi0 /root/workspace/demo/200/huggingface_data/color_spatial/random_f18
# Equivalent of run.sh but uses pi0_maniskill (pi05=False) config.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

EXP_NAME="${1:?Usage: $0 <exp_name> <repo_id_or_dataset_path>}"
REPO_ID="${2:?Usage: $0 <exp_name> <repo_id_or_dataset_path>}"

uv run scripts/compute_norm_stats.py pi0_maniskill --data.repo_id "$REPO_ID"

#!/usr/bin/env bash
# Usage: ./resume_run.sh <exp_name> <repo_id_or_dataset_path>
# Example: ./resume_run.sh verb_color_0316_1 /root/workspace/data_gen/0316_verb_color_1/lerobot_dataset
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

EXP_NAME="${1:?Usage: $0 <exp_name> <repo_id_or_dataset_path>}"
REPO_ID="${2:?Usage: $0 <exp_name> <repo_id_or_dataset_path>}"

EXTRA_STEPS="${EXTRA_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-3000}"
NPROC="${NPROC:-8}"

CKPT_DIR="checkpoints/pi05_maniskill/${EXP_NAME}"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint dir not found: $(pwd)/${CKPT_DIR}"
  exit 1
fi

LAST_STEP="$(ls -1 "${CKPT_DIR}" | grep -E '^[0-9]+$' | sort -n | tail -n 1 || true)"
if [[ -z "${LAST_STEP}" ]]; then
  echo "No numeric checkpoint folders under ${CKPT_DIR}"
  exit 1
fi

TARGET_STEPS=$((LAST_STEP + EXTRA_STEPS))

mkdir -p run
LOG="run/${EXP_NAME}_resume.log"

echo "Logging to $LOG"
echo "exp_name=${EXP_NAME} repo_id=${REPO_ID}"
echo "last_step=${LAST_STEP} -> num_train_steps=${TARGET_STEPS} save_interval=${SAVE_INTERVAL}"

uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  scripts/train_pytorch.py pi05_maniskill \
  --exp_name "${EXP_NAME}" \
  --data.repo_id "${REPO_ID}" \
  --resume \
  --num_train_steps "${TARGET_STEPS}" \
  --save_interval "${SAVE_INTERVAL}" \
  2>&1 | tee "${LOG}"

echo "Resume training finished"


cd /root/workspace
python occupy_gpu.py
echo "Successfully occupied GPU"

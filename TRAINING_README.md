# OpenPi ManiSkill Training Guide

How to fine-tune π0 / π0.5 / π0-FAST on ManiSkill data using Modal cloud (8× H100).

## Prerequisites

```bash
# 1. Activate Modal conda env
conda activate openpi_modal

# 2. Set Modal profile
export MODAL_PROFILE=modal-smilelab

# 3. Go to the train script directory
cd "/media/yu/Extreme SSD/corl_2026/openpi"
```

---

## Model Types

| Config name | Model | Architecture | Base checkpoint |
|---|---|---|---|
| `pi05_maniskill` | **π0.5** (default) | Flow-matching + VLM | `pi05-libero` (already on volume) |
| `pi0_maniskill` | **π0** | Flow-matching (smaller) | `pi0-libero-base` (already on volume) |
| `pi0_fast_maniskill` | **π0-FAST** | Autoregressive | downloads from GCS automatically |

Key difference between π0 and π0.5:
- **π0**: `Pi0Config()` — base flow-matching model, no `pi05=True`
- **π0.5**: `Pi0Config(pi05=True)` — improved VLM-conditioned version
- **π0-FAST**: `Pi0FASTConfig(...)` — completely different autoregressive architecture

---

## Step 1: Upload Training Data

```bash
# Upload local huggingface_data/ directory to Modal volume
MODAL_PROFILE=modal-smilelab modal volume put main \
  /home/yu/ext_ssd/demo_gen/color_spatial/random_f18/huggingface_data/ \
  /data/color_spatial_random_f18/

# Verify upload
MODAL_PROFILE=modal-smilelab modal volume ls main /data/
```

---

## Step 2: Run Training

### Train π0.5 (default)
```bash
MODAL_PROFILE=modal-smilelab modal run --detach train_modal.py \
  --exp-name color_spatial_random_f18 \
  --data-volume-path /data/color_spatial_random_f18 \
  --model-type pi05 \
  --num-train-steps 14000 \
  --save-interval 10000
```

### Train π0
```bash
MODAL_PROFILE=modal-smilelab modal run --detach train_modal.py \
  --exp-name color_spatial_random_f18_pi0 \
  --data-volume-path /data/color_spatial_random_f18 \
  --model-type pi0 \
  --num-train-steps 14000 \
  --save-interval 10000
```

### Train π0-FAST
```bash
MODAL_PROFILE=modal-smilelab modal run --detach train_modal.py \
  --exp-name color_spatial_random_f18_pi0fast \
  --data-volume-path /data/color_spatial_random_f18 \
  --model-type pi0_fast \
  --num-train-steps 14000 \
  --save-interval 10000
```

> `--detach` lets the job run after your local terminal disconnects.

---

## Step 3: Monitor Training

```bash
# Check running apps
MODAL_PROFILE=modal-smilelab modal app list

# Stream logs for a specific app
MODAL_PROFILE=modal-smilelab modal app logs <app-id>
```

Training also logs to **Weights & Biases** (wandb project: `openpi`).

---

## Step 4: Download Checkpoint

```bash
# List checkpoints in volume
MODAL_PROFILE=modal-smilelab modal volume ls main /checkpoints/

# Download checkpoint (replace config_name and exp_name)
MODAL_PROFILE=modal-smilelab modal volume get main \
  /checkpoints/pi0_maniskill/color_spatial_random_f18_pi0/ \
  ./local_ckpt/
```

Checkpoints are saved under:
```
/checkpoints/{config_name}/{exp_name}/
```

---

## One-time Setup: Upload Base Checkpoints

These only need to be done once per Modal volume:

```bash
# π0.5 base (already done)
# MODAL_PROFILE=modal-smilelab modal volume put main \
#   /home/yu/copy/vla/openpi/ckpt/pi05-libero/ /ckpt/pi05-libero/

# π0 base (already done)
# MODAL_PROFILE=modal-smilelab modal volume put main \
#   /home/yu/copy/vla/openpi/ckpt/pi0_libero_base/ /ckpt/pi0-libero-base/

# π0-FAST: downloads from GCS automatically, no upload needed
```

---

## Config Details

All training configs are in:
```
src/openpi/training/config.py
```

Key configs for ManiSkill:
- **`pi05_maniskill`** (line ~885): π0.5, the default
- **`pi0_maniskill`** (line ~908): π0, removes `pi05=True`, uses `pi0_base`
- **`pi0_fast_maniskill`** (line ~931): π0-FAST, `Pi0FASTConfig(action_dim=8, action_horizon=10, max_token_len=180)`

To change steps/intervals, use CLI flags `--num-train-steps` and `--save-interval` — these override the config defaults.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ImportError: libGL.so.1` | Already fixed in image (apt_install libgl1) |
| checkpoint path not found | Make sure base checkpoint is in Modal volume at `/ckpt/...` |
| OOM on H100 | Reduce `batch_size` in config.py or use LoRA variant |
| wandb not logging | Check `WANDB_API_KEY` in train_modal.py |

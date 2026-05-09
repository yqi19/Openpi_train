"""
Modal training script for pi0_maniskill (pi0, not pi0.5).

Usage:
  1. Activate env: conda activate openpi_modal
  2. Set profile:  export MODAL_PROFILE=modal-smilelab
  3. Run:
     modal run train_modal_pi0.py \
       --exp-name <name> \
       --data-volume-path /data/<name> \
       [--num-train-steps 1000] \
       [--save-interval 500] \
       [--nproc 1]

Example (quick test, 1 H100):
  MODAL_PROFILE=modal-smilelab modal run --detach train_modal_pi0.py \
    --exp-name pi0_test_run \
    --data-volume-path /data/color_spatial_random_f18 \
    --num-train-steps 100 \
    --save-interval 50 \
    --nproc 1
"""

import modal
import os

# -------------------------------------------------
# Modal app config
# -------------------------------------------------
app = modal.App("openpi-pi0-finetune")

# smilelab workspace Volume named "main"
volume = modal.Volume.from_name("main", create_if_missing=False)

VOLUME_MOUNT = "/mnt/storage"

# -------------------------------------------------
# Container image
# -------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.02-py3",
        add_python="3.11",
    )
    .apt_install(["git", "git-lfs", "curl", "clang", "libgl1", "libglib2.0-0"])
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    # Clone the Openpi_train repo (same repo, has pi0_maniskill config)
    # Cache bust: 2026-05-09d — fix tied embed_tokens weight for pi0-droid
    .run_commands(
        "echo 'build: 2026-05-09d' && "
        "git clone https://github.com/yqi19/Openpi_train.git /root/openpi_train && "
        "cd /root/openpi_train && git log --oneline -3",
    )
    .run_commands(
        "cd /root/openpi_train && "
        "export UV_LINK_MODE=copy && "
        "GIT_LFS_SKIP_SMUDGE=1 uv sync && "
        "GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . && "
        "uv pip install datasets==3.0.0 && "
        "cp -r ./src/openpi/models_pytorch/transformers_replace/* "
        "  .venv/lib/python3.11/site-packages/transformers/",
    )
    .env({
        "VIRTUAL_ENV": "/root/openpi_train/.venv",
        "PATH": "/root/openpi_train/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    })
)


# -------------------------------------------------
# Training function
# -------------------------------------------------
@app.function(
    image=image,
    gpu="H100:1",             # 1× H100 for testing; bump to H100:8 for full training
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60 * 24,    # 24 h max
    secrets=[],
)
def train(
    exp_name: str,
    data_volume_path: str,
    config_name: str = "pi0_maniskill",
    num_train_steps: int = 14000,
    save_interval: int = 2000,
    nproc: int = 1,
    ckpt_name: str = "pi0-libero-base",
):
    """Run pi0/pi0_droid training inside the Modal container.

    Args:
        ckpt_name: Name of the checkpoint folder in Modal volume /ckpt/.
                   Use 'pi0-libero-base' (default) or 'pi0-droid'.
    """
    import subprocess

    repo_dir = "/root/openpi_train"
    data_path = f"{VOLUME_MOUNT}/{data_volume_path.lstrip('/')}"

    print(f"=== Training config ===")
    print(f"  model_type:      pi0")
    print(f"  train_config:    {config_name}")
    print(f"  exp_name:        {exp_name}")
    print(f"  data_path:       {data_path}")
    print(f"  num_train_steps: {num_train_steps}")
    print(f"  save_interval:   {save_interval}")
    print(f"  nproc:           {nproc}")
    print(f"  checkpoint_dir:  {VOLUME_MOUNT}/checkpoints/{config_name}/{exp_name}")

    # Step 1: compute norm stats
    print(f"\n=== Step 1: compute_norm_stats ({config_name}) ===")
    subprocess.run(
        [
            "uv", "run",
            "scripts/compute_norm_stats.py", config_name,
            "--data.repo_id", data_path,
        ],
        cwd=repo_dir,
        check=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # Symlink checkpoint from volume to expected container path
    ckpt_src = f"{VOLUME_MOUNT}/ckpt/{ckpt_name}"
    ckpt_dst = f"/root/workspace/openpi/ckpt/{ckpt_name}"
    if not os.path.exists(ckpt_dst):
        os.makedirs("/root/workspace/openpi/ckpt", exist_ok=True)
        os.symlink(ckpt_src, ckpt_dst)
        print(f"  Symlinked {ckpt_src} → {ckpt_dst}")
    else:
        print(f"  Checkpoint already at {ckpt_dst}")

    # Step 2: training
    print(f"\n=== Step 2: torchrun training ({config_name}, {nproc} GPU) ===")
    checkpoint_out = f"{VOLUME_MOUNT}/checkpoints"
    log_dir = f"{VOLUME_MOUNT}/logs/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)

    subprocess.run(
        [
            "uv", "run",
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            "scripts/train_pytorch.py", config_name,
            "--exp_name", exp_name,
            "--data.repo_id", data_path,
            "--num_train_steps", str(num_train_steps),
            "--save_interval", str(save_interval),
        ],
        cwd=repo_dir,
        check=True,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "CHECKPOINT_BASE_DIR": checkpoint_out,
            "WANDB_API_KEY": "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",
        },
    )

    # Commit to persist volume writes
    volume.commit()
    print(f"\n✅ Training done! Checkpoint saved to volume: /checkpoints/{config_name}/{exp_name}/")


# -------------------------------------------------
# Local entrypoint
# -------------------------------------------------
@app.local_entrypoint()
def main(
    exp_name: str = "pi0_test_run",
    data_volume_path: str = "/data/color_spatial_random_f18",
    config_name: str = "pi0_maniskill",
    num_train_steps: int = 14000,
    save_interval: int = 2000,
    nproc: int = 1,
    ckpt_name: str = "pi0-libero-base",
):
    """
    Submit pi0/pi0_droid training job to Modal.

    Args:
        exp_name:           Experiment name (used for checkpoint dir)
        data_volume_path:   Path inside Modal Volume where data was uploaded
        config_name:        Training config (default: pi0_maniskill; also: pi0_droid_maniskill)
        num_train_steps:    Total training steps (default 14000)
        save_interval:      Steps between checkpoints (default 2000)
        nproc:              Number of GPUs (default 1; use 8 for full training)
        ckpt_name:          Checkpoint name in Modal volume /ckpt/ (default: pi0-libero-base)
                            Use 'pi0-droid' for DROID-pretrained weights
    """
    print(f"Submitting pi0 training job: {exp_name} ({config_name}, {nproc} GPU, ckpt={ckpt_name})")
    train.remote(
        exp_name=exp_name,
        data_volume_path=data_volume_path,
        config_name=config_name,
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        nproc=nproc,
        ckpt_name=ckpt_name,
    )

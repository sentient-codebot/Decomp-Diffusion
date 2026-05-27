#!/bin/bash
# COCO pipeline smoke test: 2-step training run on a synthetic 16-JPG dataset
# that mimics the COCO `images/train2017/*.jpg` layout.
#
# Purpose: validate that
#   - configs/coco/* parse cleanly,
#   - GlobDataset picks up JPGs (CocoDataset itself is just GlobDataset under
#     the hood, per sony/coda's recipe for the training side),
#   - the CoDA-style frozen SD2.1 UNet + DINOv3 slot encoder loads at the
#     COCO config (K=7, R=4, latent_dim=1024, resolution=256),
#   - train_lsd.py completes 2 optimizer steps without crashing.
#
# Does NOT require the real COCO download -- it fabricates random JPGs in
# scratch. The full training job (jobs/coco_coda_kv_only_train.sh) is what
# consumes the real ~/prjs0993/datasets/coco/ data.
#
# Submit from the repo root: `sbatch jobs/coco_smoketest.sh`.
#SBATCH --job-name="coco-smoketest"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion/.claude/worktrees/feat-coco-dataset

export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
export WANDB_DIR="$HOME/prjs0993/Decomp-Diffusion/wandb"
export WANDB_MODE=offline  # smoke test -- don't pollute wandb

source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

# --- 1. Build a fake COCO-shaped directory ----------------------------------
FAKE_ROOT="$TMPDIR/coco_smoke"
FAKE_TRAIN="$FAKE_ROOT/images/train2017"
mkdir -p "$FAKE_TRAIN"
echo "[coco-smoke] fake dataset root: $FAKE_TRAIN"
uv run python - "$FAKE_TRAIN" <<'PYEOF'
import os, sys
from PIL import Image
import numpy as np

out_dir = sys.argv[1]
rng = np.random.default_rng(0)
n = 32  # comfortably more than 2 * batch_size
for i in range(n):
    # COCO has variable sizes; mimic that so ResizeMinShape-style transforms
    # actually exercise the resize path.
    h = int(rng.integers(200, 480))
    w = int(rng.integers(200, 640))
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out_dir, f"{i:012d}.jpg"), quality=80)
print(f"[coco-smoke] wrote {n} synthetic JPGs to {out_dir}")
PYEOF

# --- 2. Run train_lsd.py for 2 steps ---------------------------------------
RUN_DIR="$TMPDIR/coco_smoke_out"
mkdir -p "$RUN_DIR"
echo "[coco-smoke] starting 2-step training run."

uv run accelerate launch --num_processes=1 --mixed_precision fp16 \
    --main_process_port 29502 train_lsd.py \
    --train_config configs/coco/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/coco/dinov3_slot_encoder_d1024/config.json \
    --unet_config pretrain_sd \
    --freeze_unet_except_kv \
    --scheduler_config configs/coco/scheduler/scheduler_config.json \
    --dataset_root "$FAKE_TRAIN" \
    --dataset_glob '*.jpg' \
    --resolution 256 \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --num_validation_images 2 \
    --validation_steps 999999 \
    --checkpointing_steps 999999 \
    --max_train_steps 2 \
    --report_to tensorboard
RC=$?

echo "**************** [coco-smoke] training finished (rc=$RC). ****************"

# --- 3. Verdict ------------------------------------------------------------
if [ $RC -eq 0 ]; then
    echo "[coco-smoke] PASS"
else
    echo "[coco-smoke] FAIL (rc=$RC) -- see log"
fi
exit $RC

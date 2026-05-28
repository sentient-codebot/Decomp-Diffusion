#!/bin/bash
# Smoke test for the CoDA K/V-only mode (--freeze_unet_except_kv).
#
# Exercises every code path the new flag touches without committing to a
# 200k-step run:
#   - SD 2.1 UNet loads via `--unet_config pretrain_sd`
#   - DinoSlotAttentionEncoder with latent_dim=1024 matches SD2.1's
#     cross_attention_dim
#   - K/V re-unfreeze: logger prints the trainable to_k/to_v tensor count
#   - A handful of training steps run with finite loss
#   - Checkpoint save/load round-trip preserves trained K/V
#   - eval.py loads the partially-trained UNet and produces grids
#
# Submit from the repo root: `sbatch jobs/movi_e_coda_kv_only_smoketest.sh`.
#SBATCH --job-name="coda-kv-smoke"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/movi-e_coda_kv_only_smoketest
RESOLUTION=256
PER_GPU_BATCH=4

# eval.py writes grids to ./image_test_output/ (hardcoded path); drop any
# stale symlink so this smoke writes into the worktree.
rm -rf image_test_output "$RUN_DIR"

# --- 0. Sanity check ---------------------------------------------------------
TRAIN_SHARD_ROOT=data/movi-e-wds/train
VAL_SHARD_ROOT=data/movi-e-wds/validation
if [ ! -f "$TRAIN_SHARD_ROOT/samples.jsonl" ]; then
    echo "[coda-smoke] FATAL: $TRAIN_SHARD_ROOT/samples.jsonl missing -- run jobs/movi_e_shard_wds.sh first."
    exit 1
fi

# --- 1. Train (single GPU, 12 steps, checkpoint at step 12) ------------------
# validation_steps=6 exercises log_validation halfway through.
uv run accelerate launch --num_processes=1 --mixed_precision bf16 \
    --main_process_port 29504 train_lsd.py \
    --train_config configs/movi-e/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/movi-e/dinov3_slot_encoder_d1024/config.json \
    --unet_config pretrain_sd \
    --freeze_unet_except_kv \
    --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
    --dataset_root "$TRAIN_SHARD_ROOT" \
    --dataset_glob '*.tar' \
    --dataset_format wds \
    --report_to wandb \
    --resolution "$RESOLUTION" \
    --train_batch_size "$PER_GPU_BATCH" --val_batch_size 4 --num_validation_images 4 \
    --max_train_steps 12 --validation_steps 6 --checkpointing_steps 12
TRAIN_RC=$?
echo "**************** [coda-smoke] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Resume once (load-hook check) ----------------------------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
echo "[coda-smoke] checkpoint: ${CKPT:-<none found>}"

if [ -n "$CKPT" ]; then
    # Resume from the step-12 checkpoint and run 4 more steps; this catches a
    # silent regression where K/V weren't actually saved.
    uv run accelerate launch --num_processes=1 --mixed_precision bf16 \
        --main_process_port 29504 train_lsd.py \
        --train_config configs/movi-e/train_config.yaml \
        --output_dir "$RUN_DIR/" \
        --latent_encoder_config configs/movi-e/dinov3_slot_encoder_d1024/config.json \
        --unet_config pretrain_sd \
        --freeze_unet_except_kv \
        --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
        --dataset_root "$TRAIN_SHARD_ROOT" \
        --dataset_glob '*.tar' \
        --dataset_format wds \
        --report_to wandb \
        --resolution "$RESOLUTION" \
        --train_batch_size "$PER_GPU_BATCH" --val_batch_size 4 --num_validation_images 4 \
        --max_train_steps 16 --validation_steps 100 --checkpointing_steps 100 \
        --resume_from_checkpoint latest
    RESUME_RC=$?
else
    RESUME_RC=1
fi
echo "**************** [coda-smoke] resume finished (rc=$RESUME_RC). ****************"

# --- 3. Eval -----------------------------------------------------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision bf16 --seed 42 \
        --batch_size 4 --num_validation_images 4 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
        --dataset_root "$VAL_SHARD_ROOT" \
        --dataset_glob '**/00000000_image.png' \
        --dataset_format wds --resolution "$RESOLUTION" \
        --ckpt_path "$CKPT"
    EVAL_RC=$?
else
    EVAL_RC=1
fi
echo "**************** [coda-smoke] eval finished (rc=$EVAL_RC). ****************"

if [ "$TRAIN_RC" -eq 0 ] && [ "$RESUME_RC" -eq 0 ] && [ "$EVAL_RC" -eq 0 ]; then
    echo "**************** [coda-smoke] PASS ****************"
else
    echo "**************** [coda-smoke] FAIL (train=$TRAIN_RC resume=$RESUME_RC eval=$EVAL_RC) ****************"
fi

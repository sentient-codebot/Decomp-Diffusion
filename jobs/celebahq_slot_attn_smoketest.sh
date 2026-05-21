#!/bin/bash
# Smoke test for the SlotAttentionEncoder training/eval pipeline.
#
# Runs a handful of optimization steps with frequent validation +
# checkpointing on one GPU, then evals the checkpoint, exercising every code
# path that the slot-attention encoder touches: model build via the config
# factory, train step, per-slot validation viz, checkpoint save/load hooks,
# and eval-time encoder loading. NOT a real training run.
#
# Submit from the repo root: `sbatch jobs/celebahq_slot_attn_smoketest.sh`.
#SBATCH --job-name="slot-smoke"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion-slot-encoder

# uv-managed env; keep the HF cache off the home quota (see download_data.sh)
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/celebahq_slot_smoketest
# eval.py writes grids to ./image_test_output/ (hardcoded); drop any stale
# symlink so the smoke test writes into this worktree.
rm -rf image_test_output "$RUN_DIR"

# --- Train (single GPU, 12 steps) --------------------------------------------
uv run accelerate launch --num_processes=1 --mixed_precision fp16 \
    --main_process_port 29503 train_lsd.py \
    --train_config configs/celebahq/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/celebahq/slot_encoder/config.json \
    --unet_config configs/celebahq/unet/config.json \
    --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
    --dataset_root data/celebahq_data128x128/ \
    --dataset_glob '**/*.jpg' \
    --report_to wandb \
    --train_batch_size 8 --val_batch_size 8 --num_validation_images 8 \
    --max_train_steps 12 --validation_steps 6 --checkpointing_steps 12
TRAIN_RC=$?
echo "**************** [slot-smoke] training finished (rc=$TRAIN_RC). ****************"

# --- Locate checkpoint + resume once (load-hook check) -----------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
echo "[slot-smoke] checkpoint: ${CKPT:-<none found>}"

# --- Eval (single GPU) -------------------------------------------------------
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision fp16 --seed 42 \
        --batch_size 8 --num_validation_images 8 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
        --dataset_root data/celebahq_data128x128/ \
        --dataset_glob '00000*.jpg' --resolution 128 \
        --ckpt_path "$CKPT"
    EVAL_RC=$?
else
    EVAL_RC=1
fi
echo "**************** [slot-smoke] eval finished (rc=$EVAL_RC). ****************"

if [ "$TRAIN_RC" -eq 0 ] && [ "$EVAL_RC" -eq 0 ]; then
    echo "**************** [slot-smoke] PASS ****************"
else
    echo "**************** [slot-smoke] FAIL (train=$TRAIN_RC eval=$EVAL_RC) ****************"
fi

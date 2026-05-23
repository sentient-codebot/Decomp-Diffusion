#!/bin/bash
# Eval-only run against an existing slot-attention checkpoint.
#
# Companion to jobs/celebahq_slot_attn_train_eval.sh. The full training job
# (23006944) ran ~200k of 500k steps before crashing on a home-quota
# OSError; the 200k checkpoint survived and is what this job evals to get
# image-grid output for the experiment report.
#
# Submit from the repo root: `sbatch jobs/celebahq_slot_attn_eval.sh`.
#SBATCH --job-name="slot-eval"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion-slot-encoder/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion-slot-encoder

# Redirect heavy outputs to project storage; see jobs/celebahq_slot_attn_train_eval.sh.
PRJS_DIR="$HOME/prjs0993/Decomp-Diffusion-slot-encoder"
mkdir -p "$PRJS_DIR/results" "$PRJS_DIR/wandb" "$PRJS_DIR/slurm_logs"
[ -e results ] || ln -s "$PRJS_DIR/results" results
[ -e wandb ]   || ln -s "$PRJS_DIR/wandb"   wandb
export WANDB_DIR="$PRJS_DIR/wandb"
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"

source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

# results/ is a symlink to ~/prjs0993/... -- everything below resolves through it.
RUN_DIR=results/celebahq_slot
CKPT=$(find "$RUN_DIR" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null \
    | sort -t- -k2 -n | tail -n1)
echo "[slot-eval] using checkpoint: ${CKPT:-<none found>}"

# eval.py writes grids to ./image_test_output/ (hardcoded); drop any stale
# symlink so it writes to a fresh local dir.
rm -rf image_test_output

uv run accelerate launch --num_processes=1 eval.py \
    --mixed_precision fp16 --seed 42 \
    --batch_size 32 --num_validation_images 32 \
    --output_dir "$RUN_DIR/gen_images" \
    --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
    --dataset_root data/celebahq_data128x128/ \
    --dataset_glob '000*.jpg' --resolution 128 \
    --ckpt_path "$CKPT"
EVAL_RC=$?
echo "**************** [slot-eval] eval finished (rc=$EVAL_RC). ****************"

# Keep the eval grids next to the checkpoint (image_test_output is transient).
if [ -d image_test_output ]; then
    mkdir -p "$RUN_DIR/eval_grids"
    cp image_test_output/*.jpg "$RUN_DIR/eval_grids/" 2>/dev/null
    echo "[slot-eval] grids copied to $RUN_DIR/eval_grids/"
fi

#!/bin/bash
# Smoke-test the CelebA-HQ training pipeline end to end on one GPU.
#
# Runs a handful of optimization steps with frequent validation +
# checkpointing so every code path (model build, data loading, train step,
# per-slot validation viz, checkpoint save) is exercised quickly. It is NOT a
# real training run -- output goes to results/celebahq_smoketest/.
#
# Submit from the repo root: `sbatch scripts/celebahq/smoketest.sh`.
#SBATCH --job-name="celebahq-smoke"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

# uv-managed env; keep the HF cache off the home quota (see download_data.sh)
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

uv run accelerate launch --num_processes=1 --mixed_precision bf16 \
    --main_process_port 29501 train_lsd.py \
    --train_config configs/celebahq/train_config.yaml \
    --output_dir results/celebahq_smoketest/ \
    --latent_encoder_config configs/celebahq/latent_encoder/config.json \
    --unet_config configs/celebahq/unet/config.json \
    --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
    --dataset_root data/celebahq_data128x128/ \
    --dataset_glob '**/*.jpg' \
    --report_to tensorboard \
    --train_batch_size 8 --val_batch_size 8 --num_validation_images 8 \
    --max_train_steps 12 --validation_steps 6 --checkpointing_steps 12

echo "**************** celebahq smoke test completed. ****************"

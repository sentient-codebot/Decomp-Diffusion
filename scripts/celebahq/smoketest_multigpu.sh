#!/bin/bash
# Multi-GPU (2x DDP) smoke test for the CelebA-HQ training pipeline.
#
# Same short run as smoketest.sh but with --multi_gpu --num_processes=2, to
# verify the production 2-GPU launch path (scripts/celebahq/train.sh) works.
# Output goes to results/celebahq_smoketest_mg/.
#
# Submit from the repo root: `sbatch scripts/celebahq/smoketest_multigpu.sh`.
#SBATCH --job-name="celebahq-smoke-mg"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
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

uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision bf16 \
    --main_process_port 29502 train_lsd.py \
    --train_config configs/celebahq/train_config.yaml \
    --output_dir results/celebahq_smoketest_mg/ \
    --latent_encoder_config configs/celebahq/latent_encoder/config.json \
    --unet_config configs/celebahq/unet/config.json \
    --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
    --dataset_root data/celebahq_data128x128/ \
    --dataset_glob '**/*.jpg' \
    --report_to tensorboard \
    --train_batch_size 8 --val_batch_size 8 --num_validation_images 8 \
    --max_train_steps 12 --validation_steps 6 --checkpointing_steps 12

echo "**************** celebahq multi-gpu smoke test completed. ****************"

#!/bin/bash
# Smoke-test the CelebA-HQ eval pipeline (eval.py) on one GPU.
#
# Runs eval against the checkpoint produced by smoketest_multigpu.sh, on a
# 10-image glob so it finishes fast. Verifies checkpoint loading + per-slot
# generation; it is NOT a real evaluation. Images land in ./image_test_output/.
#
# Submit from the repo root: `sbatch scripts/celebahq/eval_smoketest.sh`.
#SBATCH --job-name="celebahq-eval-smoke"
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

uv run accelerate launch --num_processes=1 eval.py \
    --mixed_precision fp16 --seed 42 \
    --batch_size 8 --num_validation_images 8 \
    --output_dir results/celebahq_smoketest/gen_images \
    --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
    --dataset_root data/celebahq_data128x128/ \
    --dataset_glob '0000*.jpg' --resolution 128 \
    --ckpt_path results/celebahq_smoketest/latent_decomposed_diffusion/checkpoint-12-last

echo "**************** celebahq eval smoke test completed. ****************"

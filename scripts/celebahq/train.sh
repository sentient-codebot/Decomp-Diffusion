# Train the CelebA-HQ latent decomposed diffusion (ldd) model.
#
# Run from the repository root: `bash scripts/celebahq/train.sh`.
# Hyperparameters live in configs/celebahq/train_config.yaml; this script only
# carries run/machine-specific paths. Override any hyperparameter by appending
# its flag, e.g. `bash scripts/celebahq/train.sh --learning_rate 1e-5`.
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --multi_gpu\
    --num_processes=2 --main_process_port 29500 train_lsd.py \
    --train_config configs/celebahq/train_config.yaml \
    --output_dir results/celebahq/ \
    --latent_encoder_config configs/celebahq/latent_encoder/config.json \
    --unet_config configs/celebahq/unet/config.json \
    --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
    --dataset_root data/celebahq_data128x128/ \
    --dataset_glob '**/*.jpg' "$@"

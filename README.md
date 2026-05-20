# 1. create the environment

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
# install uv if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# create the venv and install all dependencies (incl. CUDA 12.4 torch on Linux)
uv sync --extra wandb --extra tensorboard --extra xformers
```

Available optional extras:
- `wandb` — Weights & Biases logging
- `tensorboard` — TensorBoard logging
- `xformers` — memory-efficient attention (only used with `--enable_xformers_memory_efficient_attention`)
- `preprocess` — TensorFlow / TFDS, only needed for `scripts/data_preprocess/movi_kubric_dump_with_labels.py`

Prefix any command with `uv run` to execute it inside the project venv (e.g. `uv run python ...`, `uv run accelerate launch ...`).

# 2. train the model

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --multi_gpu --num_processes=2 --main_process_port 29500 train_lsd.py \
--enable_xformers_memory_efficient_attention --dataloader_num_workers 4 --learning_rate 2e-5 --mixed_precision fp16 \
--num_validation_images 32 --val_batch_size 32 --max_train_steps 500000 --checkpointing_steps 25000 --checkpoints_total_limit 2 \
--gradient_accumulation_steps 1 --seed 42 --encoder_lr_scale 1.0 --train_split_portion 0.9 \
--output_dir ~/Projects/latent-decomposed-diffusion/lsd/celebahq/ --backbone_config configs/celebahq/backbone/config.json \
--latent_encoder_config configs/celebahq/latent_encoder/config.json --unet_config configs/celebahq/unet/config.json \
--scheduler_config configs/celebahq/scheduler/scheduler_config.json --dataset_root /space/ywang86/celebahq_data128x128/ \
--dataset_glob '**/*.jpg' --train_batch_size 32 --resolution 128 --validation_steps 5000 --tracker_project_name latent_decomposed_diffusion
```

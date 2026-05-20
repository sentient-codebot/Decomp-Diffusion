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

Training hyperparameters live in a YAML config (`configs/celebahq/train_config.yaml`);
the launch command itself only carries run/machine-specific paths. The provided
`scripts/celebahq/train.sh` wraps the full command (run it from the repo root):

```bash
bash scripts/celebahq/train.sh
```

Equivalently, spelled out:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --multi_gpu --num_processes=2 --main_process_port 29500 train_lsd.py \
--train_config configs/celebahq/train_config.yaml \
--output_dir results/celebahq/ \
--backbone_config configs/celebahq/backbone/config.json \
--latent_encoder_config configs/celebahq/latent_encoder/config.json \
--unet_config configs/celebahq/unet/config.json \
--scheduler_config configs/celebahq/scheduler/scheduler_config.json \
--dataset_root /space/ywang86/celebahq_data128x128/ --dataset_glob '**/*.jpg'
```

Any hyperparameter in the YAML can be overridden on the command line, e.g.
appending `--learning_rate 1e-5` (the flag wins over the config value).

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

# 2. prepare the dataset

`scripts/celebahq/train.sh` trains on **CelebA-HQ** at 128×128. It expects
`.jpg` images under `data/celebahq_data128x128/` (matched recursively by the
`**/*.jpg` glob); `data/` is typically a symlink to wherever the dataset lives.

- Only the image files are needed — `GlobDataset` reads pixels only, no labels
  or masks.
- Images need not be pre-resized: the dataset transform resizes the shortest
  side to `resolution` (128) and center-crops. Pre-resizing only saves I/O.
- There is no separate validation set to prepare. `train_split_portion: 0.9`
  in `train_config.yaml` splits the single glob — first 90% (shuffled with a
  fixed seed) for training, last 10% for validation.

The standard CelebA-HQ release is 30,000 images at 1024×1024; download it and
point `--dataset_root` / `--dataset_glob` at the images (downscaling to 128 is
optional). To train on a different dataset, override those two flags.

# 3. train the model

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
--latent_encoder_config configs/celebahq/latent_encoder/config.json \
--unet_config configs/celebahq/unet/config.json \
--scheduler_config configs/celebahq/scheduler/scheduler_config.json \
--dataset_root data/celebahq_data128x128/ --dataset_glob '**/*.jpg'
```

Any hyperparameter in the YAML can be overridden on the command line, e.g.
appending `--learning_rate 1e-5` (the flag wins over the config value).

# 4. evaluate

Three offline metric scripts exist; all read a checkpoint dir saved by
`train_lsd.py`. MOVi-E is the only dataset wired up for object-discovery and
property-probing today; VOC / COCO are roadmap items.

```bash
# Task 1: object discovery -- FG-ARI, mBO, mIoU (Hungarian-matched) + viz grids.
bash scripts/movi-e/eval.sh <ckpt_path> <output_dir>
# Task 2: property-prediction probe -- position (MSE), 3D bbox (MSE),
# category (accuracy). Hungarian matches slot masks to GT, trains a frozen-slot
# MLP, reports test-split numbers.
bash scripts/movi-e/probe.sh <ckpt_path> <output_dir>
```

Training-time validation can also stream the segmentation metrics to wandb if
you pass `--movi_eval_root <root>` to `train_lsd.py` (off by default, so
CelebA-HQ runs are unaffected). The MOVi pair dataset is capped to
`--movi_eval_max_images` (default 256) for a quick read each `validation_steps`.

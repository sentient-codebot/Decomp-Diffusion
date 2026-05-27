# Decomp-Diffusion

Latent Slot Diffusion with decomposed representations: trains a UNet diffusion
model conditioned on slot-based latent encodings of input images.

## Environment

Managed by `uv`.

- Sync: `uv sync --extra wandb --extra tensorboard --extra xformers`
- Run: prefix commands with `uv run` (for example, `uv run accelerate launch ...`)
- The lock is multi-platform; on Linux, torch resolves to the CUDA 12.6 wheel
  via the `pytorch-cu126` index declared in `pyproject.toml`. cu126 wheels
  need NVIDIA driver R555+, which the Snellius H100 partitions have.

Optional extras:

- `wandb`, `tensorboard`: logging backends
- `xformers`: only needed with `--enable_xformers_memory_efficient_attention`
- `preprocess`: `tensorflow` / `tfds`, used only by
  `scripts/data_preprocess/movi_kubric_dump_with_labels.py`

## Layout

- `train_lsd.py`: training entrypoint, launched via `accelerate launch`
- `scripts/celebahq/train.sh`: training launch wrapper with run/machine paths
- `eval.py` / `eval.sh`: evaluation entrypoint
- `src/models/`: latent encoder, UNet variants, ColorMask viz utilities
- `src/pipeline/composable_stable_diffusion_pipeline.py`: custom diffusion pipeline
- `src/data/dataset.py`: `GlobDataset`
- `src/parser.py`: shared argparse for train and eval
- `configs/celebahq/`: celebahq component jsons plus `train_config.yaml`
- `scripts/environment.sh`: uv bootstrap
- `scripts/data_preprocess/`: MOVi/Kubric preprocessing
- `jobs/`: Slurm sbatch job scripts
- `docs/`: experiment reports

## Project Conventions

- Commits: follow `CONTRIBUTING.md`: `<type>(<scope>): <description>`,
  lowercase, imperative, no trailing period, under 60 chars. Types are `feat`,
  `fix`, `exp`, `refactor`, `build`, `chore`, `perf`, `docs`, `test`.
- Do not add `Co-Authored-By` trailers or any co-author sign-off to commits.
- Lint/format: `ruff.toml` configures ruff. Run `uvx ruff check` and
  `uvx ruff format`; ruff is not a project dependency.
- Launch or wrapper shell scripts belong under `scripts/<experiment>/`, not the
  repo root or `configs/`. Scripts are run from the repo root, so relative paths
  like `configs/...` and `train_lsd.py` remain valid.

## Project Tracking

- `ROADMAP.md`: longer-term direction, bigger changes, planned architecture work.
  Keep it current when project goals change. When an experiment addresses a
  roadmap point, note the run location, Slurm job id or job script, wandb/output
  path, and the result report under `docs/`.
- `TECHDEBT.md`: known shortcuts, divergences, and deferred fixes. Add entries
  when discovering or introducing debt; remove them when resolved.
- Experiment reports go under `docs/experiments/` as
  `YYYY-MM-DD-<dataset>-<topic>.md`.

## Storage And Runtime

- This project runs on Snellius HPC through Slurm `sbatch`.
- Job setup usually starts with `module load 2025`, `cd ~/projects/Decomp-Diffusion`,
  `source .venv/bin/activate`, then `uv sync`.
- Slurm notifications use `--mail-user=n.lin@tudelft.nl`.
- Partitions: `gpu_a100` for single-GPU and eval/sampling, `gpu_h100` for
  multi-GPU/multi-node DDP, and `genoa` for CPU-only jobs. There is no generic
  `gpu` partition.
- Launch training with `accelerate launch train_lsd.py`. For multi-GPU training
  launched directly, do not wrap the launcher in `srun`; use `srun` only for
  single-process sampling/eval steps.
- Datasets live under `~/prjs0993/datasets/{dataset_name}`. The repo root uses a
  gitignored `data` symlink pointing to `~/prjs0993/datasets`, so code should
  refer to `data/{dataset_name}/`.
- Keep heavy artifacts off the home filesystem. Use
  `~/prjs0993/<project>/results`, `~/prjs0993/<project>/wandb`, and
  `~/prjs0993/<project>/slurm_logs`; symlink `results` and `wandb` from the repo
  root when needed.
- Set `WANDB_DIR="$HOME/prjs0993/<project>/wandb"` and
  `HF_HOME="$HOME/prjs0993/<project>/cache/huggingface"` in job scripts.
- Jobs that use `torch.compile` / `--dynamo_backend=inductor` should also
  export `TORCHINDUCTOR_CACHE_DIR="$HOME/prjs0993/tmp/torchinductor"` (shared
  across projects). It is set in `~/.bashrc`, but sbatch does not always
  re-source bashrc; exporting in the script keeps the Inductor compile cache
  warm across runs (one-time ~170s compile becomes ~0 on cache hit; keyed by
  FX graph + shapes + dtypes + torch/triton version, so weight values and
  unrelated projects do not invalidate it).
- Slurm output paths must be literal, for example
  `#SBATCH --output=/home/nlin/prjs0993/<project>/slurm_logs/slurm_%j.log`.

## Experiment Defaults

- Prefer wandb over tensorboard for future training/eval scripts:
  pass `--report_to wandb`.
- The wandb entity is `nan-team`; `tracker_project_name` becomes the wandb
  project, for example `wandb.ai/nan-team/latent_decomposed_diffusion`.
- For MOVi-E training runs, use 200k steps by default unless requested otherwise.
- `src/parser.py` appends `tracker_project_name` to `output_dir`, so checkpoints
  land under `<output_dir>/<tracker_project_name>/checkpoint-*`, not directly
  under `--output_dir`.

## Encoder Notes

- `src/models/encoder.py` exposes interchangeable encoders through
  `ENCODER_REGISTRY`, selected by `_class_name` in the latent-encoder config.
- `LatentEncoder` is the original CNN + flatten + linear baseline. It has no
  slot attention and is intentionally retained as an ablation baseline, not tech
  debt.
- `SlotAttentionEncoder` uses the CNN front end plus `SoftPositionEmbed`, an
  MLP, and iterative Slot Attention. `return_attn=True` can return
  `[B, K, h, w]` soft segmentation at feature-map resolution.
- `DinoSlotAttentionEncoder` uses a pretrained DINO ViT backbone via
  `transformers.AutoModel` plus Slot Attention. The backbone is frozen by
  default, supports DINO v1 and DINOv3, and renormalizes dataset pixels to
  ImageNet stats inside `forward`.
- All encoders return `[B, num_components, latent_dim]`. `latent_dim` refers to
  the slot latent, not the SD-VAE latent; these encoders operate on raw pixels.
- Add new encoders to the registry instead of replacing existing classes so old
  checkpoints remain loadable.

## Known Divergences

- `configs/celebahq/unet/config.json` declares
  `_class_name: "UNet2DConditionModelWithPos"` and `pos_type: "cartesian"`, but
  `train_lsd.py` and `eval.py` currently construct plain
  `diffusers.UNet2DConditionModel` via `from_config`. `pos_type` is silently
  ignored and `src/models/unet_with_pos.py` is not imported. If the positional
  embedding is intended, instantiate `UNet2DConditionModelWithPos` in training
  and eval, then verify checkpoint save/load subfolder naming.

## Reproducibility Notes

Dependency caps in `pyproject.toml` are deliberately conservative:

- `diffusers <0.32`
- `transformers <5.0`
- `numpy <2.0`
- `torch >=2.7,<2.8` with CUDA 12.6 (bumped from 2.6+cu124 for triton 3.3 +
  `torch.compile` / inductor support on Python 3.12)

If these are bumped, rerun training to confirm metrics before trusting results.

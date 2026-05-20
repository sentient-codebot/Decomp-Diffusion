# Decomp-Diffusion

Latent Slot Diffusion with decomposed representations: trains a UNet diffusion model conditioned on slot-based latent encodings of input images.

## Environment

Managed by [uv](https://github.com/astral-sh/uv).

- Sync: `uv sync --extra wandb --extra tensorboard --extra xformers`
- Run: prefix commands with `uv run` (e.g. `uv run accelerate launch ...`)
- The lock is multi-platform; on Linux, torch resolves to the CUDA 12.4 wheel via the `pytorch-cu124` index declared in `pyproject.toml`.

Optional extras:
- `wandb`, `tensorboard` — logging backends
- `xformers` — only needed with `--enable_xformers_memory_efficient_attention`
- `preprocess` — `tensorflow` / `tfds`, used solely by `scripts/data_preprocess/movi_kubric_dump_with_labels.py`

## Layout

- `train_lsd.py` — training entrypoint, launched via `accelerate launch`
- `scripts/celebahq/train.sh` — training launch wrapper (run/machine-specific paths only)
- `eval.py` / `eval.sh` — evaluation entrypoint
- `src/models/` — latent encoder, UNet variants, ColorMask viz utils
- `src/pipeline/composable_stable_diffusion_pipeline.py` — custom diffusion pipeline
- `src/data/dataset.py` — `GlobDataset`
- `src/parser.py` — shared argparse for train + eval
- `configs/celebahq/` — celebahq model component jsons (backbone, encoder, unet, scheduler) + `train_config.yaml` training hyperparameters
- `scripts/environment.sh` — uv bootstrap
- `scripts/data_preprocess/` — MOVi/Kubric preprocessing (requires `preprocess` extra)
- `jobs/` — Slurm sbatch job scripts
- `docs/` — experiment result reports (markdown)

## Conventions

- **Commits:** see `CONTRIBUTING.md` — `<type>(<scope>): <description>`, lowercase, imperative, no trailing period, <60 chars. Types: `feat`, `fix`, `exp`, `refactor`, `build`, `chore`, `perf`, `docs`, `test`.
- **Lint/format:** `ruff.toml` configures ruff. Run via `uvx ruff check` and `uvx ruff format` (ruff is not a project dep).

## Project tracking

- `ROADMAP.md` — longer-term direction: bigger changes, methodological shifts, planned architecture work. Keep it current when project goals change. When an experiment is launched that addresses a roadmap point (partially or fully), note it under that point: how to find the run (e.g. Slurm job id, job script under `jobs/`, wandb/output path) and the result report file under `docs/` once written.
- `TECHDEBT.md` — known shortcuts, divergences, and deferred fixes. Add an entry when you discover or introduce debt; remove it when resolved.

## Jobs and experiment reports

- **Slurm jobs:** write sbatch job scripts under `jobs/`. (Distinct from `scripts/<experiment>/` launch wrappers, which hold only run/machine-specific paths.)
- **Experiment results:** write up results as markdown reports under `docs/`. Name reports for long-term maintenance, not one-offs: `docs/experiments/YYYY-MM-DD-<dataset>-<topic>.md` (date prefix sorts chronologically; `<dataset>` and `<topic>` keep them scannable, e.g. `2026-05-20-celebahq-slot-count-sweep.md`).

## Reproducibility notes

Dependency caps in `pyproject.toml` are deliberately conservative to keep experimental results close to the original conda env:
- `diffusers <0.32` (original was 0.25.1; the pipeline file requires ≥0.27 because of `FusedAttnProcessor2_0`)
- `transformers <5.0` — stays on the v4 API the code was written against
- `numpy <2.0`
- `torch >=2.5,<2.7` with CUDA 12.4 (upgraded from 2.0.1+cu118)

If you bump any of these, re-run training to confirm metrics before trusting results.

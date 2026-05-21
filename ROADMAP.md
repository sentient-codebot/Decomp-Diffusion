# Roadmap

Longer-term direction: bigger changes, methodological shifts, planned architecture work.
For known shortcuts and deferred fixes, see `TECHDEBT.md`.


## Setup validation

**Goal:** confirm the current end-to-end setup actually works before building on it.

- Run a **full training run** (not just the celebahq smoke test) with the current `LatentEncoder` baseline.
- Run **evaluation** on the resulting checkpoint and inspect the metrics and decomposition visualizations.
- Decide from the results whether the baseline is sound enough to build on, or whether anything in `TECHDEBT.md` needs fixing first.

This validation is the prerequisite for the slot-extraction work below — it establishes the baseline that the encoder + slot-attention version is compared against.

**Runs:**
- 2026-05-20 — reduced 50k-step end-to-end validation (pipeline check, not yet the full 500k run). Slurm jobs 22966115 (train, 2x H100) + 22974706 (eval); scripts `jobs/celebahq_train_eval_validation.sh` + `jobs/celebahq_eval_validation.sh`; checkpoint at `results/celebahq_validation/latent_decomposed_diffusion/checkpoint-50000-last`. Result: train + eval pipelines run cleanly — report `docs/experiments/2026-05-20-celebahq-train-eval-validation.md`. Still open: the full 500k-step run and the baseline-soundness decision.
- 2026-05-20 — full 500k-step run launched. Slurm job 22976095 (train+eval, 2x H100); script `jobs/celebahq_train_eval_full.sh`; output `results/celebahq/latent_decomposed_diffusion/`; report (on completion) `docs/experiments/2026-05-20-celebahq-full-500k-run.md`. ~25h compute. This is the run the soundness decision will be made from.

## Slot extraction: slot attention

**Goal:** move slot extraction to a proper **slot attention module**. This is
the canonical Latent Slot Diffusion pattern: encoder → feature map → slot
attention → slots.

**Done:** `SlotAttentionEncoder` (`src/models/encoder.py`, `src/models/slot_attn.py`)
keeps the convolutional feature extractor of `LatentEncoder` (image → feature
map) but replaces its `Flatten` + `Linear` slot read-out with a soft positional
embedding + an iterative Slot Attention module (Locatello et al., 2020). It is
added as a *selectable alternative*: the encoder class is chosen by the
`_class_name` field of the latent-encoder config json (`build_latent_encoder`
in `src/models/encoder.py`), so `LatentEncoder` stays available as the naive
baseline. Config: `configs/celebahq/slot_encoder/config.json`.

The earlier `UNetEncoder` backbone (removed in 178578b, recoverable from git
history) was scaffolding toward a swappable encoder but had no slot-attention
downstream — this design supersedes it.

**Runs:**
- 2026-05-21 — full 500k-step run, `SlotAttentionEncoder`. Slurm job 23006944
  (train+eval, 4x H100, wandb logging); script `jobs/celebahq_slot_attn_train_eval.sh`; output
  `results/celebahq_slot/latent_decomposed_diffusion/`; report
  `docs/experiments/2026-05-21-celebahq-slot-attention-encoder.md`. Effective
  batch, step count, LR, UNet and scheduler match the LatentEncoder baseline run
  (results/celebahq) so only the encoder differs.

## Encoder: pretrained feature extractor

**Goal:** replace the trained-from-scratch convolutional feature extractor
(image → feature map) in `SlotAttentionEncoder` with a **pretrained encoder,
e.g. DINO**. Slot Attention then binds the pretrained feature map into slots.
A frozen (or lightly fine-tuned) self-supervised backbone gives much stronger
per-patch features than a CNN trained only through the diffusion loss, which
is expected to improve decomposition quality and convergence speed.

**How to land it:** add the pretrained backbone as another selectable encoder
(same `_class_name`-dispatched factory), keeping the CNN-based
`SlotAttentionEncoder` as the comparison point. Only the feature-extractor
front end changes; the soft positional embedding + Slot Attention read-out
stay as they are.

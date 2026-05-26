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
- 2026-05-21 — 500k-step run, interrupted at ~215k steps by a home-quota
  OSError. Slurm jobs 23006944 (train, 4x H100, 10:33) + 23069034 (eval,
  1x A100, 03:00); scripts `jobs/celebahq_slot_attn_train_eval.sh` +
  `jobs/celebahq_slot_attn_eval.sh`; eval against
  `results/celebahq_slot/latent_decomposed_diffusion/checkpoint-200000`;
  report `docs/experiments/2026-05-21-celebahq-slot-attention-encoder.md`.
  Effective batch, step count, LR, UNet and scheduler match the
  LatentEncoder baseline (results/celebahq) so only the encoder differs.
  Result: diffusion loss matches the baseline at every shared step (within
  ~3%), but the four slots collapse onto the same global representation —
  slot attention alone is not sufficient to induce decomposition under the
  diffusion-loss objective. See the report for the path forward.
  Quota crash fixed by redirecting results/wandb/slurm-log to project
  storage (commits `ee295ed` on `feat/slot-attention-encoder` and
  `e02d149` on `main`).

## Dataset: MOVi-e

**Goal:** move past the single-image CelebA-HQ baseline and validate the
slot-attention pipeline on a **multi-object synthetic benchmark with ground-truth
segmentation** (Kubric MOVi-E). MOVi-E has up to 23 rigid objects per scene with
per-pixel instance masks, so the decomposition can be measured with real
object-centric metrics (FG-ARI, mBO) instead of only qualitative grids. This
makes it the first run where "did the encoder actually find objects?" is
answerable from numbers.

**How to land it:**
- Preprocess MOVi-E once into the same flat-PNG layout `GlobDataset` already
  consumes (`movi-e-{train,validation,test}-with-label/images/<vid>/<frame>_image.png`,
  segments + JSON labels alongside under `labels/`). The existing
  `scripts/data_preprocess/movi_kubric_dump_with_labels.py` already produces
  this layout; `data/movi-e/` will link into `~/prjs0993/datasets/movi-e/`.
- Add `configs/movi-e/` mirroring `configs/celebahq/` (latent_encoder,
  slot_encoder, unet, scheduler, train_config), with `num_components` raised
  to match the higher object count (11 slots) and `train_config.yaml` set to
  the 200k-step / 4x H100 budget.
- Default encoder is `SlotAttentionEncoder` — the object-centric encoder from
  the roadmap section above — since the whole point of moving to MOVi-E is to
  test slot binding. `LatentEncoder` stays as a comparable baseline if needed.
- Extend evaluation with object-centric metrics (FG-ARI, mBO) that compare
  per-slot attention masks against the dumped GT segment PNGs, in addition to
  the existing per-slot reconstruction grids.

**Runs:**
- 2026-05-23 — MOVi-E 200k-step run launched, `SlotAttentionEncoder` (11 slots,
  128 resolution). Slurm job: see `jobs/movi_e_slot_attn_train_eval.sh`
  (train+eval, 4x H100, wandb logging); output `results/movi-e_slot/`;
  report (on completion) `docs/experiments/2026-05-23-movi-e-slot-attention.md`.
  First run where FG-ARI / mBO numbers exist for this codebase.

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

**Done:** `DinoSlotAttentionEncoder` (`src/models/encoder.py`) wraps a
pretrained DINO ViT (`facebook/dino-vits8`, frozen by default) and feeds its
patch tokens — CLS dropped, positional embeddings interpolated to the run
resolution — into the same soft pos-embed + Slot Attention head as
`SlotAttentionEncoder`. Registered in `ENCODER_REGISTRY`, so it's selectable
through `_class_name` like the other encoders. Config:
`configs/movi-e/dino_slot_encoder/config.json` (11 slots, ViT-S/8, 128 res
→ 16x16 patch grid). `eval_movi.py` was relaxed to accept any
slot-attention encoder (not only the CNN one).

**Runs:**
- 2026-05-26 — MOVi-E 200k-step run launched, `DinoSlotAttentionEncoder`
  (frozen DINO ViT-S/8, 11 slots, 128 resolution). Same step budget /
  effective batch / LR as the CNN slot-attn baseline so only the encoder
  differs. Slurm job 23117268 (train+eval, 4x H100); script
  `jobs/movi_e_dino_slot_train_eval.sh`; output `results/movi-e_dino_slot/`;
  report (on completion) `docs/experiments/2026-05-26-movi-e-dino-slot-attention.md`.
  DINOv3 follow-up run pending HF gated-access approval (per user
  conversation 2026-05-26).

## Slot attention: register slots

**Goal:** add **register-style slots** to the current Slot Attention module.
On top of the K "object" slots used downstream by the diffusion decoder,
allocate R extra "register" slots that participate in the iterative attention
competition but are discarded before decoding. They give the attention a
sink for background / global / non-object content so the object slots aren't
forced to absorb it, which has been reported to reduce slot collapse and
improve binding stability.

**References:**
- Nguyen et al., 2026 — *Improved Object-centric Diffusion* (register slots
  for object-centric diffusion).
- Didolkar et al., 2024 — *Transfer of Object-centric Representations*
  (register-token usage in slot attention pipelines).

**How to land it:** extend `SlotAttention` (`src/models/slot_attn.py`) with
an R hyperparameter for register slots — initialized like the object slots
but kept separate at read-out, so only the first K slots are returned to the
encoder / diffusion conditioning. Plumb `num_registers` through the encoder
configs (`SlotAttentionEncoder` and `DinoSlotAttentionEncoder`). Keep the
no-register path (R=0) as default so the change is opt-in and existing runs
stay comparable. First test on MOVi-E with the DINO backbone, since that's
where slot binding is measurable via FG-ARI / mBO.

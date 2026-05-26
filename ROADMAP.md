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
  **Cancelled mid-run** in favour of the register-slot follow-up (see below);
  intermediate checkpoint remains on disk for qualitative comparison.
- 2026-05-26 — DINOv3 follow-up. `DinoSlotAttentionEncoder` extended to
  auto-detect register tokens and whether the backbone takes
  `interpolate_pos_encoding` (DINOv3 uses RoPE, no such kwarg; CLS + 4
  register tokens dropped). Run at 256 resolution so patch_size=16 gives the
  same 16x16 patch grid as the v1 / CNN runs at 128. Effective batch dropped
  to 16 (2 A100 x 8). First submission (23117941, 4x H100 48h) was killed
  by the Snellius budget guard at 0s -- the v1 reservation left only
  ~8.9k SBUs and the 48h H100 ask wanted ~37k. Resubmitted as 23118028
  on 2x A100 with 24h walltime (~6.1k SBUs); restart-safe via
  `--resume_from_checkpoint latest` for multi-slot continuation.
  Script `jobs/movi_e_dinov3_slot_train_eval.sh`; config
  `configs/movi-e/dinov3_slot_encoder/config.json`; output
  `results/movi-e_dinov3_slot/`; report (on completion)
  `docs/experiments/2026-05-26-movi-e-dinov3-slot-attention.md`.
  **Cancelled mid-run** at ~2h44 once the mean-aggregation loss path was
  retired in favour of register-based sum-of-deltas (90108cb); intermediate
  results in `results/movi-e_dinov3_slot/` are kept for qualitative comparison
  but loss/decomposition trajectories aren't comparable to the new objective.

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

**Hyperparameter note (Nguyen et al., 2026 settings):** the paper uses
**R = 77 register slots across all datasets** (VOC, COCO, MOVi) and
**K = 24 object slots for MOVi-E** — both substantially higher than the
values used here (R=4, K=11).

The R=77 figure is not a tuned choice — it comes from the paper using a
**frozen CLIP text encoder** to generate the register priors, and CLIP's
text encoder has a maximum sequence length of 77 tokens. So in their setup
registers aren't learned at all: the K object slots come out of slot
attention as usual, the R=77 registers are fixed CLIP-text outputs, and
the diffusion decoder cross-attends to the concatenation. The paper notes
that **learning the registers also works**, so a frozen-CLIP register
prior is one of several viable variants — the appeal being that you get a
strong "what's not an object" anchor for free without parameters to fit.

Implication for this codebase: we currently learn the registers as
`nn.Parameter`s, which is the simpler path and fine for the exploration /
development stage we're in. **Frozen-CLIP register priors are a SOTA-chase
lever**, not an exploration-stage one — defer until the loss/architecture
work is settled and we're trying to match published numbers.

K=24 is more directly motivated: MOVi-E scenes can have ~20 instances, so
the K=11 used here may be undercounting. If the in-flight R=4 / K=11 run
still shows collapse or weak binding, plausible next steps in order of
cost: raise K toward 24 first (more headroom per object), then sweep R
upward (start at ~7–16 with learned registers) before considering an
auxiliary diversity loss or switching to frozen-CLIP register priors.

**How to land it:** extend `SlotAttention` (`src/models/slot_attn.py`) with
an R hyperparameter for register slots — initialized like the object slots
but kept separate at read-out, so only the first K slots are returned to the
encoder / diffusion conditioning. Plumb `num_registers` through the encoder
configs (`SlotAttentionEncoder` and `DinoSlotAttentionEncoder`). Keep the
no-register path (R=0) as default so the change is opt-in and existing runs
stay comparable. First test on MOVi-E with the DINO backbone, since that's
where slot binding is measurable via FG-ARI / mBO.

**Done:** registers are now data-independent learned tokens appended *after*
slot attention (not extra slot queries inside it) — see commit 90108cb.
Both `SlotAttentionEncoder` and `DinoSlotAttentionEncoder` accept
`num_registers`. Training composes the epsilon as
`(1 - K) * eps_uncond + sum_k eps_slot_k` where `eps_uncond` conditions on
the registers only and each `eps_slot_k` conditions on `[slot_k, registers]`
(see `compose_eps` in `train_lsd.py`). This replaces the prior mean
aggregation, which was redundancy-collapse prone.

**Runs:**
- 2026-05-26 — DINOv3 + R=4 register slots + K=24 object slots, MOVi-E
  200k-step run. Same backbone / resolution / step budget as the cancelled
  no-register DINOv3 run; encoder config now has `num_registers=4` and
  `num_components=24`, and the loss is sum-of-deltas with the registers as
  the uncond context (90108cb). K bumped from 11 to 24 to match Nguyen et
  al. 2026's MOVi-E setting (~20 instances per scene). First submission
  (Slurm 23123487, 2x A100) failed at step 0 with `AttributeError:
  'DistributedDataParallel' object has no attribute 'num_components'` --
  fixed in c4fcb40 by caching the encoder's slot / register counts before
  `accelerator.prepare`. Resubmitted as 23124195 on 2x H100 80 GB so the
  K=24 / R=4 footprint (K+1=25 UNet forwards/step, cross-attn seq up to
  1+R=5) fits at PER_GPU_BATCH=8. Restart-safe via
  `--resume_from_checkpoint latest`. Script
  `jobs/movi_e_dinov3_registers_train_eval.sh`; config
  `configs/movi-e/dinov3_slot_encoder/config.json`; output
  `results/movi-e_dinov3_reg/`; report (on completion)
  `docs/experiments/2026-05-26-movi-e-dinov3-registers.md`. Also adds
  `val_loss` and `slot_pairwise_cos` (mean off-diagonal pairwise cosine sim
  of object slots, a collapse diagnostic) to wandb logging at every
  validation step.

## Object-centric representation metrics (planned)

**Goal:** broaden evaluation beyond per-component decoding grids and the
single FG-ARI / mBO check, so we can spot decomposition failures (slot
collapse, slot-to-object mis-binding, slot-redundancy) from numbers rather
than only by eyeballing grids — and so we can stop weighting individual
component decoding as the primary signal it currently is.

**Candidates to evaluate / add (no need to do this immediately):**
- *Slot-collapse diagnostics:* mean off-diagonal slot pairwise cosine
  similarity (already added to wandb in this branch) and slot-norm spread.
  Cheap, log every validation step.
- *Attention-mask metrics:* slot attention entropy (per slot, averaged over
  the image) and a Hungarian-matched per-object IoU on the slot attention
  masks against the GT instance masks (mBO already does the closest thing
  but at decoded-component resolution).
- *Slot purity / object-completeness* over the GT segments
  (Engelcke et al. / Greff et al. style).
- *Representational metrics on the slot vectors:* probe each slot for the
  per-object MOVi-E attributes shipped with the dump
  (`labels/*_labels.json` already has bbox / shape / material / color /
  3d-coords); a linear probe accuracy on object slots ~ "is the slot
  identifiable as a single object" without needing image decoding.
- *Decoder-free reconstruction proxies:* DINO-feature reconstruction MSE per
  slot (DINOSAUR-style), evaluable without paying the diffusion sampling
  cost.

These should be wired into `eval_movi.py` (full-validation pass) and a
subset into `train_lsd.py`'s `log_validation` (per-step wandb scalars), so
that "did the slots decompose?" is answerable from the wandb dashboard mid
training, not only post-hoc.

## Auxiliary slot diversity / disentanglement loss

**Goal:** add an explicit auxiliary loss term that *pushes* slots apart in
representation space, instead of relying only on the implicit pressure from
the sum-of-deltas diffusion objective. The compositional loss alone leaves a
local minimum at "all slots predict the mean," which slot-pairwise cosine
sim diagnostics will catch but won't itself fix.

**Reference:** Nguyen et al., 2026 — *Improved Object-centric Diffusion*
(`nguyen2026ImprovedObjectcentricDiffusion`) — same paper cited under the
register-slot section; it pairs registers with an auxiliary diversity term
on the object slots, the combination being what gives them their reported
gain over plain register slots.

**Candidate forms (decide after reading the paper carefully):**
- Pairwise repulsion on object-slot embeddings: penalise high off-diagonal
  cosine sim of the K object slots (essentially turn the diagnostic into a
  loss). Cheap; small weight to start.
- Cross-attention orthogonality: penalise overlap between the slot-attention
  masks of different slots at the encoder side, so two slots can't claim the
  same input tokens.
- Contrastive / InfoNCE on slots across images, treating "same slot index
  across images" as not-positive — discourages the model from collapsing
  slots to a fixed image-independent set of attractors.

**Where it lives:** the loss term goes alongside the diffusion MSE in
`train_lsd.py`'s training loop; weighting controlled by a new hyperparameter
in `configs/<dataset>/train_config.yaml`. Should be opt-in (default 0.0) so
register-only ablations stay clean.

**Sequencing:** depends on the DINOv3 + R=4 run (23123487) — if registers
alone solve collapse, this is lower-priority; if `slot_pairwise_cos` still
trends toward 1 there, this is the obvious next lever.

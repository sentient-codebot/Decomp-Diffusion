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

## Storage and inode pressure

**Goal:** keep `prjs0993` comfortably below inode quota so training jobs do
not fail during cache, checkpoint, or log writes. The COCO 200k-step launch on
2026-05-27 failed before step 1 when Torch Inductor/Triton attempted to write
compiled kernels into `~/prjs0993/tmp/torchinductor` while the project space was
over inode quota.

**Cleanup done:**
- 2026-05-27 - removed stale artifact-only environments and old HeatoDiff
  artifacts from `/gpfs/work2/0/prjs0993/artifacts`: `OpenSynth/.venv`,
  `FM-Energy/.venv`, and `HeatoDiff/`. This reduced artifact inode use from
  ~92.6k to 3 inodes. `tmp/torchinductor` was intentionally kept so compiled
  kernels remain warm.

**Implementation path:**
- MOVi-E now has a WebDataset-style tar shard path: `data/movi-e-wds/` with
  per-split `samples.jsonl` byte-offset indexes and manifests. Training,
  reconstruction eval, `eval_movi.py`, and `probe_movi.py` can read shards via
  `--dataset_format wds` / `--movi_eval_format wds`; MOVi job scripts use this
  path by default.
- `jobs/movi_e_shard_wds.sh` converts an existing loose MOVi-E tree to shards
  on `prjs0993`. `jobs/movi_e_preprocess.sh` now expands TFDS output on
  `/scratch-shared/nlin/...` and publishes only shards to persistent storage.
- Storage docs and a read-only inode report live in `docs/storage.md` and
  `scripts/storage/inode_report.sh`. The retention convention is to keep
  reports, final checkpoints, shard manifests, synced W&B URLs, and the warm
  Torch Inductor cache; prune redundant local W&B/debug runs after sync unless
  needed for offline debugging.

**Remaining work:**
- Run `jobs/movi_e_shard_wds.sh`, validate shard smoke training plus MOVi
  metrics, then remove the persistent loose `~/prjs0993/datasets/movi-e/` tree.
- Decide whether COCO should remain as loose JPGs. COCO currently costs about
  123k inodes, almost entirely `images/`; this is acceptable short term but
  should be sharded if repeated natural-image experiments become common.
- Keep `celebahq_data128x128` under review (~30k inodes). It is smaller than
  MOVi-E/COCO, so defer unless quota pressure returns.

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

## Adaptive epsilon composition weights

**Goal:** replace uniform per-slot epsilon composition with learned
spatially adaptive weights, so empty or irrelevant slots can contribute less
where another slot better explains the image content. The current composition
can be viewed as
`eps_comp = sum_k eps_k * w_k` with `w_k = 1 / K`, constant across spatial
locations and slots. This treats all slots as equally useful everywhere,
which is a poor match for object-centric decomposition when some slots are
empty, background-like, or redundant.

**Target formulation:** compose
`eps_comp[:, :, i, j] = sum_k eps_k[:, :, i, j] * w[:, k, i, j]`, where
`w[:, k, i, j]` is spatial, non-negative, and normalized over slots:
`w[:, k, i, j] in [0, 1]` and `sum_k w[:, k, i, j] = 1` for every spatial
location `(i, j)`. This turns reconstruction into a per-location competition:
each slot proposes an epsilon prediction, and the mask decides which slot(s)
should explain that location.

**Design choices to explore:**
- Reuse the encoder-side Slot Attention scores as masks. They already encode
  token-to-slot ownership and, for CNN / ViT feature maps, have a spatial
  layout that can be resized to the UNet latent resolution.
- Add lightweight adaptation layers on top of the Slot Attention scores
  before composition, for example a small per-slot refinement head,
  temperature / sharpening control, or a learned projection from encoder
  feature-map masks to denoiser latent masks.
- Predict masks from slot embeddings plus image / denoiser features when
  attention scores are too coarse or too tied to the encoder feature grid.
  This is higher risk because the mask path may learn shortcut behavior, so
  it should follow the attention-score baseline.
- Decide whether masks weight raw `eps_slot_k`, slot residuals relative to
  an unconditional/register prediction, or both objective variants. Keep the
  implementation explicit so mean-aggregation, sum-of-deltas, and register
  ablations remain comparable.
- Re-using slot attention scores. The slot attention scores exactly sum up to 1 for every spatial location. We can simply do an interpolation to the latent resolution and use it as our composition weights. 
- Decide gradient flow through the mask path. The first baseline should use
  interpolated Slot Attention masks with `detach()`, so diffusion gradients
  update `eps_slot_k` / slot content but do not directly rewrite the
  encoder-side routing. Direct `eps_comp -> mask` gradients are a valid
  ablation, but they should be treated carefully because the routing signal
  may learn reconstruction shortcuts rather than object ownership; if enabled,
  monitor slot mass, mask entropy, empty-slot suppression, and object-centric
  metrics.

**Validation plan:**
- Start with MOVi-E, where GT masks make the mask behavior measurable.
  Report reconstruction loss, FG-ARI / mBO / mIoU, property-probe scores,
  and qualitative mask overlays alongside the usual per-slot generations.
- Add diagnostics for mask entropy, empty-slot mass, and agreement between
  learned composition masks and Slot Attention masks / GT instance masks.
- Compare at least three variants under the same encoder and training budget:
  uniform weights, direct Slot Attention masks, and adapted / refined masks.
  Promote the simplest variant that improves decomposition without reducing
  reconstruction quality.

**Resolution caveat:** the composition masks operate in the same relatively
low spatial regime as the slots and denoiser latents: typically 16x16 for
128px inputs or 32x32 for 256px inputs in the SD VAE latent space. This is
enough for denoising competition, but it limits segmentation fidelity. If
high-resolution masks become important later, possible extensions include a
separate upsampling / refinement head or an image-space mask decoder. That
would be a larger effort and is not a priority for the current research line,
which is decomposition for compositional diffusion rather than high-resolution
segmentation.

**Sequencing:** this stage follows the register-slot and K/V-only decoder
experiments. Registers address what information enters each slot; adaptive
composition addresses how much each slot is allowed to contribute at each
denoiser spatial location.

**Runs:**
- 2026-05-28 -- first adaptive-weight baseline completed: reuse detached
  Slot Attention masks as epsilon composition weights, interpolate to
  latent/noise resolution, and warm-start weights from the CoDA K/V-only
  checkpoint. Training job `23179585`; corrected latest-checkpoint eval job
  `23192429`; script
  `jobs/movi_e_adaptive_eps_slot_attn_warmstart_train_eval.sh`; output
  `results/movi-e_adaptive_eps_slot_attn_warmstart/`; report
  `docs/experiments/2026-05-28-movi-e-adaptive-eps-slot-attn.md`. Result:
  methodological regression. The original eval reported FG-ARI 0.0087 / mBO
  0.0279 / mIoU 0.0663; after fixing eval precision and SD scheduler usage,
  the same checkpoint improved only to FG-ARI 0.0610 / mBO 0.0792 / mIoU
  0.1197, still far below the rerun CoDA mean-weight baseline (job `23192654`:
  FG-ARI 0.5160 / mBO 0.3451 / mIoU 0.3420). Do not promote detached
  point-wise slot-attention weighting as-is. Future attempts should keep the
  CoDA mean-weight baseline intact and test less destructive variants such as
  frozen-encoder routing, gradual interpolation from mean to spatial weights,
  or scalar per-slot weights.
- 2026-05-28 -- scalar pooled adaptive-weight variant prepared: add
  `--epsilon_composition slot_attn_pool`, which pools each Slot Attention mask
  over spatial dimensions to one detached scalar per slot, normalizes over
  slots, and broadcasts those weights over the UNet epsilon grid. This keeps
  adaptive empty-slot suppression but removes point-wise mask routing from the
  denoising target. Script
  `jobs/movi_e_adaptive_eps_slot_attn_pool_warmstart_train_eval.sh`; output
  `results/movi-e_adaptive_eps_slot_attn_pool_warmstart/`; report
  `docs/experiments/2026-05-28-movi-e-adaptive-eps-slot-attn-pool.md`;
  W&B project `latent_decomposed_diffusion_adaptive_eps`. Awaiting Slurm
  submission and metrics.

## Object-centric representation metrics

**Goal:** broaden evaluation beyond per-component decoding grids and the
single FG-ARI / mBO check, so we can spot decomposition failures (slot
collapse, slot-to-object mis-binding, slot-redundancy) from numbers rather
than only by eyeballing grids — and so we can stop weighting individual
component decoding as the primary signal it currently is.

**Reference for metric definitions:** sony/coda
(`src/metric/segmentation.py`, `experiment/linear_prob.py`) — mIoU uses
Hungarian matching via `scipy.optimize.linear_sum_assignment`; property
probe uses mask-cosine matching cost with the 2-layer MLP (continuous) /
linear (discrete) heads from that repo.

**Done (MOVi-E):**

- *Task 1 — Object discovery:* FG-ARI, mBO, and Hungarian-matched mIoU
  live in `src/metrics/segmentation.py`. `eval_movi.py` runs all three
  offline. `train_lsd.py:log_validation` also streams `val/fg_ari`,
  `val/mbo`, `val/miou` to wandb when the run is launched with
  `--movi_eval_root <root>` and the encoder is a slot-attention variant.
  Slot collapse diagnostic `slot_pairwise_cos` and `val_loss` continue to
  log unconditionally.
- *Task 2 — Property-prediction probe:* `probe_movi.py` freezes the slot
  encoder, Hungarian-matches slot attention masks to GT instance masks
  (mask cosine cost), and trains a small MLP per property:
  `image_positions` and `bboxes_3d` (2-layer MLP, hidden 786, MSE) and
  `category` (single linear layer, cross-entropy — sony/coda's
  discrete-head convention). Reports test-split metrics + caches matched
  (slot, property) pairs under `<output_dir>/cache_<split>.pt` so the
  probe can be re-trained without rerunning the encoder. Launch wrappers
  under `scripts/movi-e/{eval,probe}.sh`.

**Still candidate (not yet wired):**

- *Slot attention entropy* (per slot, averaged over the image).
- *Slot-norm spread* alongside `slot_pairwise_cos`.
- *DINO-feature reconstruction MSE* per slot (DINOSAUR-style), evaluable
  without paying the diffusion sampling cost.

## Compositional image generation metrics (planned)

**Goal:** report **reconstruction FID/KID** (slots from a real image,
decoded back) and **compositional FID/KID** (slots assembled across
different images, decoded) on a natural-image dataset. The compositional
score is the headline number — it tests whether the decoder can render
unseen slot combinations, which is what "compositional generation"
actually means.

**Target dataset: COCO.** Training-side loader landed; eval-side (masks,
FID/KID) is still open.

**Done (training side):**
- `configs/coco/` mirrors `configs/movi-e/` with K=7 / R=4 /
  latent_dim=1024 (CoDA-style K/V-only recipe on SD2.1).
- COCO data downloads to `~/prjs0993/datasets/coco/` via
  `jobs/coco_download.sh` (mirrors sony/coda's `preprocess/download.sh`
  layout: `images/{train2017,val2017}` + `annotations/`).
- `jobs/coco_smoketest.sh` validates the wiring with a synthetic
  16-JPG dataset before the real download is needed.
- `jobs/coco_coda_kv_only_train.sh` runs the 200k-step training job on
  the downloaded COCO train2017 (~118k JPGs). Pixels are read via the
  existing `GlobDataset` — sony/coda also uses a plain image glob for
  the training side, so no new dataset class is needed yet.

**Still open before this section is "done":**
- `CocoDataset` for mask/annotation-aware eval (instance masks from
  pycocotools, mirroring sony/coda's `COCO2017Dataset`).
- Wire FID/KID via `cleanfid` or `torchmetrics.image`; add the dep behind
  an extra in `pyproject.toml`.
- Compositional sampling: assemble cross-batch slot mixtures (e.g. roll
  slot indices across the batch dim) before the pipeline call — the
  pipeline already accepts arbitrary `prompt_embeds` per
  `src/pipeline/composable_stable_diffusion_pipeline.py`.

**Runs:**
- 2026-05-27 — COCO CoDA-style K/V-only 200k-step train job (Slurm id
  TBD, script `jobs/coco_coda_kv_only_train.sh`; config
  `configs/coco/dinov3_slot_encoder_d1024/config.json`; output
  `results/coco_coda_kv_only/`; report target
  `docs/experiments/2026-05-27-coco-coda-kv-only.md`).

## VOC support (planned)

Out of scope until a VOC dataset / loader exists. Once it does, the same
Task 1 (FG-ARI / mBO / mIoU via `src/metrics/segmentation.py`) and Task 2
(property probe via the same matching + probe template as
`probe_movi.py`) machinery applies — only the dataset class and config
need to change.

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
- **CoDA-style contrastive alignment loss:** form "mismatched" slot sets by
  replacing a subset of an image's slots with slots from another image, run
  the denoiser on the mismatched set, and *maximise* the MSE between the
  correct noise and the noise predicted from the mismatched slots. The
  symmetric counterpart to the diffusion MSE: matched slots must predict the
  noise, mismatched slots must fail to. See the CoDA paper for the exact
  mixing recipe (which subset, how many, sampling strategy).
- **Denoiser cross-attention entropy penalty:** in the UNet's cross-attention
  to slots, each image patch produces a distribution over the K slots. Use
  the entropy of that distribution as a regulariser to discourage patches
  from attending to all slots uniformly (no slot specialisation) — i.e.
  penalise high per-patch entropy. The dual view (per-slot distribution over
  patches) is also informative: a slot attending uniformly over the image is
  the collapse mode we want to penalise. Both directions are cheap to log
  before being used as a loss. The slot-attention attention scores at the
  encoder side admit the same treatment and are worth considering in tandem.

**Where it lives:** the loss term goes alongside the diffusion MSE in
`train_lsd.py`'s training loop; weighting controlled by a new hyperparameter
in `configs/<dataset>/train_config.yaml`. Should be opt-in (default 0.0) so
register-only ablations stay clean.

**Sequencing:** depends on the DINOv3 + R=4 run (23123487) — if registers
alone solve collapse, this is lower-priority; if `slot_pairwise_cos` still
trends toward 1 there, this is the obvious next lever.

## Decoder backbone: trainable surface and DiT (planned)

**Goal:** explore shrinking the trainable surface of the diffusion decoder,
and longer-term swap the UNet backbone for a DiT-based one. Right now we
train the full SD 2.1 UNet plus the encoder; the encoder is what should be
doing the object-centric work, so paying for full UNet gradients is mostly
inertia from the original LSD recipe.

**Train only the cross-attention K/V projections (in scope, queued run).**
Match the lightweight side of Sony CoDA's setup on MOVi-E: freeze the SD 2.1
UNet and only train cross-attn K / V projections (`.attn2.to_k.`,
`.attn2.to_v.`) plus the encoder. Same dataset, step budget, and
DINOv3+register-slot encoder family as the current full-UNet run; the
comparison is about how much decoder capacity the slot conditioning needs to
bind into. This landed as `--freeze_unet_except_kv` in `train_lsd.py`, so
existing runs stay reproducible.

- 2026-05-27 — MOVi-E CoDA-style K/V-only 200k-step train+eval completed:
  Slurm job `23151491`; script `jobs/movi_e_coda_kv_only_train_eval.sh`;
  log `/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23151491.log`;
  config `configs/movi-e/dinov3_slot_encoder_d1024/config.json`; output
  `results/movi-e_coda_kv_only/`; report
  `docs/experiments/2026-05-27-movi-e-coda-kv-only.md`. Uses the
  mean-of-eps objective (`eps = mean_k eps_slot_k`) with registers
  concatenated to each slot's conditioning sequence. Final checkpoint:
  `results/movi-e_coda_kv_only/latent_decomposed_diffusion/checkpoint-200000-last`.
  The train wrapper's metrics were FG-ARI 0.5032 / mBO 0.3373 / mIoU 0.3354;
  eval-only rerun `23192654` reproduced the result with bf16 mask extraction
  at FG-ARI 0.5160 / mBO 0.3451 / mIoU 0.3420. Treat this as the current
  MOVi-E SD2.1 K/V-only baseline. An earlier attempt (Slurm `23139442`) used
  the sum objective and was cancelled at ~3h / step 20000: with a
  mostly-frozen pretrained UNet, asking each per-slot eps to be ~noise/K so
  that the sum matches one noise is incompatible with the pretrained output
  scale and not learnable through K/V alone. Aborted checkpoints archived at
  `results/movi-e_coda_kv_only_sum_aborted_20260527T134321Z/`.

**DiT backbone, e.g. SD3 (out of scope for this paper, exploratory).** Same
training-surface question but with a transformer denoiser: SD3 / similar
MM-DiT models replace the cross-attention UNet entirely. Slot conditioning
would have to be re-plumbed (MM-DiT consumes text tokens as a second stream
rather than via cross-attn into a UNet), and again we would *not* train the
full denoiser — most likely only the slot-conditioning input projections /
adapters. Useful as a sanity check that the decomposition story isn't
specific to SD-1.5/2.x cross-attention UNets. Defer until the projections-
only variant on the UNet is understood; this is a "next project" item, not
something to crowd into the current write-up.

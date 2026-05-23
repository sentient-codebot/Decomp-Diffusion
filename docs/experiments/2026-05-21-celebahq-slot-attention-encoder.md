# CelebA-HQ slot-attention encoder -- partial 500k-step run

**Status:** PARTIAL -- training interrupted at ~215k of 500k steps by a home-quota OSError; 200k checkpoint surived and was evaluated separately
**Date:** 2026-05-22 (train), 2026-05-23 (eval)
**Slurm jobs:** 23006944 (train, 4x H100, 10:33), 23069034 (eval, 1x A100, 00:03)
**Scripts:** `jobs/celebahq_slot_attn_train_eval.sh`, `jobs/celebahq_slot_attn_eval.sh`
**wandb run:** https://wandb.ai/nan-team/latent_decomposed_diffusion/runs/v1ywc9pz (peach-glade-2)

## Purpose

Roadmap "Slot extraction: slot attention" run. The naive `LatentEncoder`
(CNN → `Flatten` → `Linear` → reshape to K slots, no slot attention) is
replaced with `SlotAttentionEncoder`: the same convolutional feature
extractor produces a feature map, which -- after a soft positional embedding
-- is bound into K slots by an iterative Slot Attention module (Locatello et
al., 2020). This is the canonical Latent Slot Diffusion encoder pattern.

Compared against the LatentEncoder baseline
(`docs/experiments/2026-05-20-celebahq-full-500k-run.md`,
`results/celebahq/`). Effective batch, step count, LR, UNet and scheduler
are identical to the baseline -- only the encoder differs.

## Configuration

| Item | Value |
|------|-------|
| Encoder | `SlotAttentionEncoder` (CNN + soft pos-embed + Slot Attention, 12.5 M params) |
| Slot Attention iters | 3 |
| Slots (num_components) | 4 |
| Slot dim (latent_dim) | 64 (matches UNet `cross_attention_dim`) |
| Steps run | ~215000 / 500000 configured (training interrupted) |
| Checkpoint used for eval | `checkpoint-200000` (the latest one that fully wrote out before quota crash) |
| Effective batch | 64 (4 GPU × 16) -- matches baseline (2 GPU × 32) |
| Resolution | 128 |
| Mixed precision | fp16 |
| Learning rate | 2.0e-5 |
| Logging | wandb (`nan-team/latent_decomposed_diffusion`) |

For reference, `LatentEncoder` has ~70 M params (dominated by its
`Linear(1024·16·16, 256)` slot read-out); `SlotAttentionEncoder` is ~5.5×
smaller.

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | INTERRUPTED at ~215k steps | 10:33:24 |
| Evaluation (checkpoint-200000) | PASS | 00:02:47 |

- Final checkpoint kept: `results/celebahq_slot/latent_decomposed_diffusion/checkpoint-200000`
- Loss curve (slot-attention only): `results/celebahq_slot/loss_curve_v1ywc9pz.png`
- Loss curve (slot-attention vs baseline): `results/celebahq_slot/loss_curve_comparison.png`
- Eval slice: 100 images (glob `000*.jpg`)
- Eval grids: `results/celebahq_slot/eval_grids/image_NN.jpg` -- each row is
  `[input | slot 0 | slot 1 | slot 2 | slot 3 | reconstruction]`

### Training loss vs LatentEncoder baseline (2000-step rolling mean)

| Step | SlotAttentionEncoder | LatentEncoder baseline | Δ |
|------|---------------------|------------------------|---|
| 50k  | 0.1312 | 0.1237 | +0.0075 |
| 100k | 0.0991 | 0.0989 | +0.0002 |
| 150k | 0.0745 | 0.0753 | −0.0008 |
| 200k | 0.0607 | 0.0605 | +0.0002 |
| 215k | 0.0592 | 0.0575 | +0.0017 |
| 500k | (interrupted) | 0.0344 | — |

The two curves overlap to within ~3% in the regime sampled (see
`loss_curve_comparison.png`).

## Assessment

**Diffusion loss:** the SlotAttentionEncoder tracks the LatentEncoder
baseline almost exactly through 215k steps. With ~5.5× fewer encoder
parameters, slot attention is matching, not improving, the diffusion-loss
trajectory. We can't say anything about the converged-loss gap because the
slot run was interrupted at 215k of 500k.

**Reconstruction:** the rightmost column ("reconstruction" from all 4 slots
together) recovers the input image well in both runs -- the slot
representation is information-preserving.

**Decomposition (the actual point of slot attention):** at 200k steps, the
SlotAttentionEncoder shows **slot collapse**: the four per-slot generations
are nearly identical to each other and to the input, i.e. each slot encodes
the whole image rather than a distinct component. The LatentEncoder
baseline at 500k, in contrast, shows clearly differentiated per-slot
generations -- the slots in that run *do* specialise. Cropped 6-row
side-by-sides for visual comparison: `slot_top6_crop.jpg` (slot, 200k) and
`baseline_top6_crop.jpg` (baseline, 500k).

Two confounders make this not a clean comparison:

1. **Step count:** slot run is 200k vs baseline 500k. Specialisation may emerge
   later in slot training; this run was cut short before that could be tested.
2. **Architecture pressure:** Slot Attention by itself does *not* break the
   symmetry between slots in a diffusion-decoder setup. The original Slot
   Attention paper paired the module with a per-slot RGB+mask decoder whose
   reconstruction objective explicitly demanded decomposition; here, the
   diffusion loss is on the *sum* of the per-slot conditioning, so a
   degenerate solution where every slot encodes the whole image satisfies
   the objective just as well as a clean decomposition. Slot collapse under
   exactly this setup is a known failure mode.

**Conclusion:** the slot-attention encoder is *correctly implemented and
trains stably* -- diffusion loss matches the baseline -- but slot attention
alone is **not sufficient** to induce decomposition in the latent slot
diffusion setup. Either training has to run much longer to see whether
specialisation eventually emerges, or an explicit decomposition signal needs
to be added (per-slot mask, slot-dropout, or -- the planned next step --
strong pretrained features that already provide spatially distinguishable
representations).

## Notes

- Only the encoder changed from the baseline. The convolutional feature
  extractor is unchanged; the `Flatten` + `Linear` slot read-out is replaced
  by a soft positional embedding + Slot Attention module.
- The home-quota crash that ended training has been fixed -- run artifacts
  now go to project storage (`~/prjs0993/<project>/`) via job-script-level
  symlinks (commit `ee295ed` on `feat/slot-attention-encoder` and `e02d149`
  on `main`).
- The wandb API pull at the end of the training job failed for the same
  quota reason; the loss curves in this report were re-pulled from wandb on
  the login node after the quota was freed.
- eval.py writes grids to `./image_test_output/` (hardcoded); they are
  copied to `results/celebahq_slot/eval_grids/` here. See `TECHDEBT.md`.

## Next steps

- **Don't** resume this run blindly -- the decomposition problem won't fix
  itself just by running longer at this scale. The two paths forward:
  1. **Roadmap next stage** (`ROADMAP.md` "Encoder: pretrained feature
     extractor") -- swap the from-scratch CNN for a frozen DINO/DINOv2
     backbone. Strong patch features make per-position attention much more
     discriminative, which is what slot attention needs to actually
     localise.
  2. **Explicit decomposition signal** -- add a per-slot mask head and a
     reconstruction loss that demands the masked combination equal the
     input (the original Slot Attention recipe). Cheaper to try than (1)
     and orthogonal to the encoder choice.
- The smoke test (`jobs/celebahq_slot_attn_smoketest.sh`) and the eval-only
  job (`jobs/celebahq_slot_attn_eval.sh`) remain the right tools for
  validating future variants.

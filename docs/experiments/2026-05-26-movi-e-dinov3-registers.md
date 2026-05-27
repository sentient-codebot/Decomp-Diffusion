# MOVi-E DINOv3 slot-attention encoder + register slots -- 200k-step run

**Status:** FAIL -- see slurm_23126825.log
**Date:** 2026-05-26T20:56Z
**Slurm job:** 23126825 (script: jobs/movi_e_dinov3_registers_train_eval.sh, log: slurm_23126825.log)
**Node / GPUs:** gcn108, 2x H100
**wandb run:** https://wandb.ai/nan-team/latent_decomposed_diffusion/runs/xwrxaivf

## Purpose

First run combining the DINOv3 backbone with the **register-slot
compositional denoising** objective from commit 90108cb. Tests whether the
sum-of-deltas loss (eps = (1 - K) * eps_uncond + sum_k eps_slot_k, with
eps_uncond conditioned on R=4 register tokens only) breaks the redundancy
collapse seen with the previous mean-aggregation loss.

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention + 4 register slots) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| Patch grid | 16x16 at 256 input (patch_size=16) |
| Special tokens dropped (backbone) | 5 (1 CLS + 4 register) |
| Slot Attention iters | 3 |
| Slots (num_components, K) | 24 |
| Register slots (R) | 4 |
| Slot dim (latent_dim) | 64 |
| Loss | sum-of-deltas, registers as uncond context |
| Steps run | 200000 / 200000 configured |
| Effective batch | 16 (2 GPU x 8) |
| Resolution | 256 |
| Mixed precision | fp16 |
| Learning rate | 2.0e-5 |
| Dataset | MOVi-E train split (233976 frames) |
| Eval split | MOVi-E validation split (6000 frames) |
| Output dir | results/movi-e_dinov3_reg/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | FAIL (rc=1) | 03:03:01 |
| Reconstruction eval | PASS | 00:28:24 |
| Object-centric metrics | PASS | (incl. above) |

### Object-centric metrics (eval_movi.py, validation split)

| Metric | DINOv3 + R=4 | DINOv3 (no reg) | DINO v1 @128 | CNN @128 |
|--------|-------|-------|-------|-------|
| FG-ARI | 0.0597 | cancelled | see prior report | see prior report |
| mBO    | 0.0481 | cancelled | see prior report | see prior report |
| Frames | 6000 | -- | -- | -- |

Full metrics: results/movi-e_dinov3_reg/metrics/metrics.json
Attention-mask viz: results/movi-e_dinov3_reg/metrics/viz_*.jpg

### Reconstruction grids and curves

Final checkpoint: results/movi-e_dinov3_reg/latent_decomposed_diffusion/checkpoint-20000
Loss + slot-pairwise-cos curves: results/movi-e_dinov3_reg/loss_curve_23126825.png
Reconstruction grids: results/movi-e_dinov3_reg/eval_grids/image_NN.jpg
Per-step validation viz: results/movi-e_dinov3_reg/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics + viz:
     - Did slot_pairwise_cos stay low (slots specialise) or trend toward 1
       (collapse)?
     - Did val_loss track or decouple from train_loss?
     - FG-ARI / mBO vs the cancelled DINOv3-no-register run -- where it had
       reached at the cancellation point (intermediate ckpt in results/movi-e_dinov3_slot). -->

## Notes

- Previous DINOv3 (no register) run 23118028 and DINO v1 run 23117268 were
  cancelled to free SBU for this run; their intermediate checkpoints stay on
  disk for qualitative comparison but loss-curve comparison isn't apples to
  apples since the objective changed (mean -> sum-of-deltas with uncond).
- New wandb-logged scalars (added in this branch): `val_loss` (single-step
  MSE on val batches under the same composition) and `slot_pairwise_cos`
  (mean off-diagonal pairwise cosine sim of object slots -- collapse
  diagnostic).

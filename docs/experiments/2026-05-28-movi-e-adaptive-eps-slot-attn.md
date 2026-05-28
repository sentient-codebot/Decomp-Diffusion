# MOVi-E adaptive epsilon composition -- slot-attention weights

**Status:** FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23179585.log
**Date:** 2026-05-28T16:46Z
**Slurm job:** 23179585 (script: jobs/movi_e_adaptive_eps_slot_attn_warmstart_train_eval.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23179585.log)
**Node / GPUs:** gcn87, 2x H100
**wandb run:** https://wandb.ai/nan-team/latent_decomposed_diffusion_adaptive_eps/runs/2sioipix

## Purpose

First adaptive epsilon-composition trial from ROADMAP.md. Reuses encoder Slot
Attention masks as spatial composition weights, bilinearly interpolates them to
UNet epsilon resolution, and stops gradients through the weights with detach().
The run warm-starts model weights from the CoDA K/V-only checkpoint but writes
to a separate output folder and trains for a fresh 50k steps.

## Configuration

| Item | Value |
|------|-------|
| Warm-start checkpoint | results/movi-e_coda_kv_only/latent_decomposed_diffusion/checkpoint-200000-last |
| Output dir | results/movi-e_adaptive_eps_slot_attn_warmstart/latent_decomposed_diffusion_adaptive_eps/ |
| Encoder | DinoSlotAttentionEncoder d1024, DINOv3 frozen, K=24, R=4 |
| UNet | SD2.1 pretrained, frozen except cross-attn K/V |
| Epsilon composition | detached Slot Attention masks, interpolated to latent resolution |
| Steps run | 50000 fresh continuation steps |
| Effective batch | 16 (2 GPU x 8) |
| Resolution | 256 |
| Dataset | MOVi-E train shards (233976 frames) |
| Eval split | MOVi-E validation shards (6000 frames) |

## Original job results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | FAIL (rc=1) | 04:25:31 |
| Reconstruction eval | PASS | 00:06:39 |
| Object-centric metrics | PASS | included above |

### Original object-centric metrics

| Metric | Value |
|--------|-------|
| FG-ARI | 0.0087 |
| mBO | 0.0279 |
| mIoU | 0.0663 |
| Frames | 6000 |

Full metrics: results/movi-e_adaptive_eps_slot_attn_warmstart/metrics/metrics.json
Attention-mask viz: results/movi-e_adaptive_eps_slot_attn_warmstart/metrics/viz_*.jpg
Final checkpoint: results/movi-e_adaptive_eps_slot_attn_warmstart/latent_decomposed_diffusion_adaptive_eps/checkpoint-50000-last
Loss + slot-pairwise-cos curves: results/movi-e_adaptive_eps_slot_attn_warmstart/loss_curve_23179585.png
Reconstruction grids: results/movi-e_adaptive_eps_slot_attn_warmstart/eval_grids/image_NN.jpg
Per-step validation viz: results/movi-e_adaptive_eps_slot_attn_warmstart/latent_decomposed_diffusion_adaptive_eps/logs/

## Corrected latest-checkpoint eval

The original train-attached eval used the old `eval_movi.py` default precision
and the reconstruction path used the MOVi scheduler config rather than the
SD2.1 scheduler used by `--unet_config pretrain_sd`. Follow-up eval job
`23192429` reran the final checkpoint with bf16 slot-mask extraction and the
SD2.1 scheduler for reconstruction.

| Stage | Result | Wall time |
|-------|--------|-----------|
| Reconstruction eval | PASS | 00:03:16 |
| Object-centric metrics | PASS | 00:02:44 |
| Total | PASS | 00:06:00 |

### Corrected object-centric metrics

| Metric | Value |
|--------|-------|
| FG-ARI | 0.0610 |
| mBO | 0.0792 |
| mIoU | 0.1197 |
| Frames | 6000 |

Corrected metrics: results/movi-e_adaptive_eps_slot_attn_warmstart/metrics_latest_23192429/metrics.json
Corrected attention-mask viz: results/movi-e_adaptive_eps_slot_attn_warmstart/metrics_latest_23192429/viz_*.jpg
Corrected reconstruction grids: results/movi-e_adaptive_eps_slot_attn_warmstart/eval_grids_latest_23192429/image_NN.jpg
Corrected generated images: results/movi-e_adaptive_eps_slot_attn_warmstart/gen_images_latest_23192429/

## Assessment

Follow-up checks confirm this is a methodological regression, not just an
evaluation artifact. The corrected adaptive eval job
`23192429` reran the same checkpoint with bf16 slot-mask extraction and the
SD2.1 scheduler for reconstruction. That improved the full-validation metrics
relative to the original eval, but the result remains far below the mean-weight
CoDA baseline:

| Checkpoint / eval | FG-ARI | mBO | mIoU | Frames |
|-------------------|--------|-----|------|--------|
| Adaptive slot-attn weights, original eval | 0.0087 | 0.0279 | 0.0663 | 6000 |
| Adaptive slot-attn weights, corrected eval (`23192429`) | 0.0610 | 0.0792 | 0.1197 | 6000 |
| CoDA mean-weight baseline rerun (`23192654`) | 0.5160 | 0.3451 | 0.3420 | 6000 |

The baseline rerun reproduced the strong object-centric metrics from
`jobs/movi_e_coda_kv_only_train_eval.sh`, so the slot-mask evaluation path is
healthy. The adaptive continuation appears to damage the encoder's
object-aligned Slot Attention masks. The likely failure mode is the objective
change itself: the same detached attention masks are used as spatial epsilon
routing weights, so diffusion loss can adapt slot tokens/K/V to the current
routing without directly correcting the masks toward object segmentation.

Do not treat this checkpoint as an improvement over the CoDA baseline. Future
adaptive-composition attempts should preserve the baseline with an ablation
that freezes the encoder, uses a much lower LR, or introduces the spatial
weights gradually instead of switching directly from mean composition.

## Reflection

- Bad results. Evaluation is now validated against a reproduced CoDA baseline,
  so point-wise attention weights are probably not helping in this form.
- Next step: try a less destructive variant only with a clear motivation, such
  as frozen-encoder spatial routing, gradual interpolation from mean to spatial
  weights, or scalar per-slot weights.

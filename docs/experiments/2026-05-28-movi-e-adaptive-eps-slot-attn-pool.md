# MOVi-E adaptive epsilon composition -- pooled slot-attention weights

**Status:** FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23193306.log
**Date:** 2026-05-28T23:39Z
**Slurm job:** 23193306 (script: jobs/movi_e_adaptive_eps_slot_attn_pool_warmstart_train_eval.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23193306.log)
**Node / GPUs:** gcn95, 2x H100
**wandb run:** https://wandb.ai/nan-team/latent_decomposed_diffusion_adaptive_eps/runs/gssfwz33

## Purpose

Second adaptive epsilon-composition trial from ROADMAP.md. Reuses encoder Slot
Attention masks only to estimate one scalar contribution weight per slot:
attention maps are averaged over spatial dimensions, normalized over slots,
detached, and broadcast over the UNet epsilon grid. This keeps adaptive
empty-slot suppression without letting point-wise mask routing dominate the
denoising objective.

The run warm-starts model weights from the CoDA K/V-only checkpoint but writes
to a separate output folder and trains for a fresh 50k steps.

## Configuration

| Item | Value |
|------|-------|
| Warm-start checkpoint | results/movi-e_coda_kv_only/latent_decomposed_diffusion/checkpoint-200000-last |
| Output dir | results/movi-e_adaptive_eps_slot_attn_pool_warmstart/latent_decomposed_diffusion_adaptive_eps/ |
| wandb project | nan-team/latent_decomposed_diffusion_adaptive_eps |
| Encoder | DinoSlotAttentionEncoder d1024, DINOv3 frozen, K=24, R=4 |
| UNet | SD2.1 pretrained, frozen except cross-attn K/V |
| Epsilon composition | detached Slot Attention masks pooled to scalar per-slot weights |
| Steps run | 50000 fresh continuation steps |
| Effective batch | 16 (2 GPU x 8) |
| Resolution | 256 |
| Dataset | MOVi-E train shards (233976 frames) |
| Eval split | MOVi-E validation shards (6000 frames) |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | FAIL (rc=1) | 04:26:07 |
| Reconstruction eval | PASS | 00:07:34 |
| Object-centric metrics | PASS | included above |

### Object-centric metrics

| Metric | Value |
|--------|-------|
| FG-ARI | 0.0906 |
| mBO | 0.0846 |
| mIoU | 0.1280 |
| Frames | 6000 |

Reference points:

| Run | FG-ARI | mBO | mIoU |
|-----|--------|-----|------|
| CoDA mean-weight baseline (23192654) | 0.5160 | 0.3451 | 0.3420 |
| Spatial slot-attn adaptive, corrected eval (23192429) | 0.0610 | 0.0792 | 0.1197 |
| This pooled slot-attn adaptive run | 0.0906 | 0.0846 | 0.1280 |

Full metrics: results/movi-e_adaptive_eps_slot_attn_pool_warmstart/metrics/metrics.json
Attention-mask viz: results/movi-e_adaptive_eps_slot_attn_pool_warmstart/metrics/viz_*.jpg
Final checkpoint: results/movi-e_adaptive_eps_slot_attn_pool_warmstart/latent_decomposed_diffusion_adaptive_eps/checkpoint-50000-last
Loss + slot-pairwise-cos curves: results/movi-e_adaptive_eps_slot_attn_pool_warmstart/loss_curve_23193306.png
Reconstruction grids: results/movi-e_adaptive_eps_slot_attn_pool_warmstart/eval_grids/image_NN.jpg
Per-step validation viz: results/movi-e_adaptive_eps_slot_attn_pool_warmstart/latent_decomposed_diffusion_adaptive_eps/logs/

## Assessment

Pending review after the Slurm job completes.

# COCO CoDA-style K/V-only training -- 200k-step run

**Status:** FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23177437.log
**Date:** 2026-05-28T19:15Z
**Slurm job:** 23177437 (script: jobs/coco_coda_kv_only_train.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23177437.log)
**Node / GPUs:** gcn74, 2x H100
**wandb run:** https://wandb.ai/nan-team/latent_decomposed_diffusion/runs/qrj79ksy

## Purpose

First COCO training run for this codebase. Same CoDA-style recipe as the
MOVi-E K/V-only baseline (`2026-05-27-movi-e-coda-kv-only.md`): frozen
SD2.1 UNet except for cross-attn K/V projections, DINOv3 slot encoder
with 4 register slots and latent_dim=1024. Tests whether the
object-centric encoder + frozen denoiser prior we developed on synthetic
MOVi-E transfers to natural images.

Object-discovery metrics (FG-ARI / mBO / mIoU) are reported in
`2026-05-28-coco-coda-kv-only-eval.md`. Compositional FID/KID is still
open -- see ROADMAP "Compositional image generation metrics (planned)".

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention + 4 register slots) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| Patch grid | 16x16 at 256 input (patch_size=16) |
| Slots (num_components, K) | 7 |
| Register slots (R) | 4 |
| Slot dim (latent_dim) | 1024 (matches SD2.1 cross_attention_dim) |
| UNet | sd2-community/stable-diffusion-2-1 (pretrained, frozen except K/V) |
| Trainable UNet params | cross-attn to_k + to_v only |
| Scheduler | SD2.1 DDPM (loaded from --pretrained_model_name/subfolder=scheduler) |
| Loss | mean-of-eps composition (eps = mean_k eps_slot_k, registers ride along) |
| Steps run | 200000 / 200000 configured |
| Effective batch | 16 (2 GPU x 8) |
| Resolution | 256 (32x32 latent vs SD2.1's native 64x64) |
| Mixed precision | bf16 |
| Attention backend | PyTorch SDPA (FlashAttention-3 on H100) |
| torch.compile | inductor (--dynamo_backend=inductor) |
| Learning rate | 2.0e-5 |
| Dataset | COCO 2017 train (118287 images, val=5000) |
| Output dir | results/coco_coda_kv_only/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | FAIL (rc=1) | 07:06:48 |

Final checkpoint: results/coco_coda_kv_only/latent_decomposed_diffusion/checkpoint-200000-last
Loss + slot-pairwise-cos curves: results/coco_coda_kv_only/loss_curve_23177437.png

## Notes

- K=7 is a placeholder for COCO scene complexity; revise based on
  collapse / binding diagnostics from this run.
- The job reached the configured 200k steps and wrote
  `checkpoint-200000-last`; the nonzero exit was a final DDP barrier timeout.
- Follow-up object metrics were collected by
  `jobs/coco_coda_kv_only_eval_latest.sh` in Slurm job 23193898.

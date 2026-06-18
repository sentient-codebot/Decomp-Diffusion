# COCO CoDA-style K/V-only from-scratch -- 500k-step run

**Status:** FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23465498.log
**Date:** 2026-06-05T06:25Z
**Slurm job:** 23465498 (script: jobs/coco_coda_kv_only_train_eval.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23465498.log)
**Node / GPUs:** gcn102, 2x H100
**wandb run:** https://wandb.ai/nan-team/latent_decomposed_diffusion/runs/b8tfkllv

## Purpose

Fresh 500k-step COCO CoDA-style run from pretrained SD2.1 and randomly
initialized DINOv3 slot/register components. Same recipe as the MOVi-E
K/V-only baseline (`2026-05-27-movi-e-coda-kv-only.md`):
frozen SD2.1 UNet except for cross-attn K/V projections, DINOv3 slot encoder
with 4 register slots and latent_dim=1024. Tests whether a clean 500k-step
trajectory improves natural-image slot binding relative to the earlier
continued COCO run.

Object-discovery metrics (FG-ARI / mBO / mIoU) are computed in this job
from COCO val2017 instance annotations. Compositional FID/KID is still open
-- see ROADMAP "Compositional image generation metrics (planned)".

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
| Steps target | 500000 configured (from scratch; no resume) |
| Effective batch | 16 (2 GPU x 8) |
| Resolution | 256 (32x32 latent vs SD2.1's native 64x64) |
| Mixed precision | bf16 |
| Attention backend | PyTorch SDPA (FlashAttention-3 on H100) |
| torch.compile | inductor (--dynamo_backend=inductor) |
| Learning rate | 2.0e-5 |
| Dataset | COCO 2017 train (118287 images, val=5000) |
| Eval split | COCO 2017 val2017 (5000 images) |
| Output dir | results/coco_coda_kv_only/scratch_500k_23465498/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | FAIL (rc=1) | 17:17:27 |
| Reconstruction eval | PASS | 00:08:18 |
| COCO object metrics | PASS | (incl. above) |

### Object-centric metrics (eval_coco.py, val2017)

| Metric | Value |
|--------|-------|
| FG-ARI | 0.3640 |
| mBO | 0.2546 |
| mIoU | 0.2472 |
| mIoU foreground-only | 0.2406 |
| Object-weighted mBO | 0.1621 |
| Images | 5000 |
| Objects | 31672 |
| Slot entropy | 1.0000 |

Full metrics: results/coco_coda_kv_only/scratch_500k_23465498/coco_metrics_23465498/metrics.json
Attention-mask viz: results/coco_coda_kv_only/scratch_500k_23465498/coco_metrics_23465498/viz_*.jpg

### Reconstruction grids and curves

Final checkpoint: results/coco_coda_kv_only/scratch_500k_23465498/latent_decomposed_diffusion/checkpoint-500000-last
Loss + slot-pairwise-cos curves: results/coco_coda_kv_only/scratch_500k_23465498/loss_curve_23465498.png
Reconstruction grids: results/coco_coda_kv_only/scratch_500k_23465498/eval_grids/image_NN.jpg
Per-step validation viz: results/coco_coda_kv_only/scratch_500k_23465498/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics + viz:
     - Did the clean 500k run improve COCO binding metrics?
     - Did slot_pairwise_cos stay low (slots specialise) or trend toward 1?
     - Do reconstruction grids show object-specific slots or texture/background collapse? -->

## Notes

- K=7 is a placeholder for COCO scene complexity; revise based on
  collapse / binding diagnostics from this run.
- The final DDP barrier can time out after the last checkpoint is written;
  this script still runs eval against the latest checkpoint it can find.
- Metrics are computed from COCO val2017 instance polygons. Crowd/RLE
  annotations are skipped by `eval_coco.py`.

# MOVi-E CoDA-style K/V-only training -- 200k-step run

**Status:** FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23151491.log
**Date:** 2026-05-28T09:52Z
**Slurm job:** 23151491 (script: jobs/movi_e_coda_kv_only_train_eval.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23151491.log)
**Node / GPUs:** gcn88, 2x H100
**wandb run:** https://wandb.ai/nan-team/latent_decomposed_diffusion/runs/joxslpp6

## Purpose

CoDA-style ablation against the
`2026-05-26-movi-e-dinov3-registers.md` baseline: swap the
trained-from-scratch UNet for a frozen pretrained SD 2.1 UNet, train
only the cross-attention `to_k` / `to_v` projections (warm-started
from SD's text-conditioned weights), and let all gradient pressure flow
through K/V into the slot encoder. Tests whether a strong fixed denoiser
prior pushes the encoder to learn better object-level representations.

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention + 4 register slots) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| Patch grid | 16x16 at 256 input (patch_size=16) |
| Slots (num_components, K) | 24 |
| Register slots (R) | 4 |
| Slot dim (latent_dim) | 1024 (matches SD2.1 cross_attention_dim) |
| UNet | sd2-community/stable-diffusion-2-1 (pretrained, frozen except K/V) |
| Trainable UNet params | cross-attn to_k + to_v only (warm-started from SD text-conditioning weights) |
| Scheduler | SD2.1 DDPM (loaded from --pretrained_model_name/subfolder=scheduler) |
| Loss | sum-of-eps composition (eps = sum_k eps_slot_k, registers ride along) |
| Steps run | 200000 / 200000 configured |
| Effective batch | 16 (2 GPU x 8) |
| Resolution | 256 (32x32 latent vs SD2.1's native 64x64) |
| Mixed precision | bf16 |
| Learning rate | 2.0e-5 |
| Dataset | MOVi-E train split (233976 frames) |
| Eval split | MOVi-E validation split (6000 frames) |
| Output dir | results/movi-e_coda_kv_only/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | FAIL (rc=1) | 16:56:11 |
| Reconstruction eval | PASS | 00:10:07 |
| Object-centric metrics | PASS | (incl. above) |

### Object-centric metrics (eval_movi.py, validation split)

| Metric | CoDA K/V-only | DINOv3 + R=4 (full UNet) |
|--------|---------------|--------------------------|
| FG-ARI | 0.5032 | see 2026-05-26-movi-e-dinov3-registers.md |
| mBO    | 0.3373   | see 2026-05-26-movi-e-dinov3-registers.md |
| Frames | 6000 | -- |

Full metrics: results/movi-e_coda_kv_only/metrics/metrics.json
Attention-mask viz: results/movi-e_coda_kv_only/metrics/viz_*.jpg

### Reconstruction grids and curves

Final checkpoint: results/movi-e_coda_kv_only/latent_decomposed_diffusion/checkpoint-200000-last
Loss + slot-pairwise-cos curves: results/movi-e_coda_kv_only/loss_curve_23151491.png
Reconstruction grids: results/movi-e_coda_kv_only/eval_grids/image_NN.jpg
Per-step validation viz: results/movi-e_coda_kv_only/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics + viz:
     - Did the frozen-denoiser prior improve slot quality (FG-ARI / mBO up)?
     - Did slot_pairwise_cos stay low (slots specialise) or trend toward 1?
     - Did val_loss flatten earlier than the full-UNet baseline (capacity
       bottleneck) or track it (encoder is the limiting factor in both)? -->

## Notes

- K/V are warm-started from SD2.1's CLIP-text-conditioned weights. This
  is a deliberate choice (see TECHDEBT.md) rather than a principled one;
  a re-init ablation could disentangle "frozen denoiser prior" from
  "text-style K/V prior."
- 256px input means the SD2.1 UNet runs on a 32x32 latent vs its native
  64x64. SD UNets handle non-native sizes fine but the pretraining
  prior is strongest at 64x64.

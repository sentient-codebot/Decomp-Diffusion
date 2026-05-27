# MOVi-E CoDA-style K/V-only frozen UNet -- 200k-step run

**Status:** RUNNING (resubmitted after design correction)
**Date:** 2026-05-27T13:43Z
**Slurm job:** 23146118 (script: jobs/movi_e_coda_kv_only_train_eval.sh, log: slurm_23146118.log)
**Node / GPUs:** gpu_h100, 2x H100
**wandb run:** _to be filled in once training starts_

## Purpose

First "CoDA-style" run: freeze the SD 2.1 UNet except for cross-attention
`to_k` / `to_v` projections, and ask the DINOv3 slot encoder to do all the
representational work. Same dataset, step budget, and encoder family as the
DINOv3 + registers run (`docs/experiments/2026-05-26-movi-e-dinov3-registers.md`);
the contrast is purely about decoder capacity.

## Design correction: mean vs sum eps composition

The originally launched job (Slurm `23139442`) used the existing sum-of-eps
training target (`eps_pred = sum_k eps_slot_k`) inherited from earlier
full-UNet runs. With a mostly-frozen pretrained denoiser this is wrong: SD
2.1 was trained to predict ONE noise; asking K=24 of its forward passes to
sum to one noise implicitly demands each per-slot prediction shrink by 1/K.
The only trainable knobs on the denoiser side are K/V projections — they
cannot rescale the network output. Magnitude alone makes the objective
unlearnable through the frozen path.

Switched to the mean target (`eps_pred = mean_k eps_slot_k`) before the
relaunch. Under mean each per-slot eps stays on the unit-noise scale the
pretrained UNet already produces; the average matches one noise. The change
also realigns sampling: per-slot decode now uses `guidance_scale=1.0`
instead of `guidance_scale=K` (no rescale needed when one slot already
represents the mean's K=1 case).

For from-scratch training runs the choice is roughly a learning-rate
rescale (gradients differ by 1/K), so this correction matters mainly here
and for future frozen-denoiser experiments. Going forward mean is the
project-wide default in `compose_eps` and the pipeline.

The sum-objective job ran ~3h and produced `checkpoint-20000` before being
cancelled. Aborted run dir preserved at
`results/movi-e_coda_kv_only_sum_aborted_20260527T134321Z/`. wandb run for
the aborted attempt: `latent_decomposed_diffusion/runs/joxslpp6`
("polished-sun-11").

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention + 4 register slots) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| Patch grid | 16x16 at 256 input (patch_size=16) |
| Slot Attention iters | 3 |
| Slots (num_components, K) | 24 |
| Register slots (R) | 4 |
| Slot dim (latent_dim) | 1024 (matches SD 2.1 `cross_attention_dim`) |
| UNet | sd2-community/stable-diffusion-2-1, frozen except cross-attn `to_k` / `to_v` (`--freeze_unet_except_kv`) |
| Loss | mean-of-eps: `eps_pred = mean_k eps_slot_k`, registers concatenated to every slot's conditioning |
| Steps configured | 200000 |
| Effective batch | 16 (2 GPU x 8) |
| Resolution | 256 |
| Mixed precision | fp16 |
| Dataset | MOVi-E train split (233976 frames) |
| Eval split | MOVi-E validation split (6000 frames) |
| Output dir | results/movi-e_coda_kv_only/latent_decomposed_diffusion/ |

## Results

_to be filled in when the run completes_

# MOVi-E adaptive epsilon composition -- slot-attention weights

**Status:** planned / pending Slurm run
**Date:** 2026-05-28
**Slurm job:** pending (script: `jobs/movi_e_adaptive_eps_slot_attn_warmstart_train_eval.sh`)
**wandb run:** pending

## Purpose

First adaptive epsilon-composition baseline from `ROADMAP.md`: reuse encoder
Slot Attention masks as spatial composition weights, interpolate them to the
UNet epsilon resolution, and stop gradients through the weights with `detach()`.
This tests whether per-location slot competition improves decomposition over
uniform mean-of-eps composition.

## Planned configuration

| Item | Value |
|------|-------|
| Warm start | latest checkpoint under `results/movi-e_coda_kv_only/latent_decomposed_diffusion/` |
| Output dir | `results/movi-e_adaptive_eps_slot_attn_warmstart/latent_decomposed_diffusion_adaptive_eps/` |
| Training budget | fresh 50k steps |
| Encoder / slots | DinoSlotAttentionEncoder d1024, K=24 object slots, R=4 registers |
| UNet | SD2.1 pretrained, frozen except cross-attention K/V |
| Epsilon composition | detached Slot Attention masks, bilinear interpolation to latent resolution |
| Dataset | MOVi-E WDS shards |

## Results

Pending Slurm completion. The job script overwrites this stub with the final
checkpoint, wandb URL, metrics, and reconstruction-grid paths.

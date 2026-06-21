# Technical debt

Known shortcuts, divergences, and deferred fixes. Add an entry when you discover or
introduce debt; remove it when resolved. For longer-term direction see `ROADMAP.md`.

## UNet positional embedding silently dropped

`configs/celebahq/unet/config.json` declares `_class_name: "UNet2DConditionModelWithPos"` and `pos_type: "cartesian"`, but `train_lsd.py` and `eval.py` construct a plain `diffusers.UNet2DConditionModel` via `from_config`. `from_config` ignores the unknown `pos_type` key (logged at startup as an ignored config attribute), so the cartesian positional embedding is never applied. `UNet2DConditionModelWithPos` (`src/models/unet_with_pos.py`) is not imported anywhere.

The pipeline runs without error, so the divergence is silent — but the trained UNet lacks the positional embedding the config implies, which is architecturally relevant to slot-based decomposition quality.

**Fix:** if the positional embedding is intended, import `UNet2DConditionModelWithPos` in `train_lsd.py` / `eval.py` and build the UNet with it instead of the plain class; verify the save/load hooks still line up (`model._get_name().lower()` → checkpoint subfolder name). Otherwise, drop the `_class_name` / `pos_type` keys from the config so it matches the code.

## tensorboard broken by setuptools 82 (`pkg_resources` removed)

`tensorboard` 2.20.0 still does `import pkg_resources` (`tensorboard/default.py:30`), but `setuptools` 81 deprecated and 82 removed the bundled `pkg_resources` module. The lock currently resolves setuptools to 82.0.1 (pulled in transitively by `tensorboard` and `torch`), so `uv run tensorboard ...` crashes at startup with `ModuleNotFoundError: No module named 'pkg_resources'`.

**Fix:** pin setuptools below 81 so `pkg_resources` is restored. Add to `pyproject.toml`:

```toml
[tool.uv]
constraint-dependencies = ["setuptools<81"]
```

then re-sync (`uv sync --extra wandb --extra tensorboard --extra xformers`). Alternatively, drop tensorboard entirely and standardise on wandb (`--report_to wandb`), which is the preferred logging backend going forward.

## H100 throughput knobs -- benchmarks pending

The low-risk pass (bf16 + `allow_tf32: true` + `dataloader_num_workers=8`) has landed in `configs/{celebahq,movi-e}/train_config.yaml` (commits f3794ae, 8fecab7, 1d55402). Remaining items:

- **xformers MEA -> PyTorch SDPA** -- `enable_xformers_memory_efficient_attention` is now `false` in both configs, so diffusers falls back to `AttnProcessor2_0` (calls `torch.nn.functional.scaled_dot_product_attention`); on H100 + PyTorch 2.5+ that auto-dispatches to FlashAttention-3 for bf16/fp16. A/B benchmark queued: `jobs/movi_e_attn_sdpa_smoketest.sh` runs xformers vs SDPA back-to-back on a single H100. If SDPA loses, override per-job with `--enable_xformers_memory_efficient_attention`.
- **`torch.compile` / dynamo** -- not yet adopted in any long run. Benchmark queued: `jobs/movi_e_attn_sdpa_dynamo_smoketest.sh` runs SDPA vs SDPA+`--dynamo_backend=inductor`. Expect ~1.3-1.8x on diffusion training, but compile is paid up-front and can surface dynamic-shape recompilations in the slot-attention path -- inspect the slurm log for `TorchDynamo` warnings before trusting the number.

VRAM headroom on the 128-res runs is also unused (per-GPU batch 16-32 on 80 GB H100), but raising that interacts with effective-batch / LR comparability across runs, so handle it separately from the throughput knobs above.

**Fix:** run both smoketests; if either backend wins, fold the change into the long jobs (configs already default to SDPA; dynamo would be a per-job `--dynamo_backend=inductor` flag on `accelerate launch`). Write the result up under `docs/experiments/` and remove the resolved bullet here.

## Capacity / resolution gap vs Nguyen et al. 2026 baseline

When comparing the register-slot results in this repo against Nguyen et al. 2026 (*Improved Object-centric Diffusion* -- our primary reference for the register-slot direction), keep in mind that their setup is significantly heavier than ours on several axes:

| Axis | Nguyen et al. 2026 | This repo (movi-e dinov3) |
|------|-------------------|--------------------------|
| Image resolution | 512 x 512 | 256 x 256 |
| Slot-attention input feature grid | 32 x 32 | 16 x 16 |
| Slot hidden dim | 768 | 64 |
| Register count R | 77 (frozen CLIP-text) | 4 (learned) |
| Object slots K (MOVi-E) | 24 | 24 (now matched) |

So even with matching K and the same architectural shape, this repo's slots have ~12x less capacity (64 vs 768) and operate on a 4x coarser feature grid. Their results are not a fair upper bound for what we should expect at our current settings -- any large gap in FG-ARI / mBO should first be attributed to capacity / resolution before concluding the method itself is the bottleneck.

**Fix:** before chasing SOTA numbers, raise `latent_dim` (64 -> 256 or 768), increase the feature grid (use a smaller patch / higher input resolution so the ViT produces a 32 x 32 grid), and switch to frozen-CLIP register priors. Until then, document this gap in every comparison report.

## CoDA K/V warm-started from CLIP-text projections

The `--freeze_unet_except_kv` mode (`jobs/movi_e_coda_kv_only_train_eval.sh`) freezes the SD 2.1 UNet and trains only cross-attention `to_k` / `to_v`. Those projections were pretrained on CLIP-text embeddings, which live in a totally different distribution than the DINOv3 slot tokens we feed them. Reusing them as a warm start is a deliberate convenience -- they're the right shape and presumed-better-than-random -- not a principled choice. The slot-token K/V subspace may have nothing useful in common with the text K/V subspace, in which case the warm start is wasted (or worse: a worse-than-random init biased toward text-like decoding).

**Fix / follow-up:** add a re-init ablation (Kaiming or zero-init the K/V before training) and compare against the warm-start variant. Until that is run, treat the warm-start choice as load-bearing and re-flag this in the experiment report.

## Train loss is logged after gradient-accumulation scaling

`train_lsd.py` divides `loss` by `args.gradient_accumulation_steps` before
calling `accelerator.backward(loss)`, then logs that already-scaled tensor as
`loss`. This makes the W&B train-loss curves numerically depend on the
accumulation factor. For example, the MOVi-E CoDA K/V-only 512 job uses
`PER_GPU_BATCH=2` and `GRAD_ACCUM=4` to keep the effective batch at 16, whereas
the 256 job uses `PER_GPU_BATCH=8` and accumulation 1. The 512 logged train
loss is therefore about 4x lower purely from logging scale, before considering
any real resolution effect.

`val_loss` does not have this accumulation division and is the better raw-MSE
comparison, but note that the training-time `val_dataset` is currently carved
from `--dataset_root` rather than the MOVi-E validation shard root. With
`train_split_portion=1.0`, it uses the last 10% of the training shards for
that cheap diffusion-loss diagnostic.

**Fix:** log both the optimization loss and an unscaled `train_loss_mse`
(`loss * gradient_accumulation_steps`, or the pre-divide value) so curves remain
comparable across memory-driven accumulation changes. If the metric is meant to
be validation loss, add an explicit validation dataset root instead of deriving
it from the train root.

## MOVi-E 512 runs upsample 256px source frames

`jobs/movi_e_coda_kv_only_512_train_eval.sh` sets `--resolution 512` and uses
`configs/movi-e/dinov3_slot_encoder_d1024_512/config.json`, but the MOVi-E WDS
source frames are the TFDS 256 x 256 images. The dataset transform resizes
those frames to 512 x 512 before DINOv3 and VAE encoding. So the 512 run tests
a denser DINOv3 patch grid (32 x 32 instead of 16 x 16) and a native-size SD
latent grid (64 x 64 x 4 instead of 32 x 32 x 4), not access to higher-fidelity
MOVi-E renders.

**Fix / follow-up:** keep reports explicit that this is an upsampled-resolution
comparison. Do not interpret improved loss, reconstruction quality, or
object-centric metrics as evidence from higher-resolution source data unless a
true higher-resolution MOVi-E render pipeline is added.

## Adaptive epsilon warm-start confound

The first adaptive epsilon runs, especially
`jobs/movi_e_adaptive_eps_slot_attn_warmstart_train_eval.sh` and
`jobs/movi_e_adaptive_eps_slot_attn_pool_warmstart_train_eval.sh`, warm-started
from the MOVi-E CoDA K/V-only checkpoint and then changed the epsilon
composition objective. Because both the starting point and objective changed,
the resulting collapse cannot be attributed to adaptive weights alone. The
current working suspicion is that warm-starting an already-specialized
mean-eps baseline into a different routing objective is itself a collapse
trigger.

**Fix / follow-up:** before drawing conclusions from adaptive epsilon weights,
run MOVi-E controls that remove the warm-start confound: train the same
adaptive composition from scratch, or introduce the new routing gradually while
freezing or lowering the LR on the encoder. Keep COCO secondary until the
MOVi-E behavior is understood.

## DDP cleanup timeout after main-process-only final work

`train_lsd.py` runs checkpointing and validation only on `accelerator.is_main_process`, but the other DDP ranks continue through the loop and then wait at the final `accelerator.wait_for_everyone()`. On long multi-GPU runs this can leave non-main ranks waiting in a NCCL collective while rank 0 is still doing final validation, checkpointing, or tracker shutdown. The MOVi-E CoDA K/V-only run hit this at the end of training: Slurm job 23151491 saved `results/movi-e_coda_kv_only/latent_decomposed_diffusion/checkpoint-200000-last`, then rank 1 timed out after 605s in NCCL and the wrapper reported `training finished (rc=1)`. The checkpoint is usable, but the job status is noisy and downstream eval can be cut short.

**Fix:** make all ranks enter matching synchronization points around periodic/final checkpointing and validation. In particular, avoid letting non-main ranks run ahead while rank 0 performs validation, and make final save/tracker shutdown return cleanly after `checkpoint-*-last` is written. Add a short multi-GPU smoketest that reaches a validation/checkpoint step and exits with rc=0.

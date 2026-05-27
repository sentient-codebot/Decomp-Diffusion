# Technical debt

Known shortcuts, divergences, and deferred fixes. Add an entry when you discover or
introduce debt; remove it when resolved. For longer-term direction see `ROADMAP.md`.

## Denoising target changed from mean to sum

The training composition is `eps_pred = sum_k eps_slot_k` (in both `train_lsd.py` and `composable_stable_diffusion_pipeline.py`) rather than the previous `mean_k eps_slot_k`. The sum form is the simplest compositional objective and matches the original Decomp-Diffusion paper; the mean form silently rescales the target by `1/K`. Registers (when present) are still concatenated to each slot's conditioning sequence, but there is no separate registers-only forward / `(1 - K) * eps_uncond` term -- that earlier sum-of-deltas variant was dropped because the per-slot decode it implied was underdetermined (see chat history 2026-05-26).

**Implication:** existing checkpoints trained under the old mean aggregation or the sum-of-deltas variant are NOT inference-compatible with the current pipeline; re-train rather than mixing.

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

## H100 throughput knobs left on the table

Current MOVi-E training (default 2x H100, e.g. `jobs/movi_e_dinov3_registers_train_eval.sh`) is using `mixed_precision: fp16` + xformers MEA, with `allow_tf32`, `torch.compile` and bf16 all unused. On Hopper this leaves measurable throughput unclaimed:

- **`fp16` instead of `bf16`** -- H100 has identical TFLOPs for both, but bf16's wider exponent range removes the loss-scaler / NaN risk that fp16 + diffusion training is known for. `src/parser.py:231` already exposes `bf16`; flipping `mixed_precision` in `configs/{celebahq,movi-e}/train_config.yaml` is the only change needed.
- **`allow_tf32` not set** -- wired through (`train_lsd.py:548-551`) but no job passes `--allow_tf32`, so the non-AMP matmuls (encoder, slot-attention master weights) miss out.
- **No `torch.compile`** -- UNet + slot-attention are well-suited to it on H100 (cudagraphs + Triton); typical 1.3-1.8x on diffusion training.
- **xformers MEA over PyTorch SDPA** -- on H100, SDPA selects FlashAttention-3, which is usually faster than xformers' MEA. Worth A/B-ing and dropping `enable_xformers_memory_efficient_attention` if SDPA wins.
- **`dataloader_num_workers=4`** -- low for the 256-res DINOv3 path; bump to 8-12 + `persistent_workers=True` to keep both H100s fed.

VRAM headroom on the 128-res runs is also unused (per-GPU batch 16-32 on 80 GB H100), but raising that interacts with effective-batch / LR comparability across runs, so handle it separately from the throughput knobs above.

**Fix:** land bf16 + `allow_tf32: true` + worker bump as a single low-risk pass, validated on a smoketest. Benchmark `torch.compile(unet)` and SDPA-vs-xformers separately before committing them to the long runs.

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

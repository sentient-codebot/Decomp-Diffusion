# Technical debt

Known shortcuts, divergences, and deferred fixes. Add an entry when you discover or
introduce debt; remove it when resolved. For longer-term direction see `ROADMAP.md`.

## `num_registers=0` denoising target changed from mean to sum

After the register-slot refactor, the training composition for `num_registers=0` falls back to `eps_pred = sum_k eps_slot_k` (in both `train_lsd.py` and `composable_stable_diffusion_pipeline.py`) rather than the previous `mean_k eps_slot_k`. With the new compositional objective `(1 - K) * eps_uncond + sum_k eps_slot_k`, the sum form is the natural `R == 0` limit (drop the unconditional term); the mean form would silently rescale the target by `1/K` relative to the new objective, so the divergence is deliberate.

**Implication:** legacy configs that still set `num_registers=0` (or omit the key) will train with a different loss scale than prior runs. Existing checkpoints trained under the old mean aggregation are NOT inference-compatible with the new pipeline; re-train rather than mixing.

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

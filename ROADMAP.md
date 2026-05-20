# Roadmap

Longer-term direction: bigger changes, methodological shifts, planned architecture work.
For known shortcuts and deferred fixes, see `TECHDEBT.md`.


## Setup validation

**Goal:** confirm the current end-to-end setup actually works before building on it.

- Run a **full training run** (not just the celebahq smoke test) with the current `LatentEncoder` baseline.
- Run **evaluation** on the resulting checkpoint and inspect the metrics and decomposition visualizations.
- Decide from the results whether the baseline is sound enough to build on, or whether anything in `TECHDEBT.md` needs fixing first.

This validation is the prerequisite for the slot-extraction work below — it establishes the baseline that the encoder + slot-attention version is compared against.

## Slot extraction: encoder + slot attention

**Goal:** move slot extraction to a **(trained or pretrained) encoder — e.g. DINO — followed by a slot attention module**. This is the intended object-centric design and the canonical Latent Slot Diffusion pattern: encoder → feature map → slot attention → slots.

**Current state:** the `LatentEncoder` (`src/models/encoder.py`) is a plain CNN + `Flatten` + `Linear` mapping raw pixels → K slot vectors, with **no slot attention**. It is kept deliberately as:
- a temporary solution to confirm the end-to-end training/eval setup works, and
- a naive baseline to compare the future encoder + slot-attention version against.

So the current encoder is **not** debt to rip out — it is an intentional placeholder/baseline.

**How to land it:** add the new slot encoder as a *selectable alternative* (pretrained encoder, e.g. DINO, + a `SlotAttention` module) rather than replacing `LatentEncoder` outright; keep `LatentEncoder` available as the baseline. The earlier `UNetEncoder` backbone (removed in 178578b, recoverable from git history) was scaffolding toward a swappable encoder but had no slot-attention downstream — the new design supersedes it.

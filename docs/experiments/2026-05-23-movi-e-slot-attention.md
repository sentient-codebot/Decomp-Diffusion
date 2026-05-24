# MOVi-E slot-attention encoder -- 200k-step run

**Status:** PASS -- full run completed, train + both evals clean
**Date:** 2026-05-24T00:14Z
**Slurm job:** 23071353 (script: jobs/movi_e_slot_attn_train_eval.sh, log: slurm_23071353.log)
**Node / GPUs:** gcn112, 4x H100
**wandb run:** https://wandb.ai/nan-team/latent_decomposed_diffusion/runs/0owebvjs

## Purpose

First run on MOVi-E (Kubric, up to 23 rigid objects with per-pixel
instance masks). Lets the slot decomposition be measured with proper
object-centric metrics instead of only qualitative grids -- the
`SlotAttentionEncoder` (image -> CNN -> soft pos-embed -> Slot Attention)
binds the feature map into 11 slots, and the per-slot attention masks are
compared against the dumped GT instance segmentations.

## Configuration

| Item | Value |
|------|-------|
| Encoder | SlotAttentionEncoder (CNN + soft pos-embed + Slot Attention) |
| Slot Attention iters | 3 |
| Steps run | 200000 / 200000 configured |
| Effective batch | 64 (4 GPU x 16) |
| Resolution | 128 |
| Slots (num_components) | 11 |
| Slot dim (latent_dim) | 64 |
| Mixed precision | fp16 |
| Learning rate | 2.0e-5 |
| Dataset | MOVi-E train split (233976 frames) |
| Eval split | MOVi-E validation split (6000 frames) |
| Output dir | results/movi-e_slot/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | PASS | 10:19:07 |
| Reconstruction eval | PASS | 00:07:09 |
| Object-centric metrics | PASS | (incl. above) |

### Object-centric metrics (eval_movi.py, validation split)

| Metric | Value |
|--------|-------|
| FG-ARI | 0.0420 |
| mBO    | 0.0351 |
| Frames | 6000 |

Full metrics: results/movi-e_slot/metrics/metrics.json
Attention-mask viz: results/movi-e_slot/metrics/viz_*.jpg

### Reconstruction grids

Final checkpoint: results/movi-e_slot/latent_decomposed_diffusion/checkpoint-200000-last
Loss curve: results/movi-e_slot/loss_curve_23071353.png
Reconstruction grids: results/movi-e_slot/eval_grids/image_NN.jpg -- each row is
[input | slot 0 | ... | slot 10 | reconstruction]
Per-step validation viz: results/movi-e_slot/latent_decomposed_diffusion/logs/

## Assessment

The training loop is clean: 200k steps in ~10h on 4x H100, final train MSE
0.099, no instability across the run (see `loss_curve_*.png`). Reconstruction
eval also runs end-to-end. So the pipeline -- dataset, training, both eval
paths -- is now wired up on MOVi-E.

The decomposition numbers are weak: **FG-ARI 0.042 / mBO 0.035**, well below
the 0.3-0.5 band that a from-scratch CNN + Slot Attention typically lands at
on MOVi-E in the OCL literature. Taken together with the non-trivial train
loss, the most likely read is that the diffusion objective alone is not
pushing the slot attention to bind to object-aligned regions -- slots are
partitioning the feature map but not by object identity. The viz grids under
`results/movi-e_slot/metrics/viz_*.jpg` should be inspected to confirm
whether the masks are spatially arbitrary or have collapsed onto a handful
of slots.

Contributing factors that are expected to depress the number relative to a
direct slot-attention auto-encoder baseline:

- The encoder is supervised only through the diffusion reconstruction loss
  (no explicit reconstruction-from-slots regulariser like the standard
  slot-attention training recipe).
- 11 slots vs up to 23 GT objects per scene caps the achievable FG-ARI/mBO.
- 128x128 resolution -> 16x16 feature map -> bilinearly upsampled mask
  (significant aliasing at object boundaries).

This is the baseline for the next roadmap step (DINO encoder + slot
attention): the numbers establish where the from-scratch CNN sits before a
pretrained feature extractor is brought in.

## Notes

- Eval uses pre-renormalisation slot-attention competition weights as soft
  masks; argmax over slots gives the predicted hard mask. Masks live at
  feature-map resolution (16x16 for 128 input) and are bilinearly upsampled
  to 128 before comparison with GT.
- 11 slots vs up to 23 GT objects: some scenes have more objects than
  slots, capping the achievable FG-ARI/mBO.
- eval.py writes grids to ./image_test_output/ (hardcoded); they are copied
  to results/movi-e_slot/eval_grids/ here. See TECHDEBT.md.

## Next steps

- Review the metrics + viz, fill in the Assessment section.
- Roadmap next stage: replace the CNN feature extractor with a pretrained
  encoder (e.g. DINO) -- see ROADMAP.md "Encoder: pretrained feature
  extractor".

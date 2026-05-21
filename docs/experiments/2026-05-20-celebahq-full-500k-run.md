# CelebA-HQ full 500k-step run

**Status:** PASS -- full run completed, train and eval clean
**Date:** 2026-05-21T14:57Z
**Slurm job:** 22976095 (script: jobs/celebahq_train_eval_full.sh, log: slurm_22976095.log)
**Node / GPUs:** gcn113, 2x H100

## Purpose

The full CelebA-HQ training run with the LatentEncoder baseline, following
the 50k-step pipeline validation
(docs/experiments/2026-05-20-celebahq-train-eval-validation.md). This is the
run the roadmap "Setup validation" point calls for: it produces the baseline
that the planned encoder + slot-attention version is compared against.

## Configuration

| Item | Value |
|------|-------|
| Steps run | 500000 / 500000 configured |
| Effective batch | 64 (2 GPU x 32) |
| Resolution | 128 |
| Slots (num_components) | 4 |
| Mixed precision | fp16 |
| Learning rate | 2.0e-5 |
| Output dir | results/celebahq/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | PASS | 23:48:15 |
| Evaluation | PASS | 00:06:33 |

- Final checkpoint: results/celebahq/latent_decomposed_diffusion/checkpoint-500000-last
- Eval slice: 100 images (glob 000*.jpg)
- Eval grids: image_test_output/image_NN.jpg -- each row is
  [input | slot 0 | slot 1 | slot 2 | slot 3 | reconstruction]
- Training curves / per-step viz: results/celebahq/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing image_test_output/ grids and tensorboard:
     - Reconstruction fidelity: do the all-slot reconstructions match inputs?
     - Decomposition: do the 4 slots capture distinct/interpretable components?
     - Baseline-soundness decision (roadmap "Setup validation" point). -->

## Notes

- eval.py writes grids to ./image_test_output/ (hardcoded); the
  --output_dir flag is ignored. See TECHDEBT.md.
- The LatentEncoder is a plain CNN+Linear baseline with no slot attention
  (intentional -- see ROADMAP.md); judge decomposition quality with that
  in mind.

## Next steps

- Review the eval grids and loss curve, fill in the Assessment section.
- Decide whether the baseline is sound enough to build the encoder +
  slot-attention version on (roadmap "Slot extraction" point).

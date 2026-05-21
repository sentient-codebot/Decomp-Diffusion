# CelebA-HQ train + eval pipeline validation

**Status:** PASS -- train and eval pipelines run cleanly
**Date:** 2026-05-20T14:35Z
**Slurm jobs:** 22966115 (train) + 22974706 (eval)
**Scripts:** jobs/celebahq_train_eval_validation.sh, jobs/celebahq_eval_validation.sh

## Purpose

Reduced-length end-to-end run confirming the CelebA-HQ training
(train_lsd.py) and evaluation (eval.py) pipelines execute cleanly. This
is a pipeline validation, not a converged model: it runs 50k of the
500k steps the config schedules.

## Configuration

| Item | Value |
|------|-------|
| Steps run | 50000 / 500000 configured |
| Effective batch | 64 (2 GPU x 32) |
| Resolution | 128 |
| Slots (num_components) | 4 |
| Mixed precision | fp16 |
| Learning rate | 2.0e-5 |
| Output dir | results/celebahq_validation/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | PASS | 02:27:21 |
| Evaluation | PASS | 00:02:02 |

- Training: 50000/50000 steps on 2x H100, final train loss ~0.116, exit 0.
- Final checkpoint: results/celebahq_validation/latent_decomposed_diffusion/checkpoint-50000-last
- Eval slice: 100 images (glob 000*.jpg)
- Eval grids: image_test_output/image_NN.jpg -- each row is
  [input | slot 0 | slot 1 | slot 2 | slot 3 | reconstruction]
- Training curves / per-step viz: results/celebahq_validation/latent_decomposed_diffusion/logs/

## Notes

- The training job (22966115) initially reported FAIL: its eval step was
  skipped by a checkpoint-path bug -- src/parser.py appends
  tracker_project_name to --output_dir, so checkpoints land in
  $RUN_DIR/latent_decomposed_diffusion/, not $RUN_DIR/ directly. Both
  job scripts now search recursively; this job re-ran eval against the
  checkpoint training had already produced.
- eval.py writes grids to ./image_test_output/ (hardcoded); the
  --output_dir flag is ignored. See TECHDEBT.md.
- 50k steps is ~1/10 of the configured schedule -- reconstructions
  indicate pipeline health, not final image quality.

## Next steps

- With both stages PASS, the pipeline is validated end-to-end; the full
  500k-step run can be launched via jobs/celebahq_train_eval_validation.sh
  (set --max_train_steps back to the configured 500000).
- Review image_test_output/ grids and the tensorboard loss curve for a
  sanity check on decomposition behaviour.

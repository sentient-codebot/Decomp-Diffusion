# COCO CoDA K/V-only latest-checkpoint object metrics

**Status:** PASS
**Date:** 2026-05-28T19:46Z
**Slurm job:** 23193898 (script: jobs/coco_coda_kv_only_eval_latest.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_23193898.log)
**Training job:** 23177437 (script: jobs/coco_coda_kv_only_train.sh)
**Checkpoint:** results/coco_coda_kv_only/latent_decomposed_diffusion/checkpoint-200000-last

## Results

| Metric | Value |
|--------|-------|
| FG-ARI | 0.3470 |
| mBO | 0.2558 |
| mIoU | 0.2500 |
| mIoU foreground-only | 0.2424 |
| Object-weighted mBO | 0.1609 |
| Images | 5000 |
| Objects | 31672 |
| Slot entropy | 1.0000 |
| Wall time | 00:04:16 |

Full metrics: results/coco_coda_kv_only/coco_metrics_latest_23193898/metrics.json
Attention-mask viz: results/coco_coda_kv_only/coco_metrics_latest_23193898/viz_*.jpg

## Notes

- Metrics are computed by eval_coco.py from COCO val2017 instance polygons.
- Crowd/RLE annotations are skipped; standard non-crowd COCO instances are
  rasterized into one integer label map per image.
- The report includes aggregate metrics; metrics.json also contains
  per-category mBO, COCO-size-bin mBO, foreground fraction, and slot usage.

#!/bin/bash
# CelebA-HQ eval-only validation against an existing checkpoint.
#
# Companion to jobs/celebahq_train_eval_validation.sh. The training job
# (Slurm 22966115) completed all 50k steps successfully, but its eval step was
# skipped by a checkpoint-path bug. This job runs eval against the checkpoint
# that training already produced and rewrites the validation report.
#
# Submit from the repo root: `sbatch jobs/celebahq_eval_validation.sh`.
#SBATCH --job-name="celebahq-eval-val"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

# uv-managed env; keep the HF cache off the home quota (see download_data.sh)
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/celebahq_validation
REPORT=docs/experiments/2026-05-20-celebahq-train-eval-validation.md
mkdir -p "$(dirname "$REPORT")"

# train_lsd.py appends --tracker_project_name to --output_dir (src/parser.py),
# so checkpoints land in $RUN_DIR/<tracker_project_name>/checkpoint-*.
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
echo "[validation] using checkpoint: ${CKPT:-<none found>}"

# --- Eval (single GPU, 100-image slice) --------------------------------------
EVAL_START=$(date +%s)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision fp16 --seed 42 \
        --batch_size 32 --num_validation_images 32 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
        --dataset_root data/celebahq_data128x128/ \
        --dataset_glob '000*.jpg' --resolution 128 \
        --ckpt_path "$CKPT"
    EVAL_RC=$?
else
    echo "[validation] no checkpoint found -- cannot eval."
    EVAL_RC=1
fi
EVAL_END=$(date +%s)
echo "**************** [validation] eval finished (rc=$EVAL_RC). ****************"

# --- Rewrite the experiment report -------------------------------------------
EVAL_DUR=$(date -u -d "@$((EVAL_END - EVAL_START))" +%H:%M:%S)
[ "$EVAL_RC" -eq 0 ] && EVAL_RESULT="PASS" || EVAL_RESULT="FAIL (rc=$EVAL_RC)"
if [ "$EVAL_RC" -eq 0 ]; then
    OVERALL="PASS -- train and eval pipelines run cleanly"
else
    OVERALL="FAIL -- eval failed, see slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# CelebA-HQ train + eval pipeline validation

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm jobs:** 22966115 (train) + ${SLURM_JOB_ID:-N/A} (eval)
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
| Output dir | $RUN_DIR/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | PASS | 02:27:21 |
| Evaluation | $EVAL_RESULT | $EVAL_DUR |

- Training: 50000/50000 steps on 2x H100, final train loss ~0.116, exit 0.
- Final checkpoint: ${CKPT:-<none>}
- Eval slice: 100 images (glob 000*.jpg)
- Eval grids: image_test_output/image_NN.jpg -- each row is
  [input | slot 0 | slot 1 | slot 2 | slot 3 | reconstruction]
- Training curves / per-step viz: $RUN_DIR/latent_decomposed_diffusion/logs/

## Notes

- The training job (22966115) initially reported FAIL: its eval step was
  skipped by a checkpoint-path bug -- src/parser.py appends
  tracker_project_name to --output_dir, so checkpoints land in
  \$RUN_DIR/latent_decomposed_diffusion/, not \$RUN_DIR/ directly. Both
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
EOF

echo "**************** [validation] report rewritten: $REPORT ($OVERALL) ****************"

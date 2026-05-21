#!/bin/bash
# CelebA-HQ train + eval pipeline validation (reduced 50k-step run).
#
# End-to-end check that the CelebA-HQ training (train_lsd.py) and evaluation
# (eval.py) pipelines run cleanly. Trains 50k of the configured 500k steps on
# 2x H100, evals the resulting checkpoint on a 100-image slice, then writes an
# experiment report to docs/experiments/. This is a pipeline validation, not a
# converged model.
#
# Submit from the repo root: `sbatch jobs/celebahq_train_eval_validation.sh`.
#SBATCH --job-name="celebahq-train-eval-val"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --time=08:00:00
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
MAX_STEPS=50000
mkdir -p "$(dirname "$REPORT")"

START=$(date +%s)

# --- 1. Train (DDP, 2 GPU -- no srun) -----------------------------------------
# --resume_from_checkpoint latest makes the job restart-safe: a resubmission
# after a timeout picks up the most recent checkpoint instead of starting over.
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision fp16 \
    --main_process_port 29500 train_lsd.py \
    --train_config configs/celebahq/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/celebahq/latent_encoder/config.json \
    --unet_config configs/celebahq/unet/config.json \
    --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
    --dataset_root data/celebahq_data128x128/ \
    --dataset_glob '**/*.jpg' \
    --report_to tensorboard \
    --resume_from_checkpoint latest \
    --max_train_steps "$MAX_STEPS"
TRAIN_RC=$?
TRAIN_END=$(date +%s)
echo "**************** [validation] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
# train_lsd.py appends --tracker_project_name to --output_dir (src/parser.py),
# so checkpoints land in $RUN_DIR/<tracker_project_name>/checkpoint-*, not
# directly under $RUN_DIR -- search recursively.
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
echo "[validation] using checkpoint: ${CKPT:-<none found>}"

# --- 3. Eval (single GPU, 100-image slice) -----------------------------------
# eval.py loops the whole dataset (the num_validation_images break is disabled),
# so the glob is restricted to 100 images to keep eval bounded.
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
    echo "[validation] no checkpoint found -- skipping eval."
    EVAL_RC=1
fi
END=$(date +%s)
echo "**************** [validation] eval finished (rc=$EVAL_RC). ****************"

# --- 4. Write the experiment report ------------------------------------------
fmt_dur() { date -u -d "@$1" +%H:%M:%S; }
TRAIN_DUR=$(fmt_dur $((TRAIN_END - START)))
EVAL_DUR=$(fmt_dur $((END - EVAL_START)))

[ "$TRAIN_RC" -eq 0 ] && TRAIN_RESULT="PASS" || TRAIN_RESULT="FAIL (rc=$TRAIN_RC)"
[ "$EVAL_RC" -eq 0 ]  && EVAL_RESULT="PASS"  || EVAL_RESULT="FAIL (rc=$EVAL_RC)"
if [ "$TRAIN_RC" -eq 0 ] && [ "$EVAL_RC" -eq 0 ]; then
    OVERALL="PASS -- train and eval pipelines run cleanly"
else
    OVERALL="FAIL -- see slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# CelebA-HQ train + eval pipeline validation

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/celebahq_train_eval_validation.sh, log: slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 2x H100

## Purpose

Reduced-length end-to-end run confirming the CelebA-HQ training
(train_lsd.py) and evaluation (eval.py) pipelines execute cleanly. This
is a pipeline validation, not a converged model: it runs 50k of the
500k steps the config schedules.

## Configuration

| Item | Value |
|------|-------|
| Steps run | $MAX_STEPS / 500000 configured |
| Effective batch | 64 (2 GPU x 32) |
| Resolution | 128 |
| Slots (num_components) | 4 |
| Mixed precision | fp16 |
| Learning rate | 2.0e-5 |
| Output dir | $RUN_DIR/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | $TRAIN_RESULT | $TRAIN_DUR |
| Evaluation | $EVAL_RESULT | $EVAL_DUR |

- Final checkpoint: ${CKPT:-<none>}
- Eval slice: 100 images (glob 000*.jpg)
- Eval grids: image_test_output/image_NN.jpg -- each row is
  [input | slot 0 | slot 1 | slot 2 | slot 3 | reconstruction]
- Training curves: run \`tensorboard --logdir $RUN_DIR\`

## Notes

- eval.py writes grids to ./image_test_output/ (hardcoded); the
  --output_dir flag is ignored. See TECHDEBT.md.
- 50k steps is ~1/10 of the configured schedule -- reconstructions
  indicate pipeline health, not final image quality.

## Next steps

- If both stages PASS, the pipeline is validated end-to-end; launch the
  full 500k-step run.
- Review image_test_output/ grids and the tensorboard loss curve for a
  sanity check on decomposition behaviour.
EOF

echo "**************** [validation] report written to $REPORT ($OVERALL) ****************"

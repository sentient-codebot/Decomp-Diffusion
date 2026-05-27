#!/bin/bash
# CelebA-HQ full 500k-step training run + eval.
#
# The real full run, following the 50k-step pipeline validation
# (jobs/celebahq_train_eval_validation.sh, report
# docs/experiments/2026-05-20-celebahq-train-eval-validation.md). Trains the
# configured 500k steps on 2x H100, evals the final checkpoint, then writes an
# experiment report. ~25h of compute; 36h walltime for margin.
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
#
# Submit from the repo root: `sbatch jobs/celebahq_train_eval_full.sh`.
#SBATCH --job-name="celebahq-full-500k"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --time=36:00:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

# Redirect heavy outputs to project storage (200 GiB home quota is too small
# for results/wandb/slurm-log accumulation -- a single 500k-step run produces
# >20 GB of checkpoints). Idempotent: if results/wandb already exist (either
# as a symlink or a real dir from before this hardening), leave them be.
PRJS_DIR="$HOME/prjs0993/Decomp-Diffusion"
mkdir -p "$PRJS_DIR/results" "$PRJS_DIR/wandb" "$PRJS_DIR/slurm_logs"
[ -e results ] || ln -s "$PRJS_DIR/results" results
[ -e wandb ]   || ln -s "$PRJS_DIR/wandb"   wandb
export WANDB_DIR="$PRJS_DIR/wandb"
export HF_HOME="$PRJS_DIR/cache/huggingface"

source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/celebahq
REPORT=docs/experiments/2026-05-20-celebahq-full-500k-run.md
MAX_STEPS=500000
mkdir -p "$(dirname "$REPORT")"

START=$(date +%s)

# --- 1. Train (DDP, 2 GPU -- no srun) -----------------------------------------
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision bf16 \
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
echo "**************** [full-run] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
# train_lsd.py appends --tracker_project_name to --output_dir (src/parser.py),
# so checkpoints land in $RUN_DIR/<tracker_project_name>/checkpoint-*, not
# directly under $RUN_DIR -- search recursively.
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
echo "[full-run] using checkpoint: ${CKPT:-<none found>}"

# --- 3. Eval (single GPU, 100-image slice) -----------------------------------
# eval.py loops the whole dataset (the num_validation_images break is disabled),
# so the glob is restricted to 100 images to keep eval bounded.
EVAL_START=$(date +%s)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision bf16 --seed 42 \
        --batch_size 32 --num_validation_images 32 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
        --dataset_root data/celebahq_data128x128/ \
        --dataset_glob '000*.jpg' --resolution 128 \
        --ckpt_path "$CKPT"
    EVAL_RC=$?
else
    echo "[full-run] no checkpoint found -- skipping eval."
    EVAL_RC=1
fi
END=$(date +%s)
echo "**************** [full-run] eval finished (rc=$EVAL_RC). ****************"

# --- 4. Write the experiment report ------------------------------------------
fmt_dur() { date -u -d "@$1" +%H:%M:%S; }
TRAIN_DUR=$(fmt_dur $((TRAIN_END - START)))
EVAL_DUR=$(fmt_dur $((END - EVAL_START)))

[ "$TRAIN_RC" -eq 0 ] && TRAIN_RESULT="PASS" || TRAIN_RESULT="FAIL (rc=$TRAIN_RC)"
[ "$EVAL_RC" -eq 0 ]  && EVAL_RESULT="PASS"  || EVAL_RESULT="FAIL (rc=$EVAL_RC)"
if [ "$TRAIN_RC" -eq 0 ] && [ "$EVAL_RC" -eq 0 ]; then
    OVERALL="PASS -- full run completed, train and eval clean"
else
    OVERALL="FAIL -- see slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# CelebA-HQ full 500k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/celebahq_train_eval_full.sh, log: slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 2x H100

## Purpose

The full CelebA-HQ training run with the LatentEncoder baseline, following
the 50k-step pipeline validation
(docs/experiments/2026-05-20-celebahq-train-eval-validation.md). This is the
run the roadmap "Setup validation" point calls for: it produces the baseline
that the planned encoder + slot-attention version is compared against.

## Configuration

| Item | Value |
|------|-------|
| Steps run | $MAX_STEPS / 500000 configured |
| Effective batch | 64 (2 GPU x 32) |
| Resolution | 128 |
| Slots (num_components) | 4 |
| Mixed precision | bf16 |
| Learning rate | 2.0e-5 |
| Output dir | $RUN_DIR/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | $TRAIN_RESULT | $TRAIN_DUR |
| Evaluation | $EVAL_RESULT | $EVAL_DUR |

- Final checkpoint: ${CKPT:-<none>}
- Eval slice: 100 images (glob 000*.jpg)
- Eval grids: image_test_output/image_NN.jpg -- each row is
  [input | slot 0 | slot 1 | slot 2 | slot 3 | reconstruction]
- Training curves / per-step viz: $RUN_DIR/latent_decomposed_diffusion/logs/

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
EOF

echo "**************** [full-run] report written to $REPORT ($OVERALL) ****************"

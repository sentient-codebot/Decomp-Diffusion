#!/bin/bash
# CelebA-HQ full 500k-step training run + eval -- SlotAttentionEncoder.
#
# The roadmap "Slot extraction: encoder + slot attention" run. Trains the
# SlotAttentionEncoder (CNN feature map -> soft positional embedding -> Slot
# Attention -> slots) for the configured 500k steps, evals the final
# checkpoint, then writes an experiment report. This is the object-centric
# encoder compared against the LatentEncoder baseline (results/celebahq,
# docs/experiments/2026-05-20-celebahq-full-500k-run.md).
#
# 4x H100 with --train_batch_size 16 -> effective batch 64, identical to the
# 2x H100 baseline run, so only the encoder differs. ~13h compute; 24h
# walltime for margin.
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
#
# Submit from the repo root: `sbatch jobs/celebahq_slot_attn_train_eval.sh`.
#SBATCH --job-name="celebahq-slot-500k"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion-slot-encoder

# uv-managed env; keep the HF cache off the home quota (see download_data.sh)
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/celebahq_slot
REPORT=docs/experiments/2026-05-21-celebahq-slot-attention-encoder.md
MAX_STEPS=500000
mkdir -p "$(dirname "$REPORT")"

# eval.py writes grids to ./image_test_output/ (hardcoded); drop any stale
# symlink so eval writes into this worktree, not a sibling checkout.
rm -rf image_test_output

START=$(date +%s)

# --- 1. Train (DDP, 4 GPU -- no srun) -----------------------------------------
# --train_batch_size 16 over 4 GPUs -> effective batch 64, matching the 2x H100
# LatentEncoder baseline run.
uv run accelerate launch --multi_gpu --num_processes=4 --mixed_precision fp16 \
    --main_process_port 29500 train_lsd.py \
    --train_config configs/celebahq/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/celebahq/slot_encoder/config.json \
    --unet_config configs/celebahq/unet/config.json \
    --scheduler_config configs/celebahq/scheduler/scheduler_config.json \
    --dataset_root data/celebahq_data128x128/ \
    --dataset_glob '**/*.jpg' \
    --report_to tensorboard \
    --train_batch_size 16 \
    --resume_from_checkpoint latest \
    --max_train_steps "$MAX_STEPS"
TRAIN_RC=$?
TRAIN_END=$(date +%s)
echo "**************** [slot-run] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
# train_lsd.py appends --tracker_project_name to --output_dir (src/parser.py),
# so checkpoints land in $RUN_DIR/<tracker_project_name>/checkpoint-*.
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
echo "[slot-run] using checkpoint: ${CKPT:-<none found>}"

# --- 3. Eval (single GPU, 100-image slice) -----------------------------------
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
    echo "[slot-run] no checkpoint found -- skipping eval."
    EVAL_RC=1
fi
END=$(date +%s)
echo "**************** [slot-run] eval finished (rc=$EVAL_RC). ****************"

# Keep the eval grids with the run (image_test_output is gitignored/transient).
if [ -d image_test_output ]; then
    mkdir -p "$RUN_DIR/eval_grids"
    cp image_test_output/*.jpg "$RUN_DIR/eval_grids/" 2>/dev/null
fi

# --- 4. Loss curve from the tensorboard event file ---------------------------
LOSS_PNG="$RUN_DIR/loss_curve_${SLURM_JOB_ID}.png"
uv run python - "$RUN_DIR" "$LOSS_PNG" <<'PYEOF'
import glob, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir, out_png = sys.argv[1], sys.argv[2]
events = sorted(glob.glob(f"{run_dir}/**/events.out.tfevents.*", recursive=True))
if not events:
    print("[loss-curve] no tfevents file found -- skipping")
    sys.exit(0)
steps, vals = [], []
for ev in events:
    acc = EventAccumulator(ev)
    acc.Reload()
    if "loss" not in acc.Tags().get("scalars", []):
        continue
    for s in acc.Scalars("loss"):
        steps.append(s.step)
        vals.append(s.value)
if not steps:
    print("[loss-curve] no 'loss' scalar found -- skipping")
    sys.exit(0)
order = sorted(range(len(steps)), key=lambda i: steps[i])
steps = [steps[i] for i in order]
vals = [vals[i] for i in order]
plt.figure(figsize=(8, 4))
plt.plot(steps, vals, linewidth=0.7)
plt.xlabel("step")
plt.ylabel("train loss (MSE)")
plt.title("SlotAttentionEncoder -- CelebA-HQ training loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_png, dpi=120)
print(f"[loss-curve] wrote {out_png} ({len(steps)} points, final loss {vals[-1]:.4f})")
PYEOF

# --- 5. Write the experiment report ------------------------------------------
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
# CelebA-HQ slot-attention encoder -- full 500k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/celebahq_slot_attn_train_eval.sh, log: slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 4x H100

## Purpose

The roadmap "Slot extraction: encoder + slot attention" run. The naive
\`LatentEncoder\` (CNN -> Flatten -> Linear -> reshape to K slots, no slot
attention) is replaced with \`SlotAttentionEncoder\`: the same convolutional
feature extractor produces a feature map, which -- after a soft positional
embedding -- is bound into K slots by an iterative Slot Attention module
(Locatello et al., 2020). This is the canonical Latent Slot Diffusion
encoder pattern.

The run is the object-centric encoder compared against the LatentEncoder
baseline (results/celebahq,
docs/experiments/2026-05-20-celebahq-full-500k-run.md). Effective batch,
step count, LR, UNet and scheduler are identical to the baseline; only the
encoder differs.

## Configuration

| Item | Value |
|------|-------|
| Encoder | SlotAttentionEncoder (CNN + soft pos-embed + Slot Attention) |
| Slot Attention iters | 3 |
| Steps run | $MAX_STEPS / 500000 configured |
| Effective batch | 64 (4 GPU x 16) |
| Resolution | 128 |
| Slots (num_components) | 4 |
| Slot dim (latent_dim) | 64 |
| Mixed precision | fp16 |
| Learning rate | 2.0e-5 |
| Output dir | $RUN_DIR/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | $TRAIN_RESULT | $TRAIN_DUR |
| Evaluation | $EVAL_RESULT | $EVAL_DUR |

- Final checkpoint: ${CKPT:-<none>}
- Loss curve: $LOSS_PNG
- Eval slice: 100 images (glob 000*.jpg)
- Eval grids: $RUN_DIR/eval_grids/image_NN.jpg -- each row is
  [input | slot 0 | slot 1 | slot 2 | slot 3 | reconstruction]
- Training curves / per-step viz: $RUN_DIR/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing eval_grids/ and the loss curve:
     - Reconstruction fidelity: do all-slot reconstructions match inputs?
     - Decomposition: do the 4 slots capture distinct components, and is the
       decomposition cleaner / more interpretable than the LatentEncoder
       baseline?
     - Final train loss vs the baseline run. -->

## Notes

- Only the encoder changed from the baseline. The convolutional feature
  extractor is unchanged; the Flatten + Linear slot read-out is replaced by
  a soft positional embedding + Slot Attention module.
- eval.py writes grids to ./image_test_output/ (hardcoded); they are copied
  to $RUN_DIR/eval_grids/ here. See TECHDEBT.md.

## Next steps

- Review the eval grids and loss curve, fill in the Assessment section.
- Roadmap next stage: replace the CNN feature extractor with a pretrained
  encoder (e.g. DINO) -- see ROADMAP.md.
EOF

echo "**************** [slot-run] report written to $REPORT ($OVERALL) ****************"

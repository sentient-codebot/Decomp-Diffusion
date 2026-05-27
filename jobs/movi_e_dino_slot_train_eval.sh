#!/bin/bash
# MOVi-E 200k-step training + object-centric eval -- DinoSlotAttentionEncoder.
#
# The roadmap "Encoder: pretrained feature extractor" run: same pipeline as
# jobs/movi_e_slot_attn_train_eval.sh, but the trained-from-scratch CNN feature
# extractor is replaced with a pretrained DINO ViT-S/8 (frozen). Slot
# Attention then binds the DINO patch tokens into 11 slots.
#
# 4x H100 with --train_batch_size 16 -> effective batch 64. DINO ViT-S/8 is
# small (~22M params, frozen), and the forward pass skips backbone backward,
# so step time should be comparable to or faster than the CNN baseline.
#
# Depends on data laid down by jobs/movi_e_preprocess.sh (~/prjs0993/datasets/movi-e/).
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
#
# Submit from the repo root: `sbatch jobs/movi_e_dino_slot_train_eval.sh`.
#SBATCH --job-name="movi-e-dino-slot-200k"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

# uv-managed env; keep the HF cache off the home quota. DINO weights are
# downloaded into this cache on first instantiation of the encoder.
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/movi-e_dino_slot
REPORT=docs/experiments/2026-05-26-movi-e-dino-slot-attention.md
MAX_STEPS=200000
mkdir -p "$(dirname "$REPORT")"

# Image grids from eval.py are dumped here (path is hardcoded in eval.py);
# drop any stale symlink so they land in this worktree.
rm -rf image_test_output

# --- 0. Sanity check: dataset must already be preprocessed -------------------
TRAIN_IMG_ROOT=data/movi-e/movi-e-train-with-label/images
VAL_IMG_ROOT=data/movi-e/movi-e-validation-with-label/images
if [ ! -d "$TRAIN_IMG_ROOT" ]; then
    echo "[movi-e-dino-run] FATAL: $TRAIN_IMG_ROOT is missing -- run jobs/movi_e_preprocess.sh first."
    exit 1
fi
TRAIN_N=$(find "$TRAIN_IMG_ROOT" -name '*_image.png' | wc -l)
VAL_N=$(find "$VAL_IMG_ROOT" -name '*_image.png' 2>/dev/null | wc -l)
echo "[movi-e-dino-run] train frames=$TRAIN_N  val frames=$VAL_N"

# Prime the GPFS metadata cache (same reason as the CNN slot-attn run -- 234k
# files, cold metadata races across 4 ranks).
echo "[movi-e-dino-run] priming GPFS metadata..."
find "$TRAIN_IMG_ROOT" "$VAL_IMG_ROOT" -type f >/dev/null
echo "[movi-e-dino-run] done priming."

# Raise NCCL collective timeout.
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

START=$(date +%s)

# --- 1. Train (DDP, 4 GPU -- no srun) ----------------------------------------
# --train_batch_size 16 over 4 GPUs -> effective batch 64.
uv run accelerate launch --multi_gpu --num_processes=4 --mixed_precision bf16 \
    --main_process_port 29500 train_lsd.py \
    --train_config configs/movi-e/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/movi-e/dino_slot_encoder/config.json \
    --unet_config configs/movi-e/unet/config.json \
    --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
    --dataset_root "$TRAIN_IMG_ROOT" \
    --dataset_glob '**/*.png' \
    --report_to wandb \
    --train_batch_size 16 \
    --resume_from_checkpoint latest \
    --max_train_steps "$MAX_STEPS"
TRAIN_RC=$?
TRAIN_END=$(date +%s)
echo "**************** [movi-e-dino-run] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -z "$CKPT" ]; then
    CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
fi
echo "[movi-e-dino-run] using checkpoint: ${CKPT:-<none found>}"

# --- 3a. Eval: qualitative reconstruction grids ------------------------------
EVAL_START=$(date +%s)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision bf16 --seed 42 \
        --batch_size 16 --num_validation_images 16 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
        --dataset_root "$VAL_IMG_ROOT" \
        --dataset_glob '**/00000000_image.png' --resolution 128 \
        --ckpt_path "$CKPT"
    EVAL_RECON_RC=$?
else
    echo "[movi-e-dino-run] no checkpoint found -- skipping reconstruction eval."
    EVAL_RECON_RC=1
fi

if [ -d image_test_output ]; then
    mkdir -p "$RUN_DIR/eval_grids"
    cp image_test_output/*.jpg "$RUN_DIR/eval_grids/" 2>/dev/null
fi

# --- 3b. Eval: FG-ARI + mBO on full validation split -------------------------
METRICS_DIR="$RUN_DIR/metrics"
mkdir -p "$METRICS_DIR"
if [ -n "$CKPT" ]; then
    uv run python eval_movi.py \
        --ckpt_path "$CKPT" \
        --dataset_root data/movi-e \
        --split validation \
        --resolution 128 \
        --batch_size 32 \
        --num_workers 4 \
        --output_dir "$METRICS_DIR"
    EVAL_METRICS_RC=$?
else
    EVAL_METRICS_RC=1
fi
END=$(date +%s)
echo "**************** [movi-e-dino-run] eval finished (recon rc=$EVAL_RECON_RC, metrics rc=$EVAL_METRICS_RC). ****************"

# --- 4. Loss curve + wandb run url -------------------------------------------
LOSS_PNG="$RUN_DIR/loss_curve_${SLURM_JOB_ID}.png"
WANDB_URL_FILE="$RUN_DIR/wandb_url.txt"
uv run python - "$RUN_DIR" "$LOSS_PNG" "$WANDB_URL_FILE" <<'PYEOF'
import os, sys

run_dir, out_png, url_file = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    api = wandb.Api()
    target = os.path.join(run_dir, "latent_decomposed_diffusion")
    runs = list(
        api.runs(
            "nan-team/latent_decomposed_diffusion",
            filters={"config.output_dir": target},
        )
    )
    if not runs:
        print("[loss-curve] no matching wandb run found -- skipping")
        sys.exit(0)
    run = runs[0]
    with open(url_file, "w") as f:
        f.write(run.url + "\n")
    print(f"[loss-curve] wandb run: {run.url}")
    rows = run.history(keys=["loss"], samples=200000, pandas=False)
    pts = sorted(
        (r["_step"], r["loss"]) for r in rows if r.get("loss") is not None
    )
    if not pts:
        print("[loss-curve] no 'loss' history -- skipping")
        sys.exit(0)
    steps = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, vals, linewidth=0.7)
    plt.xlabel("step")
    plt.ylabel("train loss (MSE)")
    plt.title("DinoSlotAttentionEncoder -- MOVi-E training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"[loss-curve] wrote {out_png} ({len(steps)} points, final loss {vals[-1]:.4f})")
except Exception as e:
    print(f"[loss-curve] wandb pull failed: {e}")
PYEOF
WANDB_URL=$(cat "$WANDB_URL_FILE" 2>/dev/null || echo "N/A")

read FG_ARI MBO N_IMG <<<$(uv run python - "$METRICS_DIR/metrics.json" <<'PYEOF'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("N/A N/A N/A"); raise SystemExit
m = json.load(open(p))
print(f"{m['fg_ari']:.4f} {m['mbo']:.4f} {m['n_images_ari']}")
PYEOF
)

# --- 5. Write the experiment report ------------------------------------------
fmt_dur() { date -u -d "@$1" +%H:%M:%S; }
TRAIN_DUR=$(fmt_dur $((TRAIN_END - START)))
EVAL_DUR=$(fmt_dur $((END - EVAL_START)))

[ "$TRAIN_RC" -eq 0 ] && TRAIN_RESULT="PASS" || TRAIN_RESULT="FAIL (rc=$TRAIN_RC)"
[ "$EVAL_RECON_RC" -eq 0 ]  && EVAL_RECON_RESULT="PASS"  || EVAL_RECON_RESULT="FAIL (rc=$EVAL_RECON_RC)"
[ "$EVAL_METRICS_RC" -eq 0 ] && EVAL_METRICS_RESULT="PASS" || EVAL_METRICS_RESULT="FAIL (rc=$EVAL_METRICS_RC)"
if [ "$TRAIN_RC" -eq 0 ] && [ "$EVAL_RECON_RC" -eq 0 ] && [ "$EVAL_METRICS_RC" -eq 0 ]; then
    OVERALL="PASS -- full run completed, train + both evals clean"
else
    OVERALL="FAIL -- see slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# MOVi-E DINO slot-attention encoder -- 200k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_dino_slot_train_eval.sh, log: slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 4x H100
**wandb run:** $WANDB_URL

## Purpose

Roadmap "Encoder: pretrained feature extractor" run. Replaces the
trained-from-scratch CNN feature extractor in \`SlotAttentionEncoder\` with a
pretrained DINO ViT-S/8 (frozen), keeping the soft positional embedding +
Slot Attention head identical. Tests whether stronger pretrained per-patch
features improve object decomposition over the CNN baseline
(2026-05-23-movi-e-slot-attention).

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINO ViT-S/8 frozen + soft pos-embed + Slot Attention) |
| DINO model | facebook/dino-vits8 |
| Patch grid | 16x16 at 128 input (patch_size=8) |
| Slot Attention iters | 3 |
| Steps run | $MAX_STEPS / 200000 configured |
| Effective batch | 64 (4 GPU x 16) |
| Resolution | 128 |
| Slots (num_components) | 11 |
| Slot dim (latent_dim) | 64 |
| Mixed precision | bf16 |
| Learning rate | 2.0e-5 |
| Dataset | MOVi-E train split ($TRAIN_N frames) |
| Eval split | MOVi-E validation split ($VAL_N frames) |
| Output dir | $RUN_DIR/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | $TRAIN_RESULT | $TRAIN_DUR |
| Reconstruction eval | $EVAL_RECON_RESULT | $EVAL_DUR |
| Object-centric metrics | $EVAL_METRICS_RESULT | (incl. above) |

### Object-centric metrics (eval_movi.py, validation split)

| Metric | DINO + Slot Attn | CNN + Slot Attn (2026-05-23) |
|--------|-------|-------|
| FG-ARI | $FG_ARI | see prior report |
| mBO    | $MBO | see prior report |
| Frames | $N_IMG | -- |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg

### Reconstruction grids

Final checkpoint: ${CKPT:-<none>}
Loss curve: $LOSS_PNG
Reconstruction grids: $RUN_DIR/eval_grids/image_NN.jpg
Per-step validation viz: $RUN_DIR/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics + viz:
     - Did pretrained features improve FG-ARI/mBO over the CNN baseline?
     - Do slot masks visually align with object boundaries better than the
       CNN run, where slots collapsed onto a global representation?
     - Loss curve vs CNN run: does pretraining converge faster, or just
       end at the same place? -->

## Notes

- DINO is frozen: only the soft pos-embed, layer norm, MLP projection and
  Slot Attention head are trained. The diffusion UNet trains end-to-end as
  before.
- Inside the encoder, dataset normalisation (mean=0.5, std=0.5) is converted
  to ImageNet stats before passing pixels to ViT. ViT positional embeddings
  are interpolated from the pretrained 28x28 (224 input) grid down to the
  16x16 (128 input) grid used here.
- DINO weights are saved in every checkpoint via the diffusers state_dict
  serialisation -- adds ~88 MB to each checkpoint but keeps the load path
  simple. See TECHDEBT.md if this becomes a storage problem.

## Next steps

- Compare FG-ARI / mBO numbers to the CNN baseline; fill in Assessment.
- If pretrained features help, consider DINOv2 (patch 14) at a matched
  resolution, or unfreezing the last few ViT blocks for light fine-tuning.
EOF

echo "**************** [movi-e-dino-run] report written to $REPORT ($OVERALL) ****************"

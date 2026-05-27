#!/bin/bash
# MOVi-E 200k-step training + object-centric eval -- DinoSlotAttentionEncoder
# with DINOv3 ViT-S/16 at 256 resolution PLUS 4 learned register slots.
#
# Same backbone/resolution/budget as jobs/movi_e_dinov3_slot_train_eval.sh, but
# the encoder config sets num_registers=4, num_components=24. Training uses
# the simple-sum compositional objective:
#   eps = sum_k eps_slot_k
# where each per-slot forward conditions on [slot_k, registers]. The earlier
# sum-of-deltas variant ((1 - K) * eps_uncond + sum_k eps_slot_k with a
# separate registers-only uncond forward) was dropped because per-slot decode
# under it was underdetermined; registers still ride along in each slot's
# conditioning but no longer have a dedicated uncond forward.
#
# K bumped from 11 -> 24 to match Nguyen et al. 2026 -- MOVi-E scenes can
# carry ~20 instances, so 11 may have been undercounting.
#
# Earlier MOVi-E runs (DINO v1 23117268, DINOv3 23118028 without registers,
# and the sum-of-deltas register run 23124195) were cancelled -- their
# trajectories are not comparable under the new loss, so we start fresh.
#
# 2x H100 with K=24: per-step UNet work is K = 24 forwards (vs the earlier
# K+1 = 25 under sum-of-deltas) and the cross-attn sequence is (1+R)=5 long.
# H100's 80 GB lets us keep PER_GPU_BATCH=8.
#
# Depends on data laid down by jobs/movi_e_preprocess.sh
# (~/prjs0993/datasets/movi-e/).
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
#
# Requires HF gated-access approval for facebook/dinov3-vits16-pretrain-lvd1689m
# and a token cached under $HF_HOME (huggingface-cli login once).
#
# Submit from the repo root: `sbatch jobs/movi_e_dinov3_registers_train_eval.sh`.
#SBATCH --job-name="movi-e-dinov3-k24r4-200k"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

# uv-managed env; keep the HF cache off the home quota. DINOv3 weights and
# token live in this cache.
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/movi-e_dinov3_reg
REPORT=docs/experiments/2026-05-26-movi-e-dinov3-registers.md
MAX_STEPS=200000
RESOLUTION=256
PER_GPU_BATCH=8
mkdir -p "$(dirname "$REPORT")"

# Image grids from eval.py are dumped here (hardcoded path); drop any stale
# symlink so they land in this worktree.
rm -rf image_test_output

# --- 0. Sanity check: dataset must already be preprocessed -------------------
TRAIN_IMG_ROOT=data/movi-e/movi-e-train-with-label/images
VAL_IMG_ROOT=data/movi-e/movi-e-validation-with-label/images
if [ ! -d "$TRAIN_IMG_ROOT" ]; then
    echo "[movi-e-dinov3-reg] FATAL: $TRAIN_IMG_ROOT is missing -- run jobs/movi_e_preprocess.sh first."
    exit 1
fi
TRAIN_N=$(find "$TRAIN_IMG_ROOT" -name '*_image.png' | wc -l)
VAL_N=$(find "$VAL_IMG_ROOT" -name '*_image.png' 2>/dev/null | wc -l)
echo "[movi-e-dinov3-reg] train frames=$TRAIN_N  val frames=$VAL_N"

echo "[movi-e-dinov3-reg] priming GPFS metadata..."
find "$TRAIN_IMG_ROOT" "$VAL_IMG_ROOT" -type f >/dev/null
echo "[movi-e-dinov3-reg] done priming."

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

START=$(date +%s)

# --- 1. Train (DDP, 2 GPU -- no srun) ----------------------------------------
# --train_batch_size $PER_GPU_BATCH over 2 GPUs -> effective batch 16.
# Same per-GPU batch as the no-register dinov3 run to keep results comparable;
# with K=24 + R=4 each step does K+1=25 UNet forwards (vs 12 at K=11) and the
# cross-attn sequence is 5 long (1 slot + 4 registers) for cond, 4 for uncond.
# A100 40 GB would be tight at this batch -- 2x H100 80 GB gives the headroom.
# --resolution overrides the train_config default (128) to 256.
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision bf16 \
    --main_process_port 29501 train_lsd.py \
    --train_config configs/movi-e/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/movi-e/dinov3_slot_encoder/config.json \
    --unet_config configs/movi-e/unet/config.json \
    --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
    --dataset_root "$TRAIN_IMG_ROOT" \
    --dataset_glob '**/*.png' \
    --report_to wandb \
    --resolution "$RESOLUTION" \
    --train_batch_size "$PER_GPU_BATCH" \
    --resume_from_checkpoint latest \
    --max_train_steps "$MAX_STEPS"
TRAIN_RC=$?
TRAIN_END=$(date +%s)
echo "**************** [movi-e-dinov3-reg] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -z "$CKPT" ]; then
    CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
fi
echo "[movi-e-dinov3-reg] using checkpoint: ${CKPT:-<none found>}"

# --- 3a. Eval: qualitative reconstruction grids ------------------------------
EVAL_START=$(date +%s)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision bf16 --seed 42 \
        --batch_size 8 --num_validation_images 16 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
        --dataset_root "$VAL_IMG_ROOT" \
        --dataset_glob '**/00000000_image.png' --resolution "$RESOLUTION" \
        --ckpt_path "$CKPT"
    EVAL_RECON_RC=$?
else
    echo "[movi-e-dinov3-reg] no checkpoint found -- skipping reconstruction eval."
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
        --resolution "$RESOLUTION" \
        --batch_size 16 \
        --num_workers 4 \
        --output_dir "$METRICS_DIR"
    EVAL_METRICS_RC=$?
else
    EVAL_METRICS_RC=1
fi
END=$(date +%s)
echo "**************** [movi-e-dinov3-reg] eval finished (recon rc=$EVAL_RECON_RC, metrics rc=$EVAL_METRICS_RC). ****************"

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
    rows = run.history(
        keys=["loss", "val_loss", "slot_pairwise_cos"], samples=200000, pandas=False
    )

    def series(key):
        return sorted(
            (r["_step"], r[key]) for r in rows if r.get(key) is not None
        )

    train_pts = series("loss")
    val_pts = series("val_loss")
    cos_pts = series("slot_pairwise_cos")

    if not train_pts:
        print("[loss-curve] no 'loss' history -- skipping")
        sys.exit(0)

    fig, (ax_loss, ax_cos) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_loss.plot(
        [p[0] for p in train_pts],
        [p[1] for p in train_pts],
        linewidth=0.7,
        label="train",
    )
    if val_pts:
        ax_loss.plot(
            [p[0] for p in val_pts],
            [p[1] for p in val_pts],
            marker="o",
            linewidth=1.0,
            label="val",
        )
    ax_loss.set_ylabel("MSE")
    ax_loss.set_title(
        "DinoSlotAttentionEncoder (DINOv3 ViT-S/16 @256, R=4) -- MOVi-E"
    )
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="upper right")

    if cos_pts:
        ax_cos.plot(
            [p[0] for p in cos_pts],
            [p[1] for p in cos_pts],
            marker="o",
            linewidth=1.0,
            color="tab:green",
        )
        ax_cos.set_ylabel("slot pairwise cos")
        ax_cos.set_ylim(-1.0, 1.0)
    ax_cos.set_xlabel("step")
    ax_cos.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(
        f"[loss-curve] wrote {out_png} "
        f"(train pts={len(train_pts)}, val pts={len(val_pts)}, cos pts={len(cos_pts)}, "
        f"final train loss {train_pts[-1][1]:.4f})"
    )
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
# MOVi-E DINOv3 slot-attention encoder + register slots -- 200k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_dinov3_registers_train_eval.sh, log: slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 2x H100
**wandb run:** $WANDB_URL

## Purpose

First run combining the DINOv3 backbone with the **register-slot
compositional denoising** objective from commit 90108cb. Tests whether the
sum-of-deltas loss (eps = (1 - K) * eps_uncond + sum_k eps_slot_k, with
eps_uncond conditioned on R=4 register tokens only) breaks the redundancy
collapse seen with the previous mean-aggregation loss.

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention + 4 register slots) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| Patch grid | 16x16 at 256 input (patch_size=16) |
| Special tokens dropped (backbone) | 5 (1 CLS + 4 register) |
| Slot Attention iters | 3 |
| Slots (num_components, K) | 24 |
| Register slots (R) | 4 |
| Slot dim (latent_dim) | 64 |
| Loss | sum-of-deltas, registers as uncond context |
| Steps run | $MAX_STEPS / 200000 configured |
| Effective batch | 16 (2 GPU x $PER_GPU_BATCH) |
| Resolution | 256 |
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

| Metric | DINOv3 + R=4 | DINOv3 (no reg) | DINO v1 @128 | CNN @128 |
|--------|-------|-------|-------|-------|
| FG-ARI | $FG_ARI | cancelled | see prior report | see prior report |
| mBO    | $MBO | cancelled | see prior report | see prior report |
| Frames | $N_IMG | -- | -- | -- |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg

### Reconstruction grids and curves

Final checkpoint: ${CKPT:-<none>}
Loss + slot-pairwise-cos curves: $LOSS_PNG
Reconstruction grids: $RUN_DIR/eval_grids/image_NN.jpg
Per-step validation viz: $RUN_DIR/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics + viz:
     - Did slot_pairwise_cos stay low (slots specialise) or trend toward 1
       (collapse)?
     - Did val_loss track or decouple from train_loss?
     - FG-ARI / mBO vs the cancelled DINOv3-no-register run -- where it had
       reached at the cancellation point (intermediate ckpt in results/movi-e_dinov3_slot). -->

## Notes

- Previous DINOv3 (no register) run 23118028 and DINO v1 run 23117268 were
  cancelled to free SBU for this run; their intermediate checkpoints stay on
  disk for qualitative comparison but loss-curve comparison isn't apples to
  apples since the objective changed (mean -> sum-of-deltas with uncond).
- New wandb-logged scalars (added in this branch): \`val_loss\` (single-step
  MSE on val batches under the same composition) and \`slot_pairwise_cos\`
  (mean off-diagonal pairwise cosine sim of object slots -- collapse
  diagnostic).
EOF

echo "**************** [movi-e-dinov3-reg] report written to $REPORT ($OVERALL) ****************"

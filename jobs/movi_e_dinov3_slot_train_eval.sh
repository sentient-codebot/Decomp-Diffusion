#!/bin/bash
# MOVi-E 200k-step training + object-centric eval -- DinoSlotAttentionEncoder
# with DINOv3 ViT-S/16 at 256 resolution.
#
# Follow-up to jobs/movi_e_dino_slot_train_eval.sh (DINO v1 ViT-S/8 @128).
# Both encoders end up with a 16x16 patch grid -- DINO v1 gets there with
# patch_size=8 at 128 input, DINOv3 with patch_size=16 at 256 input. The 256
# resolution keeps the feature density comparable to the v1 / CNN baselines
# while letting us use DINOv3's stronger dense features.
#
# Resolution change cascades: VAE encodes 256 -> 32x32 latents (vs 16x16 at
# 128), so each UNet forward is ~4x heavier than the v1 run.
#
# Running on 2x A100 (40 GB) instead of 4x H100 because the H100 budget is
# tied up by the v1 run reservation; A100 is cheaper per GPU-hour and the
# remaining SBU budget fits a 24h slot here. Per-GPU batch is 8 -> effective
# batch 16 (vs 64 in the v1 run; slot attention is robust to lower batch).
# Step count stays at 200k -- single 24h slot likely won't finish at A100
# throughput, so the restart-safe --resume_from_checkpoint latest path will
# carry it over multiple submissions.
#
# Depends on shards laid down by jobs/movi_e_shard_wds.sh
# (~/prjs0993/datasets/movi-e-wds/).
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
#
# Requires HF gated-access approval for facebook/dinov3-vits16-pretrain-lvd1689m
# and a token cached under $HF_HOME (huggingface-cli login once).
#
# Submit from the repo root: `sbatch jobs/movi_e_dinov3_slot_train_eval.sh`.
#SBATCH --job-name="movi-e-dinov3-slot-200k"
#SBATCH --partition=gpu_a100
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

RUN_DIR=results/movi-e_dinov3_slot
REPORT=docs/experiments/2026-05-26-movi-e-dinov3-slot-attention.md
MAX_STEPS=200000
RESOLUTION=256
PER_GPU_BATCH=8
mkdir -p "$(dirname "$REPORT")"

# Image grids from eval.py are dumped here (hardcoded path); drop any stale
# symlink so they land in this worktree.
rm -rf image_test_output

# --- 0. Sanity check: dataset must already be preprocessed -------------------
TRAIN_SHARD_ROOT=data/movi-e-wds/train
VAL_SHARD_ROOT=data/movi-e-wds/validation
if [ ! -f "$TRAIN_SHARD_ROOT/samples.jsonl" ]; then
    echo "[movi-e-dinov3-run] FATAL: $TRAIN_SHARD_ROOT/samples.jsonl missing -- run jobs/movi_e_shard_wds.sh first."
    exit 1
fi
TRAIN_N=$(wc -l < "$TRAIN_SHARD_ROOT/samples.jsonl")
VAL_N=$(wc -l < "$VAL_SHARD_ROOT/samples.jsonl" 2>/dev/null || echo 0)
echo "[movi-e-dinov3-run] train frames=$TRAIN_N  val frames=$VAL_N"


export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

START=$(date +%s)

# --- 1. Train (DDP, 2 GPU -- no srun) ----------------------------------------
# --train_batch_size $PER_GPU_BATCH over 2 GPUs -> effective batch 16.
# --resolution overrides the train_config default (128) to 256.
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision bf16 \
    --main_process_port 29501 train_lsd.py \
    --train_config configs/movi-e/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/movi-e/dinov3_slot_encoder/config.json \
    --unet_config configs/movi-e/unet/config.json \
    --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
    --dataset_root "$TRAIN_SHARD_ROOT" \
    --dataset_glob '*.tar' \
    --dataset_format wds \
    --report_to wandb \
    --resolution "$RESOLUTION" \
    --train_batch_size "$PER_GPU_BATCH" \
    --resume_from_checkpoint latest \
    --max_train_steps "$MAX_STEPS"
TRAIN_RC=$?
TRAIN_END=$(date +%s)
echo "**************** [movi-e-dinov3-run] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -z "$CKPT" ]; then
    CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
fi
echo "[movi-e-dinov3-run] using checkpoint: ${CKPT:-<none found>}"

# --- 3a. Eval: qualitative reconstruction grids ------------------------------
EVAL_START=$(date +%s)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision bf16 --seed 42 \
        --batch_size 8 --num_validation_images 16 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
        --dataset_root "$VAL_SHARD_ROOT" \
        --dataset_glob '**/00000000_image.png' \
        --dataset_format wds --resolution "$RESOLUTION" \
        --ckpt_path "$CKPT"
    EVAL_RECON_RC=$?
else
    echo "[movi-e-dinov3-run] no checkpoint found -- skipping reconstruction eval."
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
        --dataset_root data/movi-e-wds \
        --movi_eval_format wds \
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
echo "**************** [movi-e-dinov3-run] eval finished (recon rc=$EVAL_RECON_RC, metrics rc=$EVAL_METRICS_RC). ****************"

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
    plt.title("DinoSlotAttentionEncoder (DINOv3 ViT-S/16 @256) -- MOVi-E training loss")
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
# MOVi-E DINOv3 slot-attention encoder -- 200k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_dinov3_slot_train_eval.sh, log: slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 2x A100
**wandb run:** $WANDB_URL

## Purpose

Roadmap "Encoder: pretrained feature extractor" follow-up. Same drop-in
\`DinoSlotAttentionEncoder\` as the v1 run, but the backbone is DINOv3
ViT-S/16 (frozen) and resolution is 256 so the patch grid (256/16=16) again
matches the 16x16 used by DINO v1 ViT-S/8 @128. Tests whether a stronger
pretrained backbone improves object decomposition over both the CNN
baseline and the DINO v1 run.

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| Patch grid | 16x16 at 256 input (patch_size=16) |
| Special tokens dropped | 5 (1 CLS + 4 register) |
| Slot Attention iters | 3 |
| Steps run | $MAX_STEPS / 200000 configured |
| Effective batch | 16 (2 GPU x $PER_GPU_BATCH) |
| Resolution | 256 |
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

| Metric | DINOv3 @256 | DINO v1 @128 | CNN @128 |
|--------|-------|-------|-------|
| FG-ARI | $FG_ARI | see prior report | see prior report |
| mBO    | $MBO | see prior report | see prior report |
| Frames | $N_IMG | -- | -- |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg

### Reconstruction grids

Final checkpoint: ${CKPT:-<none>}
Loss curve: $LOSS_PNG
Reconstruction grids: $RUN_DIR/eval_grids/image_NN.jpg
Per-step validation viz: $RUN_DIR/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics + viz:
     - Did DINOv3's stronger features improve over DINO v1 (same 16x16 grid)?
     - Effective batch is half (32 vs 64) -- consider that when comparing.
     - Does the loss curve plateau faster than v1 / CNN? -->

## Notes

- 16x16 patch grid matches v1 / CNN, so feature density is comparable; the
  variable is backbone quality and resolution. Resolution doubles the VAE
  latent area; with batch 8 per GPU effective batch is 32 (vs 64 in the
  v1 run).
- DINOv3 uses RoPE positional encoding -- no \`interpolate_pos_encoding\`
  kwarg is needed (encoder auto-detects). 1 CLS + 4 register tokens are
  dropped before feeding patches to the slot-attention head.

## Next steps

- Compare against the v1 / CNN baselines; fill in Assessment.
- If DINOv3 helps a lot, consider DINOv3 ViT-B/16 (next size up) or
  unfreezing the last few blocks.
EOF

echo "**************** [movi-e-dinov3-run] report written to $REPORT ($OVERALL) ****************"

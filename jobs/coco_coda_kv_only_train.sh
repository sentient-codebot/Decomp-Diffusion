#!/bin/bash
# COCO 2017 training (no eval) -- CoDA-style frozen SD2.1 UNet + DINOv3 slot
# encoder + register slots. Mirrors jobs/movi_e_coda_kv_only_train_eval.sh on
# the encoder/diffusion side; differences are:
#   - Dataset is COCO train2017 (~118k JPGs at variable resolution; resized
#     to 256 by GlobDataset). Object-centric COCO metrics are run separately
#     by jobs/coco_coda_kv_only_eval_latest.sh.
#   - num_components (K) reduced from 24 to 7. COCO scenes typically have
#     2-8 prominent instances; the K=24 sweep belongs to MOVi-E's
#     synthetic-clutter regime. Revisit if slot_pairwise_cos stays low.
#   - Same R=4 register slots and latent_dim=1024 as the MOVi-E CoDA run.
#
# Depends on data laid down by jobs/coco_download.sh
# (~/prjs0993/datasets/coco/images/train2017/*.jpg).
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
#
# Submit from the repo root: `sbatch jobs/coco_coda_kv_only_train.sh`.
#SBATCH --job-name="coco-coda-kv-only-200k"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

# uv-managed env; keep caches/logs off the home quota. DINOv3 and SD2.1
# weights live in HF_HOME.
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
export WANDB_DIR="$HOME/prjs0993/Decomp-Diffusion/wandb"
# Persistent torch.compile / inductor cache (shared across projects). Set in
# ~/.bashrc too, but sbatch may not re-source bashrc, so we belt-and-suspender.
export TORCHINDUCTOR_CACHE_DIR="$HOME/prjs0993/tmp/torchinductor"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/coco_coda_kv_only
REPORT=docs/experiments/2026-05-27-coco-coda-kv-only.md
MAX_STEPS=200000
RESOLUTION=256
PER_GPU_BATCH=8
mkdir -p "$RUN_DIR" "$WANDB_DIR" "$(dirname "$REPORT")"

# --- 0. Sanity check: dataset must already be downloaded ---------------------
TRAIN_IMG_ROOT="$HOME/prjs0993/datasets/coco/images/train2017"
VAL_IMG_ROOT="$HOME/prjs0993/datasets/coco/images/val2017"
if [ ! -d "$TRAIN_IMG_ROOT" ]; then
    echo "[coco-coda] FATAL: $TRAIN_IMG_ROOT is missing -- run jobs/coco_download.sh first."
    exit 1
fi
TRAIN_N=$(find "$TRAIN_IMG_ROOT" -maxdepth 1 -name '*.jpg' | wc -l)
VAL_N=$(find "$VAL_IMG_ROOT" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
echo "[coco-coda] train images=$TRAIN_N  val images=$VAL_N"
if [ "$TRAIN_N" -lt 1000 ]; then
    echo "[coco-coda] FATAL: only $TRAIN_N train images found -- dataset looks incomplete."
    exit 1
fi

echo "[coco-coda] priming GPFS metadata..."
find "$TRAIN_IMG_ROOT" -type f >/dev/null
echo "[coco-coda] done priming."

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

START=$(date +%s)

# --- 1. Train (DDP, 2 GPU -- no srun) ----------------------------------------
# Per-step UNet work is K=7 forwards with a (1+R)=5 long cross-attn sequence.
# Scheduler config path is required by argparse but ignored: train_lsd.py
# loads SD2.1's own DDPM scheduler when --unet_config=pretrain_sd.
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision bf16 \
    --dynamo_backend=inductor \
    --main_process_port 29501 train_lsd.py \
    --train_config configs/coco/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/coco/dinov3_slot_encoder_d1024/config.json \
    --unet_config pretrain_sd \
    --freeze_unet_except_kv \
    --scheduler_config configs/coco/scheduler/scheduler_config.json \
    --dataset_root "$TRAIN_IMG_ROOT" \
    --dataset_glob '*.jpg' \
    --report_to wandb \
    --resolution "$RESOLUTION" \
    --train_batch_size "$PER_GPU_BATCH" \
    --resume_from_checkpoint latest \
    --max_train_steps "$MAX_STEPS"
TRAIN_RC=$?
TRAIN_END=$(date +%s)
echo "**************** [coco-coda] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -z "$CKPT" ]; then
    CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
fi
echo "[coco-coda] final checkpoint: ${CKPT:-<none found>}"

# --- 3. Loss curve + wandb run url -------------------------------------------
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
    ax_loss.set_title("CoDA K/V-only (SD2.1 frozen + DINOv3 @256, K=7, R=4) -- COCO")
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
        f"(train pts={len(train_pts)}, val pts={len(val_pts)}, cos pts={len(cos_pts)})"
    )
except Exception as e:
    print(f"[loss-curve] wandb pull failed: {e}")
PYEOF
WANDB_URL=$(cat "$WANDB_URL_FILE" 2>/dev/null || echo "N/A")

# --- 4. Write the experiment report ------------------------------------------
fmt_dur() { date -u -d "@$1" +%H:%M:%S; }
TRAIN_DUR=$(fmt_dur $((TRAIN_END - START)))

[ "$TRAIN_RC" -eq 0 ] && TRAIN_RESULT="PASS" || TRAIN_RESULT="FAIL (rc=$TRAIN_RC)"
if [ "$TRAIN_RC" -eq 0 ]; then
    OVERALL="PASS -- training completed"
else
    OVERALL="FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# COCO CoDA-style K/V-only training -- 200k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/coco_coda_kv_only_train.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 2x H100
**wandb run:** $WANDB_URL

## Purpose

First COCO training run for this codebase. Same CoDA-style recipe as the
MOVi-E K/V-only baseline (\`2026-05-27-movi-e-coda-kv-only.md\`): frozen
SD2.1 UNet except for cross-attn K/V projections, DINOv3 slot encoder
with 4 register slots and latent_dim=1024. Tests whether the
object-centric encoder + frozen denoiser prior we developed on synthetic
MOVi-E transfers to natural images.

Object-discovery metrics (FG-ARI / mBO / mIoU) are run separately by
\`jobs/coco_coda_kv_only_eval_latest.sh\`. Compositional FID/KID is still
open -- see ROADMAP "Compositional image generation metrics (planned)".

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention + 4 register slots) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| Patch grid | 16x16 at 256 input (patch_size=16) |
| Slots (num_components, K) | 7 |
| Register slots (R) | 4 |
| Slot dim (latent_dim) | 1024 (matches SD2.1 cross_attention_dim) |
| UNet | sd2-community/stable-diffusion-2-1 (pretrained, frozen except K/V) |
| Trainable UNet params | cross-attn to_k + to_v only |
| Scheduler | SD2.1 DDPM (loaded from --pretrained_model_name/subfolder=scheduler) |
| Loss | mean-of-eps composition (eps = mean_k eps_slot_k, registers ride along) |
| Steps run | $MAX_STEPS / 200000 configured |
| Effective batch | 16 (2 GPU x $PER_GPU_BATCH) |
| Resolution | 256 (32x32 latent vs SD2.1's native 64x64) |
| Mixed precision | bf16 |
| Attention backend | PyTorch SDPA (FlashAttention-3 on H100) |
| torch.compile | inductor (--dynamo_backend=inductor) |
| Learning rate | 2.0e-5 |
| Dataset | COCO 2017 train ($TRAIN_N images, val=$VAL_N) |
| Output dir | $RUN_DIR/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | $TRAIN_RESULT | $TRAIN_DUR |

Final checkpoint: ${CKPT:-<none>}
Loss + slot-pairwise-cos curves: $LOSS_PNG

## Notes

- K=7 is a placeholder for COCO scene complexity; revise based on
  collapse / binding diagnostics from this run.
- The final DDP barrier can time out after the last checkpoint is written;
  run \`jobs/coco_coda_kv_only_eval_latest.sh\` against the latest complete
  checkpoint to collect COCO object metrics.
EOF

echo "**************** [coco-coda] report written to $REPORT ($OVERALL) ****************"

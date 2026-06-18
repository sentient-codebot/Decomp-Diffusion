#!/bin/bash
# MOVi-E 200k-step training + object-centric eval -- CoDA-style frozen UNet
# at 512 input resolution.
#
# Controlled follow-up to jobs/movi_e_coda_kv_only_train_eval.sh:
#   - Dataset source remains MOVi-E TFDS 256x256, read from WDS shards.
#   - The dataloader upsamples frames to 512x512.
#   - DINOv3 ViT-S/16 therefore produces a 32x32 slot-attention feature grid.
#   - SD2.1 VAE latents are 64x64x4 instead of the 256 run's 32x32x4.
#
# This closes the image/grid gap to CODA 2026 while intentionally keeping this
# repo's DINOv3 + SD2.1 K/V-only recipe. It is not a full CODA reproduction.
#
# Depends on shards laid down by jobs/movi_e_shard_wds.sh
# (~/prjs0993/datasets/movi-e-wds/).
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
#
# Submit from the repo root: `sbatch jobs/movi_e_coda_kv_only_512_train_eval.sh`.
#SBATCH --job-name="movi-e-coda-kv-512-200k"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
export WANDB_DIR="$HOME/prjs0993/Decomp-Diffusion/wandb"
export TORCHINDUCTOR_CACHE_DIR="$HOME/prjs0993/tmp/torchinductor"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR="${RUN_DIR:-results/movi-e_coda_kv_only_512}"
REPORT="${REPORT:-docs/experiments/2026-06-18-movi-e-coda-kv-only-512.md}"
MAX_STEPS="${MAX_STEPS:-200000}"
RESOLUTION=512
PER_GPU_BATCH="${PER_GPU_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH="${EVAL_BATCH:-4}"
METRICS_BATCH="${METRICS_BATCH:-8}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-16}"
mkdir -p "$RUN_DIR" "$WANDB_DIR" "$(dirname "$REPORT")"

# Image grids from eval.py are dumped here (hardcoded path); drop any stale
# symlink so they land in this worktree.
rm -rf image_test_output

# --- 0. Sanity check: dataset must already be preprocessed ---------------------
TRAIN_SHARD_ROOT=data/movi-e-wds/train
VAL_SHARD_ROOT=data/movi-e-wds/validation
if [ ! -f "$TRAIN_SHARD_ROOT/samples.jsonl" ]; then
    echo "[movi-e-coda-512] FATAL: $TRAIN_SHARD_ROOT/samples.jsonl missing -- run jobs/movi_e_shard_wds.sh first."
    exit 1
fi
TRAIN_N=$(wc -l < "$TRAIN_SHARD_ROOT/samples.jsonl")
VAL_N=$(wc -l < "$VAL_SHARD_ROOT/samples.jsonl" 2>/dev/null || echo 0)
echo "[movi-e-coda-512] train frames=$TRAIN_N  val frames=$VAL_N"
echo "[movi-e-coda-512] batch per GPU=$PER_GPU_BATCH grad_accum=$GRAD_ACCUM effective batch=$((2 * PER_GPU_BATCH * GRAD_ACCUM))"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

START=$(date +%s)

# --- 1. Train (DDP, 2 GPU -- no srun) -----------------------------------------
# Per-step UNet work is K=24 forwards with a (1+R)=5 long cross-attn sequence.
# If 512 OOMs, rerun with PER_GPU_BATCH=1 GRAD_ACCUM=8; keep model settings.
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision bf16 \
    --dynamo_backend=inductor \
    --main_process_port 29501 train_lsd.py \
    --train_config configs/movi-e/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/movi-e/dinov3_slot_encoder_d1024_512/config.json \
    --unet_config pretrain_sd \
    --freeze_unet_except_kv \
    --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
    --dataset_root "$TRAIN_SHARD_ROOT" \
    --dataset_glob '*.tar' \
    --dataset_format wds \
    --report_to wandb \
    --resolution "$RESOLUTION" \
    --train_batch_size "$PER_GPU_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --resume_from_checkpoint latest \
    --max_train_steps "$MAX_STEPS"
TRAIN_RC=$?
TRAIN_END=$(date +%s)
echo "**************** [movi-e-coda-512] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint -------------------------------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -z "$CKPT" ]; then
    CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
fi
echo "[movi-e-coda-512] using checkpoint: ${CKPT:-<none found>}"

# --- 3a. Eval: qualitative reconstruction grids -------------------------------
EVAL_START=$(date +%s)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision bf16 --seed 42 \
        --batch_size "$EVAL_BATCH" --num_validation_images "$NUM_VALIDATION_IMAGES" \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config pretrain_sd \
        --dataset_root "$VAL_SHARD_ROOT" \
        --dataset_glob '**/00000000_image.png' \
        --dataset_format wds --resolution "$RESOLUTION" \
        --ckpt_path "$CKPT"
    EVAL_RECON_RC=$?
else
    echo "[movi-e-coda-512] no checkpoint found -- skipping reconstruction eval."
    EVAL_RECON_RC=1
fi

if [ -d image_test_output ]; then
    mkdir -p "$RUN_DIR/eval_grids"
    cp image_test_output/*.jpg "$RUN_DIR/eval_grids/" 2>/dev/null
fi

# --- 3b. Eval: FG-ARI + mBO on full validation split --------------------------
METRICS_DIR="$RUN_DIR/metrics"
mkdir -p "$METRICS_DIR"
if [ -n "$CKPT" ]; then
    uv run python eval_movi.py \
        --ckpt_path "$CKPT" \
        --dataset_root data/movi-e-wds \
        --movi_eval_format wds \
        --split validation \
        --resolution "$RESOLUTION" \
        --batch_size "$METRICS_BATCH" \
        --num_workers 4 \
        --mixed_precision bf16 \
        --output_dir "$METRICS_DIR"
    EVAL_METRICS_RC=$?
else
    EVAL_METRICS_RC=1
fi
END=$(date +%s)
echo "**************** [movi-e-coda-512] eval finished (recon rc=$EVAL_RECON_RC, metrics rc=$EVAL_METRICS_RC). ****************"

# --- 4. Loss curve + wandb run url --------------------------------------------
LOSS_PNG="$RUN_DIR/loss_curve_${SLURM_JOB_ID}.png"
WANDB_URL_FILE="$RUN_DIR/wandb_url.txt"
uv run python - "$RUN_DIR" "$LOSS_PNG" "$WANDB_URL_FILE" <<'PYEOF'
import os
import sys

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
        "CoDA K/V-only (SD2.1 frozen UNet + DINOv3 slots @512, R=4) -- MOVi-E"
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

read FG_ARI MBO MIOU N_IMG <<<$(uv run python - "$METRICS_DIR/metrics.json" <<'PYEOF'
import json
import os
import sys

p = sys.argv[1]
if not os.path.exists(p):
    print("N/A N/A N/A N/A")
    raise SystemExit
m = json.load(open(p))
print(f"{m['fg_ari']:.4f} {m['mbo']:.4f} {m['miou']:.4f} {m['n_images_ari']}")
PYEOF
)

# --- 5. Write the experiment report -------------------------------------------
fmt_dur() { date -u -d "@$1" +%H:%M:%S; }
TRAIN_DUR=$(fmt_dur $((TRAIN_END - START)))
EVAL_DUR=$(fmt_dur $((END - EVAL_START)))

[ "$TRAIN_RC" -eq 0 ] && TRAIN_RESULT="PASS" || TRAIN_RESULT="FAIL (rc=$TRAIN_RC)"
[ "$EVAL_RECON_RC" -eq 0 ] && EVAL_RECON_RESULT="PASS" || EVAL_RECON_RESULT="FAIL (rc=$EVAL_RECON_RC)"
[ "$EVAL_METRICS_RC" -eq 0 ] && EVAL_METRICS_RESULT="PASS" || EVAL_METRICS_RESULT="FAIL (rc=$EVAL_METRICS_RC)"
if [ "$TRAIN_RC" -eq 0 ] && [ "$EVAL_RECON_RC" -eq 0 ] && [ "$EVAL_METRICS_RC" -eq 0 ]; then
    OVERALL="PASS -- full run completed, train + both evals clean"
else
    OVERALL="FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# MOVi-E CoDA-style K/V-only 512 training -- 200k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_coda_kv_only_512_train_eval.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 2x H100
**wandb run:** $WANDB_URL

## Purpose

Controlled 512-resolution follow-up to the 256px MOVi-E CoDA-style K/V-only
run. This keeps the DINOv3 + SD2.1 frozen-UNet recipe fixed while increasing
the encoder feature grid from 16 x 16 to 32 x 32 and the diffusion latent grid
from 32 x 32 x 4 to 64 x 64 x 4.

This closes the main resolution/grid gap to CODA 2026. It does not reproduce
CODA's DINOv2 backbone, SD1.5 decoder, CLIP padding registers, 768-dim slots,
or contrastive alignment loss.

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention + 4 learned register slots) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| MOVi-E source frames | 256 x 256 TFDS release, upsampled by the dataset transform |
| Model input resolution | 512 x 512 |
| Patch grid | 32 x 32 at 512 input (patch_size=16) |
| Slots (num_components, K) | 24 |
| Register slots (R) | 4 |
| Slot dim (latent_dim) | 1024 (matches SD2.1 cross_attention_dim) |
| UNet | sd2-community/stable-diffusion-2-1 (pretrained, frozen except K/V) |
| Trainable UNet params | cross-attn to_k + to_v only (warm-started from SD text-conditioning weights) |
| Scheduler | SD2.1 DDPM (loaded from --pretrained_model_name/subfolder=scheduler) |
| Loss | mean-of-eps composition (eps = mean_k eps_slot_k, registers ride along) |
| Steps run | $MAX_STEPS / 200000 configured |
| Per-GPU batch / grad accumulation | $PER_GPU_BATCH / $GRAD_ACCUM |
| Effective batch | $((2 * PER_GPU_BATCH * GRAD_ACCUM)) (2 GPU x batch x accumulation) |
| Diffusion latent grid | 64 x 64 x 4 |
| Mixed precision | bf16 |
| Learning rate | 2.0e-5 |
| Dataset | MOVi-E train shards ($TRAIN_N frames) |
| Eval split | MOVi-E validation shards ($VAL_N frames) |
| Output dir | $RUN_DIR/latent_decomposed_diffusion/ |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | $TRAIN_RESULT | $TRAIN_DUR |
| Reconstruction eval | $EVAL_RECON_RESULT | $EVAL_DUR |
| Object-centric metrics | $EVAL_METRICS_RESULT | incl. above |

### Object-centric metrics (eval_movi.py, validation split)

| Metric | 512 DINOv3 K/V-only |
|--------|---------------------|
| FG-ARI | $FG_ARI |
| mBO | $MBO |
| mIoU | $MIOU |
| Frames | $N_IMG |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg

### Reconstruction grids and curves

Final checkpoint: ${CKPT:-<none>}
Loss + slot-pairwise-cos curves: $LOSS_PNG
Reconstruction grids: $RUN_DIR/eval_grids/image_NN.jpg
Per-step validation viz: $RUN_DIR/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics + viz:
     - Did the denser 32 x 32 DINOv3 feature grid improve FG-ARI / mBO?
     - Did slot_pairwise_cos stay low or trend toward collapse?
     - Did the 64 x 64 diffusion latent grid improve reconstructions enough to
       justify the extra runtime?
     - Compare against docs/experiments/2026-05-27-movi-e-coda-kv-only.md. -->

## Notes

- If the full 512 run OOMs, rerun with PER_GPU_BATCH=1 GRAD_ACCUM=8. Keep the
  model, resolution, and step budget fixed so the comparison remains clean.
- This run uses upsampled 256px MOVi-E frames. It tests feature-grid density and
  SD latent resolution, not access to a higher-fidelity MOVi-E render.
EOF

echo "**************** [movi-e-coda-512] report written to $REPORT ($OVERALL) ****************"

#!/bin/bash
# MOVi-E 200k-step training + object-centric eval -- CoDA-style frozen UNet.
#
# Same encoder/dataset/budget as jobs/movi_e_dinov3_registers_train_eval.sh,
# but the diffusion side is rewired:
#   - UNet: SD 2.1 pretrained (sd2-community/stable-diffusion-2-1), frozen
#     except for cross-attention to_k / to_v projections.
#   - Slot encoder: DinoSlotAttentionEncoder with latent_dim=1024 (matches
#     SD2.1 cross_attention_dim). Slot-attention head is trained; DINOv3
#     backbone stays frozen as before.
#   - Loss: mean-of-eps composition (eps = mean_k eps_slot_k). Mean (not
#     sum) keeps each per-slot eps on a unit-noise scale, which is required
#     for a mostly-frozen pretrained denoiser to stay in its trained output
#     regime. Previous runs with summed eps inflated targets by ~K and asked
#     the frozen UNet to predict ~noise/K per slot -- mostly unlearnable.
#
# Motivation: study how slot representations evolve when the denoiser is a
# fixed strong prior. All gradient pressure flows through K/V into the
# encoder; nothing else in the UNet can compensate for a weak encoder.
#
# Depends on data laid down by jobs/movi_e_preprocess.sh
# (~/prjs0993/datasets/movi-e/).
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
# When K/V are trained, the UNet IS handed to accelerator.prepare and so its
# updated K/V weights are written into each checkpoint-* directory.
#
# Submit from the repo root: `sbatch jobs/movi_e_coda_kv_only_train_eval.sh`.
#SBATCH --job-name="movi-e-coda-kv-only-200k"
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
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/movi-e_coda_kv_only
REPORT=docs/experiments/2026-05-27-movi-e-coda-kv-only.md
MAX_STEPS=200000
RESOLUTION=256
PER_GPU_BATCH=8
mkdir -p "$RUN_DIR" "$WANDB_DIR" "$(dirname "$REPORT")"

# Image grids from eval.py are dumped here (hardcoded path); drop any stale
# symlink so they land in this worktree.
rm -rf image_test_output

# --- 0. Sanity check: dataset must already be preprocessed -------------------
TRAIN_IMG_ROOT=data/movi-e/movi-e-train-with-label/images
VAL_IMG_ROOT=data/movi-e/movi-e-validation-with-label/images
if [ ! -d "$TRAIN_IMG_ROOT" ]; then
    echo "[movi-e-coda] FATAL: $TRAIN_IMG_ROOT is missing -- run jobs/movi_e_preprocess.sh first."
    exit 1
fi
TRAIN_N=$(find "$TRAIN_IMG_ROOT" -name '*_image.png' | wc -l)
VAL_N=$(find "$VAL_IMG_ROOT" -name '*_image.png' 2>/dev/null | wc -l)
echo "[movi-e-coda] train frames=$TRAIN_N  val frames=$VAL_N"

echo "[movi-e-coda] priming GPFS metadata..."
find "$TRAIN_IMG_ROOT" "$VAL_IMG_ROOT" -type f >/dev/null
echo "[movi-e-coda] done priming."

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

START=$(date +%s)

# --- 1. Train (DDP, 2 GPU -- no srun) ----------------------------------------
# Per-step UNet work is K=24 forwards with a (1+R)=5 long cross-attn sequence.
# Scheduler config path is required by argparse but ignored: train_lsd.py
# loads SD2.1's own DDPM scheduler when --unet_config=pretrain_sd.
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision fp16 \
    --main_process_port 29501 train_lsd.py \
    --train_config configs/movi-e/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/movi-e/dinov3_slot_encoder_d1024/config.json \
    --unet_config pretrain_sd \
    --freeze_unet_except_kv \
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
echo "**************** [movi-e-coda] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -z "$CKPT" ]; then
    CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
fi
echo "[movi-e-coda] using checkpoint: ${CKPT:-<none found>}"

# --- 3a. Eval: qualitative reconstruction grids ------------------------------
EVAL_START=$(date +%s)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision fp16 --seed 42 \
        --batch_size 8 --num_validation_images 16 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
        --dataset_root "$VAL_IMG_ROOT" \
        --dataset_glob '**/00000000_image.png' --resolution "$RESOLUTION" \
        --ckpt_path "$CKPT"
    EVAL_RECON_RC=$?
else
    echo "[movi-e-coda] no checkpoint found -- skipping reconstruction eval."
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
echo "**************** [movi-e-coda] eval finished (recon rc=$EVAL_RECON_RC, metrics rc=$EVAL_METRICS_RC). ****************"

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
        "CoDA K/V-only (SD2.1 frozen UNet + DINOv3 slots @256, R=4) -- MOVi-E"
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
    OVERALL="FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# MOVi-E CoDA-style K/V-only training -- 200k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_coda_kv_only_train_eval.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 2x H100
**wandb run:** $WANDB_URL

## Purpose

CoDA-style ablation against the
\`2026-05-26-movi-e-dinov3-registers.md\` baseline: swap the
trained-from-scratch UNet for a frozen pretrained SD 2.1 UNet, train
only the cross-attention \`to_k\` / \`to_v\` projections (warm-started
from SD's text-conditioned weights), and let all gradient pressure flow
through K/V into the slot encoder. Tests whether a strong fixed denoiser
prior pushes the encoder to learn better object-level representations.

## Configuration

| Item | Value |
|------|-------|
| Encoder | DinoSlotAttentionEncoder (DINOv3 ViT-S/16 frozen + soft pos-embed + Slot Attention + 4 register slots) |
| DINO model | facebook/dinov3-vits16-pretrain-lvd1689m |
| Patch grid | 16x16 at 256 input (patch_size=16) |
| Slots (num_components, K) | 24 |
| Register slots (R) | 4 |
| Slot dim (latent_dim) | 1024 (matches SD2.1 cross_attention_dim) |
| UNet | sd2-community/stable-diffusion-2-1 (pretrained, frozen except K/V) |
| Trainable UNet params | cross-attn to_k + to_v only (warm-started from SD text-conditioning weights) |
| Scheduler | SD2.1 DDPM (loaded from --pretrained_model_name/subfolder=scheduler) |
| Loss | sum-of-eps composition (eps = sum_k eps_slot_k, registers ride along) |
| Steps run | $MAX_STEPS / 200000 configured |
| Effective batch | 16 (2 GPU x $PER_GPU_BATCH) |
| Resolution | 256 (32x32 latent vs SD2.1's native 64x64) |
| Mixed precision | fp16 |
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

| Metric | CoDA K/V-only | DINOv3 + R=4 (full UNet) |
|--------|---------------|--------------------------|
| FG-ARI | $FG_ARI | see 2026-05-26-movi-e-dinov3-registers.md |
| mBO    | $MBO   | see 2026-05-26-movi-e-dinov3-registers.md |
| Frames | $N_IMG | -- |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg

### Reconstruction grids and curves

Final checkpoint: ${CKPT:-<none>}
Loss + slot-pairwise-cos curves: $LOSS_PNG
Reconstruction grids: $RUN_DIR/eval_grids/image_NN.jpg
Per-step validation viz: $RUN_DIR/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics + viz:
     - Did the frozen-denoiser prior improve slot quality (FG-ARI / mBO up)?
     - Did slot_pairwise_cos stay low (slots specialise) or trend toward 1?
     - Did val_loss flatten earlier than the full-UNet baseline (capacity
       bottleneck) or track it (encoder is the limiting factor in both)? -->

## Notes

- K/V are warm-started from SD2.1's CLIP-text-conditioned weights. This
  is a deliberate choice (see TECHDEBT.md) rather than a principled one;
  a re-init ablation could disentangle "frozen denoiser prior" from
  "text-style K/V prior."
- 256px input means the SD2.1 UNet runs on a 32x32 latent vs its native
  64x64. SD UNets handle non-native sizes fine but the pretraining
  prior is strongest at 64x64.
EOF

echo "**************** [movi-e-coda] report written to $REPORT ($OVERALL) ****************"

#!/bin/bash
# MOVi-E adaptive epsilon composition -- 50k warm-start continuation.
#
# Starts from the latest checkpoint produced by
# jobs/movi_e_coda_kv_only_train_eval.sh, but writes to a separate output tree.
# This is a weight-only warm start: optimizer, scheduler, and global step start
# fresh unless this job is resubmitted and can resume its own checkpoints.
#
# Submit from the repo root:
#   sbatch jobs/movi_e_adaptive_eps_slot_attn_warmstart_train_eval.sh
#SBATCH --job-name="movi-e-adapt-eps-50k"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --time=12:00:00
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

RUN_DIR=results/movi-e_adaptive_eps_slot_attn_warmstart
TRACKER_PROJECT_NAME=latent_decomposed_diffusion_adaptive_eps
SOURCE_RUN_DIR=results/movi-e_coda_kv_only/latent_decomposed_diffusion
REPORT=docs/experiments/2026-05-28-movi-e-adaptive-eps-slot-attn.md
MAX_STEPS=50000
RESOLUTION=256
PER_GPU_BATCH=8
mkdir -p "$RUN_DIR" "$WANDB_DIR" "$(dirname "$REPORT")"

latest_checkpoint() {
    local root=$1
    local ckpt
    ckpt=$(find "$root" -maxdepth 1 -type d -name 'checkpoint-*-last' 2>/dev/null \
        | sort -t- -k2 -n \
        | tail -n1)
    if [ -z "$ckpt" ]; then
        ckpt=$(find "$root" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
            | sort -t- -k2 -n \
            | tail -n1)
    fi
    printf '%s\n' "$ckpt"
}

WARM_CKPT=${WARM_START_CHECKPOINT:-$(latest_checkpoint "$SOURCE_RUN_DIR")}
if [ -z "$WARM_CKPT" ]; then
    echo "[movi-e-adapt-eps] FATAL: no warm-start checkpoint found under $SOURCE_RUN_DIR"
    exit 1
fi
if [ ! -d "$WARM_CKPT/dinoslotattentionencoder" ] || [ ! -d "$WARM_CKPT/unet2dconditionmodel" ]; then
    echo "[movi-e-adapt-eps] FATAL: checkpoint appears incomplete: $WARM_CKPT"
    exit 1
fi
echo "[movi-e-adapt-eps] warm-start checkpoint: $WARM_CKPT"

# Image grids from eval.py are dumped here (hardcoded path); drop any stale
# symlink so they land in this worktree.
rm -rf image_test_output

TRAIN_SHARD_ROOT=data/movi-e-wds/train
VAL_SHARD_ROOT=data/movi-e-wds/validation
if [ ! -f "$TRAIN_SHARD_ROOT/samples.jsonl" ]; then
    echo "[movi-e-adapt-eps] FATAL: $TRAIN_SHARD_ROOT/samples.jsonl missing -- run jobs/movi_e_shard_wds.sh first."
    exit 1
fi
TRAIN_N=$(wc -l < "$TRAIN_SHARD_ROOT/samples.jsonl")
VAL_N=$(wc -l < "$VAL_SHARD_ROOT/samples.jsonl" 2>/dev/null || echo 0)
echo "[movi-e-adapt-eps] train frames=$TRAIN_N  val frames=$VAL_N"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

START=$(date +%s)

# --- 1. Train (DDP, 2 GPU -- no srun) ----------------------------------------
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision bf16 \
    --dynamo_backend=inductor \
    --main_process_port 29501 train_lsd.py \
    --train_config configs/movi-e/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --tracker_project_name "$TRACKER_PROJECT_NAME" \
    --latent_encoder_config configs/movi-e/dinov3_slot_encoder_d1024/config.json \
    --unet_config pretrain_sd \
    --freeze_unet_except_kv \
    --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
    --dataset_root "$TRAIN_SHARD_ROOT" \
    --dataset_glob '*.tar' \
    --dataset_format wds \
    --report_to wandb \
    --resolution "$RESOLUTION" \
    --train_batch_size "$PER_GPU_BATCH" \
    --warm_start_checkpoint "$WARM_CKPT" \
    --epsilon_composition slot_attn \
    --resume_from_checkpoint latest \
    --max_train_steps "$MAX_STEPS"
TRAIN_RC=$?
TRAIN_END=$(date +%s)
echo "**************** [movi-e-adapt-eps] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate this run's final checkpoint -----------------------------------
OUT_SUBDIR="$RUN_DIR/$TRACKER_PROJECT_NAME"
CKPT=$(find "$OUT_SUBDIR" -maxdepth 1 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -z "$CKPT" ]; then
    CKPT=$(latest_checkpoint "$OUT_SUBDIR")
fi
echo "[movi-e-adapt-eps] using checkpoint: ${CKPT:-<none found>}"

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
        --epsilon_composition slot_attn \
        --ckpt_path "$CKPT"
    EVAL_RECON_RC=$?
else
    echo "[movi-e-adapt-eps] no checkpoint found -- skipping reconstruction eval."
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
echo "**************** [movi-e-adapt-eps] eval finished (recon rc=$EVAL_RECON_RC, metrics rc=$EVAL_METRICS_RC). ****************"

# --- 4. Loss curve + wandb run url -------------------------------------------
LOSS_PNG="$RUN_DIR/loss_curve_${SLURM_JOB_ID}.png"
WANDB_URL_FILE="$RUN_DIR/wandb_url.txt"
uv run python - "$RUN_DIR" "$TRACKER_PROJECT_NAME" "$LOSS_PNG" "$WANDB_URL_FILE" <<'PYEOF'
import os, sys

run_dir, tracker_project_name, out_png, url_file = sys.argv[1:5]
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    api = wandb.Api()
    target = os.path.join(run_dir, tracker_project_name)
    runs = list(
        api.runs(
            f"nan-team/{tracker_project_name}",
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
        keys=["loss", "val_loss", "slot_pairwise_cos"],
        samples=50000,
        pandas=False,
    )

    def series(key):
        return sorted((r["_step"], r[key]) for r in rows if r.get(key) is not None)

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
    ax_loss.set_title("Adaptive slot-attn epsilon weights -- MOVi-E")
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
    print(f"[loss-curve] wrote {out_png}")
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
[ "$EVAL_RECON_RC" -eq 0 ] && EVAL_RECON_RESULT="PASS" || EVAL_RECON_RESULT="FAIL (rc=$EVAL_RECON_RC)"
[ "$EVAL_METRICS_RC" -eq 0 ] && EVAL_METRICS_RESULT="PASS" || EVAL_METRICS_RESULT="FAIL (rc=$EVAL_METRICS_RC)"
if [ "$TRAIN_RC" -eq 0 ] && [ "$EVAL_RECON_RC" -eq 0 ] && [ "$EVAL_METRICS_RC" -eq 0 ]; then
    OVERALL="PASS -- 50k warm-start continuation completed"
else
    OVERALL="FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# MOVi-E adaptive epsilon composition -- slot-attention weights

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_adaptive_eps_slot_attn_warmstart_train_eval.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 2x H100
**wandb run:** $WANDB_URL

## Purpose

First adaptive epsilon-composition trial from ROADMAP.md. Reuses encoder Slot
Attention masks as spatial composition weights, bilinearly interpolates them to
UNet epsilon resolution, and stops gradients through the weights with detach().
The run warm-starts model weights from the CoDA K/V-only checkpoint but writes
to a separate output folder and trains for a fresh 50k steps.

## Configuration

| Item | Value |
|------|-------|
| Warm-start checkpoint | $WARM_CKPT |
| Output dir | $OUT_SUBDIR/ |
| Encoder | DinoSlotAttentionEncoder d1024, DINOv3 frozen, K=24, R=4 |
| UNet | SD2.1 pretrained, frozen except cross-attn K/V |
| Epsilon composition | detached Slot Attention masks, interpolated to latent resolution |
| Steps run | $MAX_STEPS fresh continuation steps |
| Effective batch | 16 (2 GPU x $PER_GPU_BATCH) |
| Resolution | $RESOLUTION |
| Dataset | MOVi-E train shards ($TRAIN_N frames) |
| Eval split | MOVi-E validation shards ($VAL_N frames) |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Training | $TRAIN_RESULT | $TRAIN_DUR |
| Reconstruction eval | $EVAL_RECON_RESULT | $EVAL_DUR |
| Object-centric metrics | $EVAL_METRICS_RESULT | included above |

### Object-centric metrics

| Metric | Value |
|--------|-------|
| FG-ARI | $FG_ARI |
| mBO | $MBO |
| Frames | $N_IMG |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg
Final checkpoint: ${CKPT:-<none>}
Loss + slot-pairwise-cos curves: $LOSS_PNG
Reconstruction grids: $RUN_DIR/eval_grids/image_NN.jpg
Per-step validation viz: $OUT_SUBDIR/logs/

## Assessment

Pending review after the Slurm job completes.
EOF

echo "**************** [movi-e-adapt-eps] report written to $REPORT ($OVERALL) ****************"

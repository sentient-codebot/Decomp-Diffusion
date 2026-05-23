#!/bin/bash
# MOVi-E 200k-step training + object-centric eval -- SlotAttentionEncoder.
#
# The roadmap "Dataset: MOVi-E" run: first multi-object dataset with GT
# segmentation, so the decomposition can be measured with FG-ARI and mBO
# (eval_movi.py) instead of only qualitative grids. Pipeline mirrors the
# celebahq slot-attention run, with the dataset, slot count (11) and step
# budget (200k) swapped.
#
# 4x H100 with --train_batch_size 16 -> effective batch 64. ~6h compute for
# train at the celebahq run's ~9 steps/s; 24h walltime for margin (MOVi-E
# loads from per-frame PNGs, so dataloader throughput may be the bottleneck).
#
# Depends on data laid down by jobs/movi_e_preprocess.sh (~/prjs0993/datasets/movi-e/).
#
# Restart-safe: --resume_from_checkpoint latest picks up the most recent
# checkpoint if the job is resubmitted after a timeout or node failure.
#
# Submit from the repo root: `sbatch jobs/movi_e_slot_attn_train_eval.sh`.
#SBATCH --job-name="movi-e-slot-200k"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion-movi

# uv-managed env; keep the HF cache off the home quota.
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/movi-e_slot
REPORT=docs/experiments/2026-05-23-movi-e-slot-attention.md
MAX_STEPS=200000
mkdir -p "$(dirname "$REPORT")"

# Image grids from eval.py are dumped here (path is hardcoded in eval.py);
# drop any stale symlink so they land in this worktree.
rm -rf image_test_output

# --- 0. Sanity check: dataset must already be preprocessed -------------------
TRAIN_IMG_ROOT=data/movi-e/movi-e-train-with-label/images
VAL_IMG_ROOT=data/movi-e/movi-e-validation-with-label/images
if [ ! -d "$TRAIN_IMG_ROOT" ]; then
    echo "[movi-e-run] FATAL: $TRAIN_IMG_ROOT is missing -- run jobs/movi_e_preprocess.sh first."
    exit 1
fi
TRAIN_N=$(find "$TRAIN_IMG_ROOT" -name '*_image.png' | wc -l)
VAL_N=$(find "$VAL_IMG_ROOT" -name '*_image.png' 2>/dev/null | wc -l)
echo "[movi-e-run] train frames=$TRAIN_N  val frames=$VAL_N"

START=$(date +%s)

# --- 1. Train (DDP, 4 GPU -- no srun) ----------------------------------------
# --train_batch_size 16 over 4 GPUs -> effective batch 64.
uv run accelerate launch --multi_gpu --num_processes=4 --mixed_precision fp16 \
    --main_process_port 29500 train_lsd.py \
    --train_config configs/movi-e/train_config.yaml \
    --output_dir "$RUN_DIR/" \
    --latent_encoder_config configs/movi-e/slot_encoder/config.json \
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
echo "**************** [movi-e-run] training finished (rc=$TRAIN_RC). ****************"

# --- 2. Locate the final checkpoint ------------------------------------------
# train_lsd.py appends --tracker_project_name to --output_dir (src/parser.py),
# so checkpoints land in $RUN_DIR/<tracker_project_name>/checkpoint-*.
CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*-last' 2>/dev/null | sort -t- -k2 -n | tail -n1)
if [ -z "$CKPT" ]; then
    CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -n1)
fi
echo "[movi-e-run] using checkpoint: ${CKPT:-<none found>}"

# --- 3a. Eval: qualitative reconstruction grids (existing eval.py) -----------
EVAL_START=$(date +%s)
if [ -n "$CKPT" ]; then
    uv run accelerate launch --num_processes=1 eval.py \
        --mixed_precision fp16 --seed 42 \
        --batch_size 16 --num_validation_images 16 \
        --output_dir "$RUN_DIR/gen_images" \
        --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
        --dataset_root "$VAL_IMG_ROOT" \
        --dataset_glob '**/00000000_image.png' --resolution 128 \
        --ckpt_path "$CKPT"
    EVAL_RECON_RC=$?
else
    echo "[movi-e-run] no checkpoint found -- skipping reconstruction eval."
    EVAL_RECON_RC=1
fi

# Keep the eval grids with the run (image_test_output is gitignored/transient).
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
echo "**************** [movi-e-run] eval finished (recon rc=$EVAL_RECON_RC, metrics rc=$EVAL_METRICS_RC). ****************"

# --- 4. Loss curve + wandb run url -------------------------------------------
# Metrics are logged to wandb (project: latent_decomposed_diffusion). Pull the
# loss history back via the wandb API for a self-contained PNG; best-effort, so
# a wandb/network hiccup never fails the run.
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
    run = runs[0]  # most recent matching run
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
    plt.title("SlotAttentionEncoder -- MOVi-E training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"[loss-curve] wrote {out_png} ({len(steps)} points, final loss {vals[-1]:.4f})")
except Exception as e:
    print(f"[loss-curve] wandb pull failed: {e}")
PYEOF
WANDB_URL=$(cat "$WANDB_URL_FILE" 2>/dev/null || echo "N/A")

# Pull the metric numbers out for the report header.
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
# MOVi-E slot-attention encoder -- 200k-step run

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_slot_attn_train_eval.sh, log: slurm_${SLURM_JOB_ID}.log)
**Node / GPUs:** ${SLURM_JOB_NODELIST:-N/A}, 4x H100
**wandb run:** $WANDB_URL

## Purpose

First run on MOVi-E (Kubric, up to 23 rigid objects with per-pixel
instance masks). Lets the slot decomposition be measured with proper
object-centric metrics instead of only qualitative grids -- the
\`SlotAttentionEncoder\` (image -> CNN -> soft pos-embed -> Slot Attention)
binds the feature map into 11 slots, and the per-slot attention masks are
compared against the dumped GT instance segmentations.

## Configuration

| Item | Value |
|------|-------|
| Encoder | SlotAttentionEncoder (CNN + soft pos-embed + Slot Attention) |
| Slot Attention iters | 3 |
| Steps run | $MAX_STEPS / 200000 configured |
| Effective batch | 64 (4 GPU x 16) |
| Resolution | 128 |
| Slots (num_components) | 11 |
| Slot dim (latent_dim) | 64 |
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

| Metric | Value |
|--------|-------|
| FG-ARI | $FG_ARI |
| mBO    | $MBO |
| Frames | $N_IMG |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg

### Reconstruction grids

Final checkpoint: ${CKPT:-<none>}
Loss curve: $LOSS_PNG
Reconstruction grids: $RUN_DIR/eval_grids/image_NN.jpg -- each row is
[input | slot 0 | ... | slot 10 | reconstruction]
Per-step validation viz: $RUN_DIR/latent_decomposed_diffusion/logs/

## Assessment

<!-- Fill in after reviewing metrics.json, viz_*.jpg and the loss curve:
     - FG-ARI / mBO numbers in absolute terms and vs slot-attention literature
       (Locatello FG-ARI ~ 0.6+ for MOVi-A; MOVi-E is harder, 0.3-0.5 is the
       typical band for a from-scratch CNN backbone).
     - Do slot attention masks visually align with object boundaries?
     - Reconstruction grids: do per-slot reconstructions decompose into
       distinguishable objects, or does the model collapse to averaged blobs? -->

## Notes

- Eval uses pre-renormalisation slot-attention competition weights as soft
  masks; argmax over slots gives the predicted hard mask. Masks live at
  feature-map resolution (16x16 for 128 input) and are bilinearly upsampled
  to 128 before comparison with GT.
- 11 slots vs up to 23 GT objects: some scenes have more objects than
  slots, capping the achievable FG-ARI/mBO.
- eval.py writes grids to ./image_test_output/ (hardcoded); they are copied
  to $RUN_DIR/eval_grids/ here. See TECHDEBT.md.

## Next steps

- Review the metrics + viz, fill in the Assessment section.
- Roadmap next stage: replace the CNN feature extractor with a pretrained
  encoder (e.g. DINO) -- see ROADMAP.md "Encoder: pretrained feature
  extractor".
EOF

echo "**************** [movi-e-run] report written to $REPORT ($OVERALL) ****************"

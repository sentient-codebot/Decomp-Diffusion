#!/bin/bash
# Eval-only COCO object metrics for the latest CoDA K/V-only checkpoint.
#
# Submit from the repo root:
#   sbatch jobs/coco_coda_kv_only_eval_latest.sh
#
# The training job reached checkpoint-200000-last but exited after a final DDP
# barrier timeout, so evaluation is split out here.
#SBATCH --job-name="coco-coda-metrics"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
export WANDB_DIR="$HOME/prjs0993/Decomp-Diffusion/wandb"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/coco_coda_kv_only
REPORT=docs/experiments/2026-05-28-coco-coda-kv-only-eval.md
COCO_ROOT="$HOME/prjs0993/datasets/coco"
RESOLUTION=256
BATCH_SIZE=16
EVAL_TAG="${SLURM_JOB_ID:-manual}"
METRICS_DIR="$RUN_DIR/coco_metrics_latest_$EVAL_TAG"
mkdir -p "$METRICS_DIR" "$(dirname "$REPORT")"

latest_checkpoint() {
    find "$RUN_DIR" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null \
        | while IFS= read -r path; do
            base=${path##*/checkpoint-}
            step=${base%-last}
            case "$step" in
                ''|*[!0-9]*) continue ;;
            esac
            printf '%s\t%s\n' "$step" "$path"
        done \
        | sort -n \
        | tail -n1 \
        | cut -f2-
}

if [ ! -f "$COCO_ROOT/annotations/instances_val2017.json" ]; then
    echo "[coco-coda-eval] FATAL: COCO annotations missing under $COCO_ROOT"
    exit 1
fi
VAL_N=$(find "$COCO_ROOT/images/val2017" -maxdepth 1 -name '*.jpg' | wc -l)
echo "[coco-coda-eval] COCO val images=$VAL_N"
if [ "$VAL_N" -lt 1000 ]; then
    echo "[coco-coda-eval] FATAL: only $VAL_N val images found -- dataset looks incomplete."
    exit 1
fi

CKPT=$(latest_checkpoint)
echo "[coco-coda-eval] using checkpoint: ${CKPT:-<none found>}"
if [ -z "$CKPT" ]; then
    echo "[coco-coda-eval] no checkpoint found under $RUN_DIR"
    exit 1
fi

if [ ! -f "$CKPT/dinoslotattentionencoder/diffusion_pytorch_model.safetensors" ]; then
    echo "[coco-coda-eval] checkpoint appears incomplete: $CKPT"
    exit 1
fi

START=$(date +%s)
uv run python eval_coco.py \
    --ckpt_path "$CKPT" \
    --dataset_root "$COCO_ROOT" \
    --split val2017 \
    --resolution "$RESOLUTION" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 4 \
    --mixed_precision bf16 \
    --num_viz 8 \
    --min_category_count 5 \
    --output_dir "$METRICS_DIR"
EVAL_RC=$?
END=$(date +%s)
echo "**************** [coco-coda-eval] eval finished (rc=$EVAL_RC). ****************"

read FG_ARI MBO MIOU MIOU_FG OBJ_MBO N_IMG N_OBJ SLOT_ENTROPY <<<$(uv run python - "$METRICS_DIR/metrics.json" <<'PYEOF'
import json
import os
import sys

p = sys.argv[1]
if not os.path.exists(p):
    print("N/A N/A N/A N/A N/A N/A N/A N/A")
    raise SystemExit
m = json.load(open(p))

def fmt(key):
    value = m.get(key)
    return "N/A" if value is None else f"{value:.4f}"

print(
    fmt("fg_ari"),
    fmt("mbo"),
    fmt("miou"),
    fmt("miou_fg"),
    fmt("object_mbo"),
    m.get("n_images", "N/A"),
    m.get("n_objects", "N/A"),
    fmt("slot_entropy"),
)
PYEOF
)

fmt_dur() { date -u -d "@$1" +%H:%M:%S; }
EVAL_DUR=$(fmt_dur $((END - START)))
[ "$EVAL_RC" -eq 0 ] && RESULT="PASS" || RESULT="FAIL (rc=$EVAL_RC)"

cat > "$REPORT" <<EOF
# COCO CoDA K/V-only latest-checkpoint object metrics

**Status:** $RESULT
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/coco_coda_kv_only_eval_latest.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log)
**Training job:** 23177437 (script: jobs/coco_coda_kv_only_train.sh)
**Checkpoint:** $CKPT

## Results

| Metric | Value |
|--------|-------|
| FG-ARI | $FG_ARI |
| mBO | $MBO |
| mIoU | $MIOU |
| mIoU foreground-only | $MIOU_FG |
| Object-weighted mBO | $OBJ_MBO |
| Images | $N_IMG |
| Objects | $N_OBJ |
| Slot entropy | $SLOT_ENTROPY |
| Wall time | $EVAL_DUR |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg

## Notes

- Metrics are computed by eval_coco.py from COCO val2017 instance polygons.
- Crowd/RLE annotations are skipped; standard non-crowd COCO instances are
  rasterized into one integer label map per image.
- The report includes aggregate metrics; metrics.json also contains
  per-category mBO, COCO-size-bin mBO, foreground fraction, and slot usage.
EOF

echo "[coco-coda-eval] report written to $REPORT ($RESULT)"
exit "$EVAL_RC"

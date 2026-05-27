#!/bin/bash
# Eval-only MOVi-E object metrics for the latest CoDA K/V-only checkpoint.
#
# Submit from the repo root: `sbatch jobs/movi_e_coda_kv_only_eval_latest.sh`.
# The script resolves the latest checkpoint at job start, runs full validation
# FG-ARI/mBO/mIoU with eval_movi.py, and writes a short experiment report.
#SBATCH --job-name="movi-e-coda-fgari"
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

RUN_DIR=results/movi-e_coda_kv_only
REPORT=docs/experiments/2026-05-27-movi-e-coda-kv-only-latest-eval.md
RESOLUTION=256
METRICS_DIR="$RUN_DIR/metrics_latest_${SLURM_JOB_ID:-manual}"
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

CKPT=$(latest_checkpoint)
echo "[movi-e-coda-eval] using checkpoint: ${CKPT:-<none found>}"
if [ -z "$CKPT" ]; then
    echo "[movi-e-coda-eval] no checkpoint found under $RUN_DIR"
    exit 1
fi

if [ ! -f "$CKPT/dinoslotattentionencoder/diffusion_pytorch_model.safetensors" ]; then
    echo "[movi-e-coda-eval] checkpoint appears incomplete: $CKPT"
    exit 1
fi

START=$(date +%s)
uv run python eval_movi.py \
    --ckpt_path "$CKPT" \
    --dataset_root data/movi-e \
    --split validation \
    --resolution "$RESOLUTION" \
    --batch_size 16 \
    --num_workers 4 \
    --output_dir "$METRICS_DIR"
EVAL_RC=$?
END=$(date +%s)
echo "**************** [movi-e-coda-eval] eval finished (rc=$EVAL_RC). ****************"

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

fmt_dur() { date -u -d "@$1" +%H:%M:%S; }
EVAL_DUR=$(fmt_dur $((END - START)))
[ "$EVAL_RC" -eq 0 ] && RESULT="PASS" || RESULT="FAIL (rc=$EVAL_RC)"

cat > "$REPORT" <<EOF
# MOVi-E CoDA K/V-only latest-checkpoint object metrics

**Status:** $RESULT
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_coda_kv_only_eval_latest.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log)
**Checkpoint:** $CKPT

## Results

| Metric | Value |
|--------|-------|
| FG-ARI | $FG_ARI |
| mBO | $MBO |
| mIoU | $MIOU |
| Frames | $N_IMG |
| Wall time | $EVAL_DUR |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg
EOF

echo "[movi-e-coda-eval] report written to $REPORT"
exit "$EVAL_RC"

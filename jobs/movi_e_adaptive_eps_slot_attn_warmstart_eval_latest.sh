#!/bin/bash
# Eval-only MOVi-E reconstruction + object metrics for the latest adaptive
# epsilon-composition warm-start checkpoint.
#
# Submit from the repo root:
#   sbatch jobs/movi_e_adaptive_eps_slot_attn_warmstart_eval_latest.sh
#
# The checkpoint is produced by:
#   jobs/movi_e_adaptive_eps_slot_attn_warmstart_train_eval.sh
# and resolved at job start so the script can be reused after resubmissions.
# Reconstruction eval must use --epsilon_composition slot_attn to match the
# training objective.
#SBATCH --job-name="movi-e-adapt-eps-eval"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
export WANDB_DIR="$HOME/prjs0993/Decomp-Diffusion/wandb"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_DIR=results/movi-e_adaptive_eps_slot_attn_warmstart
TRACKER_PROJECT_NAME=latent_decomposed_diffusion_adaptive_eps
OUT_SUBDIR="$RUN_DIR/$TRACKER_PROJECT_NAME"
REPORT=docs/experiments/2026-05-28-movi-e-adaptive-eps-slot-attn-latest-eval.md
RESOLUTION=256
EVAL_TAG=${SLURM_JOB_ID:-manual}
GEN_DIR="$RUN_DIR/gen_images_latest_$EVAL_TAG"
GRID_DIR="$RUN_DIR/eval_grids_latest_$EVAL_TAG"
METRICS_DIR="$RUN_DIR/metrics_latest_$EVAL_TAG"
VAL_SHARD_ROOT=data/movi-e-wds/validation
mkdir -p "$GEN_DIR" "$GRID_DIR" "$METRICS_DIR" "$WANDB_DIR" "$(dirname "$REPORT")"

latest_checkpoint() {
    find "$OUT_SUBDIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | while IFS= read -r path; do
            base=${path##*/checkpoint-}
            is_last=0
            case "$base" in
                *-last)
                    is_last=1
                    step=${base%-last}
                    ;;
                *)
                    step=$base
                    ;;
            esac
            case "$step" in
                ''|*[!0-9]*) continue ;;
            esac
            printf '%s\t%s\t%s\n' "$step" "$is_last" "$path"
        done \
        | sort -k1,1n -k2,2n \
        | tail -n1 \
        | cut -f3-
}

CKPT=${CKPT_PATH:-$(latest_checkpoint)}
echo "[movi-e-adapt-eps-eval] using checkpoint: ${CKPT:-<none found>}"
if [ -z "$CKPT" ]; then
    echo "[movi-e-adapt-eps-eval] no checkpoint found under $OUT_SUBDIR"
    exit 1
fi
if [ ! -d "$CKPT/dinoslotattentionencoder" ] || [ ! -d "$CKPT/unet2dconditionmodel" ]; then
    echo "[movi-e-adapt-eps-eval] checkpoint appears incomplete: $CKPT"
    exit 1
fi
if [ ! -f "$VAL_SHARD_ROOT/samples.jsonl" ]; then
    echo "[movi-e-adapt-eps-eval] FATAL: $VAL_SHARD_ROOT/samples.jsonl missing -- run jobs/movi_e_shard_wds.sh first."
    exit 1
fi
VAL_N=$(wc -l < "$VAL_SHARD_ROOT/samples.jsonl")
echo "[movi-e-adapt-eps-eval] validation frames=$VAL_N"

# eval.py writes grids to this hardcoded worktree path.
rm -rf image_test_output

START=$(date +%s)

# --- 1. Qualitative reconstruction grids -------------------------------------
RECON_START=$(date +%s)
uv run accelerate launch --num_processes=1 eval.py \
    --mixed_precision bf16 --seed 42 \
    --batch_size 8 --num_validation_images 16 \
    --output_dir "$GEN_DIR" \
    --scheduler_config pretrain_sd \
    --dataset_root "$VAL_SHARD_ROOT" \
    --dataset_glob '**/00000000_image.png' \
    --dataset_format wds --resolution "$RESOLUTION" \
    --epsilon_composition slot_attn \
    --ckpt_path "$CKPT"
EVAL_RECON_RC=$?
RECON_END=$(date +%s)

if [ -d image_test_output ]; then
    cp image_test_output/*.jpg "$GRID_DIR/" 2>/dev/null
fi
echo "**************** [movi-e-adapt-eps-eval] reconstruction eval finished (rc=$EVAL_RECON_RC). ****************"

# --- 2. FG-ARI + mBO + mIoU on full validation split -------------------------
METRICS_START=$(date +%s)
uv run python eval_movi.py \
    --ckpt_path "$CKPT" \
    --dataset_root data/movi-e-wds \
    --movi_eval_format wds \
    --split validation \
    --resolution "$RESOLUTION" \
    --batch_size 16 \
    --num_workers 4 \
    --mixed_precision bf16 \
    --output_dir "$METRICS_DIR"
EVAL_METRICS_RC=$?
END=$(date +%s)
echo "**************** [movi-e-adapt-eps-eval] object metrics finished (rc=$EVAL_METRICS_RC). ****************"

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
RECON_DUR=$(fmt_dur $((RECON_END - RECON_START)))
METRICS_DUR=$(fmt_dur $((END - METRICS_START)))
TOTAL_DUR=$(fmt_dur $((END - START)))

[ "$EVAL_RECON_RC" -eq 0 ] && EVAL_RECON_RESULT="PASS" || EVAL_RECON_RESULT="FAIL (rc=$EVAL_RECON_RC)"
[ "$EVAL_METRICS_RC" -eq 0 ] && EVAL_METRICS_RESULT="PASS" || EVAL_METRICS_RESULT="FAIL (rc=$EVAL_METRICS_RC)"
if [ "$EVAL_RECON_RC" -eq 0 ] && [ "$EVAL_METRICS_RC" -eq 0 ]; then
    OVERALL="PASS -- latest adaptive checkpoint evaluation completed"
else
    OVERALL="FAIL -- see /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log"
fi

cat > "$REPORT" <<EOF
# MOVi-E adaptive epsilon composition -- latest-checkpoint eval

**Status:** $OVERALL
**Date:** $(date -u +%Y-%m-%dT%H:%MZ)
**Slurm job:** ${SLURM_JOB_ID:-N/A} (script: jobs/movi_e_adaptive_eps_slot_attn_warmstart_eval_latest.sh, log: /home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_${SLURM_JOB_ID}.log)
**Training script:** jobs/movi_e_adaptive_eps_slot_attn_warmstart_train_eval.sh
**Checkpoint:** $CKPT

## Configuration

| Item | Value |
|------|-------|
| Output dir | $OUT_SUBDIR/ |
| Epsilon composition | slot_attn |
| Resolution | $RESOLUTION |
| Eval split | MOVi-E validation shards ($VAL_N frames) |
| Reconstruction images | 16 |

## Results

| Stage | Result | Wall time |
|-------|--------|-----------|
| Reconstruction eval | $EVAL_RECON_RESULT | $RECON_DUR |
| Object-centric metrics | $EVAL_METRICS_RESULT | $METRICS_DUR |
| Total | $OVERALL | $TOTAL_DUR |

### Object-centric metrics

| Metric | Value |
|--------|-------|
| FG-ARI | $FG_ARI |
| mBO | $MBO |
| mIoU | $MIOU |
| Frames | $N_IMG |

Full metrics: $METRICS_DIR/metrics.json
Attention-mask viz: $METRICS_DIR/viz_*.jpg
Reconstruction grids: $GRID_DIR/image_NN.jpg
Generated images: $GEN_DIR/
EOF

echo "[movi-e-adapt-eps-eval] report written to $REPORT ($OVERALL)"
if [ "$EVAL_RECON_RC" -eq 0 ] && [ "$EVAL_METRICS_RC" -eq 0 ]; then
    exit 0
fi
exit 1

#!/bin/bash
# Smoketest: confirm PyTorch SDPA (FlashAttention-3 on H100) trains cleanly
# and benchmark it against xformers MEA back-to-back in the same job.
#
# Driven by the TECHDEBT entry "H100 throughput knobs left on the table":
# swap xformers MEA for PyTorch SDPA so the UNet's attention layers use
# diffusers' default AttnProcessor2_0, which calls
# torch.nn.functional.scaled_dot_product_attention; on H100 + PyTorch 2.5+
# that auto-dispatches to FlashAttention-3 (bf16/fp16).
#
# Compares (xformers) vs (SDPA) on a single H100 at the dinov3_registers
# config (current production setup): same encoder, resolution, per-GPU batch
# and dtype as jobs/movi_e_dinov3_registers_train_eval.sh, just one GPU and
# 150 steps each.
#
# Submit from the repo root: `sbatch jobs/movi_e_attn_sdpa_smoketest.sh`.
#SBATCH --job-name="attn-sdpa-smoke"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_ROOT=results/movi-e_attn_sdpa_smoketest
RESOLUTION=256
PER_GPU_BATCH=8
N_STEPS=150

rm -rf image_test_output "$RUN_ROOT"

# --- 0. Sanity check ---------------------------------------------------------
TRAIN_IMG_ROOT=data/movi-e/movi-e-train-with-label/images
if [ ! -d "$TRAIN_IMG_ROOT" ]; then
    echo "[attn-sdpa-smoke] FATAL: $TRAIN_IMG_ROOT missing -- run jobs/movi_e_preprocess.sh first."
    exit 1
fi

run_train() {
    # $1: variant name (used for run dir)
    # $2..: extra args to train_lsd.py
    local variant=$1; shift
    local run_dir="$RUN_ROOT/$variant"
    echo "**************** [attn-sdpa-smoke] start variant=$variant ****************"
    local t0=$(date +%s)
    uv run accelerate launch --num_processes=1 --mixed_precision bf16 \
        --main_process_port 29505 train_lsd.py \
        --train_config configs/movi-e/train_config.yaml \
        --output_dir "$run_dir/" \
        --latent_encoder_config configs/movi-e/dinov3_slot_encoder/config.json \
        --unet_config configs/movi-e/unet/config.json \
        --scheduler_config configs/movi-e/scheduler/scheduler_config.json \
        --dataset_root "$TRAIN_IMG_ROOT" \
        --dataset_glob '**/*.png' \
        --report_to wandb \
        --resolution "$RESOLUTION" \
        --train_batch_size "$PER_GPU_BATCH" \
        --max_train_steps "$N_STEPS" \
        --validation_steps 1000000 --checkpointing_steps 1000000 \
        "$@"
    local rc=$?
    local t1=$(date +%s)
    local dur=$((t1 - t0))
    echo "**************** [attn-sdpa-smoke] variant=$variant rc=$rc wall=${dur}s ($N_STEPS steps) ****************"
    eval "${variant}_RC=$rc"
    eval "${variant}_DUR=$dur"
}

# --- 1. Baseline: xformers MEA (override config default) ---------------------
run_train xformers --enable_xformers_memory_efficient_attention

# --- 2. SDPA: config default (false) leaves diffusers on AttnProcessor2_0 ----
run_train sdpa

# --- 3. Summary --------------------------------------------------------------
echo
echo "================ [attn-sdpa-smoke] summary ================"
printf "  %-10s  rc=%s  wall=%5ss  (%.3fs/step)\n" xformers "$xformers_RC" "$xformers_DUR" "$(awk -v t=$xformers_DUR -v n=$N_STEPS 'BEGIN{print t/n}')"
printf "  %-10s  rc=%s  wall=%5ss  (%.3fs/step)\n" sdpa     "$sdpa_RC"     "$sdpa_DUR"     "$(awk -v t=$sdpa_DUR     -v n=$N_STEPS 'BEGIN{print t/n}')"
if [ "$xformers_RC" -eq 0 ] && [ "$sdpa_RC" -eq 0 ]; then
    speedup=$(awk -v a=$xformers_DUR -v b=$sdpa_DUR 'BEGIN{ if(b>0) printf "%.2f", a/b; else print "inf"}')
    echo "  speedup (sdpa vs xformers): ${speedup}x"
    echo "**************** [attn-sdpa-smoke] PASS ****************"
else
    echo "**************** [attn-sdpa-smoke] FAIL (xformers=$xformers_RC sdpa=$sdpa_RC) ****************"
fi

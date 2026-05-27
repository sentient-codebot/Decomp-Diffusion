#!/bin/bash
# Smoketest: confirm SDPA + torch.compile (accelerate --dynamo_backend=inductor)
# trains cleanly, and benchmark it against plain SDPA back-to-back.
#
# Driven by the TECHDEBT entry "H100 throughput knobs left on the table":
# torch.compile/inductor typically wins 1.3-1.8x on diffusion training when
# stacked on top of SDPA. Compile is paid up-front (~30-90s on the first few
# steps); we run 250 steps so steady-state dominates.
#
# Pairs with jobs/movi_e_attn_sdpa_smoketest.sh (xformers vs SDPA). Same H100
# single-GPU setup with the dinov3_registers production config.
#
# Submit from the repo root: `sbatch jobs/movi_e_attn_sdpa_dynamo_smoketest.sh`.
#SBATCH --job-name="attn-dynamo-smoke"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
source .venv/bin/activate
uv sync --extra wandb --extra tensorboard --extra xformers

RUN_ROOT=results/movi-e_attn_sdpa_dynamo_smoketest
RESOLUTION=256
PER_GPU_BATCH=8
N_STEPS=250

rm -rf image_test_output "$RUN_ROOT"

TRAIN_IMG_ROOT=data/movi-e/movi-e-train-with-label/images
if [ ! -d "$TRAIN_IMG_ROOT" ]; then
    echo "[attn-dynamo-smoke] FATAL: $TRAIN_IMG_ROOT missing -- run jobs/movi_e_preprocess.sh first."
    exit 1
fi

run_train() {
    # $1: variant name (used for run dir + env tag)
    # $2: extra accelerate-launch flags (string, may be empty)
    # $3..: extra train_lsd.py args
    local variant=$1; shift
    local launch_flags=$1; shift
    local run_dir="$RUN_ROOT/$variant"
    echo "**************** [attn-dynamo-smoke] start variant=$variant launch_flags='$launch_flags' ****************"
    local t0=$(date +%s)
    # shellcheck disable=SC2086
    uv run accelerate launch --num_processes=1 --mixed_precision bf16 \
        --main_process_port 29506 $launch_flags train_lsd.py \
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
    echo "**************** [attn-dynamo-smoke] variant=$variant rc=$rc wall=${dur}s ($N_STEPS steps) ****************"
    eval "${variant}_RC=$rc"
    eval "${variant}_DUR=$dur"
}

# --- 1. Baseline: SDPA only (config default after the xformers flag flip) ----
run_train sdpa ""

# --- 2. SDPA + dynamo/inductor: accelerate wraps unet with torch.compile -----
# Includes one-time compile cost in the wall-clock; with N_STEPS=250 the
# steady state should still dominate. If compile crashes, accelerate falls
# back to eager and rc may still be 0 -- check the slurm log for
# "TorchDynamo" / "Inductor" warnings before trusting the number.
run_train sdpa_dynamo "--dynamo_backend=inductor"

# --- 3. Summary --------------------------------------------------------------
echo
echo "================ [attn-dynamo-smoke] summary ================"
printf "  %-12s  rc=%s  wall=%5ss  (%.3fs/step)\n" sdpa         "$sdpa_RC"         "$sdpa_DUR"         "$(awk -v t=$sdpa_DUR         -v n=$N_STEPS 'BEGIN{print t/n}')"
printf "  %-12s  rc=%s  wall=%5ss  (%.3fs/step)\n" sdpa_dynamo  "$sdpa_dynamo_RC"  "$sdpa_dynamo_DUR"  "$(awk -v t=$sdpa_dynamo_DUR  -v n=$N_STEPS 'BEGIN{print t/n}')"
if [ "$sdpa_RC" -eq 0 ] && [ "$sdpa_dynamo_RC" -eq 0 ]; then
    speedup=$(awk -v a=$sdpa_DUR -v b=$sdpa_dynamo_DUR 'BEGIN{ if(b>0) printf "%.2f", a/b; else print "inf"}')
    echo "  speedup (sdpa+dynamo vs sdpa): ${speedup}x  [includes one-time compile]"
    echo "**************** [attn-dynamo-smoke] PASS ****************"
else
    echo "**************** [attn-dynamo-smoke] FAIL (sdpa=$sdpa_RC sdpa_dynamo=$sdpa_dynamo_RC) ****************"
fi

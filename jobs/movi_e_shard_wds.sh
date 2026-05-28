#!/bin/bash
# Convert an existing loose MOVi-E dump into WebDataset-style tar shards.
#
# Default source: data/movi-e (usually ~/prjs0993/datasets/movi-e via symlink).
# Default output: data/movi-e-wds (persistent shards on prjs0993).
# Override SOURCE_ROOT, OUTPUT_ROOT, or SAMPLES_PER_SHARD at submission time if
# converting from /scratch-shared/nlin instead.
#
# Submit from the repo root: `sbatch jobs/movi_e_shard_wds.sh`.
#SBATCH --job-name="movi-e-shard-wds"
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

source .venv/bin/activate
uv sync

SOURCE_ROOT="${SOURCE_ROOT:-data/movi-e}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/movi-e-wds}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-2048}"

if [ ! -d "$SOURCE_ROOT/movi-e-train-with-label" ]; then
    echo "[movi-e-shard] FATAL: $SOURCE_ROOT does not look like a loose MOVi-E root."
    echo "[movi-e-shard] Expected $SOURCE_ROOT/movi-e-train-with-label/."
    exit 1
fi

echo "[movi-e-shard] source=$SOURCE_ROOT"
echo "[movi-e-shard] output=$OUTPUT_ROOT"
echo "[movi-e-shard] samples_per_shard=$SAMPLES_PER_SHARD"

uv run python scripts/data_preprocess/movi_make_wds_shards.py \
    --source_root "$SOURCE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --samples_per_shard "$SAMPLES_PER_SHARD" \
    --overwrite
RC=$?

for split in train validation test; do
    index="$OUTPUT_ROOT/$split/samples.jsonl"
    if [ -f "$index" ]; then
        cnt=$(wc -l < "$index")
        shards=$(find "$OUTPUT_ROOT/$split" -maxdepth 1 -name '*.tar' | wc -l)
    else
        cnt=0
        shards=0
    fi
    echo "[movi-e-shard] $split: $cnt frames in $shards shards"
done

du -sh "$OUTPUT_ROOT" 2>/dev/null || true
exit "$RC"

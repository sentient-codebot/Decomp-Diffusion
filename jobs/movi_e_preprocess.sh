#!/bin/bash
# Preprocess MOVi-E (256x256, v1.0.0), using scratch for the loose PNG/JSON
# expansion and publishing inode-efficient tar shards to prjs0993.
#
# Scratch output is intentionally temporary: /scratch-shared is large but may
# be purged. The persistent dataset is ~/prjs0993/datasets/movi-e-wds/.
#
# Submit from the repo root: `sbatch jobs/movi_e_preprocess.sh`.
#SBATCH --job-name="movi-e-preprocess"
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion

# uv-managed env; the `preprocess` extra adds tensorflow + tfds + gcsfs.
source .venv/bin/activate
uv sync --extra preprocess

LOOSE_DATA_DIR="${LOOSE_DATA_DIR:-/scratch-shared/nlin/movi-e-loose-${SLURM_JOB_ID:-manual}}"
SHARD_ROOT="${SHARD_ROOT:-$HOME/prjs0993/datasets/movi-e-wds}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-2048}"
mkdir -p "$LOOSE_DATA_DIR" "$(dirname "$SHARD_ROOT")"

# Avoid TF eating the home quota with its default cache; keep TFDS cache in
# project storage and the expanded PNG/JSON dump on scratch.
export TFDS_DATA_DIR="$HOME/prjs0993/cache/tfds"
mkdir -p "$TFDS_DATA_DIR"

# libcurl on this host looks at the Debian-style cert path by default; point
# TF / gcsfs at the real RHEL bundle so HTTPS to gs:// works.
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "[preprocess] writing loose MOVi-E into $LOOSE_DATA_DIR/movi-e/"
df -h "$LOOSE_DATA_DIR" | tail -n1

uv run python scripts/data_preprocess/movi_kubric_dump_with_labels.py \
    --dataset_split movi-e \
    --data_dir "$LOOSE_DATA_DIR"
DUMP_RC=$?
if [ "$DUMP_RC" -ne 0 ]; then
    echo "[preprocess] loose dump failed with rc=$DUMP_RC"
    exit "$DUMP_RC"
fi

echo "[preprocess] sharding MOVi-E into $SHARD_ROOT"
uv run python scripts/data_preprocess/movi_make_wds_shards.py \
    --source_root "$LOOSE_DATA_DIR/movi-e" \
    --output_root "$SHARD_ROOT" \
    --samples_per_shard "$SAMPLES_PER_SHARD" \
    --overwrite
RC=$?

echo "[preprocess] finished with rc=$RC"
df -h "$(dirname "$SHARD_ROOT")" | tail -n1
du -sh "$SHARD_ROOT" 2>/dev/null || true

for split in train validation test; do
    index="$SHARD_ROOT/$split/samples.jsonl"
    if [ -f "$index" ]; then
        cnt=$(wc -l < "$index")
    else
        cnt=0
    fi
    echo "[preprocess] $split: $cnt sharded frames"
done

echo "[preprocess] loose scratch copy remains at $LOOSE_DATA_DIR/movi-e/ until scratch cleanup."
exit "$RC"

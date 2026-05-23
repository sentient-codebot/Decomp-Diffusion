#!/bin/bash
# Preprocess MOVi-E (256x256, v1.0.0) into the per-frame PNG layout
# `GlobDataset` consumes.
#
# Streams the dataset from `gs://kubric-public/tfds` via tfds and writes one
# PNG per frame plus a segment PNG and per-frame instances.json under
# `~/prjs0993/datasets/movi-e/movi-e-{train,validation,test}-with-label/`.
#
# Submit from the repo root: `sbatch jobs/movi_e_preprocess.sh`.
#SBATCH --job-name="movi-e-preprocess"
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/Decomp-Diffusion-movi

# uv-managed env; the `preprocess` extra adds tensorflow + tfds + gcsfs.
source .venv/bin/activate
uv sync --extra preprocess

# Dataset symlink already points at this dir; the script then writes the
# `movi-e/...` subtree there.
DATA_DIR="$HOME/prjs0993/datasets"
mkdir -p "$DATA_DIR"

# Avoid TF eating the home quota with its default cache; pin to scratch.
export TFDS_DATA_DIR="$HOME/prjs0993/cache/tfds"
mkdir -p "$TFDS_DATA_DIR"

# libcurl on this host looks at the Debian-style cert path by default; point
# TF / gcsfs at the real RHEL bundle so HTTPS to gs:// works.
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

echo "[preprocess] writing MOVi-E into $DATA_DIR/movi-e/"
df -h "$DATA_DIR" | tail -n1

uv run python scripts/data_preprocess/movi_kubric_dump_with_labels.py \
    --dataset_split movi-e \
    --data_dir "$DATA_DIR"
RC=$?

echo "[preprocess] finished with rc=$RC"
df -h "$DATA_DIR" | tail -n1
du -sh "$DATA_DIR/movi-e" 2>/dev/null || true

# Quick sanity counts so the training job can verify the dataset is in place.
for split in train validation test; do
    cnt=$(find "$DATA_DIR/movi-e/movi-e-${split}-with-label/images" -name '*_image.png' 2>/dev/null | wc -l)
    echo "[preprocess] $split: $cnt frames"
done

exit $RC

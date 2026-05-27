#!/bin/bash
# Download COCO 2017 (train + val images + instance annotations) into
# ~/prjs0993/datasets/coco/, matching the layout sony/coda's
# preprocess/download.sh produces:
#
#   coco/
#     images/
#       train2017/*.jpg     (~118k images)
#       val2017/*.jpg       (~5k images)
#     annotations/
#       instances_train2017.json
#       instances_val2017.json
#       ... (captions, person_keypoints — not used by us but bundled)
#
# Training only uses the JPGs (read via GlobDataset); the JSON annotations
# are kept around for future compositional / mask-based eval work.
#
# Idempotent: zips are downloaded with `-C -` (resume) and unzipped with
# `-n` (never overwrite), so re-submission is safe.
#
# Submit from the repo root: `sbatch jobs/coco_download.sh`.
#SBATCH --job-name="coco-download"
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/nlin/prjs0993/Decomp-Diffusion/slurm_logs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

DATA_DIR="$HOME/prjs0993/datasets/coco"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[coco-download] target dir: $(pwd)"
df -h "$DATA_DIR" | tail -n1

HAVE_ARIA=0
if command -v aria2c >/dev/null 2>&1; then
    HAVE_ARIA=1
    echo "[coco-download] using aria2c (parallel)."
else
    echo "[coco-download] aria2c not found, falling back to curl."
fi

fetch() {
    local url="$1"
    local out
    out=$(basename "$url")
    if [ -f "$out" ]; then
        echo "[coco-download] $out already present, attempting resume."
    fi
    if [ "$HAVE_ARIA" -eq 1 ]; then
        aria2c -x 16 -s 16 -c -o "$out" "$url" || return $?
    else
        curl -L --retry 5 --retry-delay 10 -C - -o "$out" "$url" || return $?
    fi
}

set -e

# --- 1. Download archives --------------------------------------------------
fetch http://images.cocodataset.org/zips/train2017.zip
fetch http://images.cocodataset.org/zips/val2017.zip
fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# --- 2. Unzip (no-overwrite so this is safe to re-run) ---------------------
# Images go under images/{train2017,val2017}; annotations sit at root, exactly
# the layout sony/coda expects in experiment/setup/dataset/coco.yaml.
mkdir -p images
echo "[coco-download] unzipping train2017.zip -> images/"
unzip -nq train2017.zip -d images
echo "[coco-download] unzipping val2017.zip -> images/"
unzip -nq val2017.zip -d images
echo "[coco-download] unzipping annotations_trainval2017.zip -> ."
unzip -nq annotations_trainval2017.zip

# --- 3. Sanity counts ------------------------------------------------------
TRAIN_N=$(find images/train2017 -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
VAL_N=$(find images/val2017   -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
echo "[coco-download] train images: $TRAIN_N (expect 118287)"
echo "[coco-download] val   images: $VAL_N (expect 5000)"
ls annotations/ 2>/dev/null | sed 's/^/[coco-download] anno: /'
df -h "$DATA_DIR" | tail -n1
du -sh "$DATA_DIR"/images/train2017 "$DATA_DIR"/images/val2017 "$DATA_DIR"/annotations 2>/dev/null || true

echo "[coco-download] done."

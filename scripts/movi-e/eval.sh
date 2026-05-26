#!/bin/bash
# Wrap `eval_movi.py` invocation -- run from the repo root:
#   bash scripts/movi-e/eval.sh <ckpt_path> <output_dir> [extra args...]
set -euo pipefail

CKPT="${1:?usage: eval.sh <ckpt_path> <output_dir> [extra args...]}"
OUT="${2:?usage: eval.sh <ckpt_path> <output_dir> [extra args...]}"
shift 2

uv run python eval_movi.py \
    --ckpt_path "$CKPT" \
    --dataset_root data/movi-e \
    --split validation \
    --resolution 128 \
    --batch_size 16 \
    --num_workers 4 \
    --output_dir "$OUT" \
    "$@"

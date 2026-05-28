#!/bin/bash
# Report inode-heavy project locations. This is intentionally read-only.
# Usage: bash scripts/storage/inode_report.sh [prjs_root]
set -euo pipefail

PRJS_ROOT="${1:-$HOME/prjs0993}"
PROJECT_ROOT="$PRJS_ROOT/Decomp-Diffusion"

paths=(
    "$PRJS_ROOT/datasets/movi-e"
    "$PRJS_ROOT/datasets/movi-e-wds"
    "$PRJS_ROOT/datasets/coco"
    "$PRJS_ROOT/datasets/celebahq_data128x128"
    "$PROJECT_ROOT/results"
    "$PROJECT_ROOT/wandb"
    "$PROJECT_ROOT/cache"
    "$PRJS_ROOT/tmp/torchinductor"
    "$PRJS_ROOT/slurm_logs"
    "$PROJECT_ROOT/slurm_logs"
)

printf '%-58s %12s %12s
' "path" "inodes" "size"
printf '%-58s %12s %12s
' "----" "------" "----"
for path in "${paths[@]}"; do
    if [ ! -e "$path" ]; then
        continue
    fi
    inodes=$(find "$path" -xdev 2>/dev/null | wc -l)
    size=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
    printf '%-58s %12s %12s
' "$path" "$inodes" "${size:-?}"
done

if [ -d "$PROJECT_ROOT/wandb" ]; then
    run_count=$(find "$PROJECT_ROOT/wandb" -maxdepth 1 -type d -name 'run-*' | wc -l)
    echo
    echo "wandb local run dirs: $run_count"
    echo "After sync, keep local run dirs only when needed for offline debugging."
fi

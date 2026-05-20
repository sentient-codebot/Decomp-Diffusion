#!/usr/bin/env bash
# Install uv (https://github.com/astral-sh/uv) then provision the venv.
# Run once from the project root.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --extra wandb --extra tensorboard --extra xformers

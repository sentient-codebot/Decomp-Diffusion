# Download the CelebA-HQ 128x128 dataset into data/celebahq_data128x128/.
#
# Pulls korexyz/celeba-hq-256x256 from the Hugging Face Hub (~3 GB) and dumps
# 30k JPGs resized to 128x128. The HF cache is pinned to prjs0993 to keep the
# 3 GB parquet download off the home quota. `datasets`/`Pillow` are pulled in
# ephemerally with `uv run --with` -- they are not project dependencies.
export HF_HOME="$HOME/prjs0993/Decomp-Diffusion/cache/huggingface"
uv run --with 'datasets,Pillow,huggingface_hub' \
    python scripts/celebahq/download_data.py "$@"

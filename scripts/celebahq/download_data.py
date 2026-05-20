"""Download CelebA-HQ and dump it as 128x128 JPGs for GlobDataset.

Source: korexyz/celeba-hq-256x256 on the Hugging Face Hub (30k images, the
standard 30k CelebA-HQ set from the Progressive GAN paper, at 256x256).

Each image is centre-resized to 128x128 and written as a flat JPG into the
output dir, matching the original `**/*.jpg` glob and `--resolution 128`.
The train/val split is done in-code by `train_split_portion`, so all 30k
images go into one directory.

Run via scripts/celebahq/download_data.sh (it pins the HF cache to prjs0993).
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image


REPO_ID = "korexyz/celeba-hq-256x256"
TARGET_SIZE = 128


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out_dir",
        default="data/celebahq_data128x128",
        help="output directory for the JPGs (default: data/celebahq_data128x128)",
    )
    ap.add_argument(
        "--size", type=int, default=TARGET_SIZE, help="output square size in px"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(REPO_ID)
    idx = 0
    written = skipped = 0
    for split in ds:
        for row in ds[split]:
            path = out_dir / f"{idx:05d}.jpg"
            idx += 1
            if path.exists():
                skipped += 1
                continue
            img = row["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != (args.size, args.size):
                img = img.resize((args.size, args.size), Image.LANCZOS)
            img.save(path, "JPEG", quality=95)
            written += 1
            if written % 2000 == 0:
                print(f"  ...{written} written")

    print(f"done: {written} written, {skipped} already present -> {out_dir}")


if __name__ == "__main__":
    main()

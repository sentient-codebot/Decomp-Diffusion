"""Object-centric evaluation for MOVi-E (FG-ARI + mBO + mIoU).

Loads a trained checkpoint produced by ``train_lsd.py`` on MOVi-E, runs the
slot-attention encoder over the validation split and compares the per-slot
attention masks against the ground-truth instance segmentations dumped
alongside the images by ``scripts/data_preprocess/movi_kubric_dump_with_labels.py``.

Reports three standard object-discovery metrics:
  - **FG-ARI** -- Adjusted Rand Index over foreground pixels only.
  - **mBO** -- per-GT-object greedy max IoU; averaged over objects then images.
  - **mIoU** -- Hungarian-matched mean per-pair IoU (sony/coda convention).

Also writes a small JSON with the aggregate numbers and an optional grid of
per-slot attention overlays for inspection.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.data.dataset import build_movi_pair_dataset
from src.metrics.segmentation import (
    per_image_fg_ari,
    per_image_mbo,
    per_image_miou,
)
from src.models.encoder import (
    DinoSlotAttentionEncoder,
    SlotAttentionEncoder,
    load_latent_encoder,
)


# Encoders that expose per-slot attention masks via forward(..., return_attn=True).
SLOT_ATTN_ENCODER_CLASSES = (SlotAttentionEncoder, DinoSlotAttentionEncoder)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument(
        "--dataset_root",
        required=True,
        help=(
            "MOVi root. For files: contains movi-e-{split}-with-label/. "
            "For wds: contains split shard directories."
        ),
    )
    p.add_argument(
        "--movi_eval_format",
        default="files",
        choices=["files", "wds"],
        help="Storage format for --dataset_root.",
    )
    p.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"]
    )
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap the number of frames evaluated (None = full split).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory for metrics.json and viz grids.",
    )
    p.add_argument("--num_viz", type=int, default=8, help="How many viz grids to save.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mixed_precision",
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    return p.parse_args()


def save_viz_grid(pixel_values, attn_up, pred_mask, gt, out_path):
    """Save a [B, 1+K+2] grid: input | per-slot attention heatmaps | pred argmax | gt."""
    bs, k, h, w = attn_up.shape
    # Per-slot heatmaps as grayscale -> RGB
    heats = attn_up.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # [B, K, 3, H, W]
    input_rgb = (pixel_values * 0.5 + 0.5).clamp(0, 1).unsqueeze(1)  # [B, 1, 3, H, W]

    def label_to_rgb(labels):
        # Deterministic colour per integer id, 0 -> black.
        labels = labels.long()
        max_id = max(int(labels.max().item()), 1)
        rng = np.random.default_rng(123)
        palette = rng.uniform(0.2, 1.0, size=(max_id + 1, 3))
        palette[0] = 0.0
        flat = palette[labels.cpu().numpy()]  # [B, H, W, 3]
        return torch.from_numpy(flat).permute(0, 3, 1, 2).float().unsqueeze(1)

    pred_rgb = label_to_rgb(pred_mask).to(input_rgb.device)
    gt_rgb = label_to_rgb(gt).to(input_rgb.device)
    grid_per_row = torch.cat(
        [input_rgb, heats, pred_rgb, gt_rgb], dim=1
    )  # [B, 1+K+2, 3, H, W]
    grid = make_grid(
        grid_per_row.reshape(bs * grid_per_row.size(1), 3, h, w),
        nrow=grid_per_row.size(1),
        padding=2,
        pad_value=1.0,
    )
    arr = (grid * 255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(arr).save(out_path, optimize=True, quality=92)


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.mixed_precision
    ]

    encoder = load_latent_encoder(args.ckpt_path)
    if not isinstance(encoder, SLOT_ATTN_ENCODER_CLASSES):
        allowed = ", ".join(c.__name__ for c in SLOT_ATTN_ENCODER_CLASSES)
        raise SystemExit(
            f"eval_movi requires a slot-attention encoder ({allowed}); "
            f"got {type(encoder).__name__}."
        )
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    print(f"loaded {type(encoder).__name__} with {encoder.num_components} slots")

    dataset = build_movi_pair_dataset(
        dataset_format=args.movi_eval_format,
        root=args.dataset_root,
        split=args.split,
        img_size=args.resolution,
        max_images=args.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"evaluating {len(dataset)} frames in {len(loader)} batches")

    aris, mbos, mious = [], [], []
    viz_saved = 0
    for batch_idx, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
        gt = batch["segment"].to(device=device)  # [B, H, W]
        _, attn = encoder(pixel_values, return_attn=True)  # attn [B, K, h, w]
        if not torch.isfinite(attn).all():
            bad = (~torch.isfinite(attn)).sum().item()
            raise RuntimeError(
                f"non-finite slot attention masks in batch {batch_idx}: "
                f"{bad} values with --mixed_precision {args.mixed_precision}"
            )
        attn_up = F.interpolate(
            attn.float(),
            size=(args.resolution, args.resolution),
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = attn_up.argmax(dim=1)  # [B, H, W]
        for b in range(pred_mask.shape[0]):
            gt_np = gt[b].cpu().numpy()
            pred_np = pred_mask[b].cpu().numpy()
            ari = per_image_fg_ari(gt_np, pred_np)
            mbo = per_image_mbo(gt_np, pred_np)
            miou = per_image_miou(gt_np, pred_np, ignore_background=False)
            if ari is not None:
                aris.append(ari)
            if mbo is not None:
                mbos.append(mbo)
            if miou is not None:
                mious.append(miou)
        if viz_saved < args.num_viz:
            viz_path = os.path.join(args.output_dir, f"viz_{batch_idx:04}.jpg")
            save_viz_grid(pixel_values.float(), attn_up, pred_mask, gt, viz_path)
            viz_saved += 1
        if (batch_idx + 1) % 10 == 0:
            print(
                f"  batch {batch_idx + 1}/{len(loader)} "
                f"running FG-ARI={np.mean(aris):.4f}  mBO={np.mean(mbos):.4f}  "
                f"mIoU={np.mean(mious):.4f}"
            )

    metrics = {
        "fg_ari": float(np.mean(aris)),
        "mbo": float(np.mean(mbos)),
        "miou": float(np.mean(mious)),
        "fg_ari_std": float(np.std(aris)),
        "mbo_std": float(np.std(mbos)),
        "miou_std": float(np.std(mious)),
        "n_images_ari": len(aris),
        "n_images_mbo": len(mbos),
        "n_images_miou": len(mious),
        "split": args.split,
        "resolution": args.resolution,
        "num_slots": encoder.num_components,
        "ckpt_path": args.ckpt_path,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()

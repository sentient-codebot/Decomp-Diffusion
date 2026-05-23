"""Object-centric evaluation for MOVi-E (FG-ARI + mBO).

Loads a trained checkpoint produced by ``train_lsd.py`` on MOVi-E, runs the
slot-attention encoder over the validation split and compares the per-slot
attention masks against the ground-truth instance segmentations dumped
alongside the images by ``scripts/data_preprocess/movi_kubric_dump_with_labels.py``.

Reports two standard object-centric metrics:
  - **FG-ARI** -- Adjusted Rand Index over foreground pixels only (the
    standard segmentation metric used in the slot-attention literature).
  - **mBO** -- mean Best Overlap: for each GT object, the highest IoU with
    any predicted slot mask; averaged over objects then over images.

Also writes a small JSON with the aggregate numbers and an optional grid of
per-slot attention overlays for inspection.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

from src.models.encoder import SlotAttentionEncoder, load_latent_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument(
        "--dataset_root",
        required=True,
        help="Path containing movi-e-{split}-with-label/{images,labels}/<vid>/...",
    )
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
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
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    return p.parse_args()


class MoviPairDataset(Dataset):
    """Reads MOVi-E (image, segment) pairs from the dumped per-frame layout."""

    def __init__(self, root, split, img_size, max_images=None):
        self.img_size = img_size
        images_root = os.path.join(root, f"movi-e-{split}-with-label", "images")
        labels_root = os.path.join(root, f"movi-e-{split}-with-label", "labels")
        image_paths = sorted(glob.glob(os.path.join(images_root, "**", "*_image.png"), recursive=True))
        if not image_paths:
            raise FileNotFoundError(f"No frames found under {images_root}")
        if max_images is not None:
            image_paths = image_paths[:max_images]
        self.image_paths = image_paths
        self.labels_root = labels_root
        self.images_root = images_root
        # Image transform: same normalisation as training (GlobDataset).
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        # Segment transform: nearest-neighbour so integer instance ids survive.
        self.segment_transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(img_size),
            ]
        )

    def _segment_path(self, image_path):
        # images/<vid>/<frame>_image.png  ->  labels/<vid>/<frame>_segment.png
        rel = os.path.relpath(image_path, self.images_root)
        rel = rel.replace("_image.png", "_segment.png")
        return os.path.join(self.labels_root, rel)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert("RGB")
        seg = Image.open(self._segment_path(self.image_paths[i]))
        pixel_values = self.image_transform(img)
        seg_resized = self.segment_transform(seg)
        segment = torch.from_numpy(np.array(seg_resized, dtype=np.int64))
        return {"pixel_values": pixel_values, "segment": segment}


def per_image_fg_ari(gt, pred):
    """Foreground-only Adjusted Rand Index for one image.

    gt, pred: int arrays of equal shape. Background pixels (gt == 0) are
    excluded. Returns None when the image has fewer than 2 foreground pixels
    or only a single GT cluster (ARI is undefined there).
    """
    mask = gt > 0
    if mask.sum() < 2:
        return None
    g = gt[mask]
    p = pred[mask]
    if np.unique(g).size < 2:
        return None
    return adjusted_rand_score(g, p)


def per_image_mbo(gt, pred):
    """Mean Best Overlap (a.k.a. mBO/mIoU per-object).

    For each GT instance (id > 0) take the maximum IoU over all predicted
    slot masks (ids in pred, including the slot covering background); return
    the mean over GT instances. Returns None when the image has no
    foreground objects.
    """
    gt_ids = [i for i in np.unique(gt) if i != 0]
    if not gt_ids:
        return None
    pred_ids = list(np.unique(pred))
    ious = []
    for gi in gt_ids:
        gm = gt == gi
        best = 0.0
        for pi in pred_ids:
            pm = pred == pi
            inter = np.logical_and(gm, pm).sum()
            if inter == 0:
                continue
            union = np.logical_or(gm, pm).sum()
            iou = inter / union
            if iou > best:
                best = iou
        ious.append(best)
    return float(np.mean(ious))


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
    grid_per_row = torch.cat([input_rgb, heats, pred_rgb, gt_rgb], dim=1)  # [B, 1+K+2, 3, H, W]
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
    if not isinstance(encoder, SlotAttentionEncoder):
        raise SystemExit(
            f"eval_movi requires a SlotAttentionEncoder checkpoint; got {type(encoder).__name__}. "
            "Re-train with configs/movi-e/slot_encoder/config.json."
        )
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    print(f"loaded SlotAttentionEncoder with {encoder.num_components} slots")

    dataset = MoviPairDataset(
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

    aris, mbos = [], []
    viz_saved = 0
    for batch_idx, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
        gt = batch["segment"].to(device=device)  # [B, H, W]
        _, attn = encoder(pixel_values, return_attn=True)  # attn [B, K, h, w]
        attn_up = F.interpolate(
            attn.float(), size=(args.resolution, args.resolution), mode="bilinear", align_corners=False
        )
        pred_mask = attn_up.argmax(dim=1)  # [B, H, W]
        for b in range(pred_mask.shape[0]):
            ari = per_image_fg_ari(gt[b].cpu().numpy(), pred_mask[b].cpu().numpy())
            mbo = per_image_mbo(gt[b].cpu().numpy(), pred_mask[b].cpu().numpy())
            if ari is not None:
                aris.append(ari)
            if mbo is not None:
                mbos.append(mbo)
        if viz_saved < args.num_viz:
            viz_path = os.path.join(args.output_dir, f"viz_{batch_idx:04}.jpg")
            save_viz_grid(pixel_values.float(), attn_up, pred_mask, gt, viz_path)
            viz_saved += 1
        if (batch_idx + 1) % 10 == 0:
            print(
                f"  batch {batch_idx + 1}/{len(loader)} "
                f"running FG-ARI={np.mean(aris):.4f}  mBO={np.mean(mbos):.4f}"
            )

    metrics = {
        "fg_ari": float(np.mean(aris)),
        "mbo": float(np.mean(mbos)),
        "fg_ari_std": float(np.std(aris)),
        "mbo_std": float(np.std(mbos)),
        "n_images_ari": len(aris),
        "n_images_mbo": len(mbos),
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

"""Object-centric evaluation for COCO instance annotations.

This mirrors ``eval_movi.py`` from main for the encoder-side metrics path:
load a trained slot-attention checkpoint, extract per-slot attention masks,
argmax them into predicted segments, and compare them to ground-truth instance
label maps. For COCO, the label maps are rasterized from
``instances_{split}.json`` polygon annotations.

Reports:
  - FG-ARI -- Adjusted Rand Index over foreground pixels only.
  - mBO -- per-GT-object greedy max IoU, averaged over objects then images.
  - mIoU -- Hungarian-matched mean IoU, with and without background.
  - Per-category and COCO-size-bin mBO summaries for binding diagnostics.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

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


SLOT_ATTN_ENCODER_CLASSES = (SlotAttentionEncoder, DinoSlotAttentionEncoder)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument(
        "--dataset_root",
        required=True,
        help="COCO root containing images/{train2017,val2017}/ and annotations/.",
    )
    p.add_argument("--split", default="val2017", choices=["train2017", "val2017"])
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap the number of images evaluated (None = full split).",
    )
    p.add_argument(
        "--min_area",
        type=float,
        default=1.0,
        help="Drop annotations with COCO area below this threshold.",
    )
    p.add_argument(
        "--max_instances",
        type=int,
        default=256,
        help="Max instance ids kept per image for category/area metadata.",
    )
    p.add_argument(
        "--min_category_count",
        type=int,
        default=1,
        help="Minimum evaluated objects required to include a category summary.",
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


def image_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize(
                img_size,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def segment_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize(
                img_size,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(img_size),
        ]
    )


class CocoInstanceDataset(Dataset):
    """COCO image + single-label instance mask dataset.

    COCO masks can overlap. We draw larger annotations first and smaller ones
    later, so smaller foreground instances retain priority in the final integer
    label map. Crowd RLE annotations are skipped; the standard COCO instance
    annotations use polygons for non-crowd objects.
    """

    def __init__(
        self,
        root,
        split,
        img_size,
        max_images=None,
        min_area=1.0,
        max_instances=256,
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.max_instances = max_instances
        self.image_root = os.path.join(root, "images", split)
        self.annotation_path = os.path.join(
            root, "annotations", f"instances_{split}.json"
        )
        if not os.path.isdir(self.image_root):
            raise FileNotFoundError(f"No COCO image directory found: {self.image_root}")
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(
                f"No COCO annotation file found: {self.annotation_path}"
            )

        with open(self.annotation_path) as f:
            data = json.load(f)

        self.category_names = {
            int(cat["id"]): cat.get("name", str(cat["id"]))
            for cat in data.get("categories", [])
        }
        self.annotations_by_image = defaultdict(list)
        self.skipped_crowd = 0
        self.skipped_tiny = 0
        self.skipped_non_polygon = 0
        for ann in data.get("annotations", []):
            if ann.get("iscrowd", 0):
                self.skipped_crowd += 1
                continue
            if float(ann.get("area", 0.0)) < min_area:
                self.skipped_tiny += 1
                continue
            if not isinstance(ann.get("segmentation"), list):
                self.skipped_non_polygon += 1
                continue
            self.annotations_by_image[int(ann["image_id"])].append(ann)

        images = sorted(data.get("images", []), key=lambda im: int(im["id"]))
        self.images = [
            im
            for im in images
            if os.path.exists(os.path.join(self.image_root, im["file_name"]))
        ]
        if max_images is not None:
            self.images = self.images[:max_images]
        if not self.images:
            raise FileNotFoundError(f"No COCO images found under {self.image_root}")

        self.image_transform = image_transform(img_size)
        self.segment_transform = segment_transform(img_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.image_root, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        label_img, category_id, area = self._label_image(image_info)
        segment = self.segment_transform(label_img)
        sample = {
            "pixel_values": self.image_transform(image),
            "segment": torch.from_numpy(np.array(segment, dtype=np.int64)),
            "category_id": torch.from_numpy(category_id),
            "area": torch.from_numpy(area),
            "image_id": torch.tensor(int(image_info["id"]), dtype=torch.int64),
        }
        return sample

    def _label_image(self, image_info):
        width, height = int(image_info["width"]), int(image_info["height"])
        label_img = Image.new("I", (width, height), 0)
        draw = ImageDraw.Draw(label_img)
        category_id = np.zeros((self.max_instances + 1,), dtype=np.int64)
        area = np.zeros((self.max_instances + 1,), dtype=np.float32)

        anns = self.annotations_by_image.get(int(image_info["id"]), [])
        anns = sorted(anns, key=lambda ann: float(ann.get("area", 0.0)), reverse=True)
        label_id = 1
        for ann in anns:
            polygons = []
            for poly in ann.get("segmentation", []):
                if len(poly) < 6:
                    continue
                polygons.append(list(zip(poly[0::2], poly[1::2], strict=False)))
            if not polygons:
                continue
            if label_id > self.max_instances:
                break
            for polygon in polygons:
                draw.polygon(polygon, fill=label_id)
            category_id[label_id] = int(ann["category_id"])
            area[label_id] = float(ann.get("area", 0.0))
            label_id += 1
        return label_img, category_id, area


def save_viz_grid(pixel_values, attn_up, pred_mask, gt, out_path):
    """Save a [B, 1+K+2] grid: input | per-slot heatmaps | pred argmax | gt."""
    bs, k, h, w = attn_up.shape
    heats = attn_up.unsqueeze(2).expand(-1, -1, 3, -1, -1)
    input_rgb = (pixel_values * 0.5 + 0.5).clamp(0, 1).unsqueeze(1)

    def label_to_rgb(labels):
        labels = labels.long()
        max_id = max(int(labels.max().item()), 1)
        rng = np.random.default_rng(123)
        palette = rng.uniform(0.2, 1.0, size=(max_id + 1, 3))
        palette[0] = 0.0
        flat = palette[labels.cpu().numpy()]
        return torch.from_numpy(flat).permute(0, 3, 1, 2).float().unsqueeze(1)

    pred_rgb = label_to_rgb(pred_mask).to(input_rgb.device)
    gt_rgb = label_to_rgb(gt).to(input_rgb.device)
    grid_per_row = torch.cat([input_rgb, heats, pred_rgb, gt_rgb], dim=1)
    grid = make_grid(
        grid_per_row.reshape(bs * grid_per_row.size(1), 3, h, w),
        nrow=grid_per_row.size(1),
        padding=2,
        pad_value=1.0,
    )
    arr = (grid * 255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(arr).save(out_path, optimize=True, quality=92)


def mean_or_none(values):
    return None if not values else float(np.mean(values))


def std_or_none(values):
    return None if not values else float(np.std(values))


def fmt_running(values):
    return "N/A" if not values else f"{np.mean(values):.4f}"


def coco_size_bin(area):
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def per_gt_best_ious(gt, pred):
    pred_ids = list(np.unique(pred))
    out = []
    for gt_id in [int(i) for i in np.unique(gt) if i != 0]:
        gt_mask = gt == gt_id
        best = 0.0
        for pred_id in pred_ids:
            pred_mask = pred == pred_id
            inter = np.logical_and(gt_mask, pred_mask).sum()
            if inter == 0:
                continue
            union = np.logical_or(gt_mask, pred_mask).sum()
            best = max(best, float(inter / union))
        out.append((gt_id, best))
    return out


def summarize_grouped(values_by_key, key_names=None, min_count=1):
    summary = {}
    for key in sorted(values_by_key):
        values = values_by_key[key]
        if len(values) < min_count:
            continue
        name = key_names.get(key, str(key)) if key_names is not None else str(key)
        summary[name] = {
            "count": len(values),
            "mbo": float(np.mean(values)),
            "mbo_std": float(np.std(values)),
        }
    return summary


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
            f"eval_coco requires a slot-attention encoder ({allowed}); "
            f"got {type(encoder).__name__}."
        )
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    print(f"loaded {type(encoder).__name__} with {encoder.num_components} slots")

    dataset = CocoInstanceDataset(
        root=args.dataset_root,
        split=args.split,
        img_size=args.resolution,
        max_images=args.max_images,
        min_area=args.min_area,
        max_instances=args.max_instances,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(
        f"evaluating {len(dataset)} COCO {args.split} images in {len(loader)} batches "
        f"(skipped crowd={dataset.skipped_crowd}, tiny={dataset.skipped_tiny}, "
        f"non-polygon={dataset.skipped_non_polygon})"
    )

    aris, mbos, mious, mious_fg = [], [], [], []
    object_best = []
    best_by_category = defaultdict(list)
    best_by_size = defaultdict(list)
    gt_object_counts = []
    foreground_fractions = []
    slot_pixel_counts = np.zeros((encoder.num_components,), dtype=np.int64)
    viz_saved = 0

    for batch_idx, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
        gt = batch["segment"].to(device=device)
        _, attn = encoder(pixel_values, return_attn=True)
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
        pred_mask = attn_up.argmax(dim=1)

        slot_pixel_counts += (
            torch.bincount(pred_mask.flatten().cpu(), minlength=encoder.num_components)
            .numpy()
            .astype(np.int64)
        )

        for b in range(pred_mask.shape[0]):
            gt_np = gt[b].cpu().numpy()
            pred_np = pred_mask[b].cpu().numpy()
            category_id = batch["category_id"][b].cpu().numpy()
            area = batch["area"][b].cpu().numpy()

            gt_ids = [int(i) for i in np.unique(gt_np) if i != 0]
            gt_object_counts.append(len(gt_ids))
            foreground_fractions.append(float((gt_np > 0).mean()))

            ari = per_image_fg_ari(gt_np, pred_np)
            mbo = per_image_mbo(gt_np, pred_np)
            miou = per_image_miou(gt_np, pred_np, ignore_background=False)
            miou_fg = per_image_miou(gt_np, pred_np, ignore_background=True)
            if ari is not None:
                aris.append(ari)
            if mbo is not None:
                mbos.append(mbo)
            if miou is not None:
                mious.append(miou)
            if miou_fg is not None:
                mious_fg.append(miou_fg)

            for gt_id, best_iou in per_gt_best_ious(gt_np, pred_np):
                object_best.append(best_iou)
                if gt_id < len(category_id) and category_id[gt_id] != 0:
                    best_by_category[int(category_id[gt_id])].append(best_iou)
                if gt_id < len(area) and area[gt_id] > 0:
                    best_by_size[coco_size_bin(float(area[gt_id]))].append(best_iou)

        if viz_saved < args.num_viz:
            viz_path = os.path.join(args.output_dir, f"viz_{batch_idx:04}.jpg")
            save_viz_grid(pixel_values.float(), attn_up, pred_mask, gt, viz_path)
            viz_saved += 1
        if (batch_idx + 1) % 10 == 0:
            print(
                f"  batch {batch_idx + 1}/{len(loader)} "
                f"FG-ARI={fmt_running(aris)}  mBO={fmt_running(mbos)}  "
                f"mIoU={fmt_running(mious)}  mIoU-fg={fmt_running(mious_fg)}"
            )

    slot_fraction = slot_pixel_counts / max(int(slot_pixel_counts.sum()), 1)
    nonzero_slot_fraction = slot_fraction[slot_fraction > 0]
    slot_entropy = 0.0
    if len(nonzero_slot_fraction) > 0:
        slot_entropy = float(
            -np.sum(nonzero_slot_fraction * np.log(nonzero_slot_fraction))
            / np.log(encoder.num_components)
        )

    metrics = {
        "fg_ari": mean_or_none(aris),
        "mbo": mean_or_none(mbos),
        "miou": mean_or_none(mious),
        "miou_fg": mean_or_none(mious_fg),
        "fg_ari_std": std_or_none(aris),
        "mbo_std": std_or_none(mbos),
        "miou_std": std_or_none(mious),
        "miou_fg_std": std_or_none(mious_fg),
        "object_mbo": mean_or_none(object_best),
        "object_mbo_std": std_or_none(object_best),
        "n_images": len(dataset),
        "n_images_ari": len(aris),
        "n_images_mbo": len(mbos),
        "n_images_miou": len(mious),
        "n_images_miou_fg": len(mious_fg),
        "n_objects": len(object_best),
        "mean_gt_objects": mean_or_none(gt_object_counts),
        "mean_foreground_fraction": mean_or_none(foreground_fractions),
        "slot_pixel_fraction": slot_fraction.tolist(),
        "slot_entropy": slot_entropy,
        "category_mbo": summarize_grouped(
            best_by_category,
            key_names=dataset.category_names,
            min_count=args.min_category_count,
        ),
        "size_mbo": summarize_grouped(best_by_size),
        "split": args.split,
        "resolution": args.resolution,
        "num_slots": encoder.num_components,
        "ckpt_path": args.ckpt_path,
        "dataset_root": args.dataset_root,
        "skipped_crowd_annotations": dataset.skipped_crowd,
        "skipped_tiny_annotations": dataset.skipped_tiny,
        "skipped_non_polygon_annotations": dataset.skipped_non_polygon,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()

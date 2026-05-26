"""Property-prediction probe on MOVi-E slot vectors.

Freezes the slot-attention encoder from a trained checkpoint, Hungarian-matches
slot attention masks to GT instance masks (cost = 1 - mask cosine similarity),
caches matched (slot, property) pairs, then trains a small MLP head per
property and reports test-set metrics:

- ``image_positions`` (2D, continuous) -- MSE.
- ``bboxes_3d`` (8 corners x 3, continuous) -- MSE.
- ``category`` (discrete) -- classification accuracy.

Follows sony/coda's ``experiment/linear_prob.py`` conventions: 4000 AdamW steps,
83/17 train/test split, mask-cosine matching, MSE / cross-entropy losses.
Continuous heads are 2-layer MLPs (hidden 786); the category head is a single
linear layer (sony/coda's choice for discrete heads).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import MoviPairDataset
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
        help="Path containing movi-e-{split}-with-label/{images,labels}/<vid>/...",
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
        help="Cap the number of frames whose slots are extracted (None = full split).",
    )
    p.add_argument(
        "--max_instances",
        type=int,
        default=24,
        help="Per-frame instance cap for padding the GT property tensors.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory for cache_<split>.pt and metrics.json.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mixed_precision",
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    # Probe training hyperparameters (sony/coda defaults).
    p.add_argument("--probe_steps", type=int, default=4000)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_batch_size", type=int, default=512)
    p.add_argument(
        "--probe_train_frac",
        type=float,
        default=0.83,
        help="Fraction of matched (slot, GT) pairs used for probe training; rest is held out for eval.",
    )
    p.add_argument(
        "--probe_hidden_dim",
        type=int,
        default=786,
        help="Hidden width of the 2-layer continuous-property MLPs.",
    )
    p.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Force re-extraction of slots even if a cache file already exists.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hungarian matching: slot attention masks <-> GT instance masks
# ---------------------------------------------------------------------------


def _mask_cosine_cost(gt_one_hot, pred_one_hot, eps=1e-8):
    """Pairwise (1 - cosine) cost between flattened binary masks.

    gt_one_hot:   [N_gt,  H*W] float
    pred_one_hot: [N_pred, H*W] float
    returns: [N_gt, N_pred] cost matrix in [0, 2].
    """
    gt_norm = gt_one_hot / (gt_one_hot.norm(dim=-1, keepdim=True) + eps)
    pr_norm = pred_one_hot / (pred_one_hot.norm(dim=-1, keepdim=True) + eps)
    cos = gt_norm @ pr_norm.t()
    return 1.0 - cos


def _match_one_image(pred_mask, gt_segment, valid_mask, max_instances):
    """Hungarian match for one image.

    pred_mask:   [K, H, W] hard one-hot from argmaxed slot attention.
    gt_segment:  [H, W] int instance ids (0 = background).
    valid_mask:  [max_instances] bool -- which property rows are real GT objects.
    Returns: matched_slot_idx_per_gt -- [max_instances] long, -1 where invalid.
    """
    K, H, W = pred_mask.shape
    matched = -np.ones(max_instances, dtype=np.int64)

    # Build GT one-hot over the padded instance index space (id i -> row i-1).
    valid_idx = np.where(valid_mask.cpu().numpy())[0]
    if valid_idx.size == 0:
        return matched
    gt_np = gt_segment.cpu().numpy()
    # MOVi instance ids in the segment PNG are 1-indexed.
    gt_one_hot = np.zeros((valid_idx.size, H * W), dtype=np.float32)
    for row, inst_idx in enumerate(valid_idx):
        gt_one_hot[row] = (gt_np == (inst_idx + 1)).reshape(-1).astype(np.float32)

    pred_one_hot = pred_mask.reshape(K, H * W).float().cpu().numpy()

    # Drop GT rows with no pixels in the cropped frame -- nothing to match.
    has_pixels = gt_one_hot.sum(axis=1) > 0
    if not has_pixels.any():
        return matched
    gt_kept = gt_one_hot[has_pixels]
    valid_kept = valid_idx[has_pixels]

    cost = _mask_cosine_cost(
        torch.from_numpy(gt_kept), torch.from_numpy(pred_one_hot)
    ).numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind, strict=False):
        matched[valid_kept[r]] = c
    return matched


# ---------------------------------------------------------------------------
# Cache build: forward encoder over split, save matched (slot, property) pairs
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_cache(args, device, dtype, cache_path):
    encoder = load_latent_encoder(args.ckpt_path)
    if not isinstance(encoder, SLOT_ATTN_ENCODER_CLASSES):
        allowed = ", ".join(c.__name__ for c in SLOT_ATTN_ENCODER_CLASSES)
        raise SystemExit(
            f"probe_movi requires a slot-attention encoder ({allowed}); "
            f"got {type(encoder).__name__}."
        )
    encoder = encoder.to(device=device, dtype=dtype).eval()
    encoder.requires_grad_(False)
    print(f"loaded {type(encoder).__name__} with {encoder.num_components} slots")

    dataset = MoviPairDataset(
        root=args.dataset_root,
        split=args.split,
        img_size=args.resolution,
        max_images=args.max_images,
        load_properties=True,
        max_instances=args.max_instances,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"extracting slots over {len(dataset)} frames in {len(loader)} batches")

    matched_slots = []
    matched_category = []
    matched_position = []
    matched_bbox3d = []

    K = encoder.num_components
    for batch_idx, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
        gt_segment = batch["segment"].to(device=device)  # [B, H, W]
        slots_tokens, attn = encoder(pixel_values, return_attn=True)
        # Drop register slots; only K object slots are matched / probed.
        slots = slots_tokens[:, :K].float()
        attn_up = F.interpolate(
            attn.float(),
            size=(args.resolution, args.resolution),
            mode="bilinear",
            align_corners=False,
        )
        pred_mask_int = attn_up.argmax(dim=1)  # [B, H, W]
        # Hard one-hot per slot.
        pred_one_hot = F.one_hot(pred_mask_int, num_classes=K).permute(0, 3, 1, 2)

        for b in range(pixel_values.shape[0]):
            matched = _match_one_image(
                pred_one_hot[b], gt_segment[b], batch["valid"][b], args.max_instances
            )
            valid_pairs = matched >= 0
            if not valid_pairs.any():
                continue
            slot_idx = matched[valid_pairs]
            matched_slots.append(slots[b, slot_idx].cpu())
            matched_category.append(batch["category"][b][valid_pairs])
            matched_position.append(batch["image_positions"][b][valid_pairs])
            matched_bbox3d.append(batch["bboxes_3d"][b][valid_pairs])

        if (batch_idx + 1) % 20 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)}")

    cache = {
        "slots": torch.cat(matched_slots, dim=0),
        "category": torch.cat(matched_category, dim=0),
        "image_positions": torch.cat(matched_position, dim=0),
        "bboxes_3d": torch.cat(matched_bbox3d, dim=0),
        "slot_dim": int(slots.shape[-1]),
        "num_slots": int(K),
    }
    torch.save(cache, cache_path)
    print(f"wrote {cache_path} with {cache['slots'].shape[0]} matched pairs")
    return cache


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


def _train_continuous_probe(
    slots_train, y_train, slots_test, y_test, hidden_dim, steps, lr, batch_size, device
):
    D = slots_train.shape[1]
    out_dim = y_train.shape[1] if y_train.dim() > 1 else 1
    head = nn.Sequential(
        nn.Linear(D, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    n = slots_train.shape[0]
    slots_train = slots_train.to(device)
    y_train = y_train.to(device).float()
    slots_test = slots_test.to(device)
    y_test = y_test.to(device).float()
    for _step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        pred = head(slots_train[idx])
        loss = F.mse_loss(pred, y_train[idx].reshape(idx.shape[0], -1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pred_test = head(slots_test)
        test_mse = F.mse_loss(pred_test, y_test.reshape(y_test.shape[0], -1)).item()
    return float(test_mse)


def _train_discrete_probe(
    slots_train, y_train, slots_test, y_test, steps, lr, batch_size, device
):
    D = slots_train.shape[1]
    num_classes = int(max(y_train.max().item(), y_test.max().item())) + 1
    # sony/coda uses a single linear head for discrete properties.
    head = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    n = slots_train.shape[0]
    slots_train = slots_train.to(device)
    y_train = y_train.to(device).long()
    slots_test = slots_test.to(device)
    y_test = y_test.to(device).long()
    for _step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        logits = head(slots_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pred_test = head(slots_test).argmax(dim=-1)
        acc = (pred_test == y_test).float().mean().item()
    return float(acc)


def train_probes(cache, args, device):
    n = cache["slots"].shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_train = int(n * args.probe_train_frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    slots = cache["slots"]
    pos = cache["image_positions"]  # [N, 2]
    bbox = cache["bboxes_3d"].reshape(n, -1)  # [N, 24]
    cat = cache["category"]

    position_mse = _train_continuous_probe(
        slots[train_idx],
        pos[train_idx],
        slots[test_idx],
        pos[test_idx],
        args.probe_hidden_dim,
        args.probe_steps,
        args.probe_lr,
        args.probe_batch_size,
        device,
    )
    bbox3d_mse = _train_continuous_probe(
        slots[train_idx],
        bbox[train_idx],
        slots[test_idx],
        bbox[test_idx],
        args.probe_hidden_dim,
        args.probe_steps,
        args.probe_lr,
        args.probe_batch_size,
        device,
    )
    category_acc = _train_discrete_probe(
        slots[train_idx],
        cat[train_idx],
        slots[test_idx],
        cat[test_idx],
        args.probe_steps,
        args.probe_lr,
        args.probe_batch_size,
        device,
    )
    return {
        "position_mse": position_mse,
        "bbox3d_mse": bbox3d_mse,
        "category_acc": category_acc,
        "n_pairs_total": int(n),
        "n_pairs_train": int(n_train),
        "n_pairs_test": int(n - n_train),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.mixed_precision
    ]

    cache_path = os.path.join(args.output_dir, f"cache_{args.split}.pt")
    if os.path.exists(cache_path) and not args.rebuild_cache:
        print(f"loading cache from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
    else:
        cache = build_cache(args, device, dtype, cache_path)

    metrics = train_probes(cache, args, device)
    metrics.update(
        {
            "split": args.split,
            "resolution": args.resolution,
            "num_slots": cache["num_slots"],
            "slot_dim": cache["slot_dim"],
            "ckpt_path": args.ckpt_path,
            "probe_steps": args.probe_steps,
            "probe_hidden_dim": args.probe_hidden_dim,
        }
    )
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()

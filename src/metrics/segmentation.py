"""Per-image object-discovery metrics: FG-ARI, mBO, mIoU.

Operate on integer label maps (`np.ndarray`, any shape, equal between gt/pred).
Background is the id 0 by convention -- matches the MOVi-E dump.

References sony/coda (`src/metric/segmentation.py`) for the definitions:
- ``per_image_fg_ari``: foreground-only Adjusted Rand Index.
- ``per_image_mbo``: greedy per-GT max IoU (mean Best Overlap).
- ``per_image_miou``: Hungarian-matched per-pair mean IoU.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def per_image_fg_ari(gt, pred):
    """Foreground-only Adjusted Rand Index for one image.

    Returns None when the image has fewer than 2 foreground pixels or only a
    single GT cluster (ARI is undefined there).
    """
    mask = gt > 0
    if mask.sum() < 2:
        return None
    g = gt[mask]
    p = pred[mask]
    if np.unique(g).size < 2:
        return None
    return float(adjusted_rand_score(g, p))


def per_image_mbo(gt, pred):
    """Mean Best Overlap (per-GT-object greedy max IoU).

    For each GT instance (id > 0) take the maximum IoU over all predicted
    masks (including the slot covering background); return the mean. Returns
    None when the image has no foreground objects.
    """
    gt_ids = [int(i) for i in np.unique(gt) if i != 0]
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


def per_image_miou(gt, pred, ignore_background=False):
    """Hungarian-matched mean IoU between GT and predicted segmentations.

    Builds the per-pair IoU matrix over the ids actually present, solves the
    assignment problem (maximising IoU), and averages IoU over matched pairs.
    Unmatched GT ids contribute 0 to the mean (only when more GT than pred
    ids); same convention as sony/coda's ``hungarian_miou``.

    With ``ignore_background=True``, the GT id 0 is excluded from rows of the
    cost matrix (matches sony/coda's default for ``hungarian_miou``).
    """
    gt_ids = list(np.unique(gt))
    pred_ids = list(np.unique(pred))
    if ignore_background:
        gt_ids = [i for i in gt_ids if i != 0]
    if not gt_ids or not pred_ids:
        return None

    iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for i, gi in enumerate(gt_ids):
        gm = gt == gi
        for j, pi in enumerate(pred_ids):
            pm = pred == pi
            inter = np.logical_and(gm, pm).sum()
            if inter == 0:
                continue
            union = np.logical_or(gm, pm).sum()
            iou[i, j] = inter / union

    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    matched = iou[row_ind, col_ind]
    # Divide by the number of GT ids so unmatched GTs (when len(gt) > len(pred))
    # are counted as zero -- consistent with sony/coda's convention.
    return float(matched.sum() / len(gt_ids))
